"""Build a minimal MDL from a PostgreSQL schema."""

import asyncio
import json
from pathlib import Path
from typing import Any

import asyncpg

from configs import config
from env import secrets

TEXT_TYPES = {"character varying", "varchar", "text"}


async def fetch_tables(conn, schema: str, table_prefix: str) -> list[str]:
    """Fetch table names matching a prefix from information_schema."""
    rows = await conn.fetch(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = $1 AND table_type = 'BASE TABLE'
          AND table_name LIKE $2
        ORDER BY table_name
        """,
        schema,
        f"{table_prefix}%",
    )
    return [r["table_name"] for r in rows]


async def fetch_columns(conn, schema: str, table_prefix: str) -> list[dict[str, Any]]:
    """Fetch column metadata for matching tables."""
    rows = await conn.fetch(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = $1
          AND table_name LIKE $2
        ORDER BY table_name, ordinal_position
        """,
        schema,
        f"{table_prefix}%",
    )
    return [
        {
            "table_name": r["table_name"],
            "column_name": r["column_name"],
            "data_type": r["data_type"],
            "not_null": r["is_nullable"] == "NO",
        }
        for r in rows
    ]


async def fetch_primary_keys(conn, schema: str, table_prefix: str) -> dict[str, str]:
    """Fetch the first primary key column per table."""
    rows = await conn.fetch(
        """
        SELECT
          kcu.table_name,
          kcu.column_name,
          kcu.ordinal_position
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = $1
          AND tc.table_name LIKE $2
        ORDER BY kcu.table_name, kcu.ordinal_position
        """,
        schema,
        f"{table_prefix}%",
    )
    pk_map = {}
    for r in rows:
        if r["table_name"] not in pk_map:
            pk_map[r["table_name"]] = r["column_name"]  # only first PK column
    return pk_map


async def fetch_table_descriptions(
    conn, schema: str, table_prefix: str
) -> dict[str, str]:
    """Fetch table descriptions from PostgreSQL comments."""
    rows = await conn.fetch(
        """
        SELECT
          c.relname AS table_name,
          obj_description(c.oid, 'pg_class') AS table_description
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1
          AND c.relkind = 'r'
          AND c.relname LIKE $2
        """,
        schema,
        f"{table_prefix}%",
    )
    return {r["table_name"]: r["table_description"] or "" for r in rows}


async def fetch_column_descriptions(
    conn, schema: str, table_prefix: str
) -> dict[tuple[str, str], str]:
    """Fetch column descriptions from PostgreSQL comments."""
    rows = await conn.fetch(
        """
        SELECT
          c.relname AS table_name,
          a.attname AS column_name,
          col_description(c.oid, a.attnum) AS column_description
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = $1
          AND c.relkind = 'r'
          AND c.relname LIKE $2
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        schema,
        f"{table_prefix}%",
    )
    return {
        (r["table_name"], r["column_name"]): r["column_description"] or "" for r in rows
    }


async def fetch_relationships(
    conn, schema: str, table_prefix: str
) -> list[dict[str, Any]]:
    """Fetch foreign-key relationships for matching tables."""
    rows = await conn.fetch(
        """
        SELECT
          tc.table_name AS from_table,
          kcu.column_name AS from_column,
          ccu.table_name AS to_table,
          ccu.column_name AS to_column,
          tc.constraint_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = $1
          AND tc.table_name LIKE $2
          AND ccu.table_name LIKE $2
        """,
        schema,
        f"{table_prefix}%",
    )

    relationships = []
    for r in rows:
        relationships.append(
            {
                "name": r["constraint_name"],
                "models": [r["from_table"], r["to_table"]],
                "joinType": "MANY_TO_ONE",
                "condition": f"\"{r['from_table']}\".{r['from_column']} = \"{r['to_table']}\".{r['to_column']}",
            }
        )
    return relationships


def _quote_ident(name: str) -> str:
    """Quote an identifier for SQL usage."""
    return f"\"{name.replace('\"', '\"\"')}\""


async def _fetch_unique_values_for_column(
    pool: asyncpg.Pool,
    schema: str,
    table: str,
    column: str,
    sem: asyncio.Semaphore,
) -> tuple[tuple[str, str], list[Any]]:
    """Fetch distinct values for a single text column."""
    async with sem:
        table_ref = f"{_quote_ident(schema)}.{_quote_ident(table)}"
        column_ref = _quote_ident(column)
        sql = f"SELECT DISTINCT {column_ref} FROM {table_ref} WHERE {column_ref} IS NOT NULL"

        values: list[Any] = []
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql)
            values = [row[0] for row in rows]

        return (table, column), values


async def fetch_unique_values(
    pool: asyncpg.Pool,
    schema: str,
    columns: list[dict[str, Any]],
    concurrency: int,
) -> dict[tuple[str, str], list[Any]]:
    """Fetch distinct values for text columns with concurrency limits."""
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _fetch_unique_values_for_column(
            pool=pool,
            schema=schema,
            table=col["table_name"],
            column=col["column_name"],
            sem=sem,
        )
        for col in columns
        if col["data_type"].lower() in TEXT_TYPES
    ]
    results = await asyncio.gather(*tasks)
    return {key: values for key, values in results}


async def fetch_views(conn, schema: str, table_prefix: str) -> list[dict[str, Any]]:
    """Fetch view definitions for matching views."""
    rows = await conn.fetch(
        """
        SELECT table_name, view_definition
        FROM information_schema.views
        WHERE table_schema = $1
          AND table_name LIKE $2
        ORDER BY table_name
        """,
        schema,
        f"{table_prefix}%",
    )
    return [
        {
            "name": r["table_name"],
            "statement": r["view_definition"],
            "properties": {"description": f"View for {r['table_name']}"},
        }
        for r in rows
    ]


def build_mdl(
    schema: str,
    tables: list[str],
    columns: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    views: list[dict[str, Any]],
    table_descriptions: dict[str, str],
    column_descriptions: dict[tuple[str, str], str],
    primary_keys: dict[str, str],
    unique_values: dict[tuple[str, str], list[Any]],
):
    """Build an MDL dict from schema metadata."""
    models = []
    for table in tables:
        table_cols = [c for c in columns if c["table_name"] == table]
        mdl_cols = []
        for c in table_cols:
            desc = column_descriptions.get((table, c["column_name"]), "")
            uniques = unique_values.get((table, c["column_name"]), [])
            col = {
                "name": c["column_name"],
                "type": c["data_type"],
                "isCalculated": False,
                "notNull": c["not_null"],
            }
            props: dict[str, Any] = {}
            if desc:
                props["description"] = desc
            if uniques:
                props["uniqueValues"] = uniques
            if props:
                col["properties"] = props
            mdl_cols.append(col)

        model = {
            "name": table,
            "tableReference": {"schema": schema, "table": table},
            "columns": mdl_cols,
        }

        if table in primary_keys:
            model["primaryKey"] = primary_keys[table]

        table_desc = table_descriptions.get(table, "")
        if table_desc:
            model["properties"] = {"description": table_desc}

        models.append(model)

    mdl = {
        "$schema": "https://raw.githubusercontent.com/Canner/WrenAI/main/wren-mdl/mdl.schema.json",
        "catalog": "default",
        "schema": schema,
        "dataSource": "POSTGRES",
        "models": models,
        "relationships": relationships,
        "views": views,
        "metrics": [],
        "enumDefinitions": [],
    }
    return mdl


async def main():
    """Connect to Postgres and write an MDL JSON file."""
    db_cfg = config.database.remote_db
    password = secrets.REMOTE_POSTGRES_PASSWORD
    print(f"[build_mdl] connecting to {db_cfg.host}:{db_cfg.port}/{db_cfg.db}")

    pool = await asyncpg.create_pool(
        host=db_cfg.host,
        port=db_cfg.port,
        user=db_cfg.user,
        password=password,
        database=db_cfg.db,
        min_size=db_cfg.min_pool_size,
        max_size=db_cfg.max_pool_size,
        timeout=db_cfg.timeout,
    )

    schema = "public"
    table_prefix = "summary_"
    unique_concurrency = config.runtime.unique_values_concurrency
    print(f"[build_mdl] schema={schema} table_prefix={table_prefix}")
    print(f"[build_mdl] unique_values: concurrency={unique_concurrency}")

    async with pool.acquire() as conn:
        tables = await fetch_tables(conn, schema, table_prefix)
        print(f"[build_mdl] tables={len(tables)}")
        columns = await fetch_columns(conn, schema, table_prefix)
        print(f"[build_mdl] columns={len(columns)}")
        primary_keys = await fetch_primary_keys(conn, schema, table_prefix)
        print(f"[build_mdl] primary_keys={len(primary_keys)}")
        table_descriptions = await fetch_table_descriptions(conn, schema, table_prefix)
        print(f"[build_mdl] table_descriptions={len(table_descriptions)}")
        column_descriptions = await fetch_column_descriptions(
            conn, schema, table_prefix
        )
        print(f"[build_mdl] column_descriptions={len(column_descriptions)}")
        relationships = await fetch_relationships(conn, schema, table_prefix)
        print(f"[build_mdl] relationships={len(relationships)}")
        views = await fetch_views(conn, schema, table_prefix)
        print(f"[build_mdl] views={len(views)}")

    unique_values = await fetch_unique_values(
        pool=pool,
        schema=schema,
        columns=columns,
        concurrency=unique_concurrency,
    )
    print(f"[build_mdl] unique_values={len(unique_values)}")

    await pool.close()

    mdl = build_mdl(
        schema,
        tables,
        columns,
        relationships,
        views,
        table_descriptions,
        column_descriptions,
        primary_keys,
        unique_values,
    )
    output_path = Path(__file__).parent / "mdl.json"
    output_path.write_text(json.dumps(mdl, indent=2), encoding="utf-8")
    print(f"[build_mdl] MDL saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
