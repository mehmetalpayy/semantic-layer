"""PGVector-backed document store and retrieval helpers."""

import json
import uuid
from typing import Any

import asyncpg
from haystack import Document
from haystack.document_stores.types import DuplicatePolicy

from configs import config
from env import secrets
from .embedder_litellm import EMBEDDING_DIM

_db_cfg = config.database.local_db
PG_HOST = _db_cfg.host
PG_PORT = _db_cfg.port
PG_USER = _db_cfg.user
PG_PASSWORD = secrets.LOCAL_POSTGRES_PASSWORD
PG_DATABASE = _db_cfg.db


async def _ensure_table(conn: asyncpg.Connection, table_name: str) -> None:
    """Ensure the pgvector extension and table exist."""
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id UUID PRIMARY KEY,
            content TEXT,
            meta JSONB,
            embedding VECTOR({EMBEDDING_DIM})
        );
        """
    )


def _build_filter_sql(
    filters: dict[str, Any] | None,
    params: list[Any],
    param_idx: int,
) -> tuple[str, list[Any], int]:
    """Build SQL WHERE clause for document filters."""
    if not filters:
        return "", params, param_idx

    if "conditions" in filters:
        operator = filters.get("operator", "AND").upper()
        parts = []
        for condition in filters["conditions"]:
            clause, params, param_idx = _build_filter_sql(
                condition, params, param_idx
            )
            if clause:
                parts.append(clause)
        if not parts:
            return "", params, param_idx
        joined = f" {operator} ".join(parts)
        return f"({joined})", params, param_idx

    field = filters.get("field")
    operator = filters.get("operator")
    value = filters.get("value")
    if not field or not operator:
        return "", params, param_idx

    column = field if field in {"id", "content"} else f"meta->>'{field}'"
    if operator == "==":
        params.append(str(value))
        clause = f"{column} = ${param_idx}"
        return clause, params, param_idx + 1
    if operator == "in":
        params.append([str(v) for v in value or []])
        clause = f"{column} = ANY(${param_idx})"
        return clause, params, param_idx + 1

    return "", params, param_idx


class PGVectorDocumentStore:
    """Minimal document store for PGVector tables."""

    def __init__(self, table_name: str) -> None:
        """Set the target table name."""
        self.table_name = table_name

    def to_dict(self) -> dict[str, Any]:
        """Return init parameters for logging/debugging."""
        return {"init_parameters": {"index": self.table_name}}

    async def write_documents(
        self,
        documents: list[Document],
        policy: DuplicatePolicy | None = DuplicatePolicy.OVERWRITE,
    ) -> int:
        """Write documents to the table, optionally overwriting duplicates."""
        if not documents:
            return 0

        conn = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
        )
        try:
            await _ensure_table(conn, self.table_name)

            rows = []
            for doc in documents:
                doc_id = doc.id or str(uuid.uuid4())
                emb = doc.embedding or []
                emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                rows.append(
                    (
                        doc_id,
                        doc.content,
                        json.dumps(doc.meta or {}),
                        emb_str,
                    )
                )

            await conn.executemany(
                f"""
                INSERT INTO {self.table_name} (id, content, meta, embedding)
                VALUES ($1, $2, $3, $4::vector)
                ON CONFLICT (id) DO UPDATE SET
                  content = EXCLUDED.content,
                  meta = EXCLUDED.meta,
                  embedding = EXCLUDED.embedding
                """,
                rows,
            )
            return len(rows)
        finally:
            await conn.close()

    async def delete_documents(self, filters: dict[str, Any] | None = None) -> None:
        """Delete documents matching optional filters."""
        conn = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
        )
        try:
            await _ensure_table(conn, self.table_name)
            params: list[Any] = []
            where_clause, params, _ = _build_filter_sql(filters, params, 1)
            sql = f"DELETE FROM {self.table_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            await conn.execute(sql, *params)
        finally:
            await conn.close()

    async def query_by_embedding(
        self,
        query_embedding: list[float],
        filters: dict[str, Any] | None = None,
        top_k: int = 10,
    ) -> list[Document]:
        """Return top-k documents by vector similarity."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        conn = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
        )
        try:
            params: list[Any] = [embedding_str]
            where_clause, params, next_idx = _build_filter_sql(filters, params, 2)
            sql = f"""
                SELECT id, content, meta,
                       embedding <=> $1::vector AS distance
                FROM {self.table_name}
            """
            if where_clause:
                sql += f" WHERE {where_clause}"
            sql += f" ORDER BY embedding <=> $1::vector LIMIT ${next_idx}"
            params.append(top_k)
            rows = await conn.fetch(sql, *params)
            docs = []
            for row in rows:
                distance = row["distance"]
                score = 1 / (1 + distance) if distance is not None else None
                docs.append(
                    Document(
                        id=row["id"],
                        content=row["content"],
                        meta=row["meta"],
                        score=score,
                    )
                )
            return docs
        finally:
            await conn.close()

    async def query_by_filters(
        self,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Return documents matching filter-only queries."""
        conn = await asyncpg.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE,
        )
        try:
            params: list[Any] = []
            where_clause, params, _ = _build_filter_sql(filters, params, 1)
            sql = f"SELECT id, content, meta FROM {self.table_name}"
            if where_clause:
                sql += f" WHERE {where_clause}"
            if top_k is not None:
                sql += f" LIMIT {top_k}"
            rows = await conn.fetch(sql, *params)
            return [
                Document(id=row["id"], content=row["content"], meta=row["meta"])
                for row in rows
            ]
        finally:
            await conn.close()


class PGVectorEmbeddingRetriever:
    """Retriever that queries the document store by embedding."""

    def __init__(self, document_store: PGVectorDocumentStore, top_k: int = 10) -> None:
        """Store the document store and default top-k."""
        self.document_store = document_store
        self.top_k = top_k

    async def run(
        self, query_embedding: list[float], filters: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fetch documents using embedding or filter-only retrieval."""
        if query_embedding:
            documents = await self.document_store.query_by_embedding(
                query_embedding=query_embedding,
                filters=filters,
                top_k=self.top_k,
            )
        else:
            documents = await self.document_store.query_by_filters(
                filters=filters, top_k=self.top_k
            )
        return {"documents": documents}


async def write_documents(table_name: str, documents: list[Document]) -> int:
    """Convenience helper to write documents into a table."""
    store = PGVectorDocumentStore(table_name=table_name)
    return await store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)
