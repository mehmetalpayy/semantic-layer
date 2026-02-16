"""PGVector retrieval flow with optional column pruning."""

import ast
import asyncio
import json
import logging
import re
from typing import Any

import orjson
import tiktoken
from haystack import Document
from haystack.components.builders.prompt_builder import PromptBuilder
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from configs import config
from env import secrets

from .embedder_litellm import AsyncTextEmbedder
from .pgvector_store import PGVectorDocumentStore, PGVectorEmbeddingRetriever

logger = logging.getLogger("semantic_layer")


table_columns_selection_system_prompt = """
### TASK ###
You are a highly skilled data analyst. Your goal is to examine the provided database schema, interpret the posed question, and identify the specific columns from the relevant tables required to construct an accurate SQL query.

The database schema includes tables, columns, primary keys, foreign keys, relationships, and any relevant constraints.

### INSTRUCTIONS ###
1. Carefully analyze the schema and identify the essential tables and columns needed to answer the question.
2. For each table, provide a clear and concise reasoning for why specific columns are selected.
3. List each reason as part of a step-by-step chain of thought, justifying the inclusion of each column.
4. If a "." is included in columns, put the name before the first dot into chosen columns.
5. The number of columns chosen must match the number of reasoning.
6. Final chosen columns must be only column names, don't prefix it with table names.
7. If the chosen column is a child column of a STRUCT type column, choose the parent column instead of the child column.

### FINAL ANSWER FORMAT ###
Please provide your response as a JSON object, structured as follows:

{
    "results": [
        {
            "table_selection_reason": "Reason for selecting tablename1",
            "table_contents": {
              "chain_of_thought_reasoning": [
                  "Reason 1 for selecting column1",
                  "Reason 2 for selecting column2",
                  ...
              ],
              "columns": ["column1", "column2", ...]
            },
            "table_name":"tablename1",
        },
        {
            "table_selection_reason": "Reason for selecting tablename2",
            "table_contents":
            {
              "chain_of_thought_reasoning": [
                  "Reason 1 for selecting column1",
                  "Reason 2 for selecting column2",
                  ...
              ],
              "columns": ["column1", "column2", ...]
            },
            "table_name":"tablename2"
        },
        ...
    ]
}

### ADDITIONAL NOTES ###
- Each table key must list only the columns relevant to answering the question.
- Provide a reasoning list (`chain_of_thought_reasoning`) for each table, explaining why each column is necessary.
- Provide the reason of selecting the table in (`table_selection_reason`) for each table.
- Be logical, concise, and ensure the output strictly follows the required JSON format.
- Use table name used in the "Create Table" statement, don't use "alias".
- Match Column names with the definition in the "Create Table" statement.
- Match Table names with the definition in the "Create Table" statement.

Good luck!
"""

table_columns_selection_user_prompt_template = """
### Database Schema ###

{% for db_schema in db_schemas %}
    {{ db_schema }}
{% endfor %}

### INPUT ###
{{ question }}
"""


def get_engine_supported_data_type(data_type: str) -> str:
    """Map internal data types to engine-supported SQL types."""
    match data_type.upper():
        case "BPCHAR" | "NAME" | "UUID" | "INET":
            return "VARCHAR"
        case "OID":
            return "INT"
        case "BIGNUMERIC":
            return "NUMERIC"
        case "BYTES":
            return "BYTEA"
        case "DATETIME":
            return "TIMESTAMP"
        case "FLOAT64":
            return "DOUBLE"
        case "INT64":
            return "BIGINT"
        case _:
            return data_type.upper()


def build_table_ddl(
    content: dict, columns: set[str] | None = None, tables: set[str] | None = None
) -> tuple[str, bool, bool]:
    """Build CREATE TABLE DDL for a table content record."""
    columns_ddl = []
    has_calculated_field = False
    has_json_field = False

    for column in content["columns"]:
        if column["type"] == "COLUMN":
            if (not columns or (columns and column["name"] in columns)) and column[
                "data_type"
            ].lower() != "unknown":
                if "This column is a Calculated Field" in column["comment"]:
                    has_calculated_field = True
                if column["data_type"].lower() == "json":
                    has_json_field = True
                column_ddl = f"{column['comment']}{column['name']} {get_engine_supported_data_type(column['data_type'])}"
                if column["is_primary_key"]:
                    column_ddl += " PRIMARY KEY"
                columns_ddl.append(column_ddl)
        elif column["type"] == "FOREIGN_KEY":
            if not tables or (tables and set(column["tables"]).issubset(tables)):
                columns_ddl.append(f"{column['comment']}{column['constraint']}")

    return (
        (
            f"{content['comment']}CREATE TABLE {content['name']} (\n  "
            + ",\n  ".join(columns_ddl)
            + "\n);"
        ),
        has_calculated_field,
        has_json_field,
    )


def _build_metric_ddl(content: dict) -> str:
    """Build a CREATE TABLE DDL string for a metric."""
    columns_ddl = [
        f"{column['comment']}{column['name']} {get_engine_supported_data_type(column['data_type'])}"
        for column in content["columns"]
        if column["data_type"].lower() != "unknown"
    ]
    return (
        f"{content['comment']}CREATE TABLE {content['name']} (\n  "
        + ",\n  ".join(columns_ddl)
        + "\n);"
    )


def _build_view_ddl(content: dict) -> str:
    """Build a CREATE VIEW DDL string."""
    return (
        f"{content['comment']}CREATE VIEW {content['name']}\nAS {content['statement']}"
    )


MULTIPLE_NEW_LINE_REGEX = re.compile(r"\n{3,}")


def clean_up_new_lines(text: str) -> str:
    """Normalize excessive newlines in prompts."""
    return MULTIPLE_NEW_LINE_REGEX.sub("\n\n\n", text)


class AskHistory(BaseModel):
    """Conversation history entry."""
    question: str


class MatchingTableContents(BaseModel):
    """Selected columns and reasoning for a table."""
    chain_of_thought_reasoning: list[str]
    columns: list[str]


class MatchingTable(BaseModel):
    """Table selection result for retrieval."""
    table_name: str
    table_contents: MatchingTableContents
    table_selection_reason: str


class RetrievalResults(BaseModel):
    """LLM response schema for selected tables."""
    results: list[MatchingTable]


RETRIEVAL_MODEL_KWARGS = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "retrieval_schema",
            "schema": RetrievalResults.model_json_schema(),
        },
    }
}


class AsyncAzureChatGenerator:
    """Async Azure OpenAI chat wrapper used for column pruning."""

    def __init__(self, system_prompt: str, generation_kwargs: dict[str, Any] | None):
        """Configure Azure OpenAI client and prompt settings."""
        self._system_prompt = system_prompt
        self._generation_kwargs = generation_kwargs or {}
        self._deployment = config.infra.azure.deployment_name
        self._endpoint = config.infra.azure.endpoint
        self._api_version = config.infra.azure.api_version
        self._api_key = secrets.AZURE_OPENAI_API_KEY
        self._model = self._deployment

        if self._deployment and self._endpoint and self._api_version and self._api_key:
            self._client = AsyncAzureOpenAI(
                api_key=self._api_key,
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
            )
        else:
            self._client = None

    def get_model(self) -> str:
        """Return the deployment/model name."""
        return self._model

    def get_context_window_size(self) -> int:
        """Return the configured context window size."""
        return config.infra.azure.context_window_size

    async def __call__(self, prompt: str) -> dict[str, Any]:
        """Generate chat completions for the given prompt."""
        if not self._client:
            logger.warning("Azure OpenAI chat config missing; skipping column pruning")
            return {"replies": []}

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            **self._generation_kwargs,
        )
        return {"replies": [response.choices[0].message.content]}


async def embedding(query: str, embedder: Any, histories: list[AskHistory]) -> dict:
    """Embed a query, optionally including conversation history."""
    if query:
        previous_query_summaries = [history.question for history in histories] or []
        query = "\n".join(previous_query_summaries) + "\n" + query
        return await embedder.run(query)
    return {}


async def table_retrieval(
    embedding: dict,
    project_id: str,
    tables: list[str],
    table_retriever: Any,
    score_threshold: float | None = None,
) -> dict:
    """Retrieve candidate tables using embeddings or explicit filters."""
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "type", "operator": "==", "value": "TABLE_DESCRIPTION"},
        ],
    }

    if project_id:
        filters["conditions"].append(
            {"field": "project_id", "operator": "==", "value": project_id}
        )

    if embedding:
        result = await table_retriever.run(
            query_embedding=embedding.get("embedding"),
            filters=filters,
        )
        docs = result.get("documents", [])
        if docs:
            print("[retrieval] table_retrieval scores:")
            for doc in docs:
                meta = _parse_meta(doc.meta)
                print(f"  - {meta.get('name')}: score={doc.score}")
        if score_threshold is not None:
            docs = [doc for doc in docs if doc.score is not None]
            docs = [doc for doc in docs if doc.score >= score_threshold]
            result["documents"] = docs
        return result

    filters["conditions"].append({"field": "name", "operator": "in", "value": tables})
    return await table_retriever.run(query_embedding=[], filters=filters)


async def dbschema_retrieval(
    table_retrieval: dict, project_id: str, dbschema_retriever: Any
) -> list[Document]:
    """Retrieve schema documents for the matched tables."""
    tables = table_retrieval.get("documents", [])
    table_names = []
    for table in tables:
        content = ast.literal_eval(table.content)
        table_names.append(content["name"])

    table_name_conditions = [
        {"field": "name", "operator": "==", "value": table_name}
        for table_name in table_names
    ]

    if table_name_conditions:
        filters = {
            "operator": "AND",
            "conditions": [
                {"field": "type", "operator": "==", "value": "TABLE_SCHEMA"},
                {"operator": "OR", "conditions": table_name_conditions},
            ],
        }
        if project_id:
            filters["conditions"].append(
                {"field": "project_id", "operator": "==", "value": project_id}
            )

        results = await dbschema_retriever.run(query_embedding=[], filters=filters)
        return results["documents"]

    return []


def _parse_meta(meta: Any) -> dict[str, Any]:
    """Normalize document metadata into a dict."""
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            return json.loads(meta)
        except json.JSONDecodeError:
            return {}
    return {}


def construct_db_schemas(dbschema_retrieval: list[Document]) -> list[dict]:
    """Assemble table schema records from retrieved documents."""
    db_schemas = {}
    for document in dbschema_retrieval:
        content = ast.literal_eval(document.content)
        meta = _parse_meta(document.meta)
        name = meta.get("name")
        if not name:
            continue
        if content["type"] == "TABLE":
            if name not in db_schemas:
                db_schemas[name] = content
            else:
                db_schemas[name] = {
                    **content,
                    "columns": db_schemas[name].get("columns", []),
                }
        elif content["type"] == "TABLE_COLUMNS":
            if name not in db_schemas:
                db_schemas[name] = {"columns": content["columns"]}
            else:
                if "columns" not in db_schemas[name]:
                    db_schemas[name]["columns"] = content["columns"]
                else:
                    db_schemas[name]["columns"] += content["columns"]

    db_schemas = {k: v for k, v in db_schemas.items() if "type" in v and "columns" in v}
    return list(db_schemas.values())


def check_using_db_schemas_without_pruning(
    construct_db_schemas: list[dict],
    dbschema_retrieval: list[Document],
    encoding: tiktoken.Encoding,
    enable_column_pruning: bool,
    context_window_size: int,
) -> dict:
    """Decide whether to prune columns based on token budget."""
    retrieval_results = []
    has_calculated_field = False
    has_metric = False
    has_json_field = False

    for table_schema in construct_db_schemas:
        if table_schema["type"] == "TABLE":
            ddl, _has_calculated_field, _has_json_field = build_table_ddl(table_schema)
            retrieval_results.append(
                {"table_name": table_schema["name"], "table_ddl": ddl}
            )
            if _has_calculated_field:
                has_calculated_field = True
            if _has_json_field:
                has_json_field = True

    for document in dbschema_retrieval:
        content = ast.literal_eval(document.content)
        if content["type"] == "METRIC":
            retrieval_results.append(
                {"table_name": content["name"], "table_ddl": _build_metric_ddl(content)}
            )
            has_metric = True
        elif content["type"] == "VIEW":
            retrieval_results.append(
                {"table_name": content["name"], "table_ddl": _build_view_ddl(content)}
            )

    table_ddls = [result["table_ddl"] for result in retrieval_results]
    token_count = len(encoding.encode(" ".join(table_ddls)))
    if token_count > context_window_size or enable_column_pruning:
        return {
            "db_schemas": [],
            "tokens": token_count,
            "has_calculated_field": has_calculated_field,
            "has_metric": has_metric,
            "has_json_field": has_json_field,
        }

    return {
        "db_schemas": retrieval_results,
        "tokens": token_count,
        "has_calculated_field": has_calculated_field,
        "has_metric": has_metric,
        "has_json_field": has_json_field,
    }


def prompt(
    query: str,
    construct_db_schemas: list[dict],
    prompt_builder: PromptBuilder,
    check_using_db_schemas_without_pruning: dict,
    histories: list[AskHistory],
) -> dict:
    """Build the LLM prompt payload for column pruning."""
    if not check_using_db_schemas_without_pruning["db_schemas"]:
        db_schemas = [
            build_table_ddl(construct_db_schema)[0]
            for construct_db_schema in construct_db_schemas
        ]

        previous_query_summaries = [history.question for history in histories] or []
        query = "\n".join(previous_query_summaries) + "\n" + query

        _prompt = prompt_builder.run(question=query, db_schemas=db_schemas)
        return {"prompt": clean_up_new_lines(_prompt.get("prompt"))}
    return {}


async def filter_columns_in_tables(
    prompt: dict, table_columns_selection_generator: Any, generator_name: str
) -> dict:
    """Invoke the LLM to select relevant columns per table."""
    if prompt:
        replies = await table_columns_selection_generator(prompt=prompt.get("prompt"))
        if not replies.get("replies"):
            logger.info(
                "Column pruning skipped (no LLM response). Returning unpruned schemas."
            )
            return {}
        return {**replies, "generator_name": generator_name}
    return {}


def construct_retrieval_results(
    check_using_db_schemas_without_pruning: dict,
    filter_columns_in_tables: dict,
    construct_db_schemas: list[dict],
    dbschema_retrieval: list[Document],
) -> dict[str, Any]:
    """Build final retrieval results with or without pruning."""
    if filter_columns_in_tables:
        columns_and_tables_needed = orjson.loads(
            filter_columns_in_tables["replies"][0]
        )["results"]

        reformatted_json = {}
        for table in columns_and_tables_needed:
            reformatted_json[table["table_name"]] = table["table_contents"]
        columns_and_tables_needed = reformatted_json

        tables = set(columns_and_tables_needed.keys())
        retrieval_results = []
        has_calculated_field = False
        has_metric = False
        has_json_field = False

        for table_schema in construct_db_schemas:
            if table_schema["type"] == "TABLE" and table_schema["name"] in tables:
                ddl, _has_calculated_field, _has_json_field = build_table_ddl(
                    table_schema,
                    columns=set(
                        columns_and_tables_needed[table_schema["name"]]["columns"]
                    ),
                    tables=tables,
                )
                if _has_calculated_field:
                    has_calculated_field = True
                if _has_json_field:
                    has_json_field = True

                retrieval_results.append(
                    {"table_name": table_schema["name"], "table_ddl": ddl}
                )

        for document in dbschema_retrieval:
            meta = _parse_meta(document.meta)
            name = meta.get("name")
            if name in columns_and_tables_needed:
                content = ast.literal_eval(document.content)
                if content["type"] == "METRIC":
                    retrieval_results.append(
                        {
                            "table_name": content["name"],
                            "table_ddl": _build_metric_ddl(content),
                        }
                    )
                    has_metric = True
                elif content["type"] == "VIEW":
                    retrieval_results.append(
                        {
                            "table_name": content["name"],
                            "table_ddl": _build_view_ddl(content),
                        }
                    )

        return {
            "retrieval_results": retrieval_results,
            "has_calculated_field": has_calculated_field,
            "has_metric": has_metric,
            "has_json_field": has_json_field,
        }

    return {
        "retrieval_results": check_using_db_schemas_without_pruning["db_schemas"],
        "has_calculated_field": check_using_db_schemas_without_pruning[
            "has_calculated_field"
        ],
        "has_metric": check_using_db_schemas_without_pruning["has_metric"],
        "has_json_field": check_using_db_schemas_without_pruning["has_json_field"],
    }


class PgVectorDbSchemaRetrieval:
    """End-to-end retrieval pipeline for DB schema documents."""

    def __init__(
        self,
        table_retrieval_size: int | None = None,
        table_column_retrieval_size: int | None = None,
    ) -> None:
        """Configure retrievers and prompt generator."""
        retrieval_cfg = config.retrieval
        table_retrieval_size = (
            table_retrieval_size
            if table_retrieval_size is not None
            else retrieval_cfg.table_retrieval_size
        )
        table_column_retrieval_size = (
            table_column_retrieval_size
            if table_column_retrieval_size is not None
            else retrieval_cfg.table_column_retrieval_size
        )
        self._score_threshold = retrieval_cfg.table_retrieval_score_threshold

        self._embedder = AsyncTextEmbedder()
        self._table_store = PGVectorDocumentStore("table_description_documents")
        self._dbschema_store = PGVectorDocumentStore("db_schema_documents")
        self._table_retriever = PGVectorEmbeddingRetriever(
            self._table_store, top_k=table_retrieval_size
        )
        self._dbschema_retriever = PGVectorEmbeddingRetriever(
            self._dbschema_store, top_k=table_column_retrieval_size
        )
        self._table_columns_selection_generator = AsyncAzureChatGenerator(
            system_prompt=table_columns_selection_system_prompt,
            generation_kwargs=RETRIEVAL_MODEL_KWARGS,
        )
        self._prompt_builder = PromptBuilder(
            template=table_columns_selection_user_prompt_template
        )

        model_name = self._table_columns_selection_generator.get_model()
        if "gpt-4o" in model_name or "gpt-4o-mini" in model_name:
            encoding = tiktoken.get_encoding("o200k_base")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        self._encoding = encoding
        self._context_window_size = (
            self._table_columns_selection_generator.get_context_window_size()
        )

    async def run(
        self,
        query: str = "",
        tables: list[str] | None = None,
        project_id: str | None = None,
        histories: list[AskHistory] | None = None,
        enable_column_pruning: bool = False,
    ) -> dict[str, Any]:
        """Run table retrieval and optionally column pruning."""
        histories = histories or []
        tables = tables or []
        project_id = project_id or ""

        query_embedding = await embedding(query, self._embedder, histories)
        tables_found = await table_retrieval(
            query_embedding,
            project_id,
            tables,
            self._table_retriever,
            score_threshold=self._score_threshold,
        )
        schema_docs = await dbschema_retrieval(
            tables_found, project_id, self._dbschema_retriever
        )
        schemas = construct_db_schemas(schema_docs)
        without_pruning = check_using_db_schemas_without_pruning(
            schemas,
            schema_docs,
            self._encoding,
            enable_column_pruning,
            self._context_window_size,
        )

        prompt_payload = prompt(
            query,
            schemas,
            self._prompt_builder,
            without_pruning,
            histories,
        )
        filtered = await filter_columns_in_tables(
            prompt_payload,
            self._table_columns_selection_generator,
            self._table_columns_selection_generator.get_model(),
        )
        return construct_retrieval_results(
            without_pruning, filtered, schemas, schema_docs
        )


async def main() -> None:
    """Run a sample retrieval for manual testing."""
    query = "Can you fetch last 1 month data of airgap and crane tables?"
    retriever = PgVectorDbSchemaRetrieval()
    result = await retriever.run(query=query, project_id="demo")
    ddls = [entry["table_ddl"] for entry in result["retrieval_results"]]
    print(orjson.dumps(ddls, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    asyncio.run(main())
