import asyncio
import logging
import sys
import uuid
from typing import Any

from hamilton import base
from hamilton.async_driver import AsyncDriver
from hamilton.function_modifiers import extract_fields
from haystack import Document, component
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from tqdm import tqdm

from . import helper as helper
from .embedder_litellm import AsyncDocumentEmbedder
from .mdl_validator import AsyncDocumentWriter, DocumentCleaner, MDLValidator
from .pgvector_store import PGVectorDocumentStore
from .pipeline import BasicPipeline

logger = logging.getLogger("semantic_layer")


@component
class DDLChunker:
    @component.output_types(documents=list[Document])
    async def run(
        self,
        mdl: dict[str, Any],
        column_batch_size: int,
        project_id: str | None = None,
    ):
        def _additional_meta() -> dict[str, Any]:
            return {"project_id": project_id} if project_id else {}

        chunks = [
            {
                "id": str(uuid.uuid4()),
                "meta": {
                    "type": "TABLE_SCHEMA",
                    "name": chunk["name"],
                    **_additional_meta(),
                },
                "content": chunk["payload"],
            }
            for chunk in await self._get_ddl_commands(
                **mdl, column_batch_size=column_batch_size
            )
        ]

        return {
            "documents": [
                Document(**chunk)
                for chunk in tqdm(
                    chunks,
                    desc=f"Project ID: {project_id}, Chunking DDL commands into documents",
                )
            ]
        }

    async def _model_preprocessor(
        self, models: list[dict[str, Any]], **kwargs
    ) -> list[dict[str, Any]]:
        def _column_preprocessor(
            column: dict[str, Any], addition: dict[str, Any]
        ) -> dict[str, Any]:
            addition = {
                key: helper(column, **addition)
                for key, helper in helper.COLUMN_PREPROCESSORS.items()
                if helper.condition(column, **addition)
            }

            return {
                "name": column.get("name", ""),
                "type": column.get("type", ""),
                **addition,
            }

        async def _preprocessor(model: dict[str, Any], **kwargs) -> dict[str, Any]:
            addition = {
                key: await helper(model, **kwargs)
                for key, helper in helper.MODEL_PREPROCESSORS.items()
                if helper.condition(model, **kwargs)
            }

            columns = [
                _column_preprocessor(column, addition)
                for column in model.get("columns", [])
                if column.get("isHidden") is not True
            ]
            return {
                "name": model.get("name", ""),
                "properties": model.get("properties", {}),
                "columns": columns,
                "primaryKey": model.get("primaryKey", ""),
            }

        tasks = [_preprocessor(model, **kwargs) for model in models]
        return await asyncio.gather(*tasks)

    async def _get_ddl_commands(
        self,
        models: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        views: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        column_batch_size: int = 50,
        **kwargs,
    ) -> list[dict]:
        return (
            self._convert_models_and_relationships(
                await self._model_preprocessor(models, **kwargs),
                relationships,
                column_batch_size,
            )
            + self._convert_views(views)
            + self._convert_metrics(metrics)
        )

    def _convert_models_and_relationships(
        self,
        models: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        column_batch_size: int,
    ) -> list[dict[str, str]]:
        def _model_command(model: dict[str, Any]) -> dict:
            properties = model.get("properties", {})
            model_properties = {
                "alias": helper.clean_display_name(properties.get("displayName", "")),
                "description": properties.get("description", ""),
            }
            comment = f"\n/* {str(model_properties)} */\n"
            table_name = model["name"]
            payload = {"type": "TABLE", "comment": comment, "name": table_name}
            return {"name": table_name, "payload": str(payload)}

        def _column_command(column: dict[str, Any], model: dict[str, Any]) -> dict:
            if column.get("relationship"):
                return None

            comments = [
                helper(column, model=model)
                for helper in helper.COLUMN_COMMENT_HELPERS.values()
                if helper.condition(column)
            ]

            return {
                "type": "COLUMN",
                "comment": "".join(comments),
                "name": column["name"],
                "data_type": column["type"],
                "is_primary_key": column["name"] == model["primaryKey"],
            }

        def _relationship_command(
            relationship: dict[str, Any],
            table_name: str,
            primary_keys_map: dict[str, str],
        ) -> dict:
            condition = relationship.get("condition", "")
            join_type = relationship.get("joinType", "")
            models = relationship.get("models", [])

            if len(models) != 2:
                return None
            if table_name not in models:
                return None
            if join_type not in ["MANY_TO_ONE", "ONE_TO_MANY", "ONE_TO_ONE"]:
                return None

            is_source = table_name == models[0]
            related_table = models[1] if is_source else models[0]
            condition_parts = condition.split(" = ")
            fk_column = condition_parts[0 if is_source else 1].split(".")[1]
            fk_constraint = f"FOREIGN KEY ({fk_column}) REFERENCES {related_table}({primary_keys_map[related_table]})"

            return {
                "type": "FOREIGN_KEY",
                "comment": f'-- {{"condition": {condition}, "joinType": {join_type}}}\n  ',
                "constraint": fk_constraint,
                "tables": models,
            }

        def _column_batch(
            model: dict[str, Any], primary_keys_map: dict[str, str]
        ) -> list[dict]:
            commands = [
                _column_command(column, model) for column in model["columns"]
            ] + [
                _relationship_command(relationship, model["name"], primary_keys_map)
                for relationship in relationships
            ]

            filtered = [command for command in commands if command is not None]

            return [
                {
                    "name": model["name"],
                    "payload": str(
                        {
                            "type": "TABLE_COLUMNS",
                            "columns": filtered[i : i + column_batch_size],
                        }
                    ),
                }
                for i in range(0, len(filtered), column_batch_size)
            ]

        primary_keys_map = {model["name"]: model["primaryKey"] for model in models}

        return [
            command
            for model in models
            for command in _column_batch(model, primary_keys_map)
            + [_model_command(model)]
        ]

    def _convert_views(self, views: list[dict[str, Any]]) -> list[dict[str, str]]:
        def _payload(view: dict[str, Any]) -> dict:
            return {
                "type": "VIEW",
                "comment": (
                    f"/* {view['properties']} */\n" if "properties" in view else ""
                ),
                "name": view["name"],
                "statement": view["statement"],
            }

        return [
            {"name": view["name"], "payload": str(_payload(view))} for view in views
        ]

    def _convert_metrics(self, metrics: list[dict[str, Any]]) -> list[dict[str, str]]:
        def _create_column(name: str, data_type: str, comment: str) -> dict:
            return {
                "type": "COLUMN",
                "comment": comment,
                "name": name,
                "data_type": data_type,
            }

        def _dimensions(metric: dict[str, Any]) -> list[dict]:
            return [
                _create_column(
                    name=dim.get("name", ""),
                    data_type=dim.get("type", ""),
                    comment="-- This column is a dimension\n  ",
                )
                for dim in metric.get("dimension", [])
            ]

        def _measures(metric: dict[str, Any]) -> list[dict]:
            return [
                _create_column(
                    name=measure.get("name", ""),
                    data_type=measure.get("type", ""),
                    comment=f"-- This column is a measure\n  -- expression: {measure['expression']}\n  ",
                )
                for measure in metric.get("measure", [])
            ]

        def _payload(metric: dict[str, Any]) -> dict:
            return {
                "type": "METRIC",
                "comment": f"\n/* This table is a metric */\n/* Metric Base Object: {metric['baseObject']} */\n",
                "name": metric["name"],
                "columns": _dimensions(metric) + _measures(metric),
            }

        return [
            {"name": metric["name"], "payload": str(_payload(metric))}
            for metric in metrics
        ]


@extract_fields(dict(mdl=dict[str, Any]))
def validate_mdl(mdl_str: str, validator: MDLValidator) -> dict[str, Any]:
    res = validator.run(mdl=mdl_str)
    return dict(mdl=res["mdl"])


async def chunk(
    mdl: dict[str, Any],
    chunker: DDLChunker,
    column_batch_size: int,
    project_id: str | None = None,
) -> dict[str, Any]:
    return await chunker.run(
        mdl=mdl, column_batch_size=column_batch_size, project_id=project_id
    )


async def embedding(chunk: dict[str, Any], embedder: Any) -> dict[str, Any]:
    return await embedder.run(documents=chunk["documents"])


async def clean(
    embedding: dict[str, Any],
    cleaner: DocumentCleaner,
    project_id: str | None = None,
) -> dict[str, Any]:
    await cleaner.run(project_id=project_id)
    return embedding


async def write(clean: dict[str, Any], writer: DocumentWriter) -> None:
    return await writer.run(documents=clean["documents"])


class DBSchema(BasicPipeline):
    def __init__(self, column_batch_size: int = 50, **kwargs) -> None:
        dbschema_store = PGVectorDocumentStore(table_name="db_schema_documents")

        self._components = {
            "cleaner": DocumentCleaner([dbschema_store]),
            "validator": MDLValidator(),
            "embedder": AsyncDocumentEmbedder(),
            "chunker": DDLChunker(),
            "writer": AsyncDocumentWriter(
                document_store=dbschema_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        }
        self._configs = {"column_batch_size": column_batch_size}
        self._final = "write"

        helper.load_helpers(package_path="semantic_layer")
        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    async def run(self, mdl_str: str, project_id: str | None = None) -> dict[str, Any]:
        logger.info(
            f"Project ID: {project_id}, DB Schema Indexing pipeline is running..."
        )
        return await self._pipe.execute(
            [self._final],
            inputs={
                "mdl_str": mdl_str,
                "project_id": project_id,
                **self._components,
                **self._configs,
            },
        )

    async def clean(self, project_id: str | None = None) -> None:
        await clean(
            embedding={"documents": []},
            cleaner=self._components["cleaner"],
            project_id=project_id,
        )
