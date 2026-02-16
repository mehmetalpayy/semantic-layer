"""Table description chunking and indexing pipeline helpers."""

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

from .embedder_litellm import AsyncDocumentEmbedder
from .mdl_validator import AsyncDocumentWriter, DocumentCleaner, MDLValidator
from .pgvector_store import PGVectorDocumentStore
from .pipeline import BasicPipeline

logger = logging.getLogger("wren-ai-service")


@component
class TableDescriptionChunker:
    """Chunk table description resources into Haystack documents."""

    @component.output_types(documents=list[Document])
    def run(self, mdl: dict[str, Any], project_id: str | None = None):
        """Build documents from model/metric/view descriptions."""
        def _additional_meta() -> dict[str, Any]:
            """Build optional project metadata."""
            return {"project_id": project_id} if project_id else {}

        chunks = [
            {
                "id": str(uuid.uuid4()),
                "meta": {
                    "type": "TABLE_DESCRIPTION",
                    "name": chunk["name"],
                    **_additional_meta(),
                },
                "content": str(chunk),
            }
            for chunk in self._get_table_descriptions(mdl)
        ]

        return {
            "documents": [
                Document(**chunk)
                for chunk in tqdm(
                    chunks,
                    desc=f"Project ID: {project_id}, Chunking table descriptions into documents",
                )
            ]
        }

    def _get_table_descriptions(self, mdl: dict[str, Any]) -> list[str]:
        """Extract simplified table description records from MDL."""
        def _structure_data(mdl_type: str, payload: dict[str, Any]) -> dict[str, Any]:
            """Normalize MDL resource into a common structure."""
            return {
                "mdl_type": mdl_type,
                "name": payload.get("name"),
                "columns": [column["name"] for column in payload.get("columns", [])],
                "properties": payload.get("properties", {}),
            }

        resources = (
            [_structure_data("MODEL", model) for model in mdl["models"]]
            + [_structure_data("METRIC", metric) for metric in mdl["metrics"]]
            + [_structure_data("VIEW", view) for view in mdl["views"]]
        )

        return [
            {
                "name": resource["name"],
                "description": resource["properties"].get("description", ""),
                "columns": ", ".join(resource["columns"]),
            }
            for resource in resources
            if resource["name"] is not None
        ]


@extract_fields(dict(mdl=dict[str, Any]))
def validate_mdl(mdl_str: str, validator: MDLValidator) -> dict[str, Any]:
    """Validate and parse MDL text."""
    res = validator.run(mdl=mdl_str)
    return dict(mdl=res["mdl"])


def chunk(
    mdl: dict[str, Any],
    chunker: TableDescriptionChunker,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Run the table description chunker."""
    return chunker.run(mdl=mdl, project_id=project_id)


async def embedding(chunk: dict[str, Any], embedder: Any) -> dict[str, Any]:
    """Embed chunked documents."""
    return await embedder.run(documents=chunk["documents"])


async def clean(
    embedding: dict[str, Any],
    cleaner: DocumentCleaner,
    project_id: str | None = None,
) -> dict[str, Any]:
    """Clean existing documents before writing new ones."""
    await cleaner.run(project_id=project_id)
    return embedding


async def write(clean: dict[str, Any], writer: DocumentWriter) -> None:
    """Write documents to the document store."""
    return await writer.run(documents=clean["documents"])


class TableDescription(BasicPipeline):
    """Pipeline for building and storing table descriptions."""

    def __init__(self, **kwargs) -> None:
        """Configure the table description pipeline components."""
        table_description_store = PGVectorDocumentStore(
            table_name="table_description_documents"
        )

        self._components = {
            "cleaner": DocumentCleaner([table_description_store]),
            "validator": MDLValidator(),
            "embedder": AsyncDocumentEmbedder(),
            "chunker": TableDescriptionChunker(),
            "writer": AsyncDocumentWriter(
                document_store=table_description_store,
                policy=DuplicatePolicy.OVERWRITE,
            ),
        }
        self._configs = {}
        self._final = "write"

        super().__init__(
            AsyncDriver({}, sys.modules[__name__], result_builder=base.DictResult())
        )

    async def run(self, mdl_str: str, project_id: str | None = None) -> dict[str, Any]:
        """Execute the pipeline with the provided MDL."""
        logger.info(
            f"Project ID: {project_id}, Table Description Indexing pipeline is running..."
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
        """Clear documents for a project."""
        await clean(
            embedding={"documents": []},
            cleaner=self._components["cleaner"],
            project_id=project_id,
        )
