"""Utilities for validating MDL and cleaning/writing documents."""

import asyncio
import json
import logging
from typing import Any

import orjson
from haystack import Document, component
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy

logger = logging.getLogger("wren-ai-service")


@component
class DocumentCleaner:
    """
    This component is used to clear all the documents in the specified document store(s).
    """

    def __init__(self, stores: list[Any]) -> None:
        """Store document stores to clean."""
        self._stores = stores

    @component.output_types()
    async def run(self, project_id: str | None = None) -> None:
        """Delete documents for the optional project ID from all stores."""
        async def _clear_documents(store: Any, project_id: str | None = None) -> None:
            """Delete documents from a single store with optional filter."""
            store_name = (
                store.to_dict().get("init_parameters", {}).get("index", "unknown")
                if hasattr(store, "to_dict")
                else "unknown"
            )
            logger.info(f"Project ID: {project_id}, Cleaning documents in {store_name}")
            filters = (
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "project_id", "operator": "==", "value": project_id},
                    ],
                }
                if project_id
                else None
            )
            await store.delete_documents(filters)

        await asyncio.gather(
            *[_clear_documents(store, project_id) for store in self._stores]
        )


@component
class MDLValidator:
    """
    Validate the MDL to check if it is a valid JSON and contains the required keys.
    """

    @component.output_types(mdl=dict[str, Any])
    def run(self, mdl: str) -> dict[str, Any]:
        """Parse and normalize the MDL JSON payload."""
        try:
            mdl_json = orjson.loads(mdl)
            logger.info("MDL JSON parsed successfully")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

        mdl_json.setdefault("models", [])
        mdl_json.setdefault("views", [])
        mdl_json.setdefault("relationships", [])
        mdl_json.setdefault("metrics", [])

        return {"mdl": mdl_json}


@component
class AsyncDocumentWriter(DocumentWriter):
    @component.output_types(documents_written=int)
    async def run(
        self, documents: list[Document], policy: DuplicatePolicy | None = None
    ):
        """Write documents to the store using the provided policy."""
        if policy is None:
            policy = self.policy

        documents_written = await self.document_store.write_documents(
            documents=documents, policy=policy
        )
        return {"documents_written": documents_written}
