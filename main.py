"""Entry point for manual semantic layer indexing and retrieval checks."""

import asyncio
import inspect
import json
from configs import config
from pathlib import Path

from haystack import Document

from semantic_layer import helper
from semantic_layer.build_mdl import main as build_mdl_main
from semantic_layer.db_schema import DDLChunker as DbSchemaChunker
from semantic_layer.embedder_litellm import EMBEDDING_DIM, AsyncDocumentEmbedder
from semantic_layer.mdl_validator import AsyncDocumentWriter, DocumentCleaner, MDLValidator
from semantic_layer.pgvector_retrieval import PgVectorDbSchemaRetrieval
from semantic_layer.pgvector_store import PGVectorDocumentStore
from semantic_layer.table_description import TableDescriptionChunker


def _log_documents(tag: str, documents: list[Document], limit: int = 3) -> None:
    """Print a few documents for debugging by tag."""
    print(f"[{tag}] documents={len(documents)}")
    for doc in documents[:limit]:
        print(f"[{tag}] meta: {doc.meta}")
        print(f"[{tag}] content: {doc.content}")
        print(f"[{tag}] ---")


async def _run_indexing(
    mdl_str: str,
    project_id: str,
    chunker,
    store_table: str,
    tag: str,
) -> None:
    """Run chunking, embedding, and storage for a given chunker."""
    validator = MDLValidator()
    embedder = AsyncDocumentEmbedder()
    store = PGVectorDocumentStore(table_name=store_table)
    cleaner = DocumentCleaner([store])
    writer = AsyncDocumentWriter(document_store=store)

    mdl = validator.run(mdl_str)["mdl"]
    if tag == "db_schema":
        result = await chunker.run(mdl=mdl, column_batch_size=50, project_id=project_id)
    else:
        maybe_result = chunker.run(mdl=mdl, project_id=project_id)
        if inspect.isawaitable(maybe_result):
            result = await maybe_result
        else:
            result = maybe_result
    documents = result["documents"]

    print(f"[{tag}] chunked documents (pre-embedding)")
    _log_documents(tag, documents)

    print(f"[{tag}] embedding_dim={EMBEDDING_DIM}")
    embedded = await embedder.run(documents=documents)
    embedded_docs = embedded["documents"]
    if embedded_docs:
        print(f"[{tag}] embedding_len={len(embedded_docs[0].embedding or [])}")

    print(f"[{tag}] cleaning existing documents (project_id={project_id})")
    await cleaner.run(project_id=project_id)

    written = await writer.run(documents=embedded_docs)
    print(f"[{tag}] written={written.get('documents_written')}")


async def run_all() -> None:
    """Run the full indexing and retrieval flow for quick validation."""
    runtime_cfg = config.runtime
    project_id = runtime_cfg.project_id
    query = runtime_cfg.retrieval_query or (
        "Can you fetch last 1 month data of all tables?"
    )
    enable_column_pruning = runtime_cfg.enable_column_pruning

    mdl_path = Path(__file__).parent / "semantic_layer" / "mdl.json"
    if not mdl_path.exists():
        await build_mdl_main()
    print(f"[main] using mdl={mdl_path}")

    mdl_str = mdl_path.read_text(encoding="utf-8")
    try:
        mdl_json = json.loads(mdl_str)
        print(f"[main] mdl models={len(mdl_json.get('models', []))}")
    except json.JSONDecodeError:
        print("[main] mdl json invalid")

    helper.load_helpers(package_path="semantic_layer")
    await _run_indexing(
        mdl_str=mdl_str,
        project_id=project_id,
        chunker=DbSchemaChunker(),
        store_table="db_schema_documents",
        tag="db_schema",
    )
    await _run_indexing(
        mdl_str=mdl_str,
        project_id=project_id,
        chunker=TableDescriptionChunker(),
        store_table="table_description_documents",
        tag="table_description",
    )

    print("[retrieval] running pgvector retrieval...")
    retriever = PgVectorDbSchemaRetrieval()
    print(f"[retrieval] enable_column_pruning={enable_column_pruning}")
    result = await retriever.run(
        query=query, project_id=project_id, enable_column_pruning=enable_column_pruning
    )
    ddls = [entry["table_ddl"] for entry in result["retrieval_results"]]
    print(json.dumps(ddls, indent=2))


if __name__ == "__main__":
    asyncio.run(run_all())
