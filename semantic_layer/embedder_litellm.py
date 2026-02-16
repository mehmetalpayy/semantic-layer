"""Async embedding helpers backed by Azure OpenAI."""

from typing import Any

from haystack import Document, component
from openai import AsyncAzureOpenAI

from configs import config
from env import secrets

_cfg = config.summary.long_term
EMBEDDING_DIM = _cfg.embedding_dim
EMBEDDING_MODEL = _cfg.model
EMBEDDINGS_AZURE_ENDPOINT = _cfg.embeddings_azure_endpoint
EMBEDDINGS_DEPLOYMENT_NAME = _cfg.embeddings_deployment_name
EMBEDDINGS_API_VERSION = _cfg.embeddings_api_version

openai_client = AsyncAzureOpenAI(
    api_key=secrets.EMBEDDINGS_AZURE_OPENAI_API_KEY,
    api_version=EMBEDDINGS_API_VERSION,
    azure_endpoint=EMBEDDINGS_AZURE_ENDPOINT,
)


def _prepare_texts_to_embed(documents: list[Document]) -> list[str]:
    """Normalize documents into plain text strings for embedding."""
    texts = []
    for doc in documents:
        text = "\n".join([doc.content or ""])
        text = text.replace("\n", " ")
        texts.append(text)
    return texts


@component
class AsyncTextEmbedder:
    """Embed a single text input asynchronously."""

    def __init__(self, timeout: float = 120.0):
        """Set request timeout for embedding calls."""
        self._timeout = timeout

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run(self, text: str):
        """Return an embedding vector for a single text."""
        if not isinstance(text, str):
            raise TypeError(
                "AsyncTextEmbedder expects a string as input. "
                "Use AsyncDocumentEmbedder for a list of Documents."
            )

        text_to_embed = text.replace("\n", " ")
        response = await openai_client.embeddings.create(
            model=EMBEDDINGS_DEPLOYMENT_NAME,
            input=[text_to_embed],
            timeout=self._timeout,
        )
        return {
            "embedding": response.data[0].embedding,
            "meta": {"model": EMBEDDING_MODEL, "deployment": EMBEDDINGS_DEPLOYMENT_NAME},
        }


@component
class AsyncDocumentEmbedder:
    """Embed documents asynchronously in batches."""

    def __init__(self, batch_size: int = 32, timeout: float = 120.0):
        """Set batch size and timeout for embedding requests."""
        self._batch_size = batch_size
        self._timeout = timeout

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text inputs."""
        response = await openai_client.embeddings.create(
            model=EMBEDDINGS_DEPLOYMENT_NAME,
            input=texts,
            timeout=self._timeout,
        )
        return [item.embedding for item in response.data]

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run(self, documents: list[Document]):
        """Embed all documents and return updated documents list."""
        if not isinstance(documents, list) or (
            documents and not isinstance(documents[0], Document)
        ):
            raise TypeError(
                "AsyncDocumentEmbedder expects a list of Documents as input."
            )

        if not documents:
            return {"documents": [], "meta": {}}

        texts = _prepare_texts_to_embed(documents)
        batches = [
            texts[i : i + self._batch_size]
            for i in range(0, len(texts), self._batch_size)
        ]

        embeddings: list[list[float]] = []
        for batch in batches:
            embeddings.extend(await self._embed_batch(batch))

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {
            "documents": documents,
            "meta": {
                "model": EMBEDDING_MODEL,
                "deployment": EMBEDDINGS_DEPLOYMENT_NAME,
            },
        }
