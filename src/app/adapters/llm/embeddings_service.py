"""Asynchronous Gemini embeddings through Vertex AI."""

import asyncio
import math
from typing import Sequence

from google import genai
from google.genai import types

from app.config.settings import settings
from app.ports.rag import EmbeddingsServicePort
from app.utils.error import ConfigurationError, EmbeddingError


class EmbeddingsClient(EmbeddingsServicePort):
    """Shared, lazily initialized Google Gen AI client for RAG embeddings."""

    _instance: "EmbeddingsClient | None" = None
    _client: genai.Client | None = None

    def __new__(cls) -> "EmbeddingsClient":
        """Reuse connections without loading credentials until the first request."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _get_client(self) -> genai.Client:
        """Use the project's Google ADC and configured Vertex AI region."""
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=(
                    settings.rag.EMBEDDING_LOCATION or settings.GOOGLE_CLOUD_LOCATION
                ),
                http_options=types.HttpOptions(api_version="v1", timeout=30000),
            )
        return self._client

    async def _embed(
        self, text: str, task_type: str, dimensions: int | None = None
    ) -> list[float]:
        dimensions = (
            settings.rag.EMBEDDING_DIMENSIONS if dimensions is None else dimensions
        )
        response = await self._get_client().aio.models.embed_content(
            model=settings.rag.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=dimensions,
                auto_truncate=False,
            ),
        )
        if not response.embeddings or len(response.embeddings) != 1:
            raise EmbeddingError("Gemini must return exactly one embedding per text")
        embedding = response.embeddings[0]
        values = embedding.values
        if values is None or len(values) != dimensions:
            raise EmbeddingError("Gemini returned an unexpected embedding dimension")
        if embedding.statistics and embedding.statistics.truncated:
            raise EmbeddingError("Gemini truncated the embedding input")
        if not all(math.isfinite(value) for value in values):
            raise EmbeddingError("Gemini returned a non-finite embedding")
        norm = math.hypot(*values)
        if norm == 0 or not math.isfinite(norm):
            raise EmbeddingError("Gemini returned an invalid embedding norm")
        return [value / norm for value in values]

    async def embed_documents(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed one text per request in bounded batches, preserving input order."""
        batch_size = (
            settings.rag.EMBEDDING_BATCH_SIZE if batch_size is None else batch_size
        )
        if batch_size < 1:
            raise ConfigurationError("Embedding batch size must be positive")
        embeddings = []
        for start in range(0, len(texts), batch_size):
            # Vertex gemini-embedding-001 accepts one input text per request:
            # https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings
            # gather preserves order even when requests complete out of order.
            batch = await asyncio.gather(
                *(
                    self._embed(text, "RETRIEVAL_DOCUMENT")
                    for text in texts[start : start + batch_size]
                )
            )
            embeddings.extend(batch)
        return embeddings

    async def embed_query(
        self, text: str, dimensions: int | None = None
    ) -> list[float]:
        """Embed the question using the retrieval-query task."""
        return await self._embed(text, "RETRIEVAL_QUERY", dimensions)

    async def close(self) -> None:
        """Close the async and sync transports without constructing a client."""
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aio.aclose()
            finally:
                client.close()
