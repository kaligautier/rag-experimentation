"""Gemini embedding requests preserve task semantics, order and vector validity."""

import asyncio
import math
from types import SimpleNamespace

import pytest
from google.genai import errors, types

from app.adapters.llm import embeddings_service

DIMENSIONS = 768


def embedding_response(values):
    return types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=values)]
    )


class FakeGeminiClient:
    """An asynchronous SDK boundary with deterministic completion ordering."""

    def __init__(self):
        self.calls = []
        self.responses = {}
        self.errors = {}
        self.delays = {}
        self.completions = []
        self.active = 0
        self.peak_active = 0
        self.async_closes = 0
        self.sync_closes = 0
        self.aio = SimpleNamespace(
            models=SimpleNamespace(embed_content=self.embed_content),
            aclose=self.aclose,
        )

    async def embed_content(self, *, model, contents, config):
        assert isinstance(contents, str), "Gemini accepts one text per request"
        self.calls.append((model, contents, config))
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        try:
            for _ in range(self.delays.get(contents, 1)):
                await asyncio.sleep(0)
            if contents in self.errors:
                raise self.errors[contents]
            self.completions.append(contents)
            return self.responses.get(
                contents, embedding_response([3.0, 4.0] + [0.0] * (DIMENSIONS - 2))
            )
        finally:
            self.active -= 1

    async def aclose(self):
        self.async_closes += 1

    def close(self):
        self.sync_closes += 1


@pytest.fixture
def gemini_sdk(monkeypatch):
    sdk = FakeGeminiClient()
    creations = []

    def create_client(**kwargs):
        creations.append(kwargs)
        return sdk

    monkeypatch.setattr(
        embeddings_service,
        "genai",
        SimpleNamespace(Client=create_client),
        raising=False,
    )
    monkeypatch.setattr(embeddings_service.EmbeddingsClient, "_instance", None)
    monkeypatch.setattr(
        embeddings_service.EmbeddingsClient, "_client", None, raising=False
    )
    monkeypatch.setattr(
        embeddings_service,
        "settings",
        SimpleNamespace(
            GOOGLE_CLOUD_PROJECT="embedding-test-project",
            GOOGLE_CLOUD_LOCATION="default-test-region",
            rag=SimpleNamespace(
                EMBEDDING_MODEL="gemini-embedding-001",
                EMBEDDING_DIMENSIONS=DIMENSIONS,
                EMBEDDING_BATCH_SIZE=5,
                EMBEDDING_LOCATION="embedding-test-region",
            ),
        ),
    )
    return sdk, creations


@pytest.mark.parametrize("embedding_location", ["embedding-test-region", ""])
async def should_create_gemini_vertex_client_lazily_and_reuse_it(
    gemini_sdk, embedding_location
):
    _, creations = gemini_sdk
    settings = embeddings_service.settings
    settings.rag.EMBEDDING_LOCATION = embedding_location

    client = embeddings_service.EmbeddingsClient()

    assert creations == []
    await client.embed_query("First question")
    await client.embed_query("Second question")
    assert len(creations) == 1
    assert creations[0]["vertexai"] is True
    assert creations[0]["project"] == settings.GOOGLE_CLOUD_PROJECT
    assert creations[0]["location"] == (
        embedding_location or settings.GOOGLE_CLOUD_LOCATION
    )


async def should_embed_gemini_queries_with_query_task_and_unit_norm(gemini_sdk):
    sdk, _ = gemini_sdk

    result = await embeddings_service.EmbeddingsClient().embed_query("What is GR20?")

    assert len(sdk.calls) == 1
    model, contents, config = sdk.calls[0]
    assert model == "gemini-embedding-001"
    assert contents == "What is GR20?"
    assert isinstance(config, types.EmbedContentConfig)
    assert config.task_type == "RETRIEVAL_QUERY"
    assert config.output_dimensionality == DIMENSIONS
    assert config.auto_truncate is False
    assert result == pytest.approx([0.6, 0.8] + [0.0] * (DIMENSIONS - 2))
    assert math.sqrt(sum(value * value for value in result)) == pytest.approx(1.0)


async def should_embed_query_at_requested_mrl_dimension(gemini_sdk):
    sdk, _ = gemini_sdk
    dimensions = 768
    sdk.responses["Compact question"] = embedding_response(
        [3.0, 4.0] + [0.0] * (dimensions - 2)
    )

    result = await embeddings_service.EmbeddingsClient().embed_query(
        "Compact question", dimensions=dimensions
    )

    _, _, config = sdk.calls[0]
    assert config.output_dimensionality == dimensions
    assert len(result) == dimensions
    assert result[:2] == pytest.approx([0.6, 0.8])


@pytest.mark.parametrize(
    ("batch_size", "expected_concurrency"), [(None, 5), (2, 2), (1, 1)]
)
async def should_embed_gemini_documents_concurrently_with_order_preserved(
    gemini_sdk, batch_size, expected_concurrency
):
    sdk, _ = gemini_sdk
    texts = [f"Section {index}" for index in range(7)]
    expected_vectors = []
    for index, text in enumerate(texts):
        values = [0.0] * DIMENSIONS
        values[index] = index + 2.0
        sdk.responses[text] = embedding_response(values)
        normalized = [0.0] * DIMENSIONS
        normalized[index] = 1.0
        expected_vectors.append(normalized)
    sdk.delays[texts[0]] = 8

    results = await embeddings_service.EmbeddingsClient().embed_documents(
        texts, batch_size=batch_size
    )

    assert results == expected_vectors
    assert len(sdk.calls) == len(texts)
    assert sorted(contents for _, contents, _ in sdk.calls) == sorted(texts)
    for model, _, config in sdk.calls:
        assert model == "gemini-embedding-001"
        assert config.task_type == "RETRIEVAL_DOCUMENT"
        assert config.output_dimensionality == DIMENSIONS
        assert config.auto_truncate is False
    assert sdk.peak_active == expected_concurrency
    assert sdk.active == 0
    if expected_concurrency > 1:
        assert sdk.completions != texts


async def should_skip_gemini_client_creation_for_empty_document_list(gemini_sdk):
    sdk, creations = gemini_sdk

    result = await embeddings_service.EmbeddingsClient().embed_documents([])

    assert result == []
    assert sdk.calls == []
    assert creations == []


@pytest.mark.parametrize("operation", ["query", "documents"])
@pytest.mark.parametrize(
    "response",
    [
        pytest.param(types.EmbedContentResponse(), id="missing-embeddings"),
        pytest.param(types.EmbedContentResponse(embeddings=[]), id="empty-embeddings"),
        pytest.param(embedding_response(None), id="missing-values"),
        pytest.param(
            embedding_response([1.0] * (DIMENSIONS - 1)), id="wrong-dimension"
        ),
        pytest.param(
            embedding_response([math.nan] + [0.0] * (DIMENSIONS - 1)), id="nan"
        ),
        pytest.param(
            embedding_response([math.inf] + [0.0] * (DIMENSIONS - 1)), id="infinity"
        ),
        pytest.param(embedding_response([0.0] * DIMENSIONS), id="zero-norm"),
    ],
)
async def should_reject_invalid_gemini_embedding_responses(
    gemini_sdk, operation, response
):
    sdk, _ = gemini_sdk
    sdk.responses["Invalid response"] = response
    client = embeddings_service.EmbeddingsClient()

    with pytest.raises(ValueError):
        if operation == "query":
            await client.embed_query("Invalid response")
        else:
            await client.embed_documents(["Invalid response"])


@pytest.mark.parametrize("operation", ["query", "documents"])
async def should_propagate_google_embedding_errors_without_local_fallback(
    gemini_sdk, operation
):
    sdk, _ = gemini_sdk
    google_error = errors.ClientError(
        403, {"error": {"message": "Permission denied", "status": "PERMISSION_DENIED"}}
    )
    sdk.errors["Restricted document"] = google_error
    client = embeddings_service.EmbeddingsClient()

    with pytest.raises(errors.ClientError) as raised:
        if operation == "query":
            await client.embed_query("Restricted document")
        else:
            await client.embed_documents(["Restricted document"])

    assert raised.value is google_error


async def should_close_both_gemini_clients_and_clear_the_cached_client(gemini_sdk):
    sdk, _ = gemini_sdk
    client = embeddings_service.EmbeddingsClient()
    await client.embed_query("Open a client")

    await client.close()

    assert sdk.async_closes == 1
    assert sdk.sync_closes == 1
    assert client._client is None


async def should_close_unused_gemini_client_without_initializing_it(gemini_sdk):
    sdk, creations = gemini_sdk
    client = embeddings_service.EmbeddingsClient()

    await client.close()

    assert creations == []
    assert sdk.async_closes == 0
    assert sdk.sync_closes == 0
    assert client._client is None


@pytest.mark.parametrize("batch_size", [0, -1])
async def should_classify_invalid_batch_size_as_server_configuration(
    gemini_sdk, batch_size
):
    from app.utils.error import ConfigurationError

    with pytest.raises(ConfigurationError) as caught:
        await embeddings_service.EmbeddingsClient().embed_documents(
            ["text"], batch_size=batch_size
        )
    assert caught.value.status_code == 500
