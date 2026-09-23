"""Typed RAG errors and the FastAPI error handlers."""

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.application import register_error_handlers
from app.utils.error import (
    AppError,
    DocumentLimitError,
    DocumentNotFoundError,
    EmbeddingError,
    EmbeddingSchemaMismatchError,
    EmptyDocumentError,
    ErrorCode,
    UnsafeStoragePathError,
    UnsupportedDocumentError,
)


@pytest.mark.parametrize(
    ("error", "code", "status"),
    [
        (UnsupportedDocumentError(), ErrorCode.UNSUPPORTED_DOCUMENT, 400),
        (UnsafeStoragePathError(), ErrorCode.UNSAFE_FILENAME, 400),
        (DocumentLimitError(), ErrorCode.DOCUMENT_TOO_LARGE, 413),
        (EmptyDocumentError(), ErrorCode.EMPTY_DOCUMENT, 400),
        (DocumentNotFoundError(), ErrorCode.DOCUMENT_NOT_FOUND, 404),
        (EmbeddingError(), ErrorCode.EMBEDDING_ERROR, 502),
        (EmbeddingSchemaMismatchError(), ErrorCode.EMBEDDING_SCHEMA_MISMATCH, 503),
    ],
)
def should_map_rag_errors_to_codes_and_http_status(error, code, status):
    assert isinstance(error, AppError)
    assert error.error_code is code
    assert error.status_code == status
    assert error.message


def should_stay_catchable_as_legacy_builtin_exceptions():
    with pytest.raises(ValueError, match="filename"):
        raise UnsafeStoragePathError("bad filename")
    with pytest.raises(RuntimeError):
        raise EmbeddingSchemaMismatchError()


def should_carry_structured_details():
    error = DocumentLimitError("too big", details={"limit_bytes": 10})
    assert error.details == {"limit_bytes": 10}
    assert str(error) == "[DOCUMENT_TOO_LARGE] too big, details={'limit_bytes': 10}"


@pytest.fixture
async def client():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/typed")
    async def typed():
        raise DocumentNotFoundError(details={"document_id": "42"})

    @app.get("/unexpected")
    async def unexpected():
        raise RuntimeError("SELECT secret FROM table [parameters: {...}]")

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://errors.test",
    ) as http:
        yield http


async def should_render_typed_errors_with_their_status(client):
    response = await client.get("/typed")

    assert response.status_code == 404
    assert response.json() == {
        "error_code": "DOCUMENT_NOT_FOUND",
        "message": "Document not found",
        "details": {"document_id": "42"},
    }


async def should_hide_internals_behind_a_generic_500(client):
    response = await client.get("/unexpected")

    assert response.status_code == 500
    body = response.json()
    assert body["error_code"] == "GENERIC_ERROR"
    assert "secret" not in response.text


async def should_classify_inconsistent_embedding_dimensions_as_server_error():
    from unittest.mock import AsyncMock

    from app.adapters.persistence.rag_repository import RagRepository
    from app.services.rag.models import SearchParameters
    from app.utils.error import ConfigurationError

    session = AsyncMock()
    with pytest.raises(ConfigurationError) as caught:
        await RagRepository(session).search(
            SearchParameters(query="query", embedding_dimensions=768), [1.0] * 128
        )
    assert caught.value.status_code == 500
    session.execute.assert_not_awaited()
