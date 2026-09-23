import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.application import register_error_handlers
from app.utils.error import (
    DocumentNotFoundError,
)


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
