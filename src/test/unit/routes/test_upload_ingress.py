"""Reject oversized upload bodies before multipart allocates temporary files."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient
from starlette import formparsers

from app.adapters.persistence.database import get_async_session
from app.application import register_error_handlers
from app.routes import rag


@pytest.fixture
async def upload_client():
    app = FastAPI()
    app.include_router(rag.router)
    register_error_handlers(app)

    async def session():
        yield AsyncMock()

    app.dependency_overrides[get_async_session] = session

    @app.post("/run_sse")
    async def stream(request: Request):
        body = await request.body()
        return StreamingResponse(
            iter([b"data: " + body + b"\n\n"]), media_type="text/event-stream"
        )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://upload.test"
    ) as client:
        yield client


@pytest.mark.parametrize("declared_length", [None, 1, 32 * 8192 + 300])
async def should_stop_oversized_bodies_before_parsing_multipart(
    upload_client, monkeypatch, declared_length
):
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "16")
    consumed = 0
    temporary_files = []
    original_temporary_file = formparsers.SpooledTemporaryFile

    def record_temporary_file(*args, **kwargs):
        temporary_files.append(True)
        return original_temporary_file(*args, **kwargs)

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", record_temporary_file)

    async def body():
        nonlocal consumed
        yield (
            b'--boundary\r\nContent-Disposition: form-data; name="file"; '
            b'filename="large.md"\r\n\r\n'
        )
        for _ in range(32):
            consumed += 8192
            yield b"x" * 8192
        yield (
            b'\r\n--boundary\r\nContent-Disposition: form-data; name="title"\r\n'
            b"\r\nLarge\r\n--boundary--\r\n"
        )

    headers = {"Content-Type": "multipart/form-data; boundary=boundary"}
    if declared_length is not None:
        headers["Content-Length"] = str(declared_length)
    response = await upload_client.post("/rag/index", content=body(), headers=headers)

    assert response.status_code == 413
    assert response.json()["error_code"] == "DOCUMENT_TOO_LARGE"
    assert consumed < 32 * 8192
    assert consumed <= 16 + 64 * 1024 + 8192
    assert temporary_files == []
    if declared_length and declared_length > 16 + 64 * 1024:
        assert consumed == 0


async def should_keep_non_upload_streaming_routes_unchanged(upload_client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "16")
    body = b"x" * (128 * 1024)
    response = await upload_client.post("/run_sse", content=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"data: " + body + b"\n\n"
