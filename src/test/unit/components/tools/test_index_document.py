"""Exercise the JSON tool boundary with the real indexing service."""

import hashlib
from importlib import import_module
from unittest.mock import AsyncMock

from google.adk.tools import FunctionTool

from app.adapters.storage.object_storage import LocalFileStorageAdapter
from app.config.settings import settings


async def should_index_json_markdown_and_enforce_utf8_byte_limit(
    monkeypatch, tmp_path, mock_tool_context
):
    tool_module = import_module("app.components.tools.custom.index_document")
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.scalar.return_value = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    repository = AsyncMock()
    repository.get_document_by_hash.return_value = None
    embeddings = AsyncMock()
    embeddings.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    storage = LocalFileStorageAdapter(str(tmp_path))
    monkeypatch.setattr(tool_module, "get_session", AsyncMock(return_value=session))
    monkeypatch.setattr(tool_module, "RagRepository", lambda _: repository)
    monkeypatch.setattr(tool_module, "EmbeddingsClient", lambda: embeddings)
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_storage", lambda: storage
    )
    content = "# Été\nRandonnée à vélo."
    raw = content.encode("utf-8")
    tool = FunctionTool(tool_module.index_document)
    args = {"title": "été.md", "content": content}
    result = await tool.run_async(args=args, tool_context=mock_tool_context)

    assert result["status"] == "success"
    document = repository.create_document.call_args.args[0]
    assert document.content_hash == hashlib.sha256(raw).hexdigest()
    assert next(tmp_path.rglob("été.md")).read_bytes() == raw
    session.commit.assert_awaited_once()

    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", str(len(content)))
    session.commit.reset_mock()
    result = await tool.run_async(args=args, tool_context=mock_tool_context)

    assert result["status"] == "error"
    assert "maximum upload size" in result["error"]
    session.commit.assert_not_awaited()
