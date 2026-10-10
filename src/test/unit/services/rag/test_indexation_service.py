"""Regression coverage for repeated uploads of the same document."""

import hashlib
from unittest.mock import AsyncMock, Mock

import pytest

from app.config.settings import settings
from app.services.rag.indexation_service import IndexationService
from app.services.rag.models import Document, DocumentStatus, ParsedChunk


async def should_reuse_an_indexed_document_without_reembedding(monkeypatch):
    content = b"# GR20\nDocument already indexed."
    document = Document(
        title="GR20",
        uri="gr20.md",
        mime_type="text/markdown",
        content_hash=hashlib.sha256(content).hexdigest(),
        status=DocumentStatus.INDEXED,
        chunk_count=26,
    )
    repository = AsyncMock()
    repository.get_document_by_hash.return_value = document
    embeddings = AsyncMock()
    embeddings.embed_documents.side_effect = AssertionError("Must reuse stored chunks")
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_storage", lambda: Mock()
    )
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_chunker", lambda: AsyncMock()
    )
    session = AsyncMock()
    session.scalar.return_value = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    service = IndexationService(embeddings, repository, session)

    result = await service.index_document(
        content, title="New upload title", mime_type="text/markdown", filename="gr20.md"
    )

    assert result.document.id == document.id
    assert result.created is False
    assert result.chunk_count == 26
    assert result.embedding_time_ms == 0


async def should_reject_oversized_content_before_database_or_embeddings(monkeypatch):
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "8")
    repository = AsyncMock()
    embeddings = AsyncMock()
    session = AsyncMock()
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_storage", lambda: Mock()
    )
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_chunker", lambda: AsyncMock()
    )
    service = IndexationService(embeddings, repository, session)

    with pytest.raises(ValueError, match="maximum upload size"):
        await service.index_document(
            b"123456789",
            title="oversized.md",
            mime_type="text/markdown",
            filename="oversized.md",
        )

    session.scalar.assert_not_awaited()
    repository.get_document_by_hash.assert_not_awaited()
    embeddings.embed_documents.assert_not_awaited()


async def should_reject_too_many_chunks_before_embeddings(monkeypatch):
    monkeypatch.setenv("RAG_MAX_CHUNKS_PER_DOC", "1")
    repository = AsyncMock()
    repository.get_document_by_hash.return_value = None
    embeddings = AsyncMock()
    session = AsyncMock()
    session.scalar.return_value = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    chunker = AsyncMock()
    chunker.split.return_value = [
        ParsedChunk(content="# First\nBody"),
        ParsedChunk(content="# Second\nBody"),
    ]
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_storage", lambda: Mock()
    )
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_chunker", lambda: chunker
    )
    service = IndexationService(embeddings, repository, session)

    with pytest.raises(ValueError, match="maximum chunk count"):
        await service.index_document(
            b"# First\nBody\n# Second\nBody",
            title="many-sections.md",
            mime_type="text/markdown",
            filename="many-sections.md",
        )

    embeddings.embed_documents.assert_not_awaited()
    repository.create_document.assert_not_awaited()
