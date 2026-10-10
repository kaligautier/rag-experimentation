"""Repository guards that must hold before any SQL reaches PostgreSQL."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.adapters.persistence.rag_repository import RagRepository
from app.services.rag.models import Chunk, Document, SearchParameters
from app.utils.error import ConfigurationError, DocumentNotFoundError


async def should_classify_inconsistent_embedding_dimensions_as_server_error():
    session = AsyncMock()
    with pytest.raises(ConfigurationError) as caught:
        await RagRepository(session).search(
            SearchParameters(query="query", embedding_dimensions=768), [1.0] * 128
        )
    assert caught.value.status_code == 500
    session.execute.assert_not_awaited()


@pytest.mark.parametrize("method", ["add_chunks", "replace_chunks"])
async def should_refuse_chunks_built_for_another_document(method):
    session = AsyncMock()
    session.add_all = MagicMock()
    document_id = uuid4()
    chunks = [
        Chunk(document_id=document_id, content="Own section.", chunk_index=0),
        Chunk(document_id=uuid4(), content="Foreign section.", chunk_index=1),
    ]

    with pytest.raises(ValueError, match="reference documents"):
        await getattr(RagRepository(session), method)(document_id, chunks)

    session.execute.assert_not_awaited()
    session.add_all.assert_not_called()


@pytest.mark.parametrize("method", ["update_document", "delete_document"])
async def should_report_missing_documents_before_writing(method):
    session = AsyncMock()
    session.get.return_value = None
    document = Document(
        title="Missing", uri="missing.md", mime_type="text/markdown", content_hash="0"
    )
    argument = document if method == "update_document" else document.id

    with pytest.raises(DocumentNotFoundError):
        await getattr(RagRepository(session), method)(argument)

    session.flush.assert_not_awaited()
    session.delete.assert_not_awaited()
