"""Tests for RAG domain models."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.services.rag.models import (
    Chunk,
    Document,
    DocumentStatus,
    IndexationResult,
    SearchHit,
    SearchParameters,
)


def should_create_chunk_model():
    """Test Chunk model creation."""
    doc_id = uuid4()
    chunk = Chunk(
        document_id=doc_id,
        content="Sample text content",
        embedding=[0.1, 0.2, 0.3],
        chunk_index=0,
    )
    assert chunk.document_id == doc_id
    assert chunk.content == "Sample text content"
    assert len(chunk.embedding) == 3


def should_create_document_model():
    """Test Document model creation."""
    doc = Document(
        title="Test Document",
        uri="file://test.pdf",
        mime_type="application/pdf",
        content_hash="abc123",
    )
    assert doc.title == "Test Document"
    assert doc.status == DocumentStatus.PENDING
    assert doc.chunk_count == 0


def should_create_search_parameters():
    """Test SearchParameters model."""
    params = SearchParameters(
        query="test query",
        top_k=10,
        similarity_threshold=0.5,
    )
    assert params.query == "test query"
    assert params.top_k == 10
    assert params.similarity_threshold == 0.5
    assert params.embedding_dimensions is None


@pytest.mark.parametrize("dimensions", [768, 1536, 3072])
def should_accept_mrl_search_dimensions(dimensions):
    params = SearchParameters(query="test query", embedding_dimensions=dimensions)

    assert params.embedding_dimensions == dimensions


@pytest.mark.parametrize("dimensions", [127, 3073])
def should_reject_search_dimensions_outside_embedding_model_bounds(dimensions):
    with pytest.raises(ValidationError):
        SearchParameters(query="test query", embedding_dimensions=dimensions)


def should_create_search_hit():
    """Test SearchHit model."""
    hit = SearchHit(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_content="Result text",
        similarity_score=0.85,
        document_title="Test Doc",
        chunk_index=0,
    )
    assert hit.similarity_score == 0.85
    assert hit.document_title == "Test Doc"


def should_create_indexation_result():
    """Test IndexationResult model."""
    doc = Document(
        title="Test",
        uri="file://test.txt",
        mime_type="text/plain",
        content_hash="xyz789",
    )
    result = IndexationResult(
        document=doc,
        created=True,
        chunk_count=5,
        embedding_time_ms=150.0,
    )
    assert result.created is True
    assert result.chunk_count == 5
