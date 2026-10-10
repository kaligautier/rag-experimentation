"""Regression tests for the query embedding passed to vector search."""

from collections.abc import Sequence
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import settings
from app.services.rag.interrogation_service import InterrogationService
from app.services.rag.models import SearchHit, SearchParameters


@pytest.mark.parametrize(
    ("options", "expected_top_k", "expected_threshold", "expected_country"),
    [
        ({}, 5, 0.0, None),
        (
            {"top_k": 3, "similarity_threshold": 0.75, "country_code": "FR"},
            3,
            0.75,
            "FR",
        ),
        ({"embedding_dimensions": 768}, 5, 0.0, None),
    ],
)
async def should_forward_query_embedding_and_search_options(
    options, expected_top_k, expected_threshold, expected_country
):
    query = "What is the return policy?"
    embedding = [2.0, 1.0] + [0.0] * 382
    hit = SearchHit(
        document_id=uuid4(),
        chunk_id=uuid4(),
        chunk_content="You can return items within 30 days.",
        similarity_score=0.9,
        document_title="Return policy",
        chunk_index=2,
        metadata={"page": 3},
    )

    class EmbeddingsStub:
        async def embed_query(
            self, text: str, dimensions: int | None = None
        ) -> list[float]:
            assert text == query
            assert dimensions == options.get(
                "embedding_dimensions", settings.rag.EMBEDDING_DIMENSIONS
            )
            return embedding

    class RepositoryStub:
        async def search(
            self, parameters: SearchParameters, query_embedding: Sequence[float]
        ) -> list[SearchHit]:
            assert parameters == SearchParameters(
                query=query,
                top_k=expected_top_k,
                similarity_threshold=expected_threshold,
                country_code=expected_country,
                embedding_dimensions=options.get(
                    "embedding_dimensions", settings.rag.EMBEDDING_DIMENSIONS
                ),
            )
            assert list(query_embedding) == embedding
            return [hit]

    session = AsyncMock(spec=AsyncSession)
    session.scalar.return_value = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    service = InterrogationService(EmbeddingsStub(), RepositoryStub(), session)

    results = await service.search(query, **options)

    assert results == [hit]


@pytest.mark.parametrize(
    ("dimensions", "message"),
    [
        (0, "greater than or equal to 128"),
        (3073, "cannot exceed stored embedding dimension"),
    ],
)
async def should_reject_invalid_search_dimensions(dimensions, message):
    session = AsyncMock(spec=AsyncSession)
    service = InterrogationService(AsyncMock(), AsyncMock(), session)

    with pytest.raises(ValueError, match=message):
        await service.search("query", embedding_dimensions=dimensions)
