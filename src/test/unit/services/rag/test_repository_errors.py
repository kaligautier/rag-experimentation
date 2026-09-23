import pytest


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
