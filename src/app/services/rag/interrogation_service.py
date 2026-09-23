"""Document search/interrogation service for RAG."""

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence.database import require_embedding_schema
from app.adapters.persistence.rag_repository import RagRepository
from app.config.settings import settings
from app.services.rag.hierarchical_retrieval import merge_parent_contexts
from app.services.rag.models import SearchHit, SearchParameters
from app.utils.error import InvalidInputError

logger = logging.getLogger(__name__)


class InterrogationService:
    """Service for searching documents via vector similarity."""

    def __init__(
        self,
        embeddings_client: EmbeddingsClient,
        repository: RagRepository,
        session: AsyncSession,
    ):
        """Initialize interrogation service.

        Args:
            embeddings_client: Embeddings service
            repository: RAG repository
            session: AsyncSQLAlchemy session
        """
        self.embeddings_client = embeddings_client
        self.repository = repository
        self.session = session

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        country_code: str | None = None,
        hierarchical: bool = True,
        embedding_dimensions: int | None = None,
    ) -> list[SearchHit]:
        """Search documents using vector similarity.

        Args:
            query: Search query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            country_code: Optional country filter
            embedding_dimensions: MRL prefix dimension used for this search

        Returns:
            List of SearchHit results ranked by similarity
        """
        stored_dimensions = settings.rag.EMBEDDING_DIMENSIONS
        search_dimensions = (
            stored_dimensions if embedding_dimensions is None else embedding_dimensions
        )
        if search_dimensions > stored_dimensions:
            raise InvalidInputError(
                "Search embedding dimension cannot exceed stored embedding dimension "
                f"({stored_dimensions})",
                details={"requested": search_dimensions, "stored": stored_dimensions},
            )

        parameters = SearchParameters(
            query=query,
            top_k=settings.rag.TOP_K if top_k is None else top_k,
            similarity_threshold=(
                settings.rag.SIMILARITY_THRESHOLD
                if similarity_threshold is None
                else similarity_threshold
            ),
            country_code=country_code,
            hierarchical=hierarchical,
            embedding_dimensions=search_dimensions,
        )

        # Embed query
        await require_embedding_schema(self.session)
        logger.info("Embedding search query")
        query_embedding = await self.embeddings_client.embed_query(
            query, dimensions=search_dimensions
        )

        # Search repository
        logger.info(
            "Searching with top_k=%s, threshold=%s, embedding_dimensions=%s",
            parameters.top_k,
            parameters.similarity_threshold,
            search_dimensions,
        )
        hits = await self.repository.search(parameters, query_embedding)

        if hierarchical:
            hits = await merge_parent_contexts(
                hits, self.repository, settings.rag.MAX_PARENT_TOKENS
            )

        logger.info(f"Found {len(hits)} search results")
        return hits
