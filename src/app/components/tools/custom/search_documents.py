"""Tool for searching documents in the RAG system."""

import logging

from google.adk.tools import ToolContext

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence.database import get_session
from app.adapters.persistence.rag_repository import RagRepository
from app.components.tools.custom._errors import tool_error_result
from app.config.settings import settings
from app.services.rag.interrogation_service import InterrogationService

logger = logging.getLogger(__name__)


async def search_documents(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
    hierarchical: bool = True,
    embedding_dimensions: int | None = None,
    tool_context: ToolContext | None = None,
) -> dict:
    """Search documents in the RAG system.

    Args:
        query: Search query text
        top_k: Number of results to return
        similarity_threshold: Minimum similarity score
        hierarchical: Include parent context when a majority of children match
        embedding_dimensions: MRL prefix size; defaults to the stored dimension
        tool_context: ADK tool context

    Returns:
        Dictionary with search results
    """
    try:
        logger.info(f"Searching documents with query: {query}")

        # Initialize services
        embeddings_client = EmbeddingsClient()
        async with await get_session() as session:
            repository = RagRepository(session)
            interrogation_service = InterrogationService(
                embeddings_client, repository, session
            )

            # Search
            hits = await interrogation_service.search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                hierarchical=hierarchical,
                embedding_dimensions=embedding_dimensions,
            )

            # Format results
            results = [
                {
                    "document_id": str(hit.document_id),
                    "document_title": hit.document_title,
                    "chunk_id": str(hit.chunk_id),
                    "chunk_index": hit.chunk_index,
                    "similarity_score": round(hit.similarity_score, 3),
                    "content": hit.chunk_content,
                    "metadata": hit.metadata,
                    "hierarchy": hit.hierarchy.model_dump(mode="json")
                    if hit.hierarchy
                    else None,
                    "matched_chunk_ids": [
                        str(chunk_id) for chunk_id in hit.matched_chunk_ids
                    ],
                    "score_type": hit.score_type,
                    "content_preview": hit.chunk_content[:200] + "..."
                    if len(hit.chunk_content) > 200
                    else hit.chunk_content,
                }
                for hit in hits
            ]

            return {
                "status": "success",
                "query": query,
                "embedding_dimensions": (
                    settings.rag.EMBEDDING_DIMENSIONS
                    if embedding_dimensions is None
                    else embedding_dimensions
                ),
                "result_count": len(results),
                "results": results,
            }
    except Exception as e:
        return tool_error_result(e, "Document search", logger)
