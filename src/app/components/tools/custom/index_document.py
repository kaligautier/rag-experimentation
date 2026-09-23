"""Tool for indexing documents into the RAG system."""

import logging

from google.adk.tools import ToolContext

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence.database import get_session
from app.adapters.persistence.rag_repository import RagRepository
from app.components.tools.custom._errors import tool_error_result
from app.services.rag.indexation_service import IndexationService

logger = logging.getLogger(__name__)


async def index_document(
    title: str,
    content: str,
    mime_type: str = "text/plain",
    uri: str = "",
    tool_context: ToolContext | None = None,
) -> dict:
    """Index a document into the RAG system.

    Args:
        title: Document title
        content: Markdown document text, encoded as UTF-8 before indexing
        mime_type: MIME type of the document
        uri: Optional document URI/path
        tool_context: ADK tool context

    Returns:
        Result with document ID, chunk count, and status
    """
    try:
        logger.info(f"Indexing document: {title}")

        # Initialize services
        embeddings_client = EmbeddingsClient()
        async with await get_session() as session:
            repository = RagRepository(session)
            indexation_service = IndexationService(
                embeddings_client, repository, session
            )

            # Index document
            result = await indexation_service.index_document(
                content=content.encode("utf-8"),
                title=title,
                uri=uri,
                mime_type=mime_type,
                filename=title,
            )

            await session.commit()

            return {
                "status": "success",
                "document_id": str(result.document.id),
                "title": result.document.title,
                "chunk_count": result.chunk_count,
                "message": f"Indexed {result.chunk_count} chunks from '{title}'",
            }
    except Exception as e:
        return tool_error_result(e, "Document indexation", logger)
