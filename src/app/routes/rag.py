"""REST API routes for RAG (Retrieval-Augmented Generation).

Error handling: handlers raise typed ``AppError`` subclasses (or let adapters
raise them) and the application-level exception handlers translate them to
HTTP responses. The session dependency rolls back unfinished transactions when
a handler exits early, so no per-route ``try/except`` is needed.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence.database import get_async_session
from app.adapters.persistence.rag_repository import RagRepository
from app.adapters.storage.object_storage import validate_storage_filename
from app.config.settings import settings
from app.services.rag.indexation_service import IndexationService
from app.services.rag.interrogation_service import InterrogationService
from app.services.rag.models import SearchParameters
from app.utils.error import DocumentLimitError, UnsupportedDocumentError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])
RagSession = Annotated[AsyncSession, Depends(get_async_session)]


@router.post("/index")
async def index_document(
    session: RagSession,
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form(max_length=500)],
    force: Annotated[bool, Form()] = False,
):
    """Index a Markdown document into the RAG system."""
    if not file.filename or not file.filename.lower().endswith((".md", ".markdown")):
        raise UnsupportedDocumentError(details={"filename": file.filename})
    filename = validate_storage_filename(file.filename)

    max_upload_bytes = settings.rag.MAX_UPLOAD_BYTES
    content = await file.read(max_upload_bytes + 1)
    if len(content) > max_upload_bytes:
        raise DocumentLimitError(
            f"Document exceeds the maximum upload size of {max_upload_bytes} bytes",
            details={"limit_bytes": max_upload_bytes},
        )

    embeddings_client = EmbeddingsClient()
    repository = RagRepository(session)
    indexation_service = IndexationService(embeddings_client, repository, session)

    result = await indexation_service.index_document(
        content=content,
        title=title or filename,
        uri=filename,
        mime_type="text/markdown",
        filename=filename,
        force=force,
    )

    # Commit before reporting success to the caller.
    await session.commit()

    return {
        "status": "success",
        "created": result.created,
        "document_id": str(result.document.id),
        "title": result.document.title,
        "chunk_count": result.chunk_count,
        "leaf_count": result.document.metadata.get("leaf_count", result.chunk_count),
        "embedding_time_ms": result.embedding_time_ms,
    }


@router.post("/search")
async def search_documents(
    params: SearchParameters,
    session: RagSession,
):
    """Search documents using vector similarity."""
    embeddings_client = EmbeddingsClient()
    repository = RagRepository(session)
    interrogation_service = InterrogationService(embeddings_client, repository, session)

    hits = await interrogation_service.search(
        query=params.query,
        top_k=params.top_k,
        similarity_threshold=params.similarity_threshold,
        country_code=params.country_code,
        hierarchical=params.hierarchical,
        embedding_dimensions=params.embedding_dimensions,
    )

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
            "matched_chunk_ids": [str(chunk_id) for chunk_id in hit.matched_chunk_ids],
            "score_type": hit.score_type,
        }
        for hit in hits
    ]

    return {
        "status": "success",
        "query": params.query,
        "embedding_dimensions": (
            settings.rag.EMBEDDING_DIMENSIONS
            if params.embedding_dimensions is None
            else params.embedding_dimensions
        ),
        "result_count": len(results),
        "results": results,
    }


@router.get("/documents")
async def list_documents(
    session: RagSession,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """List indexed documents."""
    repository = RagRepository(session)
    documents, total = await repository.list_documents(limit, offset)

    return {
        "status": "success",
        "total": total,
        "limit": limit,
        "offset": offset,
        "documents": [
            {
                "id": str(doc.id),
                "title": doc.title,
                "uri": doc.uri,
                "chunk_count": doc.chunk_count,
                "leaf_count": doc.metadata.get("leaf_count", doc.chunk_count),
                "created_at": doc.created_at.isoformat(),
            }
            for doc in documents
        ],
    }


@router.get("/documents/{document_id}")
async def get_document(document_id: UUID, session: RagSession):
    """Get document details with chunks."""
    repository = RagRepository(session)
    document = await repository.get_document(document_id)

    return {
        "status": "success",
        "document": {
            "id": str(document.id),
            "title": document.title,
            "uri": document.uri,
            "chunk_count": document.chunk_count,
            "leaf_count": sum(chunk.is_leaf for chunk in document.chunks),
            "created_at": document.created_at.isoformat(),
            "chunks": [
                {
                    "index": chunk.chunk_index,
                    "id": str(chunk.id),
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "is_leaf": chunk.is_leaf,
                    "hierarchy": (
                        chunk.hierarchy.model_dump(mode="json")
                        if chunk.hierarchy
                        else None
                    ),
                }
                for chunk in document.chunks
            ],
        },
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: UUID, session: RagSession):
    """Delete a document."""
    repository = RagRepository(session)
    await repository.delete_document(document_id)
    await session.commit()

    return {
        "status": "success",
        "message": f"Document {document_id} deleted",
    }
