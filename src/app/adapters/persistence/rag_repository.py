"""RAG repository implementation using SQLAlchemy."""

import logging
from typing import Sequence
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import cast, delete, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.adapters.persistence.entities import RagChunkEntity, RagDocumentEntity
from app.ports.rag import RagRepositoryPort
from app.services.rag.models import (
    Chunk,
    ChunkHierarchy,
    Document,
    DocumentStatus,
    SearchHit,
    SearchParameters,
)
from app.utils.error import ConfigurationError, DocumentNotFoundError

logger = logging.getLogger(__name__)


class RagRepository(RagRepositoryPort):
    """SQLAlchemy-based RAG repository."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with async session."""
        self.session = session

    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        entity = RagDocumentEntity(
            id=document.id,
            title=document.title,
            uri=document.uri,
            mime_type=document.mime_type,
            content_hash=document.content_hash,
            doc_metadata=document.metadata,
            status=document.status.value,
            chunk_count=document.chunk_count,
            created_at=document.created_at,
            updated_at=document.updated_at,
        )
        self.session.add(entity)
        await self.session.flush()  # Flush to get any DB-generated values
        logger.info(f"Created document: {document.id}")
        return document

    async def get_document(self, document_id: UUID) -> Document:
        """Get document by ID with chunks."""
        stmt = (
            select(RagDocumentEntity)
            .where(RagDocumentEntity.id == document_id)
            .options(selectinload(RagDocumentEntity.chunks))
        )
        result = await self.session.execute(stmt)
        entity = result.scalar_one_or_none()

        if not entity:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}",
                details={"document_id": str(document_id)},
            )

        return self._entity_to_model(entity)

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[Document], int]:
        """List documents with pagination."""
        # Count total
        count_stmt = select(func.count(RagDocumentEntity.id))
        count_result = await self.session.execute(count_stmt)
        total = count_result.scalar() or 0

        # Fetch paginated
        stmt = (
            select(RagDocumentEntity)
            .order_by(desc(RagDocumentEntity.created_at))
            .limit(limit)
            .offset(offset)
            .options(selectinload(RagDocumentEntity.chunks))
        )
        result = await self.session.execute(stmt)
        entities = result.scalars().all()

        documents = [self._entity_to_model(entity) for entity in entities]
        return documents, total

    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Find a previously indexed document without re-embedding its content."""
        result = await self.session.execute(
            select(RagDocumentEntity)
            .where(RagDocumentEntity.content_hash == content_hash)
            .options(selectinload(RagDocumentEntity.chunks))
        )
        entity = result.scalar_one_or_none()
        return self._entity_to_model(entity) if entity is not None else None

    async def update_document(self, document: Document) -> Document:
        """Update document metadata."""
        entity = await self.session.get(RagDocumentEntity, document.id)
        if not entity:
            raise DocumentNotFoundError(
                f"Document not found: {document.id}",
                details={"document_id": str(document.id)},
            )

        entity.title = document.title
        entity.doc_metadata = document.metadata
        entity.status = document.status.value
        entity.chunk_count = document.chunk_count

        await self.session.flush()
        logger.info(f"Updated document: {document.id}")
        return document

    async def delete_document(self, document_id: UUID) -> None:
        """Delete document and chunks."""
        entity = await self.session.get(RagDocumentEntity, document_id)
        if not entity:
            raise DocumentNotFoundError(
                f"Document not found: {document_id}",
                details={"document_id": str(document_id)},
            )

        await self.session.delete(entity)
        await self.session.flush()
        logger.info(f"Deleted document: {document_id}")

    async def add_chunks(self, document_id: UUID, chunks: Sequence[Chunk]) -> None:
        """Add chunks to document."""
        chunk_entities = [
            RagChunkEntity(
                id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                embedding=chunk.embedding,
                chunk_index=chunk.chunk_index,
                chunk_metadata={
                    **chunk.metadata,
                    **(
                        {"hierarchy": chunk.hierarchy.model_dump(mode="json")}
                        if chunk.hierarchy
                        else {}
                    ),
                },
                created_at=chunk.created_at,
            )
            for chunk in chunks
        ]
        self.session.add_all(chunk_entities)
        await self.session.flush()
        logger.info(f"Added {len(chunk_entities)} chunks to document {document_id}")

    async def get_chunks(self, chunk_ids: Sequence[UUID]) -> list[Chunk]:
        if not chunk_ids:
            return []
        entities = await self.session.scalars(
            select(RagChunkEntity).where(RagChunkEntity.id.in_(chunk_ids))
        )
        return [self._chunk_to_model(entity) for entity in entities]

    async def search(
        self, parameters: SearchParameters, query_embedding: Sequence[float]
    ) -> list[SearchHit]:
        """Rank indexed chunks by cosine similarity using pgvector's <=> operator."""
        dimensions = parameters.embedding_dimensions or len(query_embedding)
        if len(query_embedding) != dimensions:
            raise ConfigurationError(
                "Query embedding dimension does not match requested search dimension",
                details={"embedding": len(query_embedding), "requested": dimensions},
            )
        stored_prefix = cast(
            func.subvector(RagChunkEntity.embedding, 1, dimensions),
            Vector(dimensions),
        )
        distance = stored_prefix.cosine_distance(query_embedding)
        similarity = (1 - distance).label("similarity_score")
        statement = (
            select(RagChunkEntity, RagDocumentEntity, similarity)
            .join(RagDocumentEntity, RagChunkEntity.document_id == RagDocumentEntity.id)
            .where(
                RagDocumentEntity.status == DocumentStatus.INDEXED.value,
                RagChunkEntity.embedding.is_not(None),
            )
        )
        if parameters.similarity_threshold > 0:
            statement = statement.where(similarity >= parameters.similarity_threshold)
        if parameters.country_code is not None:
            statement = statement.where(
                RagDocumentEntity.doc_metadata["country_code"].astext
                == parameters.country_code
            )
        statement = statement.order_by(
            distance, RagChunkEntity.document_id, RagChunkEntity.chunk_index
        ).limit(parameters.top_k)
        result = await self.session.execute(statement)
        return [
            SearchHit(
                document_id=document.id,
                document_title=document.title,
                chunk_id=chunk.id,
                chunk_content=chunk.content,
                chunk_index=chunk.chunk_index,
                similarity_score=float(score),
                hierarchy=(chunk.chunk_metadata or {}).get("hierarchy"),
                matched_chunk_ids=[chunk.id],
                metadata={
                    **(document.doc_metadata or {}),
                    **{
                        k: v
                        for k, v in (chunk.chunk_metadata or {}).items()
                        if k != "hierarchy"
                    },
                },
            )
            for chunk, document, score in result.all()
        ]

    async def replace_chunks(self, document_id: UUID, chunks: Sequence[Chunk]) -> None:
        """Replace indexed sections atomically when the caller commits."""
        await self.session.execute(
            delete(RagChunkEntity).where(RagChunkEntity.document_id == document_id)
        )
        await self.add_chunks(document_id, chunks)

    @staticmethod
    def _chunk_to_model(chunk: RagChunkEntity) -> Chunk:
        metadata = dict(chunk.chunk_metadata or {})
        hierarchy = metadata.pop("hierarchy", None)
        return Chunk(
            id=chunk.id,
            document_id=chunk.document_id,
            content=chunk.content,
            embedding=chunk.embedding,
            chunk_index=chunk.chunk_index,
            metadata=metadata,
            hierarchy=ChunkHierarchy.model_validate(hierarchy) if hierarchy else None,
            created_at=chunk.created_at,
        )

    def _entity_to_model(self, entity: RagDocumentEntity) -> Document:
        """Convert ORM entity to domain model."""
        chunks = [self._chunk_to_model(chunk) for chunk in entity.chunks or []]
        chunks.sort(key=lambda chunk: chunk.chunk_index)
        return Document(
            id=entity.id,
            title=entity.title,
            uri=entity.uri,
            mime_type=entity.mime_type,
            content_hash=entity.content_hash,
            metadata=entity.doc_metadata or {},
            status=entity.status,
            chunk_count=entity.chunk_count,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
            chunks=chunks,
        )
