"""Document indexation service for RAG."""

import hashlib
import logging
import time
from typing import BinaryIO
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.parsing.chunker import get_chunker
from app.adapters.parsing.extractors import get_extractor
from app.adapters.persistence.database import require_embedding_schema
from app.adapters.storage.object_storage import get_storage, validate_storage_filename
from app.config.settings import settings
from app.ports.rag import RagRepositoryPort
from app.services.rag.models import Chunk, Document, DocumentStatus, IndexationResult
from app.utils.error import (
    DocumentLimitError,
    DocumentNotIndexedError,
    EmptyDocumentError,
)

logger = logging.getLogger(__name__)


class IndexationService:
    """Service for indexing documents (extract → chunk → embed → persist)."""

    def __init__(
        self,
        embeddings_client: EmbeddingsClient,
        repository: RagRepositoryPort,
        session: AsyncSession,
    ):
        """Initialize indexation service.

        Args:
            embeddings_client: Embeddings service
            repository: RAG repository
            session: AsyncSQLAlchemy session
        """
        self.embeddings_client = embeddings_client
        self.repository = repository
        self.session = session
        self.storage = get_storage()
        self.chunker = get_chunker()

    async def index_document(
        self,
        content: bytes | BinaryIO,
        title: str,
        uri: str = "",
        mime_type: str = "application/octet-stream",
        filename: str = "",
        force: bool = False,
    ) -> IndexationResult:
        """Index a document (extract → chunk → embed → persist).

        Args:
            content: Document content (bytes or file-like)
            title: Document title
            uri: Original URI/path
            mime_type: MIME type
            filename: Original filename
            force: Rebuild chunks and embeddings for an existing content hash

        Returns:
            IndexationResult with created flag and chunk count
        """
        start_time = time.time()
        filename = validate_storage_filename(filename)
        max_upload_bytes = settings.rag.MAX_UPLOAD_BYTES

        if isinstance(content, bytes):
            content_bytes = content
        else:
            content_bytes = content.read(max_upload_bytes + 1)
        if len(content_bytes) > max_upload_bytes:
            raise DocumentLimitError(
                f"Document exceeds the maximum upload size of {max_upload_bytes} bytes",
                details={"limit_bytes": max_upload_bytes},
            )

        await require_embedding_schema(self.session)

        # Calculate content hash for idempotence
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        logger.info(f"Indexing document: {title} (hash: {content_hash})")

        existing = await self.repository.get_document_by_hash(content_hash)
        if existing is not None and not force:
            if existing.status != DocumentStatus.INDEXED:
                raise DocumentNotIndexedError(
                    details={
                        "document_id": str(existing.id),
                        "status": existing.status.value,
                    }
                )
            return IndexationResult(
                document=existing,
                created=False,
                chunk_count=existing.chunk_count,
            )

        # Extract text
        extractor = await get_extractor(mime_type, filename)
        logger.info(f"Using {extractor.__class__.__name__} for extraction")
        raw_text = await extractor.extract(content_bytes)
        logger.info(f"Extracted {len(raw_text)} characters")

        # Chunk text
        sections = await self.chunker.split(raw_text)
        logger.info(f"Created {len(sections)} chunks")

        if not sections:
            raise EmptyDocumentError()
        max_chunks = settings.rag.MAX_CHUNKS_PER_DOC
        if len(sections) > max_chunks:
            raise DocumentLimitError(
                f"Document exceeds the maximum chunk count of {max_chunks}",
                details={"limit_chunks": max_chunks, "chunks": len(sections)},
            )

        # Embed chunks
        leaves = [section for section in sections if section.is_leaf]
        logger.info("Embedding %s leaves (%s total nodes)", len(leaves), len(sections))
        embeddings_start = time.time()
        embeddings = await self.embeddings_client.embed_documents(
            [section.embedding_text for section in leaves]
        )
        leaf_embeddings = dict(
            zip((section.id for section in leaves), embeddings, strict=True)
        )
        embedding_time_ms = (time.time() - embeddings_start) * 1000
        logger.info(f"Embeddings completed in {embedding_time_ms:.0f}ms")

        # Create domain models
        doc_id = existing.id if existing is not None else uuid4()
        document = Document(
            id=doc_id,
            title=title,
            uri=uri or f"file://{filename}",
            mime_type=mime_type,
            content_hash=content_hash,
            status=DocumentStatus.INDEXED,
            chunk_count=len(sections),
            metadata={
                **(existing.metadata if existing is not None else {}),
                "embedding_model": settings.rag.EMBEDDING_MODEL,
                "embedding_dimensions": settings.rag.EMBEDDING_DIMENSIONS,
                "hierarchy_version": 1,
                "leaf_count": len(leaves),
            },
            **({"created_at": existing.created_at} if existing is not None else {}),
        )

        chunks = [
            Chunk(
                id=section.id,
                document_id=doc_id,
                content=section.content,
                embedding=leaf_embeddings.get(section.id),
                chunk_index=idx,
                metadata={"filename": filename, **section.metadata},
                hierarchy=section.hierarchy,
            )
            for idx, section in enumerate(sections)
        ]

        # Persist
        if existing is None:
            await self.repository.create_document(document)
            await self.repository.add_chunks(doc_id, chunks)
        else:
            await self.repository.update_document(document)
            await self.repository.replace_chunks(doc_id, chunks)

        # Store original content
        storage_key = self.storage.generate_key(str(doc_id), content_hash, filename)
        await self.storage.put(storage_key, content_bytes)
        logger.info(f"Stored original content at: {storage_key}")

        total_time = time.time() - start_time
        logger.info(f"Document indexation completed in {total_time:.2f}s")

        return IndexationResult(
            document=document,
            created=existing is None,
            chunk_count=len(chunks),
            embedding_time_ms=embedding_time_ms,
        )

    async def get_document(self, document_id) -> Document:
        """Get document with chunks."""
        return await self.repository.get_document(document_id)

    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[Document], int]:
        """List documents."""
        return await self.repository.list_documents(limit, offset)

    async def delete_document(self, document_id) -> None:
        """Delete document."""
        await self.repository.delete_document(document_id)
