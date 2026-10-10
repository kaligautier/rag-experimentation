"""Document indexation service for RAG."""

import asyncio
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, BinaryIO
from uuid import UUID, uuid4

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
    ConfigurationError,
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
        """Stage indexing; the caller owns commit and rollback.

        Use ``indexing_document`` around the caller's commit to compensate source
        writes if that commit fails. This lower-level method only compensates
        failures occurring before it returns.
        """
        async with self.indexing_document(
            content, title, uri, mime_type, filename, force
        ) as result:
            return result

    @asynccontextmanager
    async def indexing_document(
        self,
        content: bytes | BinaryIO,
        title: str,
        uri: str = "",
        mime_type: str = "application/octet-stream",
        filename: str = "",
        force: bool = False,
    ) -> AsyncIterator[IndexationResult]:
        """Index and compensate source changes if the caller's commit fails.

        Commit inside this block; rollback remains the caller's responsibility.
        Exceptions and cancellation restore the previous source, or remove a new
        source. This is local compensation, not crash/distributed atomicity:
        failed restoration or an ambiguous database commit needs manual recovery.
        A failing storage restore is propagated, chained to the original failure.

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
            yield IndexationResult(
                document=existing,
                created=False,
                chunk_count=existing.chunk_count,
            )
            return

        # A force upload rebuilds the index but retains the original source's
        # location and media type, even when the incoming display name changes.
        doc_id = existing.id if existing is not None else uuid4()
        storage_key = (
            await self._source_key(existing)
            if existing is not None
            else self.storage.generate_key(str(doc_id), content_hash, filename)
        )
        if existing is not None:
            uri = existing.uri
            mime_type = existing.mime_type
            filename = filename or storage_key.rsplit("/", 1)[-1]
            if storage_key == f"{doc_id}/{content_hash}" and not filename.endswith(
                (".md", ".markdown")
            ):
                filename = "source.md"

        # Extract text
        extractor = get_extractor(mime_type, filename)
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
        document = Document(
            id=doc_id,
            title=title,
            uri=existing.uri if existing is not None else uri or f"file://{filename}",
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
                "storage_key": storage_key,
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
            document = await self.repository.update_document(document)
            await self.repository.replace_chunks(doc_id, chunks)

        async with self._changing_source(storage_key, content_bytes):
            logger.info(
                "Document indexation completed in %.2fs", time.time() - start_time
            )
            yield IndexationResult(
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

    @asynccontextmanager
    async def deleting_document(self, document_id: UUID) -> AsyncIterator[None]:
        """Remove rows and source; the caller must commit inside this block.

        Failure/cancellation restores the source while caller rollback restores
        the rows. Missing sources are harmless. As for indexing, compensation
        cannot guarantee recovery from process crashes or failed storage writes.
        """
        document = await self.repository.get_document(document_id)
        storage_key = await self._source_key(document)
        async with self._changing_source(storage_key, None):
            await self.repository.delete_document(document_id)
            yield

    async def _source_key(self, document: Document) -> str:
        """Resolve a source only from its stored key or unambiguous chunk metadata.

        Legacy documents without either need manual source reconciliation; their
        URI is never a filesystem authority. Historical orphan copies created by
        older force-index behavior cannot be discovered safely from these fields.
        """
        prefix = f"{document.id}/{document.content_hash}"
        key = document.metadata.get("storage_key")
        if isinstance(key, str):
            if key == prefix:
                return key
            if key.startswith(prefix + "/"):
                filename = validate_storage_filename(key[len(prefix) + 1 :])
                if filename:
                    return key
            raise ConfigurationError("Document source key is invalid")

        if not document.chunks:
            document = await self.repository.get_document(document.id)
        filenames = {
            chunk.metadata["filename"]
            for chunk in document.chunks
            if isinstance(chunk.metadata.get("filename"), str)
        }
        if len(filenames) != 1:
            raise ConfigurationError(
                "Legacy document source cannot be identified; reconcile its source key",
                details={"document_id": str(document.id)},
            )
        filename = validate_storage_filename(filenames.pop())
        return self.storage.generate_key(
            str(document.id), document.content_hash, filename
        )

    @asynccontextmanager
    async def _changing_source(
        self, key: str, content: bytes | None
    ) -> AsyncIterator[None]:
        """Compensate one bounded source change, including caller cancellation."""
        try:
            original = await self.storage.get(
                key, max_bytes=settings.rag.MAX_UPLOAD_BYTES
            )
        except FileNotFoundError:
            original = None

        if original == content:
            # Same-hash force indexing does not need to rewrite unchanged bytes.
            yield
            return

        async def restore():
            if original is not None:
                await self.storage.put(key, original)
            else:
                try:
                    await self.storage.delete(key)
                except FileNotFoundError:
                    pass

        try:
            if content is None:
                await self.storage.delete(key)
            else:
                await self.storage.put(key, content)
            yield
        except BaseException:
            # Shield cleanup and wait for it even if a second cancellation arrives.
            # Never leave a background restore running after the session exits.
            restoration = asyncio.create_task(restore())
            while not restoration.done():
                try:
                    await asyncio.shield(restoration)
                except asyncio.CancelledError:
                    continue
            restoration.result()
            raise
