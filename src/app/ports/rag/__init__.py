"""RAG port interfaces for hexagonal architecture."""

from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from app.services.rag.models import (
    Chunk,
    Document,
    ParsedChunk,
    SearchHit,
    SearchParameters,
)


class RagRepositoryPort(ABC):
    """Port for RAG document and chunk persistence."""

    @abstractmethod
    async def create_document(self, document: Document) -> Document:
        """Create a new document."""
        pass

    @abstractmethod
    async def get_document(self, document_id: UUID) -> Document:
        """Get document by ID."""
        pass

    @abstractmethod
    async def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Find an existing document by its content hash."""
        pass

    @abstractmethod
    async def list_documents(
        self, limit: int = 100, offset: int = 0
    ) -> tuple[list[Document], int]:
        """List documents with pagination."""
        pass

    @abstractmethod
    async def update_document(self, document: Document) -> Document:
        """Update document metadata."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: UUID) -> None:
        """Delete document and its chunks."""
        pass

    @abstractmethod
    async def add_chunks(self, document_id: UUID, chunks: Sequence) -> None:
        """Add chunks to a document."""
        pass

    @abstractmethod
    async def replace_chunks(self, document_id: UUID, chunks: Sequence) -> None:
        """Replace a document's chunks inside the caller's transaction."""
        pass

    @abstractmethod
    async def get_chunks(self, chunk_ids: Sequence[UUID]) -> list[Chunk]:
        """Load linked parent contexts in a batch."""
        pass

    @abstractmethod
    async def search(
        self, parameters: SearchParameters, query_embedding: Sequence[float]
    ) -> list[SearchHit]:
        """Vector similarity search."""
        pass


class EmbeddingsServicePort(ABC):
    """Port for text embeddings."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of documents."""
        pass

    @abstractmethod
    async def embed_query(
        self, text: str, dimensions: int | None = None
    ) -> list[float]:
        """Embed a query."""
        pass


class ObjectStoragePort(ABC):
    """Port for object storage."""

    @abstractmethod
    async def get(self, key: str) -> bytes:
        """Get file from storage."""
        pass

    @abstractmethod
    async def put(self, key: str, content: bytes) -> str:
        """Put file to storage, return path."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete file from storage."""
        pass


class TextExtractorPort(ABC):
    """Port for text extraction from documents."""

    @abstractmethod
    async def extract(self, content: bytes) -> str:
        """Extract text from document content."""
        pass


class TextChunkerPort(ABC):
    """Port for text chunking."""

    @abstractmethod
    async def split(self, text: str) -> list[ParsedChunk]:
        """Split text into chunks."""
        pass


class RagUnitOfWork(ABC):
    """Port for transaction management."""

    @abstractmethod
    async def __aenter__(self):
        """Enter async context."""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context."""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """Commit transaction."""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback transaction."""
        pass

    @property
    @abstractmethod
    def rag(self) -> RagRepositoryPort:
        """Get RAG repository."""
        pass
