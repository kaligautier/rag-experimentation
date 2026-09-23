"""Domain models for RAG (Retrieval-Augmented Generation)."""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class DocumentStatus(str, Enum):
    """Document processing status."""

    INDEXED = "indexed"
    PENDING = "pending"
    FAILED = "failed"


class ChunkHierarchy(BaseModel):
    """Persisted links between a context node and its direct children."""

    version: Literal[1] = 1
    parent_id: UUID | None = None
    child_ids: list[UUID] = Field(default_factory=list)
    level: int = Field(default=0, ge=0)
    token_count: int = Field(default=0, ge=0)


class ParsedChunk(BaseModel):
    """A Markdown context or a searchable leaf section."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    metadata: dict = Field(default_factory=dict)
    hierarchy: ChunkHierarchy = Field(default_factory=ChunkHierarchy)

    @property
    def is_leaf(self) -> bool:
        return not self.hierarchy.child_ids

    @property
    def embedding_text(self) -> str:
        """Include parent headings in retrieval without changing the source text."""
        header_path = self.metadata.get("header_path", "/")
        if header_path == "/":
            return self.content
        return f"{header_path}\n\n{self.content}"


class Chunk(BaseModel):
    """A chunk of text from a document with embedding."""

    id: UUID = Field(default_factory=uuid4, description="Unique chunk identifier")
    document_id: UUID = Field(description="Parent document identifier")
    content: str = Field(description="Chunk text content")
    embedding: list[float] | None = Field(
        default=None, description="Leaf vector; parent contexts have no embedding"
    )
    chunk_index: int = Field(description="Sequence position in document")
    metadata: dict = Field(default_factory=dict, description="Custom metadata")
    hierarchy: ChunkHierarchy | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_leaf(self) -> bool:
        return self.hierarchy is None or not self.hierarchy.child_ids


class Document(BaseModel):
    """A document with metadata and chunks."""

    id: UUID = Field(default_factory=uuid4, description="Unique document identifier")
    title: str = Field(description="Document title")
    uri: str = Field(description="Original document URI/path")
    mime_type: str = Field(description="MIME type (application/pdf, etc.)")
    content_hash: str = Field(description="SHA256 hash of original content")
    metadata: dict = Field(default_factory=dict, description="Custom metadata")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    chunk_count: int = Field(default=0, description="Number of chunks created")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    chunks: list[Chunk] = Field(
        default_factory=list, description="Nested chunks (optional)"
    )


class SearchHit(BaseModel):
    """A search result hit from vector similarity search."""

    document_id: UUID = Field(description="Document identifier")
    chunk_id: UUID = Field(description="Chunk identifier")
    chunk_content: str = Field(description="Chunk text")
    similarity_score: float = Field(description="Cosine similarity score [-1, 1]")
    document_title: str = Field(description="Document title for context")
    chunk_index: int = Field(description="Chunk sequence position")
    metadata: dict = Field(default_factory=dict, description="Chunk/document metadata")
    hierarchy: ChunkHierarchy | None = None
    matched_chunk_ids: list[UUID] = Field(default_factory=list)
    score_type: Literal["cosine", "max_leaf_cosine"] = "cosine"


class SearchParameters(BaseModel):
    """Parameters for a RAG search query."""

    query: str = Field(min_length=1, description="Search query text")
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of results to return"
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity threshold (0.0 = no filter)",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="Optional country code filter",
    )
    hierarchical: bool = Field(
        default=True,
        description=(
            "Merge matching siblings into their parent context when a majority matches"
        ),
    )
    embedding_dimensions: int | None = Field(
        default=None,
        ge=128,
        le=3072,
        description=(
            "MRL prefix dimension used for this search; defaults to the full "
            "stored embedding dimension"
        ),
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        """Reject whitespace-only questions before running the embedding model."""
        query = query.strip()
        if not query:
            raise ValueError("Search query must not be blank")
        return query


class IndexationResult(BaseModel):
    """Result of document indexation operation."""

    document: Document = Field(description="Indexed document")
    created: bool = Field(description="True if newly created, False if updated")
    chunk_count: int = Field(description="Number of chunks created")
    embedding_time_ms: float = Field(
        default=0.0, description="Time spent on embeddings"
    )
