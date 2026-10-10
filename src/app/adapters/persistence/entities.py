"""SQLAlchemy ORM entities for RAG domain models.

Liquibase owns the schema; these mappings mirror it so tests can build the same
tables with ``create_all``. The Liquibase integration suite checks they match.
"""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, deferred, relationship

from app.config.settings import settings

Base = declarative_base()


class RagDocumentEntity(Base):
    """ORM entity for Document."""

    __tablename__ = "rag_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(500), nullable=False)
    uri = Column(String(1000), nullable=False)
    mime_type = Column(String(50), nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False, index=True)
    doc_metadata = Column(JSONB, nullable=False, default=dict)
    status = Column(String(20), nullable=False, default="pending")
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Loaded only on request; the database deletes chunks with their document.
    chunks = relationship(
        "RagChunkEntity",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        Index("ix_rag_documents_status", "status"),
        Index("ix_rag_documents_created_at", "created_at"),
    )


class RagChunkEntity(Base):
    """ORM entity for Chunk."""

    __tablename__ = "rag_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey(
            "rag_documents.id", name="fk_rag_chunks_document_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    # Write-only: reads never need vectors, which transfer as ~40 KB of text each.
    embedding = deferred(
        Column(Vector(settings.rag.EMBEDDING_DIMENSIONS), nullable=True),
        raiseload=True,
    )
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship back to document; async code must load it explicitly.
    document = relationship("RagDocumentEntity", back_populates="chunks", lazy="raise")

    __table_args__ = (
        Index("ix_rag_chunks_document_index", "document_id", "chunk_index"),
    )
