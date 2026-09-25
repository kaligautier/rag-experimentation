"""SQLAlchemy ORM entities for RAG domain models."""

from datetime import datetime
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

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
    doc_metadata = Column(JSONB, nullable=False, default={})
    status = Column(String(20), nullable=False, default="pending")
    chunk_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationship to chunks
    chunks = relationship(
        "RagChunkEntity", back_populates="document", cascade="all, delete-orphan"
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
        UUID(as_uuid=True), ForeignKey("rag_documents.id"), nullable=False
    )
    content = Column(Text, nullable=False)
    embedding = Column(Vector(settings.rag.EMBEDDING_DIMENSIONS), nullable=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationship back to document
    document = relationship("RagDocumentEntity", back_populates="chunks")

    __table_args__ = (
        Index("ix_rag_chunks_document_id", "document_id"),
        Index("ix_rag_chunks_chunk_index", "document_id", "chunk_index"),
    )
