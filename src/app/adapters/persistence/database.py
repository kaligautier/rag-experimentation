"""Async SQLAlchemy engine and sessions for the Liquibase-managed RAG schema."""

import logging
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config.settings import settings
from app.utils.error import ConfigurationError, EmbeddingSchemaMismatchError

logger = logging.getLogger(__name__)


async def require_embedding_schema(connection: AsyncConnection | AsyncSession) -> None:
    """Reject missing or incompatible vectors before calling the embedding provider."""
    vector_type = await connection.scalar(
        text(
            "SELECT t.typname || '(' || a.atttypmod || ')' "
            "FROM pg_attribute a JOIN pg_type t ON t.oid = a.atttypid "
            "WHERE a.attrelid = to_regclass('rag_chunks') "
            "AND a.attname = 'embedding' AND NOT a.attisdropped "
            "AND to_regclass('rag_documents') IS NOT NULL"
        )
    )
    if vector_type is None:
        raise ConfigurationError(
            "RAG schema not found. Apply the Liquibase migrations with `just migrate`.",
            details={"tables": ["rag_documents", "rag_chunks"]},
        )
    expected_type = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    if vector_type != expected_type:
        raise EmbeddingSchemaMismatchError(
            f"RAG schema uses {vector_type}, expected {expected_type}. "
            "Migrate stored vectors to the configured embedding dimension "
            "before retrying.",
            details={"stored": vector_type, "expected": expected_type},
        )


def _get_async_engine():
    """Create the async engine; the Vector SQLAlchemy type handles conversion."""
    engine = create_async_engine(
        settings.rag.DB_URL,
        echo=settings.DEBUG,
        pool_size=settings.rag.DB_POOL_SIZE,
        max_overflow=settings.rag.DB_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=3600,
        connect_args={
            "timeout": 30,
        },
    )

    return engine


# Global async engine and session factory
_async_engine = _get_async_engine()
_async_session_factory = async_sessionmaker(
    bind=_async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: release the session on success, errors or cancellation.

    Write handlers commit explicitly before returning. Closing the session rolls
    back unfinished transactions, including failed writes and read transactions.
    """
    async with _async_session_factory() as session:
        yield session


async def get_session() -> AsyncSession:
    """Create a new async session (one-shot, caller responsible for close)."""
    return _async_session_factory()


async def close_db() -> None:
    """Close the database connection pool."""
    await _async_engine.dispose()


async def init_db() -> None:
    """Check the migrated schema at startup; Liquibase alone creates and alters it.

    Missing tables stop startup. Incompatible vectors only block embedding
    operations, so stored documents remain browsable until they are migrated.
    """
    async with _async_engine.connect() as connection:
        try:
            await require_embedding_schema(connection)
        except EmbeddingSchemaMismatchError as error:
            logger.warning("%s Document browsing remains available.", error)

    logger.info("Database schema check complete")
