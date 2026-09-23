"""AsyncSQLAlchemy database configuration with pgvector support."""

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
from app.utils.error import EmbeddingSchemaMismatchError

logger = logging.getLogger(__name__)


async def require_embedding_schema(connection: AsyncConnection | AsyncSession) -> None:
    """Reject incompatible vectors before contacting the embedding provider."""
    vector_type = await connection.scalar(
        text(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
        )
    )
    expected_type = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    if vector_type != expected_type:
        raise EmbeddingSchemaMismatchError(
            f"RAG schema uses {vector_type}, expected {expected_type}. "
            "Migrate stored vectors to the configured embedding dimension "
            "before retrying.",
            details={"stored": vector_type, "expected": expected_type},
        )


async def setup_pgvector() -> None:
    """Initialize pgvector extension in PostgreSQL."""
    async with _async_engine.begin() as connection:
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    logger.info("pgvector extension initialized")


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
    """Initialize database (create tables, extensions, etc.)."""
    await setup_pgvector()

    # Create all tables from ORM entities
    from app.adapters.persistence.entities import Base

    async with _async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Additive compatibility migration: parents carry text and links, no vector.
        await conn.execute(
            text("ALTER TABLE rag_chunks ALTER COLUMN embedding DROP NOT NULL")
        )
        try:
            await require_embedding_schema(conn)
        except EmbeddingSchemaMismatchError as error:
            logger.warning("%s Document browsing remains available.", error)

    logger.info("Database initialization complete")
