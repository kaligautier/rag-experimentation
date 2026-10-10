from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from app.adapters.persistence.entities import Base
from app.config.settings import settings


def vector(x: float, y: float) -> list[float]:
    """Use the real storage dimension while keeping cosine scores transparent."""
    return [x, y] + [0.0] * (settings.rag.EMBEDDING_DIMENSIONS - 2)


@asynccontextmanager
async def orm_schema(database_url: str) -> AsyncIterator[AsyncEngine]:
    """Yield an engine bound to a disposable schema holding the ORM tables."""
    schema = f"rag_test_{uuid4().hex}"
    engine = create_async_engine(
        database_url,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": f"{schema},public"}},
        execution_options={"schema_translate_map": {None: schema}},
    )
    schema_created = False
    try:
        async with engine.begin() as connection:
            # Database-wide and shared: keep it out of the schema dropped below.
            await connection.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            )
            await connection.execute(CreateSchema(schema))
            await connection.run_sync(
                lambda sync_connection: Base.metadata.create_all(
                    sync_connection, checkfirst=False
                )
            )
            assert await connection.scalar(text("SELECT current_schema()")) == schema
            tables = await connection.scalars(
                text("SELECT tablename FROM pg_tables WHERE schemaname = :schema"),
                {"schema": schema},
            )
            assert set(tables) == {"rag_documents", "rag_chunks"}
        schema_created = True
        yield engine
    finally:
        try:
            if schema_created:
                async with engine.begin() as connection:
                    await connection.execute(DropSchema(schema, cascade=True))
        finally:
            await engine.dispose()
