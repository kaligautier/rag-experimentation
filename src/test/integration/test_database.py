"""The startup check reads the Liquibase-managed schema and never changes it."""

import logging

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.adapters.persistence import database
from app.config.settings import settings
from app.utils.error import ConfigurationError, EmbeddingSchemaMismatchError


@pytest.fixture
def schema_engine(repository, monkeypatch):
    """Point the application engine at the isolated test schema."""
    engine = repository.session.bind
    monkeypatch.setattr(database, "_async_engine", engine)
    return engine


async def should_accept_the_configured_embedding_dimension(schema_engine, caplog):
    with caplog.at_level(logging.WARNING, logger=database.logger.name):
        await database.init_db()

    assert caplog.records == []


async def should_block_embeddings_and_only_warn_at_startup_when_dimensions_differ(
    schema_engine, monkeypatch, caplog
):
    stored = f"vector({settings.rag.EMBEDDING_DIMENSIONS})"
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "768")

    async with schema_engine.connect() as connection:
        with pytest.raises(EmbeddingSchemaMismatchError) as caught:
            await database.require_embedding_schema(connection)
    with caplog.at_level(logging.WARNING, logger=database.logger.name):
        await database.init_db()

    assert caught.value.details == {"stored": stored, "expected": "vector(768)"}
    assert "Document browsing remains available" in caplog.text


@pytest.mark.parametrize("table", ["rag_chunks", "rag_documents"])
async def should_refuse_to_start_before_migrations_create_the_tables(
    schema_engine, monkeypatch, table
):
    schema = schema_engine.get_execution_options()["schema_translate_map"][None]
    async with schema_engine.begin() as connection:
        await connection.execute(text(f"DROP TABLE {table} CASCADE"))
    # Only the test schema: a migrated public.rag_chunks must not answer for it.
    isolated = create_async_engine(
        schema_engine.url,
        poolclass=NullPool,
        connect_args={"server_settings": {"search_path": schema}},
    )
    monkeypatch.setattr(database, "_async_engine", isolated)

    try:
        with pytest.raises(ConfigurationError, match="just migrate"):
            await database.init_db()
    finally:
        await isolated.dispose()
