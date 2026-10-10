"""Run the real changelog against an explicitly selected disposable container.

Requires RAG_TEST_DB_URL and RAG_TEST_LIQUIBASE_CONTAINER. Each test creates and
drops only its own schema. The container must expose PostgreSQL on its port 5432.
"""

import asyncio
import hashlib
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy.engine import make_url

from test.integration.support import orm_schema

CHANGELOG_DIR = Path(__file__).resolve().parents[2] / "liquibase"
RAG_TABLES = ["rag_documents", "rag_chunks"]
# Applied on a shared database, where Liquibase rejects edited changesets.
# Add each changelog file here once it has been deployed.
APPLIED_CHANGELOG_SHA256 = {
    "changelog/001-create-rag-tables.yaml": (
        "bcbe15ff0146e63ca7f611d3494195c1c87d6c9a04c8b3dc23b101b3ff7d267d"
    ),
}


@pytest.fixture
async def migration_db():
    container = os.getenv("RAG_TEST_LIQUIBASE_CONTAINER")
    if not container:
        pytest.skip("Set RAG_TEST_LIQUIBASE_CONTAINER to test real Liquibase")
    url = make_url(os.environ["RAG_TEST_DB_URL"])
    connection = await asyncpg.connect(
        url.set(drivername="postgresql").render_as_string(hide_password=False)
    )
    schema = f"rag_migration_{uuid4().hex}"
    try:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}", public')

        async def migrate(*command, java_timezone="UTC"):
            return await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    f"container:{container}",
                    # The JDBC session time zone follows the JVM default.
                    "-e",
                    f"JAVA_OPTS=-Duser.timezone={java_timezone}",
                    "-v",
                    f"{CHANGELOG_DIR}:/liquibase/changelog:ro",
                    "-w",
                    "/liquibase/changelog",
                    "liquibase/liquibase:4.33",
                    f"--url=jdbc:postgresql://localhost:5432/{url.database}"
                    f"?currentSchema={schema},public",
                    f"--username={url.username}",
                    f"--password={url.password}",
                    f"--default-schema-name={schema}",
                    "--changelog-file=db.changelog-master.yaml",
                    *command,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )

        yield connection, migrate, schema
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def embedding_type(connection):
    return await connection.fetchval(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
    )


async def schema_shape(connection, schema):
    """Structure the repository relies on: columns, indexes and foreign keys."""
    columns = await connection.fetch(
        "SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod), "
        "a.attnotnull FROM pg_attribute a "
        "JOIN pg_class c ON c.oid = a.attrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 AND c.relname = ANY($2::text[]) "
        "AND a.attnum > 0 AND NOT a.attisdropped",
        schema,
        RAG_TABLES,
    )
    indexes = await connection.fetch(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE schemaname = $1 AND tablename = ANY($2::text[])",
        schema,
        RAG_TABLES,
    )
    foreign_keys = await connection.fetch(
        "SELECT con.conname, pg_get_constraintdef(con.oid) FROM pg_constraint con "
        "JOIN pg_namespace n ON n.oid = con.connamespace "
        "WHERE n.nspname = $1 AND con.contype = 'f'",
        schema,
    )
    return {
        "columns": sorted(tuple(row) for row in columns),
        "indexes": sorted(
            (name, definition.replace(f"{schema}.", "")) for name, definition in indexes
        ),
        "foreign_keys": sorted(
            (name, definition.replace(f"{schema}.", ""))
            for name, definition in foreign_keys
        ),
    }


def should_never_edit_changesets_applied_on_shared_databases():
    applied = {
        path: hashlib.sha256((CHANGELOG_DIR / path).read_bytes()).hexdigest()
        for path in APPLIED_CHANGELOG_SHA256
    }

    assert applied == APPLIED_CHANGELOG_SHA256, "Add a new changeset instead"


async def should_create_nullable_3072_vectors_and_apply_changesets_once(
    migration_db,
):
    connection, migrate, _ = migration_db
    # The fourth changeset creates rag_chunks: dimensions must already be right.
    result = await migrate("update-count", "--count=4")
    assert result.returncode == 0, result.stderr
    assert await embedding_type(connection) == "vector(3072)"
    result = await migrate("update")
    assert result.returncode == 0, result.stderr
    assert not await connection.fetchval(
        "SELECT attnotnull FROM pg_attribute "
        "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
    )
    applied = await connection.fetchval("SELECT count(*) FROM databasechangelog")
    result = await migrate("update")
    assert result.returncode == 0, result.stderr
    assert (
        await connection.fetchval("SELECT count(*) FROM databasechangelog") == applied
    )


async def should_backfill_legacy_nulls_in_utc_before_requiring_values(
    migration_db,
):
    connection, migrate, _ = migration_db
    # Shared databases already ran the seven changesets of the first file.
    result = await migrate("update-count", "--count=7")
    assert result.returncode == 0, result.stderr
    document_id = uuid4()
    await connection.execute(
        "INSERT INTO rag_documents (id, title, uri, mime_type, content_hash, "
        "doc_metadata, status, chunk_count, created_at, updated_at) "
        "VALUES ($1, 'Legacy', 'legacy.md', 'text/markdown', 'legacy', "
        "NULL, NULL, NULL, NULL, NULL)",
        document_id,
    )
    await connection.execute(
        "INSERT INTO rag_chunks (id, document_id, content, chunk_index, "
        "chunk_metadata, created_at) VALUES ($1, $2, 'Legacy', 0, NULL, NULL)",
        uuid4(),
        document_id,
    )

    # UTC+14: a session-local NOW() would land far from the app's utcnow().
    result = await migrate("update", java_timezone="Pacific/Kiritimati")

    assert result.returncode == 0, result.stderr
    document = await connection.fetchrow(
        "SELECT doc_metadata, status, chunk_count, created_at, updated_at "
        "FROM rag_documents"
    )
    chunk = await connection.fetchrow(
        "SELECT chunk_metadata, created_at FROM rag_chunks"
    )
    assert (document["doc_metadata"], document["status"], document["chunk_count"]) == (
        "{}",
        "pending",
        0,
    )
    assert chunk["chunk_metadata"] == "{}"
    utc_now = datetime.utcnow()
    for timestamp in (
        document["created_at"],
        document["updated_at"],
        chunk["created_at"],
    ):
        assert abs(timestamp - utc_now) < timedelta(minutes=5)


async def should_migrate_to_the_schema_repository_tests_build(migration_db):
    connection, migrate, schema = migration_db
    result = await migrate("update")
    assert result.returncode == 0, result.stderr

    async with orm_schema(os.environ["RAG_TEST_DB_URL"]) as engine:
        orm_schema_name = engine.get_execution_options()["schema_translate_map"][None]
        assert await schema_shape(connection, orm_schema_name) == await schema_shape(
            connection, schema
        )


@pytest.mark.parametrize("dimensions", [128, 768, 3072])
async def should_provision_configured_vectors_and_preserve_them_on_repeat(
    migration_db, dimensions
):
    connection, migrate, _ = migration_db
    parameter = f"-Drag.embedding.dimensions={dimensions}"
    result = await migrate("update", parameter)
    assert result.returncode == 0, result.stderr
    assert await embedding_type(connection) == f"vector({dimensions})"
    document_id, chunk_id = uuid4(), uuid4()
    await connection.execute(
        "INSERT INTO rag_documents (id, title, uri, mime_type, content_hash) "
        "VALUES ($1, 'Configured', 'configured.md', 'text/markdown', 'configured')",
        document_id,
    )
    vector = "[" + ",".join(["1"] + ["0"] * (dimensions - 1)) + "]"
    await connection.execute(
        "INSERT INTO rag_chunks (id, document_id, content, chunk_index, embedding) "
        "VALUES ($1, $2, 'Configured', 0, $3::vector)",
        chunk_id,
        document_id,
        vector,
    )
    result = await migrate("update", parameter)
    assert result.returncode == 0, result.stderr
    assert (
        await connection.fetchval(
            "SELECT vector_dims(embedding) FROM rag_chunks WHERE id = $1", chunk_id
        )
        == dimensions
    )
    assert await connection.fetchval(
        "SELECT embedding <=> $1::vector FROM rag_chunks WHERE id = $2",
        vector,
        chunk_id,
    ) == pytest.approx(0)


async def should_refuse_to_resize_existing_vectors_without_losing_data(migration_db):
    connection, migrate, _ = migration_db
    result = await migrate("update-count", "--count=7")
    assert result.returncode == 0, result.stderr
    document_id, chunk_id = uuid4(), uuid4()
    await connection.execute(
        "INSERT INTO rag_documents (id, title, uri, mime_type, content_hash) "
        "VALUES ($1, 'Retained', 'retained.md', 'text/markdown', 'retained')",
        document_id,
    )
    vector = "[" + ",".join(["1"] + ["0"] * 3071) + "]"
    await connection.execute(
        "INSERT INTO rag_chunks (id, document_id, content, chunk_index, embedding) "
        "VALUES ($1, $2, 'Retained', 0, $3::vector)",
        chunk_id,
        document_id,
        vector,
    )

    result = await migrate("update", "-Drag.embedding.dimensions=768")

    assert result.returncode != 0
    assert await embedding_type(connection) == "vector(3072)"
    assert (
        await connection.fetchval(
            "SELECT embedding::text FROM rag_chunks WHERE id = $1", chunk_id
        )
        == vector
    )
