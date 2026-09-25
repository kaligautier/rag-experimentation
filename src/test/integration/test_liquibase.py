"""Run the real changelog against an explicitly selected disposable container.

Requires RAG_TEST_DB_URL and RAG_TEST_LIQUIBASE_CONTAINER. Each test creates and
drops only its own schema. The container must expose PostgreSQL on its port 5432.
"""

import asyncio
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from sqlalchemy.engine import make_url


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

        async def migrate(*command):
            changelog = Path(__file__).resolve().parents[2] / "liquibase"
            return await asyncio.to_thread(
                subprocess.run,
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    f"container:{container}",
                    "-v",
                    f"{changelog}:/liquibase/changelog:ro",
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

        yield connection, migrate
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


async def embedding_type(connection):
    return await connection.fetchval(
        "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
        "WHERE attrelid = 'rag_chunks'::regclass AND attname = 'embedding'"
    )


async def should_create_3072_vectors_in_initial_changeset(migration_db):
    connection, migrate = migration_db
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
    assert applied == 7
    result = await migrate("update")
    assert result.returncode == 0, result.stderr
    assert (
        await connection.fetchval("SELECT count(*) FROM databasechangelog") == applied
    )
