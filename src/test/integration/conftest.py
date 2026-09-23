import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.schema import CreateSchema, DropSchema

from app.adapters.persistence.entities import Base
from app.adapters.persistence.rag_repository import RagRepository
from app.services.rag.models import Chunk, Document, DocumentStatus
from test.integration.support import vector


@pytest.fixture
async def repository():
    database_url = os.getenv("RAG_TEST_DB_URL")
    if not database_url:
        pytest.skip("Set RAG_TEST_DB_URL to run PostgreSQL/pgvector integration tests")

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

        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield RagRepository(session)
    finally:
        try:
            if schema_created:
                async with engine.begin() as connection:
                    await connection.execute(DropSchema(schema, cascade=True))
        finally:
            await engine.dispose()


@pytest.fixture
async def searchable_chunks(repository):
    samples = [
        ("aligned", vector(20.0, 0.0), "FR", DocumentStatus.INDEXED),
        ("british", vector(12.0, 5.0), "GB", DocumentStatus.INDEXED),
        ("near_l2", vector(2.0, 2.0), "FR", DocumentStatus.INDEXED),
        ("without_country", vector(3.0, 4.0), None, DocumentStatus.INDEXED),
        ("orthogonal", vector(0.0, 3.0), "FR", DocumentStatus.INDEXED),
        ("opposite", vector(-4.0, 0.0), "FR", DocumentStatus.INDEXED),
        ("pending", vector(2.0, 0.0), "FR", DocumentStatus.PENDING),
        ("failed", vector(2.0, 0.0), "FR", DocumentStatus.FAILED),
    ]
    chunks = {}
    for name, embedding, country_code, status in samples:
        document = Document(
            title=f"Document {name}",
            uri=f"file:///{name}.txt",
            mime_type="text/plain",
            content_hash=uuid4().hex,
            metadata={"country_code": country_code} if country_code else {},
            status=status,
            chunk_count=1,
        )
        await repository.create_document(document)
        chunk = Chunk(
            document_id=document.id,
            content=f"Content for {name}",
            embedding=embedding,
            chunk_index=2,
            # A different chunk country must never override the document filter.
            metadata={"country_code": "CH", "page": 3},
        )
        await repository.add_chunks(document.id, [chunk])
        chunks[name] = chunk

    await repository.session.commit()
    repository.session.expunge_all()
    return chunks
