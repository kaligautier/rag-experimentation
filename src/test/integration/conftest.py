import os
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.adapters.persistence.rag_repository import RagRepository
from app.services.rag.models import Chunk, Document, DocumentStatus
from test.integration.support import orm_schema, vector


@pytest.fixture
async def repository():
    database_url = os.getenv("RAG_TEST_DB_URL")
    if not database_url:
        pytest.skip("Set RAG_TEST_DB_URL to run PostgreSQL/pgvector integration tests")

    async with orm_schema(database_url) as engine:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield RagRepository(session)


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
