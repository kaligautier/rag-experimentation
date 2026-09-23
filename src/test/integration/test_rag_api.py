import json
from datetime import datetime
from hashlib import sha256
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence import database
from app.adapters.persistence.rag_repository import RagRepository
from app.application import register_error_handlers
from app.components.tools.custom.index_document import index_document as index_tool
from app.components.tools.custom.search_documents import search_documents as search_tool
from app.config.settings import settings
from app.routes import rag
from app.services.rag import indexation_service
from app.services.rag.models import Chunk, Document, DocumentStatus
from test.integration.support import vector


@pytest.fixture
def rag_runtime(repository, monkeypatch):
    """Keep real DB sessions while replacing model and filesystem boundaries."""
    engine = repository.session.bind
    assert engine is not None
    stats = {"active": 0, "checkouts": 0, "checkins": 0}
    embedded_queries = []

    def checkout(connection, record, proxy):
        stats["active"] += 1
        stats["checkouts"] += 1

    def checkin(connection, record):
        stats["active"] -= 1
        stats["checkins"] += 1

    async def embed_query(self, query, dimensions=None):
        embedded_queries.append((query, dimensions))
        return vector(2.0, 0.0)[:dimensions]

    async def embed_documents(self, texts):
        return [vector(2.0, 0.0) for _ in texts]

    class MemoryStorage:
        def __init__(self):
            self.objects = {}

        @staticmethod
        def generate_key(document_id, content_hash, filename):
            return f"{document_id}/{content_hash}/{filename}"

        async def put(self, key, content):
            self.objects[key] = content
            return f"memory://{key}"

    storage = MemoryStorage()
    monkeypatch.setattr(
        database,
        "_async_session_factory",
        async_sessionmaker(engine, expire_on_commit=False),
    )
    monkeypatch.setattr(EmbeddingsClient, "embed_query", embed_query)
    monkeypatch.setattr(EmbeddingsClient, "embed_documents", embed_documents)
    monkeypatch.setattr(indexation_service, "get_storage", lambda: storage)
    event.listen(engine.sync_engine, "checkout", checkout)
    event.listen(engine.sync_engine, "checkin", checkin)
    try:
        yield stats, embedded_queries, storage
    finally:
        event.remove(engine.sync_engine, "checkout", checkout)
        event.remove(engine.sync_engine, "checkin", checkin)


@pytest.fixture
async def rag_client(rag_runtime):
    app = FastAPI()
    app.include_router(rag.router)
    register_error_handlers(app)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://rag.test"
    ) as client:
        yield client


async def should_release_http_sessions_on_success_and_errors(
    searchable_chunks, rag_runtime, rag_client
):
    stats, embedded_queries, storage = rag_runtime
    previous_checkouts = stats["checkouts"]

    response = await rag_client.get("/rag/documents")

    assert response.status_code == 200
    assert response.json()["total"] == 8
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    response = await rag_client.get(f"/rag/documents/{uuid4()}")

    assert response.status_code == 404
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    response = await rag_client.post(
        "/rag/search",
        json={"query": "return policy", "top_k": 10, "country_code": "GB"},
    )

    assert response.status_code == 200
    assert response.json()["result_count"] == 1
    assert response.json()["embedding_dimensions"] == settings.rag.EMBEDDING_DIMENSIONS
    assert response.json()["results"][0]["chunk_id"] == str(
        searchable_chunks["british"].id
    )
    assert embedded_queries == [("return policy", settings.rag.EMBEDDING_DIMENSIONS)]
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    response = await rag_client.post(
        "/rag/search",
        json={
            "query": "return policy",
            "top_k": 10,
            "country_code": "GB",
            "embedding_dimensions": 768,
        },
    )

    assert response.status_code == 200
    assert response.json()["embedding_dimensions"] == 768
    assert response.json()["results"][0]["chunk_id"] == str(
        searchable_chunks["british"].id
    )
    assert embedded_queries[-1] == ("return policy", 768)
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    response = await rag_client.post(
        "/rag/index",
        data={"title": "Invalid UTF-8"},
        files={"file": ("invalid.md", b"\xff\xfe", "text/markdown")},
    )

    assert response.status_code == 400
    assert response.json()["error_code"] == "UNSUPPORTED_DOCUMENT"
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    assert stats["checkins"] == stats["checkouts"]
    assert storage.objects == {}


async def should_reindex_existing_content_without_changing_document_id(
    repository, rag_runtime, rag_client
):
    content = b"# GR20\n\n## Tips\n\n### Equipment\n- Hiking boots\n- Backpack"
    document = Document(
        title="GR20",
        uri="gr20.md",
        mime_type="text/markdown",
        content_hash=sha256(content).hexdigest(),
        status=DocumentStatus.INDEXED,
        chunk_count=1,
    )
    old_chunk = Chunk(
        document_id=document.id,
        content="## Tips",
        embedding=vector(2.0, 0.0),
        chunk_index=0,
    )
    await repository.create_document(document)
    await repository.add_chunks(document.id, [old_chunk])
    await repository.session.commit()

    response = await rag_client.post(
        "/rag/index",
        data={"title": "GR20", "force": "true"},
        files={"file": ("gr20.md", content, "text/markdown")},
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == str(document.id)
    assert response.json()["created"] is False
    repository.session.expunge_all()
    stored = await repository.get_document(document.id)
    assert all(chunk.id != old_chunk.id for chunk in stored.chunks)
    assert stored.chunk_count == 3
    leaves = [chunk for chunk in stored.chunks if chunk.is_leaf]
    assert len(leaves) == 1
    assert leaves[0].content == "### Equipment\n- Hiking boots\n- Backpack"
    assert leaves[0].metadata["header_path"] == "/GR20/Tips/"


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("POST", "/rag/search", {"query": ""}),
        ("POST", "/rag/search", {"query": "   "}),
        ("POST", "/rag/search", {"query": "returns", "top_k": 0}),
        ("POST", "/rag/search", {"query": "returns", "top_k": -1}),
        ("POST", "/rag/search", {"query": "returns", "top_k": 101}),
        ("POST", "/rag/search", {"query": "returns", "embedding_dimensions": 127}),
        ("POST", "/rag/search", {"query": "returns", "embedding_dimensions": 3073}),
        ("GET", "/rag/documents?limit=-1", None),
        ("GET", "/rag/documents?offset=-1", None),
    ],
)
async def should_reject_invalid_http_rag_requests(
    rag_runtime, rag_client, method, path, payload
):
    stats, embedded_queries, _ = rag_runtime

    response = await rag_client.request(method, path, json=payload)

    assert response.status_code == 422
    assert stats == {"active": 0, "checkouts": 0, "checkins": 0}
    assert embedded_queries == []


async def should_reject_unsafe_rest_and_adk_filenames_before_writes(
    repository, rag_runtime, rag_client
):
    _, _, storage = rag_runtime
    content = b"# Unsafe filename\n\nMust never escape storage."
    unsafe_filename = "../../../../tmp/outside.md"

    response = await rag_client.post(
        "/rag/index",
        data={"title": "Unsafe filename"},
        files={"file": (unsafe_filename, content, "text/markdown")},
    )
    tool_result = await index_tool(
        unsafe_filename,
        content.decode("utf-8"),
        mime_type="text/markdown",
    )

    assert response.status_code == 400
    assert tool_result["status"] == "error"
    assert "filename" in tool_result["error"]
    assert storage.objects == {}
    assert await repository.get_document_by_hash(sha256(content).hexdigest()) is None


async def should_reject_oversized_rest_and_adk_uploads_before_writes(
    repository, rag_runtime, rag_client, monkeypatch
):
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "16")
    _, _, storage = rag_runtime
    content = b"# Oversized\n\n" + b"x" * 32

    response = await rag_client.post(
        "/rag/index",
        data={"title": "Oversized"},
        files={"file": ("oversized.md", content, "text/markdown")},
    )
    tool_result = await index_tool(
        "oversized.md",
        content.decode("utf-8"),
        mime_type="text/markdown",
    )

    assert response.status_code == 413
    assert tool_result["status"] == "error"
    assert "maximum upload size" in tool_result["error"]
    assert storage.objects == {}
    assert await repository.get_document_by_hash(sha256(content).hexdigest()) is None


async def should_release_adk_tool_sessions_and_rollback_failed_indexation(
    searchable_chunks, rag_runtime, monkeypatch
):
    stats, _, storage = rag_runtime
    previous_checkouts = stats["checkouts"]

    result = await search_tool("return policy", top_k=2)

    assert result["status"] == "success"
    assert [hit["chunk_id"] for hit in result["results"]] == [
        str(searchable_chunks["aligned"].id),
        str(searchable_chunks["british"].id),
    ]
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    async def failed_search(self, parameters, query_embedding):
        await self.session.execute(text("SELECT 1 / 0"))

    with monkeypatch.context() as patch:
        patch.setattr(RagRepository, "search", failed_search)
        result = await search_tool("return policy")

    assert result["status"] == "error"
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    content = b"# Returns\n\nKeep your receipt."
    result = await index_tool(
        "returns.md", content.decode("utf-8"), mime_type="text/markdown"
    )

    assert result["status"] == "success"
    assert result["chunk_count"] > 0
    assert list(storage.objects.values()) == [content]
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    async with database._async_session_factory() as session:
        stored = await RagRepository(session).get_document_by_hash(
            sha256(content).hexdigest()
        )
        assert stored is not None
        assert str(stored.id) == result["document_id"]
    assert stats["active"] == 0
    previous_checkouts = stats["checkouts"]

    async def failed_storage(key, content):
        raise RuntimeError("Storage unavailable")

    failed_content = b"# Failed indexing\n\nThis write must be rolled back."
    with monkeypatch.context() as patch:
        patch.setattr(storage, "put", failed_storage)
        result = await index_tool(
            "failed.md", failed_content.decode("utf-8"), mime_type="text/markdown"
        )

    assert result["status"] == "error"
    assert result["error_code"] == "GENERIC_ERROR"
    assert "Storage unavailable" not in result["error"]
    assert stats["checkouts"] > previous_checkouts
    assert stats["active"] == 0
    async with database._async_session_factory() as session:
        assert (
            await RagRepository(session).get_document_by_hash(
                sha256(failed_content).hexdigest()
            )
            is None
        )
    assert stats["active"] == 0
    assert stats["checkins"] == stats["checkouts"]


async def legacy_state(engine, schema):
    """Read all persisted fields and the vector constraint in the owned schema."""
    async with engine.connect() as connection:
        assert await connection.scalar(text("SELECT current_schema()")) == schema
        state = {}
        for key, table in (("documents", "rag_documents"), ("chunks", "rag_chunks")):
            rows = await connection.execute(
                text(f'SELECT * FROM "{schema}".{table} ORDER BY id')
            )
            state[key] = [dict(row) for row in rows.mappings()]
        state["embedding_type"] = await connection.scalar(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
            ),
            {"table": f"{schema}.rag_chunks"},
        )
        return state


@pytest.fixture
async def legacy_vectors(repository):
    """Create incompatible vectors in the existing isolated test schema."""
    engine = repository.session.bind
    assert engine is not None
    schema = engine.get_execution_options()["schema_translate_map"][None]
    async with engine.begin() as connection:
        assert await connection.scalar(text("SELECT current_schema()")) == schema
        assert (
            await connection.scalar(
                text("SELECT count(*) FROM pg_tables WHERE schemaname = :schema"),
                {"schema": schema},
            )
            == 2
        )
        await connection.execute(
            text(
                f'ALTER TABLE "{schema}".rag_chunks '
                "ALTER COLUMN embedding TYPE vector(384)"
            )
        )

    document = Document(
        title="GR20 migration fixture",
        uri="gr20-migration.md",
        mime_type="text/markdown",
        content_hash=sha256(b"migration fixture").hexdigest(),
        status=DocumentStatus.INDEXED,
        chunk_count=2,
        metadata={
            "country_code": "FR",
            "source": {"revision": 3},
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
        },
        created_at=datetime(2026, 1, 1),
        updated_at=datetime(2026, 1, 2),
    )
    await repository.create_document(document)
    await repository.session.commit()
    repository.session.expunge_all()
    chunks = [
        {
            "id": uuid4(),
            "document_id": document.id,
            "content": "### Equipment\n\n- Boots\n- Backpack",
            "embedding": json.dumps([3.0, 4.0] + [0.0] * 382),
            "chunk_index": 0,
            "metadata": json.dumps(
                {
                    "header_path": "/GR20/Tips/",
                    "section_title": "Equipment",
                    "header_level": 3,
                    "filename": "gr20-migration.md",
                }
            ),
            "created_at": datetime(2026, 1, 1),
        },
        {
            "id": uuid4(),
            "document_id": document.id,
            "content": "Introduction without ancestor headings.",
            "embedding": json.dumps([0.0, 2.0] + [0.0] * 382),
            "chunk_index": 1,
            "metadata": json.dumps({"filename": "gr20-migration.md", "page": 2}),
            "created_at": datetime(2026, 1, 1),
        },
    ]
    async with engine.begin() as connection:
        await connection.execute(
            text(
                f'INSERT INTO "{schema}".rag_chunks '
                "(id, document_id, content, embedding, chunk_index, "
                "chunk_metadata, created_at) VALUES "
                "(:id, :document_id, :content, CAST(:embedding AS vector), "
                ":chunk_index, CAST(:metadata AS jsonb), :created_at)"
            ),
            chunks,
        )
    original = await legacy_state(engine, schema)
    assert original["embedding_type"] == "vector(384)"
    return engine, schema, original


async def should_allow_browsing_legacy_chunks_while_blocking_embedding_operations(
    legacy_vectors, monkeypatch
):
    from app.application import rag_lifespan

    engine, schema, original = legacy_vectors
    monkeypatch.setattr(database, "_async_engine", engine)
    monkeypatch.setattr(
        database,
        "_async_session_factory",
        async_sessionmaker(engine, expire_on_commit=False),
    )
    embed_query = AsyncMock(side_effect=AssertionError("Must not call Gemini"))
    embed_documents = AsyncMock(side_effect=AssertionError("Must not call Gemini"))
    monkeypatch.setattr(EmbeddingsClient, "embed_query", embed_query)
    monkeypatch.setattr(EmbeddingsClient, "embed_documents", embed_documents)

    app = FastAPI(lifespan=rag_lifespan)
    app.include_router(rag.router)
    register_error_handlers(app)
    document_id = original["documents"][0]["id"]
    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://rag.test"
        ) as client:
            listing = await client.get("/rag/documents")
            assert listing.status_code == 200
            assert listing.json()["total"] == 1

            detail = await client.get(f"/rag/documents/{document_id}")
            assert detail.status_code == 200
            chunks = detail.json()["document"]["chunks"]
            assert {chunk["id"]: chunk["content"] for chunk in chunks} == {
                str(chunk["id"]): chunk["content"] for chunk in original["chunks"]
            }

            search = await client.post("/rag/search", json={"query": "Equipment"})
            upload = await client.post(
                "/rag/index",
                data={"title": "Blocked upload"},
                files={"file": ("test.md", b"# Test\nBody", "text/markdown")},
            )
            for response in (search, upload):
                assert response.status_code == 503
                assert "Migrate stored vectors" in response.json()["message"]

        for result in (
            await search_tool("Equipment"),
            await index_tool("test.md", "# Test\nBody", mime_type="text/markdown"),
        ):
            assert result["status"] == "error"
            assert "Migrate stored vectors" in result["error"]

    embed_query.assert_not_awaited()
    embed_documents.assert_not_awaited()
    assert await legacy_state(engine, schema) == original


async def should_return_413_for_a_heading_larger_than_the_token_budget(
    rag_client, rag_runtime, monkeypatch
):
    from app.adapters.parsing.chunker import HierarchicalChunker

    monkeypatch.setattr(
        indexation_service,
        "get_chunker",
        lambda: HierarchicalChunker(chunk_size=16, chunk_overlap=0),
    )
    response = await rag_client.post(
        "/rag/index",
        data={"title": "Long heading"},
        files={
            "file": ("long.md", "# " + "heading " * 40 + "\n\nBody.", "text/markdown")
        },
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "DOCUMENT_TOO_LARGE"
    assert response.json()["details"]["limit_tokens"] == 16
    assert rag_runtime[2].objects == {}


@pytest.mark.parametrize("status", [DocumentStatus.PENDING, DocumentStatus.FAILED])
async def should_return_conflict_and_allow_forced_reindex_of_unindexed_document(
    repository, rag_client, status
):
    content = b"# Retry\n\nDocument body."
    document = Document(
        title="Retry",
        uri="retry.md",
        mime_type="text/markdown",
        content_hash=sha256(content).hexdigest(),
        status=status,
    )
    await repository.create_document(document)
    await repository.session.commit()
    response = await rag_client.post(
        "/rag/index",
        data={"title": "Retry"},
        files={"file": ("retry.md", content, "text/markdown")},
    )
    assert response.status_code == 409
    assert response.json()["error_code"] == "DOCUMENT_NOT_INDEXED"
    assert response.json()["details"] == {
        "document_id": str(document.id),
        "status": status.value,
    }
    response = await rag_client.post(
        "/rag/index",
        data={"title": "Retry", "force": "true"},
        files={"file": ("retry.md", content, "text/markdown")},
    )
    assert response.status_code == 200
    assert response.json()["document_id"] == str(document.id)
    repository.session.expunge_all()
    assert (await repository.get_document(document.id)).status == DocumentStatus.INDEXED
