"""Service transactions verified against PostgreSQL and real source files."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.adapters.persistence.entities import RagChunkEntity, RagDocumentEntity
from app.adapters.persistence.rag_repository import RagRepository
from app.adapters.storage.object_storage import LocalFileStorageAdapter
from app.services.rag.indexation_service import IndexationService
from app.services.rag.models import Chunk, Document, DocumentStatus
from app.utils.error import (
    ConfigurationError,
    DocumentLimitError,
    DocumentNotFoundError,
)
from test.integration.support import vector

CONTENT = b"# Source\n\nKeep this original document."


@pytest.fixture
def source_service(repository, monkeypatch, tmp_path):
    storage = LocalFileStorageAdapter(str(tmp_path))
    embeddings = AsyncMock()
    embeddings.embed_documents.side_effect = lambda texts: [vector(1, 0) for _ in texts]
    monkeypatch.setattr(
        "app.services.rag.indexation_service.get_storage", lambda: storage
    )
    service = IndexationService(embeddings, repository, repository.session)
    return service, storage, embeddings


async def stored_document(repository, document_id):
    """Read committed state with an independent session."""
    async with async_sessionmaker(
        repository.session.bind, expire_on_commit=False
    )() as session:
        return await RagRepository(session).get_document(document_id)


async def upload(service, repository):
    async with service.indexing_document(
        CONTENT,
        title="Original",
        uri="https://example.test/original",
        mime_type="text/markdown",
        filename="original.md",
    ) as result:
        await repository.session.commit()
    return result


@pytest.mark.parametrize("filename", ["renamed.md", ""])
async def should_reindex_in_place_and_preserve_source_identity(
    repository, source_service, tmp_path, filename
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    before = await stored_document(repository, original.document.id)
    original_files = {path for path in tmp_path.rglob("*") if path.is_file()}

    updated = await service.index_document(
        CONTENT,
        title="Updated",
        uri="https://example.test/replacement",
        mime_type="text/plain",
        filename=filename,
        force=True,
    )
    await repository.session.commit()
    persisted = await stored_document(repository, original.document.id)

    assert updated.created is False
    assert persisted.uri == before.uri
    assert persisted.mime_type == before.mime_type
    assert updated.document.uri == before.uri
    assert updated.document.mime_type == before.mime_type
    assert persisted.created_at == before.created_at
    assert persisted.updated_at == updated.document.updated_at
    assert persisted.title == "Updated"
    assert persisted.metadata["storage_key"] == before.metadata["storage_key"]
    assert await storage.get(persisted.metadata["storage_key"]) == CONTENT
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == original_files
    assert {chunk.id for chunk in persisted.chunks}.isdisjoint(
        chunk.id for chunk in before.chunks
    )


async def should_reuse_duplicate_upload_without_new_source_or_embeddings(
    repository, source_service, tmp_path
):
    service, storage, embeddings = source_service
    original = await upload(service, repository)
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    embeddings.embed_documents.reset_mock()

    duplicate = await service.index_document(
        CONTENT, title="Duplicate", mime_type="text/markdown", filename="duplicate.md"
    )
    await repository.session.commit()

    assert duplicate.document.id == original.document.id
    assert duplicate.created is False
    embeddings.embed_documents.assert_not_awaited()
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == files
    assert await storage.get(duplicate.document.metadata["storage_key"]) == CONTENT


async def should_rollback_database_when_source_publication_fails(
    repository, source_service, monkeypatch, tmp_path
):
    service, storage, _ = source_service
    monkeypatch.setattr(
        storage, "put", AsyncMock(side_effect=OSError("Disk unavailable"))
    )
    with pytest.raises(OSError, match="Disk unavailable"):
        await service.index_document(
            CONTENT, title="Failed", mime_type="text/markdown", filename="failed.md"
        )
    await repository.session.rollback()

    async with async_sessionmaker(repository.session.bind)() as session:
        assert (
            await session.scalar(select(func.count()).select_from(RagDocumentEntity))
            == 0
        )
        assert (
            await session.scalar(select(func.count()).select_from(RagChunkEntity)) == 0
        )
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.parametrize(
    "error", [RuntimeError("Commit failed"), asyncio.CancelledError()]
)
async def should_remove_new_source_when_index_commit_fails_or_is_cancelled(
    repository, source_service, monkeypatch, tmp_path, error
):
    service, _, _ = source_service
    monkeypatch.setattr(repository.session, "commit", AsyncMock(side_effect=error))
    with pytest.raises(type(error)):
        async with service.indexing_document(
            CONTENT, title="Original", mime_type="text/markdown", filename="original.md"
        ):
            await repository.session.commit()
    await repository.session.rollback()

    async with async_sessionmaker(repository.session.bind)() as session:
        assert (
            await session.scalar(select(func.count()).select_from(RagDocumentEntity))
            == 0
        )
        assert (
            await session.scalar(select(func.count()).select_from(RagChunkEntity)) == 0
        )
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


async def should_preserve_source_and_chunks_when_forced_index_commit_fails(
    repository, source_service, monkeypatch, tmp_path
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    before = await stored_document(repository, original.document.id)
    files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(
        repository.session,
        "commit",
        AsyncMock(side_effect=RuntimeError("Commit failed")),
    )

    with pytest.raises(RuntimeError, match="Commit failed"):
        async with service.indexing_document(
            CONTENT,
            title="Updated",
            mime_type="text/markdown",
            filename="renamed.md",
            force=True,
        ):
            await repository.session.commit()
    await repository.session.rollback()

    persisted = await stored_document(repository, original.document.id)
    assert persisted.title == before.title
    assert {chunk.id for chunk in persisted.chunks} == {
        chunk.id for chunk in before.chunks
    }
    assert {
        path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()
    } == files


async def should_delete_committed_document_chunks_and_original(
    repository, source_service, tmp_path
):
    service, _, _ = source_service
    original = await upload(service, repository)

    async with service.deleting_document(original.document.id):
        await repository.session.commit()

    with pytest.raises(DocumentNotFoundError):
        await stored_document(repository, original.document.id)
    async with async_sessionmaker(repository.session.bind)() as session:
        assert (
            await session.scalar(select(func.count()).select_from(RagChunkEntity)) == 0
        )
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.parametrize(
    "error", [RuntimeError("Commit failed"), asyncio.CancelledError()]
)
async def should_restore_source_when_delete_commit_fails_or_is_cancelled(
    repository, source_service, monkeypatch, error
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    key = original.document.metadata["storage_key"]
    monkeypatch.setattr(repository.session, "commit", AsyncMock(side_effect=error))

    with pytest.raises(type(error)):
        async with service.deleting_document(original.document.id):
            with pytest.raises(FileNotFoundError):
                await storage.get(key)
            await repository.session.commit()
    await repository.session.rollback()

    assert await storage.get(key) == CONTENT
    persisted = await stored_document(repository, original.document.id)
    assert persisted.chunk_count > 0
    assert persisted.chunks


async def should_delete_database_rows_when_original_is_already_missing(
    repository, source_service
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    await storage.delete(original.document.metadata["storage_key"])

    async with service.deleting_document(original.document.id):
        await repository.session.commit()
    with pytest.raises(DocumentNotFoundError):
        await stored_document(repository, original.document.id)


async def should_refuse_delete_when_source_backup_exceeds_the_limit(
    repository, source_service, monkeypatch
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    key = original.document.metadata["storage_key"]
    monkeypatch.setenv("RAG_MAX_UPLOAD_BYTES", "4")

    with pytest.raises(DocumentLimitError):
        async with service.deleting_document(original.document.id):
            await repository.session.commit()
    await repository.session.rollback()
    assert await storage.get(key) == CONTENT
    assert (await stored_document(repository, original.document.id)).chunks


async def should_delete_legacy_source_using_validated_chunk_filename(
    repository, source_service
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    document = await repository.get_document(original.document.id)
    key = document.metadata.pop("storage_key")
    document.metadata.pop("source_filename", None)
    await repository.update_document(document)
    await repository.session.commit()

    async with service.deleting_document(document.id):
        await repository.session.commit()
    with pytest.raises(FileNotFoundError):
        await storage.get(key)


@pytest.mark.parametrize(
    ("legacy_filename", "new_filename"),
    [("", "renamed.md"), ("", ""), ("legacy.md", ""), ("legacy.md", "renamed.md")],
)
async def should_reuse_legacy_source_paths_across_filename_transitions(
    repository, source_service, tmp_path, legacy_filename, new_filename
):
    import hashlib

    service, storage, _ = source_service
    document = Document(
        title="Legacy",
        uri="",
        mime_type="text/markdown",
        content_hash=hashlib.sha256(CONTENT).hexdigest(),
        status=DocumentStatus.INDEXED,
        chunk_count=1,
    )
    await repository.create_document(document)
    await repository.add_chunks(
        document.id,
        [
            Chunk(
                document_id=document.id,
                content=CONTENT.decode(),
                embedding=vector(1, 0),
                chunk_index=0,
                metadata={"filename": legacy_filename},
            )
        ],
    )
    key = storage.generate_key(str(document.id), document.content_hash, legacy_filename)
    await storage.put(key, CONTENT)
    await repository.session.commit()
    files = {path for path in tmp_path.rglob("*") if path.is_file()}

    async with service.indexing_document(
        CONTENT,
        title="Reindexed",
        mime_type="text/plain",
        filename=new_filename,
        force=True,
    ) as result:
        await repository.session.commit()

    assert result.document.uri == ""
    assert result.document.mime_type == "text/markdown"
    assert result.document.metadata["storage_key"] == key
    assert await storage.get(key) == CONTENT
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == files
    assert (await stored_document(repository, document.id)).metadata[
        "storage_key"
    ] == key


async def should_restore_deleted_source_before_cancellation_finishes(
    repository, source_service
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    key = original.document.metadata["storage_key"]
    deleting = asyncio.Event()

    async def delete_then_wait():
        async with service.deleting_document(original.document.id):
            deleting.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(delete_then_wait())
    await deleting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await repository.session.rollback()

    assert await storage.get(key) == CONTENT
    assert (await stored_document(repository, original.document.id)).chunks


async def should_refuse_unidentifiable_legacy_source_without_reading_uri(
    repository, source_service, tmp_path
):
    service, storage, _ = source_service
    original = await upload(service, repository)
    document = await repository.get_document(original.document.id)
    key = document.metadata.pop("storage_key")
    document.metadata.pop("source_filename", None)
    for chunk in document.chunks:
        chunk.metadata.pop("filename", None)
    await repository.update_document(document)
    await repository.replace_chunks(document.id, document.chunks)
    await repository.session.commit()

    with pytest.raises(ConfigurationError, match="source"):
        async with service.deleting_document(document.id):
            await repository.session.commit()
    await repository.session.rollback()
    assert await storage.get(key) == CONTENT
    assert (
        await stored_document(repository, document.id)
    ).uri == "https://example.test/original"
