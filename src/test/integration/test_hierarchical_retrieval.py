"""Persisted search provenance survives promotion to a parent context."""

from uuid import uuid4

from app.services.rag.hierarchical_retrieval import merge_parent_contexts
from app.services.rag.models import (
    Chunk,
    ChunkHierarchy,
    Document,
    DocumentStatus,
    SearchParameters,
)
from test.integration.support import vector


async def should_preserve_persisted_document_metadata_on_parent_promotion(repository):
    document = Document(
        title="Policy",
        uri="file:///policy.md",
        mime_type="text/markdown",
        content_hash=uuid4().hex,
        status=DocumentStatus.INDEXED,
        chunk_count=3,
        metadata={"country_code": "FR", "source": "authoritative", "edition": 2},
    )
    parent_id, first_id, second_id = uuid4(), uuid4(), uuid4()
    parent = Chunk(
        id=parent_id,
        document_id=document.id,
        content="Both sections.",
        chunk_index=0,
        metadata={"country_code": "GB", "page": 1, "source": "chunk"},
        hierarchy=ChunkHierarchy(child_ids=[first_id, second_id], token_count=10),
    )
    leaves = [
        Chunk(
            id=id_,
            document_id=document.id,
            content=f"Section {i}",
            chunk_index=i,
            embedding=vector(1.0, 0.0),
            metadata={"leaf_only": True, "page": i + 5},
            hierarchy=ChunkHierarchy(parent_id=parent_id, level=1, token_count=5),
        )
        for i, id_ in enumerate([first_id, second_id], 1)
    ]
    await repository.create_document(document)
    await repository.add_chunks(document.id, [parent, *leaves])
    await repository.session.commit()
    repository.session.expunge_all()

    hits = await repository.search(
        SearchParameters(query="policy", country_code="FR"), vector(1.0, 0.0)
    )
    results = await merge_parent_contexts(hits, repository, 4096)

    assert len(results) == 1
    assert results[0].chunk_id == parent_id
    assert results[0].metadata == {**parent.metadata, **document.metadata}
    assert set(results[0].matched_chunk_ids) == {first_id, second_id}
    assert "leaf_only" not in results[0].metadata
    assert "document_metadata" not in results[0].model_dump()


async def should_merge_persisted_ancestor_and_partial_branch_without_duplicate_context(
    repository,
):
    from app.adapters.parsing.chunker import HierarchicalChunker

    document = Document(
        title="Policy",
        uri="file:///policy.md",
        mime_type="text/markdown",
        content_hash=uuid4().hex,
        status=DocumentStatus.INDEXED,
        metadata={"country_code": "FR"},
    )
    parsed = await HierarchicalChunker().split(
        "# Root\n\n"
        + "\n\n".join(
            f"## {branch}\n\n"
            + "\n\n".join(f"### {branch}{i}\nBody {branch}{i}." for i in range(3))
            for branch in "ABC"
        )
    )
    selected_titles = {"A0", "A1", "B0", "B1", "C0"}
    chunks = [
        Chunk(
            id=node.id,
            document_id=document.id,
            content=node.content,
            chunk_index=i,
            hierarchy=node.hierarchy,
            metadata=node.metadata,
            embedding=(
                vector(1.0, 0.0)
                if node.metadata.get("section_title") in selected_titles
                else vector(0.0, 1.0)
            )
            if node.is_leaf
            else None,
        )
        for i, node in enumerate(parsed)
    ]
    document.chunk_count = len(chunks)
    await repository.create_document(document)
    await repository.add_chunks(document.id, chunks)
    await repository.session.commit()
    repository.session.expunge_all()

    hits = await repository.search(
        SearchParameters(query="policy", similarity_threshold=0.5), vector(1.0, 0.0)
    )
    results = await merge_parent_contexts(hits, repository, 4096)

    assert len(hits) == 5
    assert [hit.chunk_id for hit in results] == [chunks[0].id]
    assert set(results[0].matched_chunk_ids) == {hit.chunk_id for hit in hits}
    assert results[0].metadata == {**chunks[0].metadata, **document.metadata}
