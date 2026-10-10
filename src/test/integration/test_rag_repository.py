import math
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.adapters.persistence.entities import RagChunkEntity, RagDocumentEntity
from app.services.rag.models import (
    Chunk,
    ChunkHierarchy,
    Document,
    DocumentStatus,
    SearchParameters,
)
from app.utils.error import DocumentNotFoundError
from test.integration.support import vector


def policy_document(**fields) -> Document:
    return Document(
        title="Draft policy",
        uri="file:///returns.txt",
        mime_type="text/plain",
        content_hash=uuid4().hex,
        metadata={"country_code": "FR"},
        **fields,
    )


def without_embedding(chunk: Chunk) -> Chunk:
    """Reads never load stored vectors."""
    return chunk.model_copy(update={"embedding": None})


@pytest.fixture
async def hierarchical_document(repository):
    """An indexed parent context without a vector and its two embedded leaves."""
    document = policy_document(status=DocumentStatus.INDEXED, chunk_count=3)
    parent_id, leaf_ids = uuid4(), [uuid4(), uuid4()]
    parent = Chunk(
        id=parent_id,
        document_id=document.id,
        content="# Returns\n\nWithin 30 days.\n\nKeep the receipt.",
        chunk_index=0,
        metadata={"header_path": "/"},
        hierarchy=ChunkHierarchy(child_ids=leaf_ids, token_count=12),
    )
    leaves = [
        Chunk(
            id=leaf_id,
            document_id=document.id,
            content=content,
            embedding=embedding,
            chunk_index=index,
            metadata={"header_path": "/Returns/"},
            hierarchy=ChunkHierarchy(parent_id=parent_id, level=1, token_count=6),
        )
        for index, (leaf_id, content, embedding) in enumerate(
            zip(
                leaf_ids,
                ["Within 30 days.", "Keep the receipt."],
                [vector(2.0, 0.0), vector(2.0, 1.0)],
                strict=True,
            ),
            start=1,
        )
    ]
    await repository.create_document(document)
    await repository.add_chunks(document.id, [parent, *leaves])
    await repository.session.commit()
    repository.session.expunge_all()
    return document, parent, leaves


async def should_rank_indexed_chunks_by_cosine_without_discarding_negative_scores(
    repository, searchable_chunks
):
    hits = await repository.search(
        SearchParameters(query="return policy", top_k=10, similarity_threshold=0.0),
        vector(2.0, 0.0),
    )

    expected_names = [
        "aligned",
        "british",
        "near_l2",
        "without_country",
        "orthogonal",
        "opposite",
    ]
    assert [hit.chunk_id for hit in hits] == [
        searchable_chunks[name].id for name in expected_names
    ]
    assert [hit.similarity_score for hit in hits] == pytest.approx(
        [1.0, 12 / 13, 1 / math.sqrt(2), 0.6, 0.0, -1.0], abs=1e-6
    )
    best_chunk = searchable_chunks["aligned"]
    assert hits[0].document_id == best_chunk.document_id
    assert hits[0].chunk_content == best_chunk.content
    assert hits[0].document_title == "Document aligned"
    assert hits[0].chunk_index == 2


async def should_search_a_prefix_of_full_mrl_vectors(repository, searchable_chunks):
    dimensions = 768

    hits = await repository.search(
        SearchParameters(
            query="return policy", top_k=2, embedding_dimensions=dimensions
        ),
        vector(2.0, 0.0)[:dimensions],
    )

    assert [hit.chunk_id for hit in hits] == [
        searchable_chunks["aligned"].id,
        searchable_chunks["british"].id,
    ]


async def should_limit_search_to_the_top_k_cosine_matches(
    repository, searchable_chunks
):
    hits = await repository.search(
        SearchParameters(query="return policy", top_k=2), vector(2.0, 0.0)
    )

    assert [hit.chunk_id for hit in hits] == [
        searchable_chunks["aligned"].id,
        searchable_chunks["british"].id,
    ]


@pytest.mark.parametrize(
    ("threshold", "expected_names"),
    [(0.8, ["aligned", "british"]), (1.0, ["aligned"])],
)
async def should_apply_positive_similarity_threshold_inclusively(
    repository, searchable_chunks, threshold, expected_names
):
    hits = await repository.search(
        SearchParameters(
            query="return policy", top_k=10, similarity_threshold=threshold
        ),
        vector(2.0, 0.0),
    )

    assert [hit.chunk_id for hit in hits] == [
        searchable_chunks[name].id for name in expected_names
    ]


@pytest.mark.parametrize(
    ("country_code", "expected_names"),
    [
        ("FR", ["aligned", "near_l2", "orthogonal", "opposite"]),
        ("GB", ["british"]),
        ("CH", []),
    ],
)
async def should_filter_country_using_document_metadata(
    repository, searchable_chunks, country_code, expected_names
):
    hits = await repository.search(
        SearchParameters(query="return policy", top_k=10, country_code=country_code),
        vector(2.0, 0.0),
    )

    assert [hit.chunk_id for hit in hits] == [
        searchable_chunks[name].id for name in expected_names
    ]
    # Chunks claim "CH": hits report the document country the filter matched.
    assert [hit.metadata for hit in hits] == [
        {"country_code": country_code, "page": 3}
    ] * len(expected_names)


async def should_return_no_hits_when_no_documents_are_indexed(repository):
    hits = await repository.search(
        SearchParameters(query="return policy"), vector(2.0, 0.0)
    )

    assert hits == []


async def should_search_leaves_only_and_return_their_hierarchy(
    repository, hierarchical_document
):
    _, _, leaves = hierarchical_document

    hits = await repository.search(
        SearchParameters(query="returns", top_k=10), vector(2.0, 0.0)
    )

    assert [hit.chunk_id for hit in hits] == [leaf.id for leaf in leaves]
    assert [hit.hierarchy for hit in hits] == [leaf.hierarchy for leaf in leaves]
    assert [hit.metadata for hit in hits] == [
        {"header_path": "/Returns/", "country_code": "FR"}
    ] * 2


async def should_load_linked_parent_contexts_by_id(repository, hierarchical_document):
    _, parent, leaves = hierarchical_document

    loaded = await repository.get_chunks([parent.id, leaves[0].id])

    assert {chunk.id: chunk for chunk in loaded} == {
        parent.id: parent,
        leaves[0].id: without_embedding(leaves[0]),
    }
    assert await repository.get_chunks([]) == []


async def should_replace_every_chunk_of_a_document(repository, hierarchical_document):
    document, _, _ = hierarchical_document
    replacement = Chunk(
        document_id=document.id,
        content="Refunds take five days.",
        embedding=vector(1.0, 0.0),
        chunk_index=0,
    )
    # Same session throughout: the loaded collection must not survive the swap.
    entity = await repository.session.get(RagDocumentEntity, document.id)
    assert len((await repository.get_document(document.id)).chunks) == 3
    assert len(entity.chunks) == 3

    await repository.replace_chunks(document.id, [replacement])

    stored = await repository.get_document(document.id)
    assert stored.chunks == [without_embedding(replacement)]
    hits = await repository.search(SearchParameters(query="refunds"), vector(2.0, 0.0))
    assert [hit.chunk_id for hit in hits] == [replacement.id]


async def should_find_a_document_by_content_hash_without_its_chunks(
    repository, searchable_chunks
):
    existing = await repository.get_document(searchable_chunks["aligned"].document_id)
    repository.session.expunge_all()

    found = await repository.get_document_by_hash(existing.content_hash)

    assert existing.chunks
    assert found == existing.model_copy(update={"chunks": []})
    assert await repository.get_document_by_hash(uuid4().hex) is None


async def should_persist_a_document_with_its_chunks_in_index_order(repository):
    document = policy_document(created_at=datetime(2026, 1, 1))
    chunks = [
        Chunk(
            document_id=document.id,
            content="Keep the receipt.",
            embedding=vector(0.0, 2.0),
            chunk_index=1,
            metadata={"page": 2},
        ),
        Chunk(
            document_id=document.id,
            content="Return within 30 days.",
            embedding=vector(3.0, 4.0),
            chunk_index=0,
            metadata={"page": 1},
        ),
    ]
    await repository.create_document(document)
    await repository.add_chunks(document.id, chunks)
    await repository.session.commit()
    repository.session.expunge_all()

    stored = await repository.get_document(document.id)

    assert stored.model_dump(exclude={"chunks"}) == document.model_dump(
        exclude={"chunks"}
    )
    assert stored.chunks == [without_embedding(chunk) for chunk in reversed(chunks)]


async def should_update_document_metadata_and_return_the_stored_timestamp(
    repository,
):
    document = policy_document()
    await repository.create_document(document)
    await repository.session.commit()
    changed = document.model_copy(
        update={
            "title": "Return policy",
            "status": DocumentStatus.INDEXED,
            "chunk_count": 2,
            "metadata": {"country_code": "FR", "version": 2},
        }
    )

    updated = await repository.update_document(changed)
    await repository.session.commit()
    repository.session.expunge_all()

    stored = await repository.get_document(document.id)
    assert updated == stored
    assert stored.updated_at > document.updated_at
    assert (stored.title, stored.status, stored.chunk_count, stored.metadata) == (
        "Return policy",
        DocumentStatus.INDEXED,
        2,
        {"country_code": "FR", "version": 2},
    )


async def should_page_documents_newest_first_with_stable_ties(repository):
    older = policy_document(created_at=datetime(2025, 12, 31))
    tied = [policy_document(created_at=datetime(2026, 1, 1)) for _ in range(2)]
    # Insert ties against the expected id order so heap order cannot pass.
    for document in [*sorted(tied, key=lambda item: item.id, reverse=True), older]:
        await repository.create_document(document)
    # A stored chunk, so empty page chunks prove listing never loads them.
    await repository.add_chunks(
        older.id,
        [Chunk(document_id=older.id, content="Stored section.", chunk_index=0)],
    )
    await repository.session.commit()
    repository.session.expunge_all()

    pages = [
        await repository.list_documents(limit=1, offset=offset) for offset in range(3)
    ]

    assert [total for _, total in pages] == [3, 3, 3]
    assert [page[0].id for page, _ in pages] == [
        *sorted(item.id for item in tied),
        older.id,
    ]
    assert [page[0].chunks for page, _ in pages] == [[], [], []]


async def should_delete_a_document_and_let_the_database_cascade_its_chunks(
    repository,
):
    document, other = policy_document(), policy_document()
    await repository.create_document(document)
    await repository.create_document(other)
    await repository.add_chunks(
        document.id,
        [
            Chunk(
                document_id=document.id,
                content="Return within 30 days.",
                embedding=vector(3.0, 4.0),
                chunk_index=0,
            )
        ],
    )
    await repository.session.commit()
    # Unloaded chunks: the ORM must neither load nor delete them itself.
    repository.session.expunge_all()

    await repository.delete_document(document.id)
    await repository.session.commit()

    with pytest.raises(DocumentNotFoundError):
        await repository.get_document(document.id)
    remaining_chunks = await repository.session.scalar(
        select(func.count(RagChunkEntity.id)).where(
            RagChunkEntity.document_id == document.id
        )
    )
    assert remaining_chunks == 0
    remaining_documents, total = await repository.list_documents()
    assert total == 1
    assert [item.id for item in remaining_documents] == [other.id]
