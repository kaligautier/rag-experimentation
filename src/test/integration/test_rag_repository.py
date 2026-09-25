import math
from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from app.adapters.persistence.entities import RagChunkEntity
from app.services.rag.models import Chunk, Document, DocumentStatus, SearchParameters
from test.integration.support import vector


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


async def should_return_no_hits_when_no_documents_are_indexed(repository):
    hits = await repository.search(
        SearchParameters(query="return policy"), vector(2.0, 0.0)
    )

    assert hits == []


async def should_find_a_document_and_its_chunks_by_content_hash(
    repository, searchable_chunks
):
    existing = await repository.get_document(searchable_chunks["aligned"].document_id)
    repository.session.expunge_all()

    found = await repository.get_document_by_hash(existing.content_hash)

    assert found == existing
    assert await repository.get_document_by_hash(uuid4().hex) is None


async def should_persist_documents_and_chunks_and_delete_chunks_with_their_document(
    repository,
):
    document = Document(
        title="Draft policy",
        uri="file:///returns.txt",
        mime_type="text/plain",
        content_hash=uuid4().hex,
        metadata={"country_code": "FR"},
        created_at=datetime(2026, 1, 1),
    )
    newer_document = Document(
        title="Unrelated policy",
        uri="file:///other.txt",
        mime_type="text/plain",
        content_hash=uuid4().hex,
        created_at=datetime(2026, 1, 2),
    )
    chunks = [
        Chunk(
            document_id=document.id,
            content="Return within 30 days.",
            embedding=vector(3.0, 4.0),
            chunk_index=0,
            metadata={"page": 1},
        ),
        Chunk(
            document_id=document.id,
            content="Keep the receipt.",
            embedding=vector(0.0, 2.0),
            chunk_index=1,
            metadata={"page": 2},
        ),
    ]
    await repository.create_document(document)
    await repository.create_document(newer_document)
    await repository.add_chunks(document.id, chunks)
    document.title = "Return policy"
    document.status = DocumentStatus.INDEXED
    document.chunk_count = len(chunks)
    document.metadata = {"country_code": "FR", "version": 2}
    await repository.update_document(document)
    await repository.session.commit()
    repository.session.expunge_all()

    stored = await repository.get_document(document.id)

    assert stored.title == "Return policy"
    assert stored.status == DocumentStatus.INDEXED
    assert stored.uri == document.uri
    assert stored.content_hash == document.content_hash
    assert stored.metadata == {"country_code": "FR", "version": 2}
    assert stored.chunk_count == 2
    assert {chunk.id: chunk for chunk in stored.chunks} == {
        chunk.id: chunk for chunk in chunks
    }
    page, total = await repository.list_documents(limit=1, offset=1)
    assert total == 2
    assert [item.id for item in page] == [document.id]

    repository.session.expunge_all()
    await repository.delete_document(document.id)
    await repository.session.commit()
    repository.session.expunge_all()

    with pytest.raises(ValueError, match="Document not found"):
        await repository.get_document(document.id)
    remaining_chunks = await repository.session.scalar(
        select(func.count(RagChunkEntity.id)).where(
            RagChunkEntity.document_id == document.id
        )
    )
    assert remaining_chunks == 0
    remaining_documents, total = await repository.list_documents()
    assert total == 1
    assert [item.id for item in remaining_documents] == [newer_document.id]
