"""Retrieve precise leaves and merge only well-supported, bounded contexts."""

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.adapters.parsing.chunker import HierarchicalChunker
from app.services.rag.hierarchical_retrieval import merge_parent_contexts
from app.services.rag.models import Chunk, SearchHit


async def tree_and_hits(indices=(2, 3)):
    parsed = await HierarchicalChunker().split(
        "# GR20\n\n## Conseils\n\n### Période\nÉté.\n\n"
        "### Équipement\nChaussures.\n\n### Eau\nFiltration.\n\n"
        "## Autre\nUn autre sujet."
    )
    document_id = uuid4()
    nodes = [
        Chunk(
            id=node.id,
            document_id=document_id,
            content=node.content,
            embedding=[1.0] if node.is_leaf else None,
            chunk_index=i,
            metadata=node.metadata,
            hierarchy=node.hierarchy,
        )
        for i, node in enumerate(parsed)
    ]
    hits = [
        SearchHit(
            document_id=document_id,
            document_title="GR20",
            chunk_id=nodes[i].id,
            chunk_content=nodes[i].content,
            chunk_index=i,
            similarity_score=0.9 - rank * 0.1,
            hierarchy=nodes[i].hierarchy,
            matched_chunk_ids=[nodes[i].id],
        )
        for rank, i in enumerate(indices)
    ]
    repository = AsyncMock()
    repository.get_chunks.side_effect = lambda ids: [n for n in nodes if n.id in ids]
    return nodes, hits, repository


async def should_merge_matching_siblings_and_keep_their_evidence():
    nodes, hits, repository = await tree_and_hits()

    results = await merge_parent_contexts(hits, repository, max_parent_tokens=4096)

    assert len(results) == 1
    assert results[0].chunk_id == nodes[1].id
    assert results[0].chunk_content == nodes[1].content
    assert results[0].matched_chunk_ids == [hit.chunk_id for hit in hits]
    assert results[0].similarity_score == 0.9
    assert results[0].score_type == "max_leaf_cosine"
    assert results[0].hierarchy == nodes[1].hierarchy


@pytest.mark.parametrize("indices,budget", [((2,), 4096), ((2, 3), 1)])
async def should_keep_precise_leaves_without_majority_or_with_oversized_parent(
    indices, budget
):
    _, hits, repository = await tree_and_hits(indices)

    assert await merge_parent_contexts(hits, repository, budget) == hits


async def should_merge_recursively_without_duplicate_parent_and_child_results():
    nodes, hits, repository = await tree_and_hits((2, 3, 5))

    results = await merge_parent_contexts(hits, repository, 4096)

    assert len(results) == 1
    assert results[0].chunk_id == nodes[0].id
    assert set(results[0].matched_chunk_ids) == {hit.chunk_id for hit in hits}


async def should_refuse_a_parent_from_another_document():
    nodes, hits, repository = await tree_and_hits()
    nodes[1].document_id = uuid4()

    assert await merge_parent_contexts(hits, repository, 4096) == hits


async def should_keep_leaves_when_parent_is_missing():
    _, hits, repository = await tree_and_hits()
    repository.get_chunks.side_effect = lambda ids: []

    assert await merge_parent_contexts(hits, repository, 4096) == hits
