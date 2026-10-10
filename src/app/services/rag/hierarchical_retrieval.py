"""Promote matching siblings to their persisted parent context."""

from collections import defaultdict
from uuid import UUID

from app.ports.rag import RagRepositoryPort
from app.services.rag.models import Chunk, ChunkHierarchy, SearchHit


async def merge_parent_contexts(
    hits: list[SearchHit], repository: RagRepositoryPort, max_parent_tokens: int
) -> list[SearchHit]:
    """Merge a strict majority of direct children; keep original leaf evidence.

    Similarity remains the maximum matching leaf cosine score, not a fabricated
    embedding score for the parent. A failed/missing link leaves the hit intact.
    """
    current = {hit.chunk_id: hit for hit in hits}
    parents, links = await _load_ancestors(hits, repository)
    promoted: set[UUID] = set()
    while True:
        groups: dict[UUID, list[SearchHit]] = defaultdict(list)
        for hit in current.values():
            if hit.hierarchy and hit.hierarchy.parent_id:
                groups[hit.hierarchy.parent_id].append(hit)
        changed = False
        for parent_id, children in groups.items():
            parent = parents.get(parent_id)
            if parent is None or parent_id in promoted or parent.is_leaf:
                continue
            info = parent.hierarchy
            if not 0 < info.token_count <= max_parent_tokens:
                continue
            if any(links.get(child.chunk_id) != parent_id for child in children):
                continue
            if len(children) / len(info.child_ids) <= 0.5:
                continue
            best = max(children, key=lambda child: child.similarity_score)
            evidence = list(
                dict.fromkeys(
                    leaf_id
                    for child in children
                    for leaf_id in (child.matched_chunk_ids or [child.chunk_id])
                )
            )
            for child in children:
                current.pop(child.chunk_id)
            current[parent_id] = SearchHit(
                document_id=parent.document_id,
                document_title=best.document_title,
                chunk_id=parent.id,
                chunk_content=parent.content,
                chunk_index=parent.chunk_index,
                similarity_score=best.similarity_score,
                score_type="max_leaf_cosine",
                metadata={**parent.metadata, **best.document_metadata},
                document_metadata=best.document_metadata,
                hierarchy=info,
                matched_chunk_ids=evidence,
            )
            promoted.add(parent_id)
            changed = True
        if not changed:
            return sorted(
                _remove_overlaps(current, links),
                key=lambda hit: (
                    -hit.similarity_score,
                    str(hit.document_id),
                    hit.chunk_index,
                ),
            )


async def _load_ancestors(
    hits: list[SearchHit], repository: RagRepositoryPort
) -> tuple[dict[UUID, Chunk], dict[UUID, UUID]]:
    """Batch only selected-hit ancestry; each valid step strictly reduces level."""
    parents: dict[UUID, Chunk] = {}
    links: dict[UUID, UUID] = {}
    attempted: set[UUID] = set()
    frontier: dict[UUID, tuple[UUID, ChunkHierarchy]] = {
        hit.chunk_id: (hit.document_id, hit.hierarchy)
        for hit in hits
        if hit.hierarchy is not None
    }
    while frontier:
        missing = {
            info.parent_id for _, info in frontier.values() if info.parent_id
        } - attempted
        if missing:
            parents.update(
                {node.id: node for node in await repository.get_chunks(missing)}
            )
            attempted.update(missing)
        following = {}
        for child_id, (document_id, info) in frontier.items():
            parent = parents.get(info.parent_id)
            if (
                parent is None
                or parent.is_leaf
                or parent.document_id != document_id
                or child_id not in parent.hierarchy.child_ids
                or info.level != parent.hierarchy.level + 1
            ):
                continue
            links[child_id] = parent.id
            if parent.id not in links:
                following[parent.id] = (parent.document_id, parent.hierarchy)
        frontier = following
    return parents, links


def _remove_overlaps(
    current: dict[UUID, SearchHit], links: dict[UUID, UUID]
) -> list[SearchHit]:
    """Fold remaining hits into selected ancestors along validated links only."""
    results = dict(current)
    for chunk_id, hit in current.items():
        ancestor_id = chunk_id
        selected_id = chunk_id
        while ancestor_id in links:
            ancestor_id = links[ancestor_id]
            if ancestor_id in current:
                selected_id = ancestor_id
        if selected_id == chunk_id:
            continue
        selected = results[selected_id]
        results[selected_id] = selected.model_copy(
            update={
                "similarity_score": max(
                    selected.similarity_score, hit.similarity_score
                ),
                "matched_chunk_ids": list(
                    dict.fromkeys(
                        [
                            *selected.matched_chunk_ids,
                            *(hit.matched_chunk_ids or [hit.chunk_id]),
                        ]
                    )
                ),
            }
        )
        results.pop(chunk_id, None)
    return list(results.values())
