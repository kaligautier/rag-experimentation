"""Promote matching siblings to their persisted parent context."""

from collections import defaultdict
from uuid import UUID

from app.ports.rag import RagRepositoryPort
from app.services.rag.models import Chunk, SearchHit


async def merge_parent_contexts(
    hits: list[SearchHit], repository: RagRepositoryPort, max_parent_tokens: int
) -> list[SearchHit]:
    """Merge a strict majority of direct children; keep original leaf evidence.

    Similarity remains the maximum matching leaf cosine score, not a fabricated
    embedding score for the parent. A failed/missing link leaves the hit intact.
    """
    current = {hit.chunk_id: hit for hit in hits}
    parents: dict[UUID, Chunk] = {}
    attempted: set[UUID] = set()
    promoted: set[UUID] = set()
    while True:
        groups: dict[UUID, list[SearchHit]] = defaultdict(list)
        for hit in current.values():
            if hit.hierarchy and hit.hierarchy.parent_id:
                groups[hit.hierarchy.parent_id].append(hit)
        missing = set(groups) - attempted
        if missing:
            parents.update(
                {node.id: node for node in await repository.get_chunks(missing)}
            )
            attempted.update(missing)
        changed = False
        for parent_id, children in groups.items():
            parent = parents.get(parent_id)
            if parent is None or parent_id in promoted or parent.is_leaf:
                continue
            info = parent.hierarchy
            if not 0 < info.token_count <= max_parent_tokens:
                continue
            if any(
                child.document_id != parent.document_id
                or child.chunk_id not in info.child_ids
                or child.hierarchy.level != info.level + 1
                for child in children
            ):
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
                metadata=parent.metadata,
                hierarchy=info,
                matched_chunk_ids=evidence,
            )
            promoted.add(parent_id)
            changed = True
        if not changed:
            return sorted(
                current.values(),
                key=lambda hit: (
                    -hit.similarity_score,
                    str(hit.document_id),
                    hit.chunk_index,
                ),
            )
