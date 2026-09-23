"""Heading-based parent contexts with bounded, searchable Markdown leaves."""

import logging
import re
from dataclasses import dataclass, field
from uuid import UUID

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document, MetadataMode
from llama_index.core.utils import get_tokenizer

from app.config.settings import settings
from app.services.rag.models import ChunkHierarchy, ParsedChunk
from app.utils.error import DocumentLimitError

logger = logging.getLogger(__name__)


@dataclass
class _Section:
    own_text: str
    metadata: dict
    has_body: bool
    children: list["_Section"] = field(default_factory=list)

    @property
    def content(self) -> str:
        return "\n\n".join([self.own_text, *(c.content for c in self.children)])


class HierarchicalChunker:
    """Build a heading tree; split only oversized leaves with LlamaIndex."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Configure leaf size and overlap in tokens, excluding ancestor metadata."""
        self.chunk_size = settings.rag.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = (
            settings.rag.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        )
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("Require 0 <= chunk_overlap < chunk_size")
        self.parser = MarkdownNodeParser()
        self.tokenizer = get_tokenizer()

    async def split(self, text: str) -> list[ParsedChunk]:
        """Return pre-order chunks with explicit parent and child identifiers."""
        if not text:
            return []

        roots: list[_Section] = []
        stack: list[_Section] = []
        for node in self.parser.get_nodes_from_documents([Document(text=text)]):
            content = node.get_content(metadata_mode=MetadataMode.NONE).strip()
            if not content:
                continue
            heading, _, body = content.partition("\n")
            match = re.match(r"^(#{1,6})\s+(.+)$", heading)
            metadata = {"header_path": node.metadata.get("header_path", "/")}
            if match:
                metadata.update(
                    section_title=match.group(2).strip(),
                    header_level=len(match.group(1)),
                )
            section = _Section(content, metadata, bool(body.strip()) if match else True)
            level = metadata.get("header_level", 0)
            while stack and stack[-1].metadata.get("header_level", 0) >= level:
                stack.pop()
            (stack[-1].children if stack else roots).append(section)
            if match:
                stack.append(section)

        def prune(section: _Section) -> bool:
            section.children = [child for child in section.children if prune(child)]
            return section.has_body or bool(section.children)

        chunks = [
            node
            for root in roots
            if prune(root)
            for node in self._emit(root, parent_id=None, level=0)
        ]
        logger.info(
            "Split Markdown into %s nodes (%s leaves)",
            len(chunks),
            sum(node.is_leaf for node in chunks),
        )
        return chunks

    def _emit(
        self, section: _Section, parent_id: UUID | None, level: int
    ) -> list[ParsedChunk]:
        content = section.content
        node = ParsedChunk(
            content=content,
            metadata=dict(section.metadata),
            hierarchy=ChunkHierarchy(
                parent_id=parent_id,
                level=level,
                token_count=len(self.tokenizer(content)),
            ),
        )
        children = list(section.children)
        if children and section.has_body:
            # Parent introductions also need a searchable vector of their own.
            children.insert(
                0,
                _Section(section.own_text, dict(section.metadata), has_body=True),
            )
        elif not children and node.hierarchy.token_count > self.chunk_size:
            heading, _, body = content.partition("\n")
            prefix = f"{heading}\n\n" if "header_level" in section.metadata else ""
            budget = self.chunk_size - len(self.tokenizer(prefix)) - 1
            if budget < 1:
                raise DocumentLimitError(
                    "Section heading exceeds the configured token budget",
                    details={
                        "limit_tokens": self.chunk_size,
                        "heading_tokens": len(self.tokenizer(prefix)),
                    },
                )
            splitter = SentenceSplitter(
                chunk_size=budget,
                chunk_overlap=min(self.chunk_overlap, budget - 1),
                tokenizer=self.tokenizer,
            )
            children = [
                _Section(
                    prefix + part,
                    {**section.metadata, "section_part": i},
                    has_body=True,
                )
                for i, part in enumerate(
                    splitter.split_text(body if prefix else content)
                )
            ]

        descendants = []
        for child in children:
            subtree = self._emit(child, parent_id=node.id, level=level + 1)
            node.hierarchy.child_ids.append(subtree[0].id)
            descendants.extend(subtree)
        return [node, *descendants]


# Singleton instance
_chunker_instance: HierarchicalChunker | None = None


def get_chunker(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> HierarchicalChunker:
    """Get or create hierarchical chunker instance."""
    global _chunker_instance
    if _chunker_instance is None:
        _chunker_instance = HierarchicalChunker(chunk_size, chunk_overlap)
    return _chunker_instance
