"""Text extractors for Markdown documents."""

import logging

from app.utils.error import UnsupportedDocumentError

logger = logging.getLogger(__name__)


class MarkdownExtractor:
    """Extract text from Markdown files."""

    supported_mimes = ["text/markdown", "text/x-markdown"]
    supported_extensions = [".md", ".markdown"]

    async def extract(self, content: bytes) -> str:
        """Extract text from Markdown (UTF-8 decode)."""
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as e:
            raise UnsupportedDocumentError(
                "Markdown document must be valid UTF-8",
                details={"reason": e.reason, "offset": e.start},
            ) from e
        logger.info(f"Extracted markdown: {len(text)} chars")
        return text


async def get_extractor(mime_type: str, filename: str = "") -> MarkdownExtractor:
    """Get Markdown extractor (only format supported)."""
    # Validate that it's markdown
    if mime_type not in ["text/markdown", "text/x-markdown", "text/plain"]:
        raise UnsupportedDocumentError(
            f"Unsupported MIME type: {mime_type}. Only Markdown (.md) is supported.",
            details={"mime_type": mime_type},
        )

    filename_lower = filename.lower()
    if not filename_lower.endswith((".md", ".markdown")):
        raise UnsupportedDocumentError(
            f"Unsupported file type: {filename}. "
            "Only Markdown (.md) files are supported.",
            details={"filename": filename},
        )

    return MarkdownExtractor()
