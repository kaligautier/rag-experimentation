"""Text extractors for Markdown documents."""

import logging

from app.ports.rag import TextExtractorPort
from app.utils.error import UnsupportedDocumentError

logger = logging.getLogger(__name__)


class MarkdownExtractor(TextExtractorPort):
    """Extract text from Markdown files."""

    # Browsers and curl commonly label .md uploads as text/plain.
    supported_mimes = ("text/markdown", "text/x-markdown", "text/plain")
    supported_extensions = (".md", ".markdown")

    async def extract(self, content: bytes) -> str:
        """Decode UTF-8 Markdown, dropping a leading byte-order mark."""
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise UnsupportedDocumentError(
                "Markdown document must be valid UTF-8",
                details={"reason": e.reason, "offset": e.start},
            ) from e
        logger.info("Extracted markdown: %s chars", len(text))
        return text


def get_extractor(mime_type: str, filename: str) -> MarkdownExtractor:
    """Get the Markdown extractor, the only supported format."""
    if mime_type not in MarkdownExtractor.supported_mimes:
        raise UnsupportedDocumentError(
            f"Unsupported MIME type: {mime_type}. Only Markdown (.md) is supported.",
            details={"mime_type": mime_type},
        )
    if not filename.lower().endswith(MarkdownExtractor.supported_extensions):
        raise UnsupportedDocumentError(
            f"Unsupported file type: {filename}. "
            "Only Markdown (.md) files are supported.",
            details={"filename": filename},
        )
    return MarkdownExtractor()
