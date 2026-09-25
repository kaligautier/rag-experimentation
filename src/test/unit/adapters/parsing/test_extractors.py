"""Only UTF-8 Markdown is accepted, and a byte-order mark never reaches the chunker."""

import pytest

from app.adapters.parsing.extractors import MarkdownExtractor, get_extractor
from app.utils.error import UnsupportedDocumentError


@pytest.mark.parametrize(
    "mime_type", ["text/markdown", "text/x-markdown", "text/plain"]
)
def should_return_the_markdown_extractor_for_markdown_files(mime_type):
    extractor = get_extractor(mime_type, "Guide.MD")

    assert isinstance(extractor, MarkdownExtractor)


@pytest.mark.parametrize(
    ("mime_type", "filename", "detail"),
    [
        ("application/pdf", "guide.pdf", "mime_type"),
        ("text/markdown", "guide.txt", "filename"),
        ("text/plain", "guide", "filename"),
    ],
)
def should_reject_non_markdown_documents(mime_type, filename, detail):
    with pytest.raises(UnsupportedDocumentError) as caught:
        get_extractor(mime_type, filename)
    assert 400 <= caught.value.status_code < 500
    assert detail in caught.value.details


async def should_decode_utf8_and_drop_the_byte_order_mark():
    text = await MarkdownExtractor().extract("\ufeff# Titre\n\nCorps".encode())

    assert text == "# Titre\n\nCorps"


async def should_reject_invalid_utf8():
    with pytest.raises(UnsupportedDocumentError) as caught:
        await MarkdownExtractor().extract(b"# Titre\n\n\xff\xfe")
    assert caught.value.details["offset"] == 9
