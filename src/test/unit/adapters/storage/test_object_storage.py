"""Local storage must keep every generated path below its configured root."""

from pathlib import Path

import pytest

from app.adapters.storage.object_storage import LocalFileStorageAdapter


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.md",
        "../../../../tmp/outside.md",
        "nested/document.md",
        r"nested\document.md",
    ],
)
def should_reject_path_components_in_uploaded_filenames(tmp_path, filename):
    storage = LocalFileStorageAdapter(str(tmp_path))

    with pytest.raises(ValueError, match="filename"):
        storage.generate_key("document-id", "content-hash", filename)


def should_reject_storage_keys_that_escape_the_configured_root(tmp_path):
    storage = LocalFileStorageAdapter(str(tmp_path))

    with pytest.raises(ValueError, match="storage root"):
        storage._get_file_path("../../outside.md")


async def should_store_a_safe_basename_below_the_configured_root(tmp_path):
    storage = LocalFileStorageAdapter(str(tmp_path))
    key = storage.generate_key("document-id", "content-hash", "guide.md")

    await storage.put(key, b"# Safe document")

    stored_path = (tmp_path / key).resolve()
    assert stored_path.is_relative_to(Path(tmp_path).resolve())
    assert stored_path.read_bytes() == b"# Safe document"
