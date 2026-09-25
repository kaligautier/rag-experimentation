"""Local storage must keep every generated path below its configured root."""

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.adapters.storage import object_storage
from app.adapters.storage.object_storage import LocalFileStorageAdapter
from app.utils.error import ConfigurationError


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


async def should_read_and_delete_stored_files(tmp_path):
    storage = LocalFileStorageAdapter(str(tmp_path))
    key = storage.generate_key("document-id", "content-hash", "guide.md")

    await storage.put(key, io.BytesIO(b"# Doc"))

    assert await storage.get(key) == b"# Doc"
    await storage.delete(key)
    with pytest.raises(FileNotFoundError):
        await storage.get(key)


async def should_publish_complete_contents_with_an_atomic_replace(
    tmp_path, monkeypatch
):
    storage = LocalFileStorageAdapter(str(tmp_path))
    target = tmp_path / "guide.md"
    target.write_bytes(b"old contents")
    replace = Path.replace
    replacements = []

    def inspect_replace(source, destination):
        assert target.read_bytes() == b"old contents"
        assert source.parent == target.parent
        assert source.read_bytes() == b"new contents"
        replacements.append(destination)
        return replace(source, destination)

    monkeypatch.setattr(Path, "replace", inspect_replace)

    await storage.put("guide.md", b"new contents")

    assert replacements == [target]
    assert await storage.get("guide.md") == b"new contents"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("existing", [False, True])
async def should_preserve_the_destination_and_clean_up_on_publish_failure(
    tmp_path, monkeypatch, existing
):
    storage = LocalFileStorageAdapter(str(tmp_path))
    target = tmp_path / "guide.md"
    if existing:
        target.write_bytes(b"old contents")

    def fail_replace(source, destination):
        raise OSError("Cannot publish file")

    monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(OSError, match="Cannot publish file"):
        await storage.put("guide.md", b"new contents")

    if existing:
        assert target.read_bytes() == b"old contents"
    assert set(tmp_path.iterdir()) == ({target} if existing else set())


def should_reject_unimplemented_storage_backends(monkeypatch, tmp_path):
    monkeypatch.setattr(object_storage, "_storage_instance", None)
    monkeypatch.setattr(
        object_storage,
        "settings",
        SimpleNamespace(
            rag=SimpleNamespace(STORAGE_TYPE="gcs", STORAGE_PATH=str(tmp_path))
        ),
    )

    with pytest.raises(ConfigurationError) as caught:
        object_storage.get_storage()
    assert caught.value.details == {"storage_type": "gcs"}
