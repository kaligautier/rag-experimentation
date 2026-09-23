"""Object storage adapter for persisting documents."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO

from app.config.settings import settings
from app.ports.rag import ObjectStoragePort
from app.utils.error import ConfigurationError, UnsafeStoragePathError

logger = logging.getLogger(__name__)


def validate_storage_filename(filename: str) -> str:
    """Accept a plain filename only, never a path from an external caller."""
    if not filename:
        return ""
    if (
        filename in {".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or Path(filename).name != filename
    ):
        raise UnsafeStoragePathError("Uploaded filename must not contain a path")
    return filename


class LocalFileStorageAdapter(ObjectStoragePort):
    """Local file system storage for documents."""

    def __init__(self, base_path: str | None = None):
        """Initialize local file storage.

        Args:
            base_path: Base directory for storage (defaults to RAG_STORAGE_PATH)
        """
        self.base_path = Path(base_path or settings.rag.STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info("Local file storage initialized at: %s", self.base_path)

    def _get_file_path(self, key: str) -> Path:
        """Get the full file path for a storage key."""
        storage_root = self.base_path.resolve()
        file_path = (storage_root / key).resolve()
        if file_path == storage_root or not file_path.is_relative_to(storage_root):
            raise UnsafeStoragePathError(
                "Storage key escapes the configured storage root"
            )
        return file_path

    async def get(self, key: str) -> bytes:
        """Retrieve a file from storage.

        Args:
            key: Storage key (path-like identifier)

        Returns:
            File contents as bytes

        Raises:
            FileNotFoundError: If the key doesn't exist
        """
        file_path = self._get_file_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        return file_path.read_bytes()

    async def put(
        self,
        key: str,
        content: bytes | BinaryIO,
        content_type: str | None = None,
    ) -> str:
        """Atomically publish a complete file in storage.

        Args:
            key: Storage key (path-like identifier)
            content: File contents (bytes or file-like object)
            content_type: MIME type (for metadata, not used locally)

        Returns:
            Storage URL/path (file:// scheme for local)
        """
        file_path = self._get_file_path(key)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        temporary_path = None
        try:
            # A sibling temporary file keeps the rename on the same filesystem.
            with NamedTemporaryFile(
                dir=file_path.parent, prefix=".rag-", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(
                    content if isinstance(content, bytes) else content.read()
                )
            temporary_path.replace(file_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        logger.info("Stored file: %s at %s", key, file_path)
        return f"file://{file_path.absolute()}"

    async def delete(self, key: str) -> None:
        """Delete a file from storage.

        Args:
            key: Storage key to delete

        Raises:
            FileNotFoundError: If the key doesn't exist
        """
        file_path = self._get_file_path(key)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
        file_path.unlink()
        logger.info("Deleted file: %s", key)

    @staticmethod
    def generate_key(document_id: str, content_hash: str, filename: str = "") -> str:
        """Generate a storage key from document metadata.

        Args:
            document_id: Unique document identifier
            content_hash: SHA256 hash of content (for idempotence)
            filename: Original filename (optional)

        Returns:
            Storage key
        """
        # Format: {document_id}/{content_hash}/{filename}
        key_parts = [document_id, content_hash]
        if filename:
            key_parts.append(validate_storage_filename(filename))
        return "/".join(key_parts)


# Singleton instance
_storage_instance: ObjectStoragePort | None = None


def get_storage() -> ObjectStoragePort:
    """Get or create the storage adapter selected by RAG_STORAGE_TYPE."""
    global _storage_instance
    if _storage_instance is None:
        storage_type = settings.rag.STORAGE_TYPE
        if storage_type != "local":
            raise ConfigurationError(
                "Only the local storage backend is implemented",
                details={"storage_type": storage_type},
            )
        _storage_instance = LocalFileStorageAdapter()
    return _storage_instance
