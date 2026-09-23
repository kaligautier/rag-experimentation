import pytest

from app.utils.error import (
    AppError,
    DocumentLimitError,
    DocumentNotFoundError,
    EmbeddingError,
    EmbeddingSchemaMismatchError,
    EmptyDocumentError,
    ErrorCode,
    UnsafeStoragePathError,
    UnsupportedDocumentError,
)


@pytest.mark.parametrize(
    ("error", "code", "status"),
    [
        (UnsupportedDocumentError(), ErrorCode.UNSUPPORTED_DOCUMENT, 400),
        (UnsafeStoragePathError(), ErrorCode.UNSAFE_FILENAME, 400),
        (DocumentLimitError(), ErrorCode.DOCUMENT_TOO_LARGE, 413),
        (EmptyDocumentError(), ErrorCode.EMPTY_DOCUMENT, 400),
        (DocumentNotFoundError(), ErrorCode.DOCUMENT_NOT_FOUND, 404),
        (EmbeddingError(), ErrorCode.EMBEDDING_ERROR, 502),
        (EmbeddingSchemaMismatchError(), ErrorCode.EMBEDDING_SCHEMA_MISMATCH, 503),
    ],
)
def should_map_rag_errors_to_codes_and_http_status(error, code, status):
    assert isinstance(error, AppError)
    assert error.error_code is code
    assert error.status_code == status
    assert error.message


def should_stay_catchable_as_legacy_builtin_exceptions():
    with pytest.raises(ValueError, match="filename"):
        raise UnsafeStoragePathError("bad filename")
    with pytest.raises(RuntimeError):
        raise EmbeddingSchemaMismatchError()


def should_carry_structured_details():
    error = DocumentLimitError("too big", details={"limit_bytes": 10})
    assert error.details == {"limit_bytes": 10}
    assert str(error) == "[DOCUMENT_TOO_LARGE] too big, details={'limit_bytes': 10}"
