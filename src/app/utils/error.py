"""Error handling with structured exceptions and HTTP status mapping."""

from enum import Enum


class ErrorCode(Enum):
    """Error codes for application exceptions."""

    # Generic and input errors (1xxx)
    GENERIC_ERROR = 1000
    INVALID_INPUT = 1001
    CONFIGURATION_ERROR = 1002
    INSTRUCTION_ERROR = 1003
    UNSUPPORTED_DOCUMENT = 1004
    UNSAFE_FILENAME = 1005
    DOCUMENT_TOO_LARGE = 1006
    EMPTY_DOCUMENT = 1007
    DOCUMENT_NOT_FOUND = 1008
    DOCUMENT_NOT_INDEXED = 1009

    # External dependency errors (3xxx)
    TOOL_EXECUTION_ERROR = 3001
    EMBEDDING_ERROR = 3002

    # Storage errors (5xxx)
    EMBEDDING_SCHEMA_MISMATCH = 5001


HTTP_STATUS_CODES = {
    ErrorCode.GENERIC_ERROR: 500,
    ErrorCode.INVALID_INPUT: 400,
    ErrorCode.CONFIGURATION_ERROR: 500,
    ErrorCode.INSTRUCTION_ERROR: 500,
    ErrorCode.UNSUPPORTED_DOCUMENT: 400,
    ErrorCode.UNSAFE_FILENAME: 400,
    ErrorCode.DOCUMENT_TOO_LARGE: 413,
    ErrorCode.EMPTY_DOCUMENT: 400,
    ErrorCode.DOCUMENT_NOT_FOUND: 404,
    ErrorCode.DOCUMENT_NOT_INDEXED: 409,
    ErrorCode.TOOL_EXECUTION_ERROR: 502,
    ErrorCode.EMBEDDING_ERROR: 502,
    ErrorCode.EMBEDDING_SCHEMA_MISMATCH: 503,
}


ERROR_MESSAGES = {
    ErrorCode.GENERIC_ERROR: "An unexpected error occurred",
    ErrorCode.INVALID_INPUT: "Invalid input provided",
    ErrorCode.CONFIGURATION_ERROR: "Configuration error",
    ErrorCode.INSTRUCTION_ERROR: "Instruction loading or rendering failed",
    ErrorCode.UNSUPPORTED_DOCUMENT: "Only Markdown (.md) documents are supported",
    ErrorCode.UNSAFE_FILENAME: "Uploaded filename must not contain a path",
    ErrorCode.DOCUMENT_TOO_LARGE: "Document exceeds a configured size limit",
    ErrorCode.EMPTY_DOCUMENT: "No text extracted from document",
    ErrorCode.DOCUMENT_NOT_FOUND: "Document not found",
    ErrorCode.DOCUMENT_NOT_INDEXED: (
        "Document exists but is not indexed; retry with force=true"
    ),
    ErrorCode.TOOL_EXECUTION_ERROR: "Tool execution failed",
    ErrorCode.EMBEDDING_ERROR: "Embedding provider returned an invalid response",
    ErrorCode.EMBEDDING_SCHEMA_MISMATCH: (
        "Stored vectors must be migrated before embedding operations can run"
    ),
}


class AppError(Exception):
    """
    Base application error with automatic HTTP status mapping.

    Attributes:
        error_code: ErrorCode enum value
        message: Human-readable error message
        details: Optional dictionary with error context
        status_code: HTTP status code (automatically mapped)
    """

    def __init__(
        self,
        error_code: ErrorCode,
        message: str = None,
        details: dict = None,
    ):
        self.error_code = error_code
        self.status_code = HTTP_STATUS_CODES.get(error_code, 500)
        self.message = message or ERROR_MESSAGES.get(error_code, "An error occurred")
        self.details = details or {}

        super().__init__(self.message)

    def __str__(self):
        """String representation with error code and details."""
        detail_str = f", details={self.details}" if self.details else ""
        return f"[{self.error_code.name}] {self.message}{detail_str}"

    def to_dict(self):
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code.name,
            "message": self.message,
            "status_code": self.status_code,
            "details": self.details,
        }


class InvalidInputError(AppError, ValueError):
    """Raised when input validation fails."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.INVALID_INPUT,
            message=message,
            details=details,
        )


class ConfigurationError(AppError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.CONFIGURATION_ERROR,
            message=message,
            details=details,
        )


class InstructionError(AppError):
    """Raised when instruction loading or rendering fails."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.INSTRUCTION_ERROR,
            message=message,
            details=details,
        )


class ToolExecutionError(AppError):
    """Raised when tool execution fails."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.TOOL_EXECUTION_ERROR,
            message=message,
            details=details,
        )


# --- RAG errors ---
#
# Client-facing RAG errors also subclass ValueError so callers that predate the
# typed hierarchy (and tests using ``pytest.raises(ValueError)``) keep working.


class UnsupportedDocumentError(AppError, ValueError):
    """Raised when the MIME type or file extension is not Markdown."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.UNSUPPORTED_DOCUMENT,
            message=message,
            details=details,
        )


class UnsafeStoragePathError(AppError, ValueError):
    """A client-provided name or key would escape the configured storage root."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.UNSAFE_FILENAME,
            message=message,
            details=details,
        )


class DocumentLimitError(AppError, ValueError):
    """The document exceeds a configured resource limit (bytes or chunks)."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.DOCUMENT_TOO_LARGE,
            message=message,
            details=details,
        )


class EmptyDocumentError(AppError, ValueError):
    """The document produced no indexable text."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.EMPTY_DOCUMENT,
            message=message,
            details=details,
        )


class DocumentNotFoundError(AppError, ValueError):
    """Raised when a document ID does not exist."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=message,
            details=details,
        )


class EmbeddingError(AppError, ValueError):
    """The embedding provider returned an unusable response."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.EMBEDDING_ERROR,
            message=message,
            details=details,
        )


class EmbeddingSchemaMismatchError(AppError, RuntimeError):
    """Stored vectors must be migrated before embedding operations can run."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.EMBEDDING_SCHEMA_MISMATCH,
            message=message,
            details=details,
        )


class DocumentNotIndexedError(AppError):
    """An existing document needs forced reindexing before it can be reused."""

    def __init__(self, message: str = None, details: dict = None):
        super().__init__(
            error_code=ErrorCode.DOCUMENT_NOT_INDEXED,
            message=message,
            details=details,
        )
