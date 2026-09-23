"""Error payloads returned by ADK tools."""

import logging

from app.components.tools.custom._errors import tool_error_result
from app.utils.error import DocumentLimitError

logger = logging.getLogger("test.tools")


def should_expose_code_and_message_for_typed_errors():
    error = DocumentLimitError("Document exceeds 10 bytes", details={"limit": 10})

    result = tool_error_result(error, "Document indexation", logger)

    assert result == {
        "status": "error",
        "error_code": "DOCUMENT_TOO_LARGE",
        "error": "Document exceeds 10 bytes",
        "details": {"limit": 10},
        "message": "Document indexation failed: Document exceeds 10 bytes",
    }


def should_hide_unexpected_exception_text():
    result = tool_error_result(RuntimeError("dsn=postgres://u:p@h"), "Search", logger)

    assert result["status"] == "error"
    assert result["error_code"] == "GENERIC_ERROR"
    assert "postgres" not in str(result)
