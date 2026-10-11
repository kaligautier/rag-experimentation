"""Shared error payload for ADK tools.

Tool results are read by the language model, so they carry a stable
``error_code`` and a human-readable message. Unexpected exceptions are logged
with their traceback and reported generically, never as ``str(exception)``.
"""

import logging

from app.utils.error import AppError, ErrorCode


def tool_error_result(error: Exception, action: str, logger: logging.Logger) -> dict:
    """Build the ``status: error`` payload for a failed tool call."""
    if isinstance(error, AppError):
        log = logger.warning if error.status_code < 500 else logger.error
        log("%s failed: %s", action, error, exc_info=error.status_code >= 500)
        return {
            "status": "error",
            "error_code": error.error_code.name,
            "error": error.message,
            "details": error.details,
            "message": f"{action} failed: {error.message}",
        }
    logger.error("%s failed unexpectedly", action, exc_info=error)
    return {
        "status": "error",
        "error_code": ErrorCode.GENERIC_ERROR.name,
        "error": "An unexpected error occurred",
        "details": {},
        "message": f"{action} failed: an unexpected error occurred",
    }
