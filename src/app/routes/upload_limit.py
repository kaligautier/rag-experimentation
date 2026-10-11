"""Bound upload ingress before FastAPI's multipart parser opens temporary files."""

from fastapi.routing import APIRoute
from starlette.datastructures import Headers
from starlette.types import Message, Receive, Scope, Send

from app.config.settings import settings
from app.utils.error import DocumentLimitError, InvalidInputError

# File bytes have their own exact limit in the upload handler. Bound the whole
# request too, allowing a small fixed budget for multipart headers and fields.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class BoundedUploadRoute(APIRoute):
    """Buffer one bounded upload; other routes, including ADK SSE, are untouched."""

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.methods and scope["method"] not in self.methods:
            await super().handle(scope, receive, send)
            return

        limit = settings.rag.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES

        def too_large() -> DocumentLimitError:
            return DocumentLimitError(
                f"Upload request exceeds the maximum body size of {limit} bytes",
                details={"limit_request_bytes": limit},
            )

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError as error:
                raise InvalidInputError("Invalid Content-Length header") from error
            if declared_length < 0:
                raise InvalidInputError("Invalid Content-Length header")
            if declared_length > limit:
                raise too_large()

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > limit:
                raise too_large()
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        payload = bytes(body)
        del body
        consumed = False

        async def replay() -> Message:
            nonlocal payload, consumed
            if consumed:
                return await receive()
            consumed = True
            message = {"type": "http.request", "body": payload, "more_body": False}
            payload = b""
            return message

        await super().handle(scope, replay, send)
