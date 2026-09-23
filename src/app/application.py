"""FastAPI application factory."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app

from app.adapters.llm.embeddings_service import EmbeddingsClient
from app.adapters.persistence.database import close_db, init_db
from app.config.settings import settings
from app.routes import rag
from app.utils.error import AppError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def rag_lifespan(app: FastAPI):
    """Initialize the local RAG schema and dispose the pool on shutdown."""
    try:
        await init_db()
        yield
    finally:
        try:
            await EmbeddingsClient().close()
        finally:
            await close_db()


async def app_error_handler(request: Request, error: AppError) -> JSONResponse:
    """Map typed application errors to their HTTP status without leaking internals."""
    log = logger.warning if error.status_code < 500 else logger.error
    log(
        "%s %s failed: %s",
        request.method,
        request.url.path,
        error,
        exc_info=error.status_code >= 500,
    )
    payload = error.to_dict()
    payload.pop("status_code")
    return JSONResponse(status_code=error.status_code, content=payload)


async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    """Return a generic 500; the traceback stays in the logs."""
    # Starlette re-raises unexpected errors; the ASGI server logs the traceback.
    logger.error("%s %s failed unexpectedly", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "GENERIC_ERROR",
            "message": "An unexpected error occurred",
            "details": {},
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Apply to all app routes, including ADK: typed status or a generic 500."""
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    # Create FastAPI app with ADK integration
    app: FastAPI = get_fast_api_app(
        agents_dir=settings.AGENT_DIR,
        web=True,  # Enable web UI
        lifespan=rag_lifespan,
    )

    # Set application metadata
    app.title = settings.APP_NAME
    app.description = settings.APP_DESCRIPTION
    app.version = settings.APP_VERSION

    # Include RAG routes and the shared error handlers
    app.include_router(rag.router)
    register_error_handlers(app)

    # Add custom health check endpoint
    @app.get("/health", tags=["Health"], summary="Health Check")
    async def health_check():
        """
        Health check endpoint for monitoring systems.

        Returns:
            JSONResponse: A simple JSON response with status "ok"
        """
        return JSONResponse(
            content={
                "status": "ok",
                "app": settings.APP_NAME,
                "version": settings.APP_VERSION,
            },
            status_code=200,
        )

    logger.info(
        f"FastAPI application created: {settings.APP_NAME} v{settings.APP_VERSION}"
    )
    logger.info(f"Agent directory: {settings.AGENT_DIR}")
    logger.info("Web UI enabled: True")

    return app
