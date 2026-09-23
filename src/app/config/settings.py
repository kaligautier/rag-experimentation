"""Application configuration using Pydantic BaseSettings."""

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.error import ConfigurationError


class RagSettings(BaseSettings):
    """RAG-specific configuration for document indexing and retrieval."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=True,
        extra="ignore",
    )

    # Database configuration
    DB_URL: str = Field(
        default="postgresql+asyncpg://rag_user:secret@127.0.0.1:15433/rag_db",
        description="PostgreSQL connection URL with asyncpg driver",
    )
    DB_POOL_SIZE: int = Field(
        default=5,
        description="Database connection pool size",
    )
    DB_MAX_OVERFLOW: int = Field(
        default=2,
        description="Maximum overflow connections beyond pool size",
    )

    EMBEDDING_MODEL: str = Field(
        default="gemini-embedding-001",
        description="Gemini embedding model served by Vertex AI",
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=3072,
        ge=128,
        le=3072,
        description="Output dimension; re-embed all documents when changing it",
    )
    EMBEDDING_LOCATION: str = Field(
        default="",
        description="Vertex AI region; defaults to GOOGLE_CLOUD_LOCATION",
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent single-text Gemini embedding requests",
    )

    CHUNK_SIZE: int = Field(
        default=1200,
        ge=1,
        description="Maximum searchable leaf size in tokens (LlamaIndex tokenizer)",
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        ge=0,
        description="Token overlap when subdividing a long Markdown section",
    )
    MIN_CHUNK_SIZE: int = Field(
        default=200,
        ge=1,
        description="Minimum chunk size (smaller chunks are merged)",
    )
    MAX_CHUNKS_PER_DOC: int = Field(
        default=2000,
        ge=1,
        description="Maximum number of chunks per document",
    )

    # Retrieval configuration
    TOP_K: int = Field(
        default=5,
        description="Number of top results to return in search",
    )
    SIMILARITY_THRESHOLD: float = Field(
        default=0.0,
        description="Minimum cosine similarity threshold (0.0 = disabled)",
    )
    MAX_PARENT_TOKENS: int = Field(
        default=4096,
        ge=1,
        description="Maximum parent context size returned by hierarchical retrieval",
    )
    MAX_UPLOAD_BYTES: int = Field(
        default=5242880,
        ge=1,
        description="Maximum file upload size in bytes (5 MiB default)",
    )

    # Storage configuration
    STORAGE_TYPE: str = Field(
        default="local",
        description="Storage backend: 'local' or 'gcs'",
    )
    STORAGE_PATH: str = Field(
        default="/tmp/rag_storage",
        description="Local file storage path (used if STORAGE_TYPE='local')",
    )
    STORAGE_BUCKET_NAME: str = Field(
        default="",
        description="GCS bucket name (used if STORAGE_TYPE='gcs')",
    )


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    Configuration is loaded from:
    1. Environment variables
    2. .env file (in non-Docker environments)
    3. Default values defined below

    All settings are type-safe and validated by Pydantic.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    # Auto-load .env file in non-Docker environments
    if not os.getenv("DOCKER_ENV"):
        load_dotenv(find_dotenv(".env"))

    # Application metadata
    APP_NAME: str = Field(
        default="ADK Agent Template",
        description="Application name displayed in API documentation",
    )
    APP_DESCRIPTION: str = Field(
        default="Production-ready ADK agent template with best practices",
        description="Application description for API documentation",
    )
    APP_VERSION: str = Field(
        default="0.1.0",
        description="Application version",
    )
    PROJECT_NAME: str = Field(
        default="adk-agent-template",
        description="Project identifier used in paths and naming",
    )

    # Server configuration
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host address (0.0.0.0 for all interfaces)",
    )
    PORT: int = Field(
        default=8000,
        description="Server port",
    )
    DEBUG: bool = Field(
        default=False,
        description="Enable debug mode (disable in production)",
    )

    # Logging configuration
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    # Agent configuration
    AGENT_NAME: str = Field(
        default="template_agent",
        description="Primary agent name",
    )
    MODEL: str = Field(
        default="gemini-2.5-flash",
        description="AI model to use for the agent",
    )

    # Agent directory (computed from project structure)
    @property
    def AGENT_DIR(self) -> str:  # noqa: N802
        """Get the absolute path to the agents directory."""
        return str(Path(__file__).parent.parent / "components" / "agents")

    # Instructions directory (computed from project structure)
    @property
    def INSTRUCTIONS_DIR(self) -> str:  # noqa: N802
        """Get the absolute path to the instructions templates directory."""
        return str(Path(__file__).parent.parent / "instructions" / "templates")

    # Google Cloud Platform configuration (required for Vertex AI)
    GOOGLE_GENAI_USE_VERTEXAI: bool = Field(
        description="Enable Vertex AI for Google Generative AI (required)",
    )
    GOOGLE_CLOUD_PROJECT: str = Field(
        description="GCP project ID (required)",
    )
    GOOGLE_CLOUD_LOCATION: str = Field(
        description="GCP region (e.g., us-central1, europe-west1)",
    )

    # Session management
    USER_ID: str = Field(
        default="api_user",
        description="Default user ID for session management",
    )

    # RAG configuration
    @property
    def rag(self) -> RagSettings:  # noqa: N802
        """Get RAG-specific configuration."""
        return RagSettings()


# Singleton settings instance with error handling
try:
    settings = Settings()
except ValidationError as e:
    raise ConfigurationError(
        message="Configuration validation failed",
        details={"pydantic_errors": e.errors()},
    ) from e
