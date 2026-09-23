"""
Tests for configuration module.

Tests:
- Settings loading
- Environment variable handling
- Configuration validation
"""

import pytest
from pydantic import ValidationError


def test_settings_loads_successfully():
    """Test that settings load with required environment variables."""
    from app.config.settings import settings

    # Test that required settings are present and non-empty
    assert settings.GOOGLE_CLOUD_PROJECT is not None
    assert len(settings.GOOGLE_CLOUD_PROJECT) > 0
    assert settings.GOOGLE_CLOUD_LOCATION is not None
    assert len(settings.GOOGLE_CLOUD_LOCATION) > 0
    assert settings.GOOGLE_GENAI_USE_VERTEXAI is True


def test_settings_has_defaults():
    """Test that settings have appropriate default values."""
    from app.config.settings import settings

    assert settings.APP_NAME == "ADK Agent Template"
    assert settings.MODEL == "gemini-3.8-flash"
    assert settings.HOST == "0.0.0.0"
    assert settings.PORT == 8000


def should_default_to_current_gemini_models():
    """Model defaults target Gemini 3.8 and the full-quality embedding space."""
    from app.config.settings import RagSettings, Settings

    assert Settings.model_fields["MODEL"].default == "gemini-3.8-flash"
    assert RagSettings.model_fields["EMBEDDING_MODEL"].default == (
        "gemini-embedding-001"
    )
    assert RagSettings.model_fields["EMBEDDING_DIMENSIONS"].default == 3072


def test_agent_dir_property():
    """Test that AGENT_DIR property returns correct path."""
    from pathlib import Path

    from app.config.settings import settings

    agent_dir = settings.AGENT_DIR

    assert isinstance(agent_dir, str)
    assert agent_dir.endswith("components/agents")
    assert Path(agent_dir).name == "agents"


def test_settings_validation():
    """Test that settings validate required fields."""
    import os
    from unittest.mock import patch

    # Remove required env var temporarily
    with patch.dict(os.environ, {}, clear=True):
        # Should raise ValidationError for missing required fields
        with pytest.raises(ValidationError):
            from app.config.settings import Settings

            Settings()
