"""Judge configuration and grading without contacting Vertex AI."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.config.settings import Settings


@pytest.mark.parametrize("override", [None, "independent-judge"])
def should_grade_with_an_independently_configured_model(monkeypatch, override):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    metric = import_module("eval.rag.correctness.correctness")
    monkeypatch.delenv("EVAL_MODEL", raising=False)
    monkeypatch.setenv("MODEL", "agent-under-test")
    if override:
        monkeypatch.setenv("EVAL_MODEL", override)
    config = Settings(_env_file=None)
    monkeypatch.setattr(metric, "settings", config)
    judge = metric.VertexCorrectnessJudge()
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        text='{"correct": true, "explanation": "Matches the reference."}'
    )
    judge._client = client

    assert judge.grade(question="Question?", answer="Answer", reference="Answer")
    assert client.models.generate_content.call_args.kwargs["model"] == (
        override or "gemini-3.1-pro-preview"
    )
    assert config.MODEL == "agent-under-test"
