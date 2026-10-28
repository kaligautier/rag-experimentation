"""Check evidence integrity and keep references outside the ADK invocation."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4]))
    return (
        import_module("eval.rag.promptfoo.capture"),
        import_module("eval.rag.promptfoo.provider"),
    )


def event(calls=(), responses=(), answer="", partial=False, error_code=None):
    return SimpleNamespace(
        error_code=error_code,
        partial=partial,
        get_function_calls=lambda: calls,
        get_function_responses=lambda: responses,
        is_final_response=lambda: bool(answer),
        content=SimpleNamespace(parts=[SimpleNamespace(text=answer, thought=False)]),
    )


def evidence(results=None):
    if results is None:
        results = [{"document_id": "doc", "chunk_id": "leaf", "content": "Fact"}]
    call = SimpleNamespace(name="search_documents", id="a", args={"query": "Query"})
    response = SimpleNamespace(
        name="search_documents",
        id="a",
        response={
            "status": "success",
            "result_count": len(results),
            "results": results,
        },
    )
    return [event(calls=[call]), event(responses=[response]), event(answer="Answer")]


def should_capture_actual_passages_without_another_search(modules):
    result = modules[0].capture(evidence())
    assert result["answer"] == "Answer"
    assert result["documents"][0]["content"] == "Fact"
    assert result["searches"] == [{"call_id": "a", "query": "Query"}]


def should_accept_completed_empty_search_for_abstention(modules):
    assert modules[0].capture(evidence([]))["documents"] == []


@pytest.mark.parametrize(
    "broken",
    [
        "no_call",
        "no_response",
        "no_answer",
        "error",
        "count",
        "passage",
        "duplicate",
        "unmatched",
    ],
)
def should_reject_incomplete_or_invalid_evidence(modules, broken):
    events = evidence()
    if broken == "no_call":
        events.pop(0)
    elif broken == "no_response":
        events.pop(1)
    elif broken == "no_answer":
        events.pop()
    elif broken == "error":
        events[0].error_code = "FAILED"
    elif broken == "count":
        events[1].get_function_responses()[0].response["result_count"] = 2
    elif broken == "passage":
        events[1].get_function_responses()[0].response["results"][0]["content"] = ""
    elif broken == "duplicate":
        events.insert(2, events[1])
    elif broken == "unmatched":
        events[1].get_function_responses()[0].id = "other"
    with pytest.raises(ValueError):
        modules[0].capture(events)


async def should_only_send_the_question_to_adk(modules, monkeypatch):
    provider = modules[1]
    run = AsyncMock(return_value={"answer": "Answer", "documents": []})
    monkeypatch.setattr(provider, "run_question", run)
    result = await provider.call_api("Question", {}, {"vars": {"reference": "SECRET"}})
    run.assert_awaited_once_with("Question")
    assert result["output"]["answer"] == "Answer"


async def should_report_provider_failure_without_raw_error(modules, monkeypatch):
    provider = modules[1]
    monkeypatch.setattr(
        provider, "run_question", AsyncMock(side_effect=RuntimeError("secret"))
    )
    assert await provider.call_api("Question", {}, {}) == {
        "error": "ADK evaluation failed (RuntimeError)"
    }
