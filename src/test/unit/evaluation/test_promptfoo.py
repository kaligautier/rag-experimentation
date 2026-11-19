"""Check OTEL parent propagation and integrity of retrieved RAG evidence."""

import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "eval"))
    module = import_module("provider")
    monkeypatch.setattr(module, "_tracer_provider_state", module._TracerProviderState())
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(module, "OTLPSpanExporter", Mock(return_value=exporter))
    monkeypatch.setattr(module.trace, "set_tracer_provider", Mock())
    monkeypatch.setattr(
        module.trace,
        "get_tracer",
        lambda *args: module._tracer_provider_state.provider.get_tracer(*args),
    )
    yield module
    if module._tracer_provider_state.provider:
        module._tracer_provider_state.provider.shutdown()


def event(answer="Answer", partial=False, error_code=None):
    return SimpleNamespace(
        error_code=error_code,
        error_message="fixture error",
        author="agent",
        usage_metadata=None,
        partial=partial,
        is_final_response=lambda: bool(answer),
        content=SimpleNamespace(parts=[SimpleNamespace(text=answer, thought=False)]),
    )


def evidence(results=None, *, trace_id=42, schema="legacy"):
    if results is None:
        results = [{"document_id": "doc", "chunk_id": "leaf", "content": "Fact"}]
    args_key, result_key = (
        ("gcp.vertex.agent.tool_call_args", "gcp.vertex.agent.tool_response")
        if schema == "legacy"
        else ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result")
    )
    return SimpleNamespace(
        context=SimpleNamespace(trace_id=trace_id),
        start_time=1,
        status=SimpleNamespace(status_code=trace.StatusCode.OK),
        attributes={
            "gen_ai.tool.name": "search_documents",
            "gen_ai.tool.call.id": "a",
            args_key: json.dumps({"query": "Query"}),
            result_key: json.dumps(
                {"status": "success", "result_count": len(results), "results": results}
            ),
        },
    )


@pytest.mark.parametrize("schema", ["legacy", "v2"])
def should_capture_only_actual_passages_in_current_trace(provider, schema):
    unrelated = evidence(trace_id=99)
    result = provider._capture_documents([unrelated, evidence(schema=schema)], 42)
    assert result == {
        "documents": ["Fact"],
        "searches": [{"call_id": "a", "query": "Query"}],
    }


def should_accept_completed_empty_search(provider):
    assert provider._capture_documents([evidence([])], 42)["documents"] == []


def should_ignore_partial_final_text(provider):
    assert (
        provider._final_text_from_events([event(), event("Incomplete", partial=True)])
        == "Answer"
    )


@pytest.mark.parametrize(
    "broken",
    ["missing", "args", "result", "count", "passage", "duplicate", "id", "error"],
)
def should_reject_missing_or_invalid_span_evidence(provider, broken):
    item = evidence()
    spans = [item]
    if broken == "missing":
        spans = []
    elif broken == "args":
        item.attributes["gcp.vertex.agent.tool_call_args"] = "{}"
    elif broken == "result":
        item.attributes["gcp.vertex.agent.tool_response"] = "{}"
    elif broken == "count":
        item.attributes["gcp.vertex.agent.tool_response"] = json.dumps(
            {"status": "success", "result_count": 2, "results": []}
        )
    elif broken == "passage":
        spans = [evidence([{"document_id": "doc", "chunk_id": "leaf", "content": ""}])]
    elif broken == "duplicate":
        spans.append(item)
    elif broken == "id":
        item.attributes.pop("gen_ai.tool.call.id")
    elif broken == "error":
        item.status.status_code = trace.StatusCode.ERROR
    with pytest.raises(ValueError):
        provider._capture_documents(spans, 42)


@pytest.mark.parametrize("events", [[], [event(error_code="FAILED")]])
def should_reject_missing_or_failed_answer(provider, events):
    with pytest.raises(RuntimeError):
        provider._final_text_from_events(events)


def should_reuse_exporter_and_refuse_endpoint_switch(provider, caplog):
    first = provider._ensure_tracer_provider("http://localhost:4318/")
    assert provider._ensure_tracer_provider("http://localhost:9999") is first
    provider.OTLPSpanExporter.assert_called_once_with(
        endpoint="http://localhost:4318/v1/traces"
    )
    provider.trace.set_tracer_provider.assert_called_once_with(first)
    assert "ignoring request" in caplog.text


@pytest.mark.parametrize("fail", [False, True])
def should_link_parent_flush_and_clear_spans_between_rows(provider, fail):
    exporter = provider.OTLPSpanExporter.return_value
    previous = trace.get_current_span()
    for index in range(2):
        context = {
            "traceparent": f"00-{index + 1:032x}-{7:016x}-01",
            "evaluationId": "eval",
            "testCaseId": str(index),
        }
        try:
            with provider._provider_span(context, provider.DEFAULT_OTLP_ENDPOINT):
                assert provider._tracer_provider_state.spans.get_finished_spans() == ()
                if fail:
                    raise RuntimeError("failure")
        except RuntimeError:
            pass
        assert provider._tracer_provider_state.spans.get_finished_spans() == ()
        assert trace.get_current_span() is previous
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for index, span in enumerate(spans):
        assert span.context.trace_id == index + 1
        assert span.parent.span_id == 7
        assert span.attributes["promptfoo.eval.id"] == "eval"
        assert span.attributes["promptfoo.test.id"] == str(index)
        assert (span.status.status_code == trace.StatusCode.ERROR) == fail


def should_report_provider_failure_without_raw_error(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_run_workflow_provider",
        AsyncMock(side_effect=RuntimeError("secret")),
    )
    assert provider.call_api("Question", {}, {}) == {
        "error": "ADK evaluation failed (RuntimeError)"
    }


@pytest.mark.parametrize("fail", [False, True])
async def should_run_adapter_keep_reference_out_and_close_database(
    provider, monkeypatch, fail
):
    messages = []

    class Runner:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def run_async(self, **kwargs):
            messages.append(kwargs["new_message"])
            if fail:
                raise RuntimeError("synthetic failure")
            with trace.get_tracer("test").start_as_current_span(
                "execute_tool search_documents"
            ) as span:
                span.set_attributes(evidence().attributes)
            yield event()

    monkeypatch.setattr(provider, "Runner", Runner)
    init_db, close_db = AsyncMock(), AsyncMock()
    monkeypatch.setattr(provider, "init_db", init_db)
    monkeypatch.setattr(provider, "close_db", close_db)
    with provider._provider_span({}, provider.DEFAULT_OTLP_ENDPOINT):
        if fail:
            with pytest.raises(RuntimeError):
                await provider._run_workflow_provider("Question", {}, {})
        else:
            result = await provider._run_workflow_provider(
                "Question", {}, {"vars": {"reference": "REFERENCE_ONLY"}}
            )
            assert result["output"] == "Answer"
            assert result["metadata"]["documents"] == ["Fact"]
            assert result["metadata"]["searches"] == [
                {"call_id": "a", "query": "Query"}
            ]
            assert "REFERENCE_ONLY" not in str(result)
    assert len(messages) == 1
    assert messages[0].parts[0].text == "Question"
    init_db.assert_awaited_once()
    close_db.assert_awaited_once()
