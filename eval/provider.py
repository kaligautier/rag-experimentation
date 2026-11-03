"""Direct ADK provider with native OTEL tracing and retrieved RAG evidence."""

import asyncio
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.adapters.persistence.database import close_db, init_db
from app.components.agents.root.agent import root_agent

APP_NAME = "root"
DEFAULT_USER_ID = "promptfoo-user"
DEFAULT_OTLP_ENDPOINT = "http://localhost:4318"

_logger = logging.getLogger(__name__)


@dataclass
class _TracerProviderState:
    provider: TracerProvider | None = None
    otlp_endpoint: str | None = None
    spans: InMemorySpanExporter | None = None


_tracer_provider_state = _TracerProviderState()


def _otlp_endpoint(options: dict[str, Any]) -> str:
    config = options.get("config", {})
    return str(config.get("otlp_endpoint") or DEFAULT_OTLP_ENDPOINT)


def _ensure_tracer_provider(otlp_endpoint: str) -> TracerProvider:
    """Install a TracerProvider that exports to ``otlp_endpoint`` exactly once.

    OpenTelemetry's global ``set_tracer_provider`` is set-once: a subsequent
    call is silently rejected with an "Overriding of current TracerProvider is
    not allowed" warning. We mirror that by installing on the first call and
    refusing to swap endpoints later, which avoids stacking a second
    SimpleSpanProcessor onto the same provider (which would cause every span
    to be exported twice).
    """

    if _tracer_provider_state.provider is not None:
        if _tracer_provider_state.otlp_endpoint != otlp_endpoint:
            _logger.warning(
                "TracerProvider already configured for %s; "
                "ignoring request to switch to %s",
                _tracer_provider_state.otlp_endpoint,
                otlp_endpoint,
            )
        return _tracer_provider_state.provider

    exporter = OTLPSpanExporter(endpoint=f"{otlp_endpoint.rstrip('/')}/v1/traces")
    provider = TracerProvider(
        resource=Resource.create({"service.name": "promptfoo-google-adk-example"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    spans = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    trace.set_tracer_provider(provider)

    _tracer_provider_state.spans = spans
    _tracer_provider_state.provider = provider
    _tracer_provider_state.otlp_endpoint = otlp_endpoint
    return provider


@contextmanager
def _provider_span(context: dict[str, Any], otlp_endpoint: str) -> Iterator[None]:
    provider = _ensure_tracer_provider(otlp_endpoint)
    # Suites run sequentially; retain spans only for the current invocation.
    _tracer_provider_state.spans.clear()
    tracer = trace.get_tracer("promptfoo.google-adk.example", "1.0.0")
    parent_context = extract({"traceparent": context.get("traceparent", "")})
    token = otel_context.attach(parent_context)
    try:
        with tracer.start_as_current_span(
            "promptfoo_google_adk_provider",
            attributes={
                "promptfoo.eval.id": str(context.get("evaluationId") or ""),
                "promptfoo.test.id": str(context.get("testCaseId") or ""),
            },
        ):
            yield
    finally:
        otel_context.detach(token)
        provider.force_flush()
        _tracer_provider_state.spans.clear()


def _session_id(context: dict[str, Any], vars_dict: dict[str, Any]) -> str:
    if explicit := vars_dict.get("session_id"):
        return str(explicit)

    evaluation_id = context.get("evaluationId", "local-eval")
    test_case_id = context.get("testCaseId", "default-test")
    repeat_index = context.get("repeatIndex")
    if repeat_index is None:
        return f"rag-{evaluation_id}-{test_case_id}"
    return f"rag-{evaluation_id}-{test_case_id}-repeat-{repeat_index}"


def _final_text_from_events(
    events: list[Any], *, allow_safety_block: bool = False
) -> str | None:
    safety_blocked = False
    for event in events:
        if event.error_code:
            if allow_safety_block and event.error_code == "SAFETY":
                safety_blocked = True
                continue
            raise RuntimeError(f"ADK {event.error_code}: {event.error_message or ''}")
    if safety_blocked:
        return None
    for event in reversed(events):
        if (
            event.partial
            or event.author == "user"
            or not event.is_final_response()
            or not event.content
            or not event.content.parts
        ):
            continue
        text = "".join(
            part.text or "" for part in event.content.parts if not part.thought
        )
        if text.strip():
            return text.strip()
    raise RuntimeError("ADK returned no final model response")


def _extract_token_usage(events: list[Any]) -> dict[str, int]:
    """Sum the agent token usage reported by ADK."""
    total = prompt = completion = cached = 0
    for event in events:
        meta = event.usage_metadata
        if meta is None:
            continue
        total += meta.total_token_count or 0
        prompt += meta.prompt_token_count or 0
        completion += meta.candidates_token_count or 0
        cached += meta.cached_content_token_count or 0
    return {
        "total": total,
        "prompt": prompt,
        "completion": completion,
        "cached": cached,
    }


def _capture_documents(spans, trace_id: int) -> dict:
    """Extract real search results from ended ADK tool spans in this trace."""
    searches, documents, call_ids = [], [], set()
    for span in sorted(spans, key=lambda span: span.start_time):
        attributes = span.attributes or {}
        if (
            span.context.trace_id != trace_id
            or attributes.get("gen_ai.tool.name") != "search_documents"
        ):
            continue
        if span.status.status_code == trace.StatusCode.ERROR:
            raise ValueError("Search span failed")
        call_id = attributes.get("gen_ai.tool.call.id")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in call_ids:
            raise ValueError("Invalid search span call ID")
        # ADK legacy and v2 both emit JSON tool payloads; never use event fallbacks.
        args = json.loads(
            attributes.get("gen_ai.tool.call.arguments")
            or attributes.get("gcp.vertex.agent.tool_call_args", "{}")
        )
        data = json.loads(
            attributes.get("gen_ai.tool.call.result")
            or attributes.get("gcp.vertex.agent.tool_response", "{}")
        )
        query = args.get("query") if isinstance(args, dict) else None
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Missing search arguments in OTEL span")
        if not isinstance(data, dict) or data.get("status") != "success":
            raise ValueError("Missing or failed search result in OTEL span")
        results, count = data.get("results"), data.get("result_count")
        if (
            not isinstance(results, list)
            or type(count) is not int
            or count != len(results)
        ):
            raise ValueError("Incomplete search results")
        for hit in results:
            if not isinstance(hit, dict) or any(
                not isinstance(hit.get(key), str) or not hit[key].strip()
                for key in ("document_id", "chunk_id", "content")
            ):
                raise ValueError("Invalid passage")
        documents.extend(hit["content"] for hit in results)
        searches.append({"call_id": call_id, "query": query})
        call_ids.add(call_id)
    if not searches:
        raise ValueError("Missing completed search span")
    return {"documents": documents, "searches": searches}


async def _run_workflow_provider(prompt, options, context):
    session_id = _session_id(context, context.get("vars", {}))
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=DEFAULT_USER_ID,
        session_id=session_id,
        state={},
    )
    events = []
    try:
        await init_db()
        async with Runner(
            app=App(name=APP_NAME, root_agent=root_agent),
            session_service=session_service,
        ) as runner:
            async for event in runner.run_async(
                user_id=DEFAULT_USER_ID,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
            ):
                events.append(event)
        return {
            "output": _final_text_from_events(events),
            "metadata": {
                **_capture_documents(
                    _tracer_provider_state.spans.get_finished_spans(),
                    trace.get_current_span().get_span_context().trace_id,
                ),
                "event_count": len(events),
            },
            "tokenUsage": _extract_token_usage(events),
        }
    finally:
        # asyncio.run creates a new loop per invocation; do not reuse its DB pool.
        await close_db()


def call_api(prompt, options, context):
    """Run one application turn, keeping reference answers outside the agent."""
    try:
        with _provider_span(context, _otlp_endpoint(options)):
            return asyncio.run(_run_workflow_provider(prompt, options, context))
    except Exception as error:
        return {"error": f"ADK evaluation failed ({type(error).__name__})"}
