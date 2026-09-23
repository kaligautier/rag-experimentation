"""ADK custom metric that judges RAG answer correctness with Vertex AI."""

from __future__ import annotations

import json
from typing import Protocol

from google import genai
from google.adk.evaluation.eval_case import ConversationScenario, Invocation
from google.adk.evaluation.eval_metrics import EvalMetric, EvalStatus
from google.adk.evaluation.evaluator import EvaluationResult, PerInvocationResult
from google.genai import types

from app.config.settings import settings

CORRECTNESS_THRESHOLD = 1.0

_INSTRUCTIONS = """You are a strict evaluator of a RAG assistant response.
Compare the assistant answer against the reference answer for the user question.
Mark correct=true only when the answer is factually consistent with the reference,
answers the question, and does not introduce a material contradiction. Equivalent
wording and harmless additional context are allowed. Do not reward unsupported
claims. Return JSON only: {"correct": boolean, "explanation": string}."""


class CorrectnessJudge(Protocol):
    """Boundary used to keep metric tests independent from Vertex AI."""

    def grade(self, *, question: str, answer: str, reference: str) -> bool: ...


class VertexCorrectnessJudge:
    """Synchronous Vertex AI judge, created lazily to preserve ADC behaviour."""

    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                vertexai=True,
                project=settings.GOOGLE_CLOUD_PROJECT,
                location=settings.GOOGLE_CLOUD_LOCATION,
                http_options=types.HttpOptions(api_version="v1", timeout=30000),
            )
        return self._client

    def grade(self, *, question: str, answer: str, reference: str) -> bool:
        response = self._get_client().models.generate_content(
            model=settings.EVAL_MODEL,
            contents=(
                f"Question:\n{question}\n\n"
                f"Reference answer:\n{reference}\n\n"
                f"Assistant answer:\n{answer}"
            ),
            config=types.GenerateContentConfig(
                system_instruction=_INSTRUCTIONS,
                temperature=0,
                response_mime_type="application/json",
            ),
        )
        if not response.text:
            raise ValueError("Vertex correctness judge returned an empty response")
        try:
            verdict = json.loads(response.text)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Vertex correctness judge returned invalid JSON"
            ) from error
        if not isinstance(verdict, dict) or not isinstance(
            verdict.get("correct"), bool
        ):
            raise ValueError("Vertex correctness judge omitted boolean 'correct'")
        return verdict["correct"]


_judge: CorrectnessJudge | None = None


def _get_judge() -> CorrectnessJudge:
    global _judge
    if _judge is None:
        _judge = VertexCorrectnessJudge()
    return _judge


def _text(invocation: Invocation, attribute: str) -> str:
    content = getattr(invocation, attribute)
    if not content or not content.parts:
        return ""
    return "\n".join(part.text for part in content.parts if part.text)


def _status(score: float) -> EvalStatus:
    return EvalStatus.PASSED if score >= CORRECTNESS_THRESHOLD else EvalStatus.FAILED


def correctness(
    eval_metric: EvalMetric,
    actual_invocations: list[Invocation],
    expected_invocations: list[Invocation] | None,
    conversation_scenario: ConversationScenario | None = None,
) -> EvaluationResult:
    """Score final responses against human-authored reference answers."""
    del eval_metric, conversation_scenario
    if expected_invocations is None:
        return EvaluationResult(overall_eval_status=EvalStatus.NOT_EVALUATED)
    if len(actual_invocations) != len(expected_invocations):
        raise ValueError(
            "actual_invocations and expected_invocations must have the same length"
        )

    results = []
    for actual, expected in zip(actual_invocations, expected_invocations, strict=True):
        score = float(
            _get_judge().grade(
                question=_text(actual, "user_content"),
                answer=_text(actual, "final_response"),
                reference=_text(expected, "final_response"),
            )
        )
        results.append(
            PerInvocationResult(
                actual_invocation=actual,
                expected_invocation=expected,
                score=score,
                eval_status=_status(score),
            )
        )

    if not results:
        return EvaluationResult(overall_eval_status=EvalStatus.NOT_EVALUATED)
    overall_score = sum(result.score or 0.0 for result in results) / len(results)
    return EvaluationResult(
        overall_score=overall_score,
        overall_eval_status=_status(overall_score),
        per_invocation_results=results,
    )
