# RAG evaluation with Promptfoo

[README](README.md) · [Contributing](CONTRIBUTING.md)

RAG evaluation checks both retrieved documents and the answer generated from
them. The [LangSmith RAG tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
describes the general workflow: create a dataset of questions and expected answers,
run the application, and evaluate answer and retrieval quality.

The [Promptfoo RAG guide](https://www.promptfoo.dev/docs/guides/evaluate-rag/)
applies this separation through metrics for the answer and the retrieved context.
The project's [Promptfoo evaluation PR](https://github.com/kaligautier/rag-experimentation/pull/1)
proposes four independent suites:

| Dimension | Promptfoo assertion | What it evaluates |
| --- | --- | --- |
| Correctness | `factuality` | The answer against a reference answer. |
| Answer relevance | `answer-relevance` | How directly the answer addresses the question. |
| Groundedness | `context-faithfulness` | Whether the answer is supported by the retrieved passages. |
| Retrieval relevance | `context-relevance` | How much retrieved context is needed to answer the question. |

These dimensions answer different questions: a relevant answer can still be
unsupported or incorrect. Context relevance is not a measure of retrieval recall.
The proposed dataset contains two GR20 questions, so results and thresholds need
calibration on a broader dataset before drawing general conclusions. LangSmith is
a conceptual reference here; the proposed implementation uses Promptfoo.

## Google ADK provider

The provider proposed in the PR runs the Google ADK agent directly and returns
its answer alongside passages extracted from the agent's `search_documents` trace.
The context assertions grade passages retrieved during that invocation, and the
provider exports ADK traces to Promptfoo's local OTLP collector.

It adapts Promptfoo's [Google ADK provider example](https://github.com/promptfoo/promptfoo/blob/main/examples/integration-google-adk/provider.py)
to this project's retrieval evidence and evaluation suites. The upstream example
is a reference, not the provider executed by the proposed implementation.

The Promptfoo integration is still under review and is not yet part of `main`.
Its PR describes setup, commands and validation. Live runs call models and can
incur costs; CI does not run them. The existing reference-answer evaluation is
documented in its [overview](eval/rag/correctness/OVERVIEW.md).
