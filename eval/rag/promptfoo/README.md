# Local RAG evaluation with Promptfoo

This scaffold implements the four comparisons in the
[LangSmith RAG tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
using the existing ADK agent and four Promptfoo `llm-rubric` assertions.
It does not require LangChain or LangSmith credentials.

| Named metric | Judge input | Binary pass criterion |
| --- | --- | --- |
| `correctness` | Question, answer, human reference | Factually consistent with the reference, without contradictions |
| `relevance` | Question, answer | Addresses the question helpfully |
| `groundedness` | Answer, actual retrieved passages | Factual claims are supported by the passages |
| `retrieval_relevance` | Question, actual retrieved passages | Some keyword or semantic relevance, following the tutorial's permissive rubric |

The rubrics are paraphrased, not the tutorial's exact prompts or model settings.
Each judge returns 0 or 1 with a short justification; all four assertions have
threshold 1. The retrieval criterion is neither precision@k nor recall@k.
An empty context scores zero for retrieval relevance. A factual abstention can
pass groundedness. If you add an out-of-corpus case, expect retrieval relevance
to fail even when abstention is correct: read the four scores separately, not
only the overall PASS/FAIL. Keep answerable and unanswerable cohorts distinct.

## Quick offline check

From the checkout root, with Python/uv/just and **Node >=22.22.0** installed
(Node 24 LTS recommended):

```sh
just install
just eval-check
just eval-smoke
```

The CLI version is pinned to `0.125.1` in `justfile`; `npx` downloads it on first
use. Offline means no model calls, credentials, database or running API are
needed after dependencies are installed. The smoke run uses synthetic Python
outputs and a deterministic judge. It verifies all four scores, isolates the
reference from the other judges, includes one deliberately failing axis and a
provider error. The wrapper checks those expected outcomes and exits zero.
**These synthetic scores are not evidence of RAG quality.**

## Run against your local RAG

1. Copy `.env.example` to `.env` and configure your own Vertex project/location,
   agent model, embeddings and local database. Set `RAG_EVAL_JUDGE_MODEL` to an
   available Vertex model different from `MODEL`. Authenticate with
   `gcloud auth application-default login`.
2. Follow the root README to start PostgreSQL, run `just migrate`, then index
   `docs/fixtures/gr20.md` under the title **GR20 Randonnée** using the existing
   upload command or Postman collection. Use a dedicated local corpus for
   reproducible comparisons. This scaffold does not reset or populate a DB.
3. Run:

```sh
just eval-rag
just eval-view
```

No API server is needed during evaluation: the Python provider invokes the
actual `root_agent` in process, with a fresh in-memory session for each question.
It uses the configured database and the normal tools. Only the question is sent
to the agent; the reference remains in Promptfoo. The collector matches actual
`search_documents` calls and responses and captures their complete passages.
It never performs a second search to manufacture grading context.
Missing/failed searches or missing final answers produce provider errors.

The configuration runs questions serially, disables result caching, and limits
each agent invocation to 180 seconds and 10 LLM calls. The judge is selected
explicitly through the Vertex provider; it is not chosen from available API keys.
Live runs incur normal Vertex charges. No live run is triggered in CI.

## Inspect and extend

- `.promptfoo/rag.json`: local answers, passages, four grades and explanations.
- `.promptfoo/smoke.json`: synthetic contract-test report.
- `just eval-view`: local viewer at the address printed by Promptfoo, bound to
  `127.0.0.1`; select the **live** run rather than the offline smoke run.
- `cases.json`: two answerable GR20 examples to extend with reviewed references.
- `config.cjs`: four rubrics, field selection and thresholds. Each assertion
  receives only its declared inputs, not the entire agent response.

Results and the local Promptfoo database are ignored by Git and Docker. They
contain raw questions, answers and passages: keep them local. Sharing and CLI
telemetry are disabled by the supplied commands. Model inputs still go to your
configured Vertex project. Do not attach raw reports to a public PR.

Run `just test test/unit` and `just lint` for Python checks, and `just eval-smoke`
after changing the Promptfoo configuration. Two fixture questions demonstrate
integration only; expand the dataset before treating this as a quality gate.

This standalone scaffold reuses the capture approach from the local
`feat/rag-quality-evaluation` experiment without importing that unmerged suite.

References: [Python providers](https://www.promptfoo.dev/docs/providers/python/),
[LLM rubrics](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/llm-rubric/),
[assertion transforms](https://www.promptfoo.dev/docs/configuration/expected-outputs/).
