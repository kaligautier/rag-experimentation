# RAG experimentation

Local Markdown retrieval using FastAPI, Google ADK, PostgreSQL/pgvector and
Vertex AI embeddings. HTTP endpoints and agent tools share the same RAG services.

## Setup and checks

```bash
just install
cp .env.example .env
# Configure the GOOGLE_* settings in .env.
just lint
just test test/unit
gcloud auth application-default login
docker compose up -d postgres
just migrate
just api 8000
```

The existing API listens on http://127.0.0.1:8000. RAG models, ports, typed errors
and settings are defined in `src/app/services/rag/models.py`, `src/app/ports/rag`,
`src/app/utils/error.py` and `src/app/config/settings.py`. Dependencies and the
lockfile are updated together. The optional local PostgreSQL container uses
127.0.0.1:15433 (`docker compose up -d postgres`).

## Document adapters

Storage validates filenames and atomically publishes files below a local root;
the GCS backend is not implemented yet. UTF-8 Markdown (BOM tolerated)
sections form linked parent/child chunks with bounded leaf token counts. Vertex
embeddings use `gemini-embedding-001` at 3072 dimensions by default. Vertex accepts one text
per request for this model; bounded concurrent calls preserve input order and
reject truncated or invalid vectors. Unit tests mock cloud calls.

Chunk sizes use LlamaIndex's tiktoken tokenizer, not Gemini's tokenizer, and
exclude the ancestor heading path added to embedding input. They do not guarantee
that every input fits Gemini's token limit; automatic truncation is disabled.
Headings deeper than six levels are accepted to match LlamaIndex's parsing
behavior, even though they are outside standard Markdown heading levels.

## Persistence

The SQLAlchemy repository supports documents, chunks and cosine prefix search.
Reads never load stored embeddings.

Liquibase owns the schema. `just migrate` provisions
`vector(RAG_EMBEDDING_DIMENSIONS)` (3072 by default) and waits for completion;
`docker compose up -d` applies migrations in the background. Configure the same
dimension in `.env` for migration and application. For standalone Liquibase, use
`update -Drag.embedding.dimensions=768`, for example. Keep that value on subsequent
runs: changing an applied migration parameter is not a data migration. Existing
vectors with a different width cause migration to fail without modifying them.
To change embedding width, provision a separate database and reindex retained
source files; do not truncate vectors or clear migration checksums.

Run `just migrate` before starting the API. Startup validates the migrated
schema and never creates tables. A missing schema blocks startup; mismatched
vector dimensions block embedding operations while stored documents remain browsable.
Databases created by the ORM have no Liquibase history: use a fresh database and
reindex retained source files. `docker compose down -v` deletes **all local
database data** and is only suitable for a disposable database.

Integration tests require `RAG_TEST_DB_URL` and build the ORM tables in isolated
schemas. Liquibase tests additionally require `RAG_TEST_LIQUIBASE_CONTAINER`,
the PostgreSQL container name, and check the ORM tables match the migrated schema.

```bash
RAG_TEST_DB_URL=postgresql+asyncpg://rag_user:secret@127.0.0.1:15433/rag_db RAG_TEST_LIQUIBASE_CONTAINER=rag_postgres just test-integration
```

## RAG services

`IndexationService` extracts, chunks, embeds and persists documents. Identical
indexed content is reused; forced indexing retains the document ID.
`InterrogationService` searches vectors and merges linked parent context.
HTTP and ADK callers commit before reporting success and close their sessions
on completion, errors or cancellation.

Use `async with service.indexing_document(...) as result:` or
`async with service.deleting_document(document_id):` and commit the database
transaction inside the context. These contexts compensate local source-file
changes if the operation or commit raises, including cancellation. Forced indexing
retains the stored source key and provenance. The lower-level `index_document()`
only compensates errors before it returns. Filesystem and PostgreSQL writes are
not a distributed transaction: process crashes, ambiguous commit outcomes and
failed compensation require reconciliation from retained sources and DB state.

## HTTP API and ADK tools

The local API and ADK interface are at <http://127.0.0.1:8000>; OpenAPI is at
<http://127.0.0.1:8000/docs>. Use `just api 7778` to select another port.

```bash
curl -F 'file=@docs/fixtures/gr20.md;type=text/markdown' \
  -F 'title=GR20 Randonnée' http://127.0.0.1:8000/rag/index
curl -H 'Content-Type: application/json' \
  -d '{"query":"étapes principales GR20","embedding_dimensions":768}' \
  http://127.0.0.1:8000/rag/search
```

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | HTTP availability |
| POST | `/rag/index` | Markdown upload: `file`, `title`, optional `force` |
| GET | `/rag/documents` | List with `limit` and `offset` |
| GET | `/rag/documents/{id}` | Document content and chunk metadata |
| POST | `/rag/search` | Cosine search, optional dimension/country filters |
| DELETE | `/rag/documents/{id}` | Delete document, chunks and stored source |

Identical indexed content returns `created=false`; `force=true` regenerates its
chunks while retaining the document ID and source provenance. Search dimensions
range from 128 up to the configured stored dimension; omitting the field uses
the configured dimension. `top_k` is 1–100; a zero similarity threshold disables
filtering. HTTP uploads do not expose custom country metadata.

The ADK agent exposes `index_document` (Markdown text and a title ending in
`.md` or `.markdown`) and `search_documents`. Unsafe filenames are rejected.
The file limit is `RAG_MAX_UPLOAD_BYTES` (5 MiB by default). Before multipart
parsing, `/rag/index` buffers at most that limit plus 64 KiB for multipart fields
and headers; it rejects larger bodies even without `Content-Length`. Other
routes, including ADK SSE, do not use this upload-specific buffer.
Errors return stable `error_code`, `message` and `details` fields: 400 for invalid
documents, 404 for unknown IDs, 409 for unindexed duplicates, 413 for document
limits, 502 for invalid embedding responses, 503 for incompatible vector schema
and a generic 500 for unexpected errors. ADK tools return the same error codes.
Existing framework validation errors retain their normal response.

## Continuous integration

Every PR and push to `main` runs Ruff, the full unit and real PostgreSQL/Liquibase
suite (skipped tests fail CI), Python package and Docker builds, and a dependency
audit. The database is disposable and uses a generated password. Tests do not
need cloud credentials; live embedding and agent evaluation remain separate
checks. Reviewed dependency exceptions are recorded in
[`docs/dependency-security.md`](docs/dependency-security.md).

## Evaluation and examples

The [Postman collection](docs/postman/RAG-GR20.postman_collection.json) contains
10 requests covering the six HTTP operations, including four searches at
variable, 768, 1536 and 3072 dimensions and forced reindexing. These examples
assume the default 3072-dimensional database. Import its
[environment](docs/postman/RAG-GR20.postman_environment.json).

On a fresh migrated database, upload `docs/fixtures/gr20.md` using the command
above or the collection's indexing request before browsing its chunks or
searching. Successful indexing updates `document_id`; its prefilled value refers
to a previous local run. DELETE uses the separate `delete_document_id` variable,
which must be supplied explicitly. Search assertions require nonempty results.

For answer-quality evaluation, see the [evaluation overview](eval/rag/correctness/OVERVIEW.md)
and [run instructions](eval/rag/correctness/RUN.md). A passing reference-answer
score does not itself prove retrieval occurred or establish general RAG quality.

### Promptfoo RAG evaluation

Four independent configurations in `eval/online/` apply the native assertions from
the [Promptfoo RAG guide](https://www.promptfoo.dev/docs/guides/evaluate-rag/).
Each loads the same two GR20 questions from `cases.yaml` and checks that the answer
is nonblank alongside its RAG metric. No custom grading prompts or scorers
are used.

| Configuration | Native assertion | Initial threshold | Command |
| --- | --- | --- | --- |
| `promptfooconfig.correctness.yaml` | `factuality` | 1 | `just eval-correctness` |
| `promptfooconfig.relevance.yaml` | `answer-relevance` | 0.7 | `just eval-relevance` |
| `promptfooconfig.groundedness.yaml` | `context-faithfulness` | 1 | `just eval-groundedness` |
| `promptfooconfig.retrieval_relevance.yaml` | `context-relevance` | 0.3 | `just eval-retrieval-relevance` |

These are starting thresholds, not calibrated quality gates. Expand the two-case
dataset and review judge explanations before using the scores to block a release.
The [LangSmith tutorial](https://docs.langchain.com/langsmith/evaluate-rag-tutorial)
inspired the four axes, but its binary rubrics are no longer the scoring method:

- [Factuality](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/factuality/)
  compares the answer with the reference. Only this assertion uses `reference`.
- [Answer relevance](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/answer-relevance/)
  generates three candidate questions from the answer and compares them with the
  input using embeddings. It needs both a text judge and an embedding provider.
- [Context faithfulness](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/context-faithfulness/)
  extracts factual claims and checks their support in the actual retrieved passages.
  The threshold of 1 requires all extracted claims to be supported.
- [Context relevance](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/context-relevance/)
  measures the fraction of context minimally needed to answer the question. It is
  stricter than the old any-related-passage rubric; supporting detail can lower the
  score even when the answer is present. It is not retrieval recall.

The single [`eval/provider.py`](eval/provider.py) adapter follows the
[direct Google ADK integration](https://www.promptfoo.dev/docs/guides/evaluate-google-adk/):
it imports the application's `root_agent`, runs one turn with `Runner` and a fresh
in-memory session, and returns the final answer as plain `output`. Actual
`search_documents` passages and search calls are extracted from native ADK tool
spans into `metadata`, alongside the event count;
token usage is reported separately. It does not grade answers or use the reference.
The two context assertions use the documented extraction hook:
`contextTransform: context.metadata.documents`. This inline JavaScript expression
only selects the passages; there is no JavaScript file or custom Python grader.
No second retrieval is performed for grading. Incomplete search evidence and
execution failures become provider errors. A completed empty search is preserved
as `documents: []`; Promptfoo 0.125.1 rejects empty context for its context assertions,
so those rows cannot pass even if the agent abstains. Read the error before treating
it as a semantic failure. Faithfulness can also fail to grade an answer with no
extractable factual claims; the previous rubric's automatic abstention pass is gone.
The database pool is closed after each invocation to avoid reusing connections
across the event loops created by the Python worker.

Tracing follows the direct ADK provider pattern: the provider attaches Promptfoo's
W3C `traceparent`, creates `promptfoo_google_adk_provider`, and exports the native
ADK workflow, agent, model and tool spans using the OTLP HTTP exporter. The tracer
provider is initialized once per Python worker and flushed after each invocation.
Each YAML enables Promptfoo's local OTLP receiver on port **4318** (keep it free).
No external collector is required. `just eval-view` exposes the stored trace timeline.
Each row also checks that exactly one provider span exists, `search_documents` ran,
and no span reports an error. These checks accompany the four RAG metrics.

The installed ADK 2.6.1 emits tool inputs/results under the legacy
`gcp.vertex.agent.tool_call_args` / `gcp.vertex.agent.tool_response` attributes;
the adapter also understands the v2 `gen_ai.tool.call.arguments` / `.result` names.
A second SDK exporter retains ended spans in memory for the current invocation,
so the context assertions use the same retrieved text as the exported trace.
Only `search_documents` spans matching the current trace ID are accepted; the
buffer is cleared between invocations. Keep evaluation concurrency at **1**.
Missing payloads or a missing search span fail the provider, with no event fallback.

The CLI sets `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` and
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=true` to capture tool arguments and results.
Traces therefore contain questions, model/tool content and retrieved documents;
they stay in the ignored local Promptfoo database unless you explicitly export them.
`PROMPTFOO_ENABLE_OTEL=false` disables extra Python-worker instrumentation, while
the provider's native ADK exporter remains active. The OTLP HTTP exporter is a dev
dependency, pinned to the existing OpenTelemetry SDK version 1.42.1.

Install dependencies with `just install` and Node >=22.22.0 (Promptfoo is pinned to
0.125.1). Configure `.env`, Google ADC, and `PROMPTFOO_GRADER_MODEL` (default
`gemini-2.5-pro`, independent of the application's `MODEL`). For answer relevance,
`PROMPTFOO_EMBEDDING_MODEL` defaults to `gemini-embedding-001` and
`PROMPTFOO_EMBEDDING_REGION` to `europe-west1`; the embedding task is
`SEMANTIC_SIMILARITY`. This provider is independent of the corpus index embeddings.
Start PostgreSQL, run
`just migrate`, and index `docs/fixtures/gr20.md` with title **GR20 Randonnée** using
the API or the existing Postman collection. The evaluation itself runs ADK directly
and does not need the HTTP API to remain running.

```sh
just eval-check              # Validate the four YAML configurations, no model calls
just test test/unit -q        # Application and adapter tests, no model calls
just eval-correctness        # One live axis (billable)
just eval-rag                # All four live axes, sequentially (billable)
just eval-view               # Local results UI on port 15500
```

Each axis runs the agent independently: all four together invoke it four times per
question, so compare per-run evidence rather than assuming an identical answer.
All suites run even if an earlier one fails; the command keeps a nonzero exit status.
Reports are `eval/.promptfoo/results/<metric>.json`. Inspect `results.stats` to
separate assertion failures from technical errors, and inspect each named RAG metric
rather than the aggregate score that also includes tracing and nonblank-answer checks.
The answer is in `response.output`; evidence is in `response.metadata.documents`
and `response.metadata.searches`. A selected run clears only its own
old report before starting. Reports and the local database are ignored by Git and
Docker; sharing and CLI telemetry are disabled. Live judge inputs still reach the
configured model provider. CI does not run model evaluations.
