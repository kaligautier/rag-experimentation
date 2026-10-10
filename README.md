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
printf '# Local RAG test\n\nThe meeting starts at 09:00.\n' > /tmp/rag-example.md
curl -F 'file=@/tmp/rag-example.md;type=text/markdown' \
  -F 'title=Meeting' http://127.0.0.1:8000/rag/index
curl -H 'Content-Type: application/json' \
  -d '{"query":"When does the meeting start?","embedding_dimensions":768}' \
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
