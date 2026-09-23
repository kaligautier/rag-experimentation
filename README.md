# RAG experimentation

This branch introduces the RAG layers incrementally. HTTP RAG endpoints and
agent tools are not enabled yet; the existing agent starts without PostgreSQL.

## Setup and checks

```bash
just install
cp .env.example .env
# Configure the GOOGLE_* settings in .env.
just lint
just test test/unit
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

The existing agent still starts without PostgreSQL on this branch. Before using
the repository, run `just migrate`. Its `init_db()` check (wired into the API in
the later API PR) validates the migrated schema and never creates tables.
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
Callers own transaction commit/rollback; HTTP and agent wiring come separately.
