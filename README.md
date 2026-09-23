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

Storage validates filenames and supports local files or GCS. UTF-8 Markdown
sections form linked parent/child chunks with bounded leaf token counts. Vertex
embeddings use `gemini-embedding-001` at 3072 dimensions; batched calls preserve
input order and reject truncated or invalid vectors. Unit tests mock cloud calls.

## Persistence

The SQLAlchemy repository supports documents, chunks and cosine prefix search.
Liquibase `001` creates `vector(3072)` in a fresh database. Do not apply it over
tables already created by the ORM. Existing incompatible dimensions require
recreating a disposable database and reindexing retained source files.

Integration tests require `RAG_TEST_DB_URL`; Liquibase additionally requires
`RAG_TEST_LIQUIBASE_CONTAINER`. Tests own isolated schemas.

```bash
RAG_TEST_DB_URL=postgresql+asyncpg://rag_user:secret@127.0.0.1:15433/rag_db just test-integration
```

## Dependencies and deployment credentials

Keep `uv.lock` in version control alongside `pyproject.toml` to reproduce exact
resolved dependency versions, as recommended by the
[uv documentation](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile).
The generated `.venv` directory stays ignored.

The Compose database credentials are for disposable local development only.
Deployments must retrieve credentials from Secret Manager or Vault and inject
them into the application's `RAG_DB_URL` at runtime; do not ship the local password
or commit deployment secrets. This repository does not provision that integration.
