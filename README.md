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

## Dependencies and deployment credentials

Keep `uv.lock` in version control alongside `pyproject.toml` to reproduce exact
resolved dependency versions, as recommended by the
[uv documentation](https://docs.astral.sh/uv/concepts/projects/layout/#the-lockfile).
The generated `.venv` directory stays ignored.

The Compose database credentials are for disposable local development only.
Deployments must retrieve credentials from Secret Manager or Vault and inject
them into the application's `RAG_DB_URL` at runtime; do not ship the local password
or commit deployment secrets. This repository does not provision that integration.
