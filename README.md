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
just api 7777
```

The existing API listens on http://127.0.0.1:7777. RAG models, ports, typed errors
and settings are defined in `src/app/services/rag/models.py`, `src/app/ports/rag`,
`src/app/utils/error.py` and `src/app/config/settings.py`. Dependencies and the
lockfile are updated together. The optional local PostgreSQL container uses
127.0.0.1:15433 (`docker compose up -d postgres`).
