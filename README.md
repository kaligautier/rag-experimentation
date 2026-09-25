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
embeddings use `gemini-embedding-001` at 3072 dimensions. Vertex accepts one text
per request for this model; bounded concurrent calls preserve input order and
reject truncated or invalid vectors. Unit tests mock cloud calls.

Chunk sizes use LlamaIndex's tiktoken tokenizer, not Gemini's tokenizer, and
exclude the ancestor heading path added to embedding input. They do not guarantee
that every input fits Gemini's token limit; automatic truncation is disabled.
Headings deeper than six levels are accepted to match LlamaIndex's parsing
behavior, even though they are outside standard Markdown heading levels.
