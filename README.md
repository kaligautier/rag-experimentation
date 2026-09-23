# RAG experimentation

Markdown document retrieval experiment using FastAPI, Google ADK,
async SQLAlchemy, PostgreSQL/pgvector, and Gemini `gemini-embedding-001`
embeddings on Vertex AI (3072 dimensions). The conversational agent separately
uses `gemini-3.8-flash` through the global endpoint.

## Local setup

```bash
just install
cp .env.example .env
# Set the ADK template's GOOGLE_* settings in .env.
gcloud auth application-default login
docker compose up -d postgres
just api
```

- API and ADK interface: <http://127.0.0.1:8000>.
- OpenAPI: <http://127.0.0.1:8000/docs>.
- Docker PostgreSQL: `127.0.0.1:15433`, database `rag_db`.
- Use `just api 7778` to select another HTTP port.

The separate PostgreSQL port avoids connecting to a Homebrew installation on
`localhost:5432`. At startup, the API creates the `vector` extension and missing
tables; it closes its pool at shutdown. This local initialization does not migrate
existing vector dimensions. The Liquibase changelog creates the initial schema; do not
apply it after SQLAlchemy has created the same tables.

Embeddings use `GOOGLE_CLOUD_PROJECT`, `RAG_EMBEDDING_LOCATION`
(`europe-west1` in the example), and Google Application Default Credentials.
Documents use `RETRIEVAL_DOCUMENT`; questions use `RETRIEVAL_QUERY`.
Each Gemini request contains one text; `RAG_EMBEDDING_BATCH_SIZE` bounds
concurrent calls. Responses are validated and normalized before use.

API startup checks that the stored vector dimension matches the configuration.
Existing vectors require a separately managed reindexing when dimensions change;
no automatic re-embedding script is included. Document lists and chunk details remain available
on a dimension mismatch. Search and indexing return HTTP 503 before any model
call; ADK tools also return an explicit error.

Errors are typed (`src/app/utils/error.py`) and rendered by shared FastAPI handlers
as `{"error_code", "message", "details"}`: 400 for unsupported or empty documents and
unsafe filenames, 404 for unknown documents, 409 for existing unindexed documents
(retry with `force=true`), 413 for byte/chunk/heading token limits, 502 for invalid
embedding responses, 503 for a schema mismatch, and a generic 500 otherwise. ADK tool
results carry the same `error_code`. These handlers apply to every application route,
including ADK-generated endpoints. Existing HTTP and validation exceptions retain
their handlers; errors after streaming starts cannot replace the response body.

## Indexing and search

```bash
printf '# Local RAG test\n\nThe meeting starts at 09:00.\n' > /tmp/rag-example.md
curl -F 'file=@/tmp/rag-example.md;type=text/markdown' \
  -F 'title=GR20 Randonnée' http://127.0.0.1:8000/rag/index

curl -H 'Content-Type: application/json' \
  -d '{"query":"étapes principales GR20","top_k":5,"embedding_dimensions":768}' \
  http://127.0.0.1:8000/rag/search
```

Chunking follows Markdown sections: a heading and its body form a chunk.
A parent heading with no text provides context for its descendants and does not
become a standalone result. For example, the source sections “Meilleure Période”
and “Équipement Essentiel” become separate chunks with their complete lists.

Sections exceeding `RAG_CHUNK_SIZE` are split into bounded leaf chunks with
`RAG_CHUNK_OVERLAP`, preserving their heading and ancestor path in the text sent
to the model. `RAG_MAX_CHUNKS_PER_DOC` stops indexing before embedding calls if
the document would produce too many nodes.

`metadata.header_path` preserves ancestors, for example
`/GR20 - Grande Randonnée en Corse/Conseils Pratiques/`. This path is included in
the embedding input; `content` retains the source section. Tables and code blocks
remain within their section.

Search uses [pgvector cosine distance](https://github.com/pgvector/pgvector#querying)
`<=>`: `similarity_score = 1 - distance`, ranging from -1 to 1. `top_k` ranges
from 1 to 100; `similarity_threshold=0` disables filtering, while a positive
threshold keeps scores at or above that value. `country_code` filters document
metadata. REST uploads do not yet expose these metadata fields.

Documents are indexed once in the full 3072-dimensional space. Because
`gemini-embedding-001` uses Matryoshka Representation Learning (MRL),
`/rag/search` accepts `embedding_dimensions` from 128 to 3072: the query vector
is generated at that size and compared with the matching prefix of stored
vectors using `subvector`. Omitting the field uses 3072. Dimensions 768, 1536,
and 3072 provide useful quality/cost comparison points; changing the search
dimension does not require reindexing.

An identical upload reuses the document and returns `created=false`. To regenerate
chunks after a chunking change, add `-F 'force=true'` to the indexing command.
The document retains its ID; chunk replacement and document updates commit in
the same transaction. Filenames containing paths are rejected. The API reads at
most `RAG_MAX_UPLOAD_BYTES + 1` bytes and returns HTTP 413 when the limit is
exceeded; the ADK tool enforces the same limit.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | HTTP availability |
| POST | `/rag/index` | Markdown upload; `file`, `title`, optional `force` |
| GET | `/rag/documents` | List with `limit` and `offset` |
| GET | `/rag/documents/{id}` | Document, full content, and chunk metadata |
| POST | `/rag/search` | Cosine search |
| DELETE | `/rag/documents/{id}` | Delete the document and its database chunks |

## Checks

```bash
just lint
just test
just test-integration
```

`just test` collects both unit and integration tests. PostgreSQL tests are skipped
unless `RAG_TEST_DB_URL` is set; Liquibase tests additionally require
`RAG_TEST_LIQUIBASE_CONTAINER`. To run only unit tests, use
`just test test/unit`. `just test-integration` uses `RAG_TEST_DB_URL`, falling back to `RAG_DB_URL`,
and creates a unique schema per test. SQL tables are explicitly qualified with
that schema, which is dropped after the test.

## Current limitations

### Liquibase schema

`src/liquibase/db.changelog-master.yaml` targets the default
`RAG_EMBEDDING_DIMENSIONS=3072`, as does ORM initialization. The creation changeset
in `001-create-rag-tables.yaml` declares `vector(3072)` directly. This experimental
project uses a fresh database and reindexing instead of an upgrade migration for
old vector dimensions. If `001` was already applied, its checksum has changed:
recreate the disposable database and reindex the source documents. Clearing
Liquibase checksums alone does not change an existing vector column. Resetting a
database deletes its indexed documents and vectors; confirm the target and retain
the source files first. MRL search dimensions do not change the storage schema.

To run real Liquibase regressions with Docker available, use a disposable database:

```bash
docker run --rm -d --name rag-pr1-migration-test \
  -e POSTGRES_PASSWORD=review_test -e POSTGRES_DB=rag_review \
  -p 127.0.0.1:25433:5432 pgvector/pgvector:pg16
RAG_TEST_DB_URL=postgresql+asyncpg://postgres:review_test@127.0.0.1:25433/rag_review \
  RAG_TEST_LIQUIBASE_CONTAINER=rag-pr1-migration-test just test-integration
docker stop rag-pr1-migration-test
```

The tests run Liquibase 4.33 in a container and own a unique schema each. Without
`RAG_TEST_LIQUIBASE_CONTAINER`, only these migration tests are skipped. Never target
an application database with the disposable-database commands above.

### Other limitations

- UTF-8 Markdown only; storage uses 3072 dimensions, with MRL search from
  128 to 3072 dimensions (3072 by default).
- Sections remain intact while they fit `RAG_CHUNK_SIZE`; longer sections are
  split with overlap.
- Retrieval quality must be evaluated separately from API functionality.
  Gemini rejects sections exceeding its input limit; text is never silently
  truncated.
- DELETE removes database records; local source files remain in `RAG_STORAGE_PATH`.
- Deduplication covers successive uploads; concurrent uploads of identical
  content may still hit the unique hash constraint.

Reference: [Gemini embeddings on Vertex AI](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings).
