# RAG implementation

This project indexes Markdown documents and retrieves relevant passages for
questions. The HTTP API and Google ADK tools share the same services, with
PostgreSQL and pgvector storing the document chunks and embeddings.

## Indexing

1. A document is supplied through the HTTP API or the ADK `index_document` tool.
2. The local storage adapter validates the filename and stores the source file.
3. The Markdown adapter splits the document into linked parent and child chunks.
4. Vertex AI creates embeddings for the chunks. The default model is
   `gemini-embedding-001`, with 3072 dimensions.
5. The indexing service persists document metadata, chunks and vectors in
   PostgreSQL. Re-indexing identical content reuses the existing document;
   forced indexing keeps its ID while regenerating its chunks.

The storage and database operations are coordinated with compensation for
failures, but they are not a distributed transaction. See the
[README](../README.md#rag-services) for the associated limitations.

## Retrieval

The HTTP `/rag/search` endpoint and the ADK `search_documents` tool use the
interrogation service. It embeds the query, searches the stored vectors by
cosine similarity, and merges linked parent context for the results. Search can
use fewer embedding dimensions than those stored; the requested dimension must
fit the configured vector width.

## Boundaries and operation

- The current source-file storage backend is local; GCS is not implemented.
- Liquibase manages the PostgreSQL schema. Run `just migrate` before the API.
- Markdown chunk sizes use the LlamaIndex tiktoken tokenizer and do not guarantee
  that an embedding input fits Gemini's token limit.
- The [Postman collection](postman/RAG-GR20.postman_collection.json) illustrates
  indexing, browsing, searching, forced reindexing and deletion.
- The existing [answer-quality evaluation](../eval/rag/correctness/OVERVIEW.md)
  compares generated answers against references; a passing score alone does not
  establish that retrieval occurred.

See the [README](../README.md) for setup, API routes, configuration and tests.
