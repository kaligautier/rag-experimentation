# Run the ADK / Vertex evaluation

From the repository root, copy this command **as a single line**:

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root eval/rag/correctness/datasets/rag_correctness.test.json --config_file_path eval/rag/correctness/eval_config.json --print_detailed_results
```

Equivalent multiline version: each `\` must be the final character on its line.

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root \
  eval/rag/correctness/datasets/rag_correctness.test.json \
  --config_file_path eval/rag/correctness/eval_config.json \
  --print_detailed_results
```

Evaluate RAG questions against the GR20 document already indexed locally:

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root eval/rag/correctness/datasets/rag_correctness_gr20.test.json --config_file_path eval/rag/correctness/eval_config.json --print_detailed_results
```

Vertex evaluation prerequisites: `.env` must define
`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION`. `MODEL` and `EVAL_MODEL` can override the defaults
`gemini-3.8-flash` and `gemini-3.1-pro-preview`. Application Default Credentials
must be valid (`gcloud auth application-default login`).

For the GR20 dataset, PostgreSQL must be running, `RAG_DB_URL` must select the
initialized database, and the GR20 fixture must already be indexed. See the
[setup and indexing instructions](../../../README.md#local-setup). Evaluation
calls the ADK tools directly; the HTTP API is only needed for the initial upload,
not for the evaluation itself.
