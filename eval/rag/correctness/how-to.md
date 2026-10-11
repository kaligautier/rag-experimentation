# Evaluate RAG correctness with ADK and Vertex AI

`rag_correctness` is an ADK custom metric that compares the agent's final answer
with a reference answer using Gemini on Vertex AI.

## Contract

- Input: user question, generated answer, reference answer.
- Output: `1.0` if the judge considers the answer correct; otherwise `0.0`.
- Status: passes when the overall score meets `criterion.threshold` from the
  evaluation config (`1.0` by default).
- Actual and expected questions must match and reference answers must be nonempty.
  Empty generated answers score zero; malformed judge output raises an error.
- References must be reviewed and annotated by a human; the judge does not
  generate them during evaluation.

The judge uses the same Vertex variables as the application:
`GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`,
and local ADC. The agent runs with `MODEL`; the independent evaluation judge
uses `EVAL_MODEL` (default: `gemini-3.1-pro-preview`).

## Run the Vertex smoke test

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root \
  eval/rag/correctness/datasets/rag_correctness.test.json \
  --config_file_path eval/rag/correctness/eval_config.json \
  --print_detailed_results
```

The supplied dataset checks end-to-end integration with a question that does not
depend on the corpus. To evaluate RAG, add domain questions with factual
reference answers, then use the same command.

## Evaluate the locally indexed GR20 document

`rag_correctness_gr20.test.json` checks two answers based on the `GR20 Randonnée`
document already indexed in the local database and one out-of-corpus question.
Start PostgreSQL (`docker compose up -d postgres`) and ensure `RAG_DB_URL` points
to that initialized database. The ADK evaluation invokes the tools and database
directly, so the HTTP API does not need to run during evaluation. For a fresh
database, first initialize it and index the fixture using the
[local setup commands](../../../README.md#setup-and-checks). Then run:

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root \
  eval/rag/correctness/datasets/rag_correctness_gr20.test.json \
  --config_file_path eval/rag/correctness/eval_config.json \
  --print_detailed_results
```
