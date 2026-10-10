# RAG correctness evaluation

This evaluation asks: **does the agent answer agree with the supplied reference?**

It runs the agent, compares its answer with a human reference, and produces
an ADK score.

```text
Test question
  -> Gemini Flash agent
  -> search_documents
  -> Gemini embedding + PostgreSQL pgvector
  -> document chunks
  -> agent answer
  -> Gemini 3.1 Pro judge
  -> ADK score and status
```

## Model roles

| Role | Default model | Responsibility |
| --- | --- | --- |
| RAG agent | `MODEL=gemini-2.5-flash` | Calls tools and writes the answer. |
| Evaluation judge | `EVAL_MODEL=gemini-3.1-pro-preview` | Compares the actual answer with the reference. |
| Embeddings | `RAG_EMBEDDING_MODEL=gemini-embedding-001` | Converts the search query into a vector. |

The agent and judge are separate: an independent Pro model scores the answer
produced by Flash.

## Files

| File | Purpose |
| --- | --- |
| `correctness.py` | ADK metric and Vertex judge call. |
| `eval_config.json` | Registers `rag_correctness` with ADK. |
| `datasets/rag_correctness.test.json` | Smoke test independent of the corpus. |
| `datasets/rag_correctness_gr20.test.json` | Questions about the locally indexed GR20 document. |
| `RUN.md` | Commands ready to copy and run. |

## Dataset: reference answers

Each case contains a question and a reference answer written or reviewed by a
human. For example, this excerpt retains the original French test data:

```json
{
  "user_content": "Quel est le dénivelé négatif Carrozzu à Ciottulu ?",
  "final_response": "Le dénivelé négatif est de 700 m (-700 m)."
}
```

The reference is not sent to the agent. ADK uses it only after the agent answers,
during judging.

## Case execution

1. ADK sends the question to `root_agent`.
2. Flash may rephrase the search before calling `search_documents`.
3. `search_documents` encodes the query with the embedding model.
4. pgvector finds the nearest chunks in PostgreSQL.
5. Flash receives the chunks and generates its final answer.
6. The Pro judge receives the question, reference, and final answer.
7. The judge returns this JSON:

   ```json
   {"correct": true, "explanation": "..."}
   ```

8. The metric converts `true` to `1.0` and `false` to `0.0`.
9. ADK calculates the mean score and final status.

## Score and threshold

The configured threshold is `1.0`.

```text
score >= 1.0  -> PASSED
score < 1.0   -> FAILED
```

Each question currently receives a binary score. Across multiple questions,
ADK uses the mean: three correct answers and one incorrect answer yield `0.75`,
which is `FAILED` at the current threshold.

The metric reads ADK's `criterion.threshold`, populated from `eval_config.json`.
`CORRECTNESS_THRESHOLD` supplies the `1.0` default when no criterion is supplied.
Thresholds must be between 0 and 1. ADK clears the deprecated `threshold` field
before invoking a custom metric, so that field is not used.

Actual and reference questions must match, with nonempty questions and reference
answers. Malformed pairs fail before any judge call. An empty generated answer
scores zero without calling the judge. Empty or malformed judge output raises an
error rather than silently producing a passing score.

Questions, references, and answers are sent as JSON data. The judge is instructed
to ignore instructions embedded in these fields. This reduces prompt confusion;
it does not establish resistance to all prompt injection attacks.

## Questions outside the corpus

This metric checks agreement with the human reference, rather than topic
relevance. For a question unrelated to the indexed corpus, a reference such as
“I have no information about this topic” can pass if the agent gives the same
answer and the Vertex judge returns `correct=true`.

## Commands

Vertex smoke test:

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root \
  eval/rag/correctness/datasets/rag_correctness.test.json \
  --config_file_path eval/rag/correctness/eval_config.json \
  --print_detailed_results
```

RAG evaluation against the already indexed GR20 document:

```bash
PYTHONPATH=src:. uv run adk eval src/app/components/agents/root \
  eval/rag/correctness/datasets/rag_correctness_gr20.test.json \
  --config_file_path eval/rag/correctness/eval_config.json \
  --print_detailed_results
```

## Recorded evaluation evidence

The previously recorded local GR20 runs showed the agent calling
`search_documents`, pgvector returning chunks, and the Pro judge considering
the final answers correct. These are historical results, not a new validation.

A new live retrieval gate must separately inspect the ADK invocation events:
require a `search_documents` call, a successful tool response containing relevant
fixture chunks for each in-corpus case, a final answer, and the judge result.
A green correctness score alone can also result from an answer without retrieval.
The arithmetic dataset is only a Vertex integration smoke test; the three GR20
cases are a narrow corpus smoke test, not evidence of general RAG quality.

## Limitations and future metrics

`rag_correctness` scores the final answer. It does not automatically guarantee:

- A `search_documents` call for every case.
- The best chunk ranking first.
- Every claim being strictly supported by a chunk.
- Accurate citations.
- Results generalizing to a real domain corpus.

Additional metrics to consider:

- **Retrieval recall**: is the expected chunk in the top-k?
- **Tool use**: was `search_documents` called when required?
- **Groundedness**: is each claim supported by the context?
- **Latency and cost**: does the system remain practical at scale?

Each live case typically involves Flash calls, an embedding call, and a Pro call.
Useful, versioned, reviewed datasets help control costs and support reliable
comparisons.
