# Online RAG evaluation: dataset and metrics

All four configs (`promptfooconfig.*.yaml`) run the same agent on the same
dataset, `golden-dataset.yaml`. Each config adds one metric assertion on top of
the shared checks (single provider span, `search_documents` called, no error
spans, non-empty answer).

## Dataset fields

| Field | Meaning | Sent to the agent |
|---|---|---|
| `query` | The question asked to the agent. | Yes |
| `reference` | The expected answer, written by hand (ground truth). It must be a fact the indexed document really contains. | No |

## Metrics

| Config | Assertion | Compares | Question answered | Threshold |
|---|---|---|---|---|
| `correctness` | `factuality` | answer ↔ `reference` | Is the answer consistent with the expected one? | 1 |
| `relevance` | `answer-relevance` | answer ↔ `query` | Does the answer address the question? | 0.7 |
| `groundedness` | `context-faithfulness` | answer ↔ retrieved chunks | Does the answer stay faithful to what was retrieved? | 1 |
| `retrieval_relevance` | `context-relevance` | chunks ↔ `query` | Are the retrieved chunks useful for the question? | 0.5 |
| `retrieval_relevance` | `context-recall` | chunks ↔ `reference` | Do the retrieved chunks contain the expected answer? | 1 |

The retrieved chunks come from the `search_documents` span in the ADK OTEL trace.
`eval/provider.py` exposes them as `metadata.documents`, and the assertions read
them with `contextTransform: context.metadata.documents`.

## Precision vs recall (retrieval)

- `context-relevance` is a **precision** signal: how much of the retrieved
  context is useful for the `query`. A low score means noise (`top_k` too large,
  chunks too broad).
- `context-recall` is a **recall** signal: how many facts of the `reference`
  appear in the retrieved context. A low score means the information was missed
  (weak embedding match, poor chunking, similarity threshold too strict).

Example with `query` = "How many refuges does the GR20 have?" and
`reference` = "The GR20 has 17 refuges.":

| Retrieved context | Relevance | Recall |
|---|---|---|
| 5 chunks, only one states "17 refuges" | low | high |
| 1 chunk about refuges, without the number | high | low |
| 1 chunk stating "17 refuges" | high | high |

Both are judged by an LLM (`PROMPTFOO_GRADER_MODEL`). Neither checks for a
specific chunk ID; there is no gold chunk in the dataset.
