"""Synthetic outputs for testing the Promptfoo wiring, never RAG quality."""


def call_api(prompt, options, context):
    if prompt == "missing-trace":
        return {"error": "Synthetic missing completed search"}
    return {
        "output": {
            "answer": "ANSWER_ONLY " + prompt,
            "documents": [
                {"document_id": "fixture", "chunk_id": "leaf", "content": "DOC_ONLY"}
            ],
            "searches": [{"call_id": "call-1", "query": prompt}],
        }
    }
