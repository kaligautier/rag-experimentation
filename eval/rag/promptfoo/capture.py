"""Capture actual ADK search responses, adapted from the quality experiment."""


def capture(events) -> dict:
    """Reject incomplete tool evidence instead of judging a second retrieval."""
    calls, searches, documents = {}, [], []
    answer = ""
    for event in events:
        if event.error_code:
            raise ValueError("Agent returned an error event")
        if event.partial:
            continue
        for call in event.get_function_calls():
            if call.name != "search_documents":
                continue
            query = (call.args or {}).get("query")
            if not call.id or call.id in calls or not isinstance(query, str):
                raise ValueError("Invalid search call")
            if not query.strip():
                raise ValueError("Empty search query")
            calls[call.id] = {"query": query, "answered": False}
        for response in event.get_function_responses():
            if response.name != "search_documents":
                continue
            call = calls.get(response.id)
            if call is None or call["answered"]:
                raise ValueError("Unmatched search response")
            data = response.response
            if not isinstance(data, dict) or data.get("status") != "success":
                raise ValueError("Search failed")
            results = data.get("results")
            count = data.get("result_count")
            if (
                not isinstance(results, list)
                or type(count) is not int
                or count != len(results)
            ):
                raise ValueError("Incomplete search results")
            for hit in results:
                if not isinstance(hit, dict) or any(
                    not isinstance(hit.get(key), str) or not hit[key].strip()
                    for key in ("document_id", "chunk_id", "content")
                ):
                    raise ValueError("Invalid passage")
            documents.extend(results)
            searches.append({"call_id": response.id, "query": call["query"]})
            call["answered"] = True
        if event.is_final_response() and event.content:
            answer = "\n".join(
                part.text
                for part in event.content.parts or []
                if part.text and not part.thought
            ).strip()
    if not calls or not all(call["answered"] for call in calls.values()):
        raise ValueError("Missing completed search")
    if not answer:
        raise ValueError("Missing final answer")
    return {"answer": answer, "documents": documents, "searches": searches}
