# Orion

Fully local RAG assistant — retrieves from a local knowledge base, generates grounded answers via Ollama, with citations back to source documents.

## Language

**Search results**:
Raw hits returned by `OrionRetriever.query()` — one `SearchResult` per retrieved chunk, before deduplication or citation formatting.
_Avoid_: sources, contexts (those name the later, prepared stage)

**Sources**:
The deduplicated, cleaned contexts actually handed to the LLM as grounding — the output of `ContextPreparer.prepare()`. What the LLM was grounded on and what the client is shown must always be the same set; a candidate that surfaces "sources" from raw search results instead is showing the client something the LLM didn't actually see.
_Avoid_: results, hits, raw retrieval
