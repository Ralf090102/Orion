"""
QueryTrace: one structured record per chat turn, keyed by the request's
correlation ID (src/utilities/request_context.py).

This exists because a generic log stream would not have caught the
2026-09-11 "not citing my knowledge base" incident -- that failure lived
entirely in Python retrieval logic (which candidates were fused, which
survived reranking, what actually landed in the prompt), not in anything a
Rust-shell-level or even a plain "RAG triggered" log line would show. A
QueryTrace records exactly those stage-by-stage decisions so a future
diagnosis doesn't need to hand-roll temporary instrumentation again.

Contains retrieved chunk text and prompt content in plaintext by design --
see Eru's Orion-Roadmap.md / README.md's "Logs & Diagnostics" section for
the retention/redaction decision (no redaction: single local user, single
local machine, no cloud sync -- the only reader already owns the source
documents) before changing what this captures.

Deliberately not surfaced in any UI yet (persisted only, via
SessionManager.save_query_trace()) -- a "why this answer" panel is a
separate, later decision.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class QueryTrace:
    request_id: str
    session_id: Optional[str]
    query_text: str
    search_type: str = "hybrid"

    # Pre-fusion, per retriever: [{"retriever": "semantic"|"keyword", "document_id", "score", "rank"}, ...]
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    # Post-fusion: [{"document_id", "score", "search_type"}, ...]
    fused: list[dict[str, Any]] = field(default_factory=list)
    # Cross-encoder survivors: [{"document_id", "score"}, ...]
    reranked: list[dict[str, Any]] = field(default_factory=list)
    # Post-MMR survivors: [{"document_id", "score"}, ...]
    mmr: list[dict[str, Any]] = field(default_factory=list)
    # What actually entered the prompt: [{"source_file", "citation_text", "final_score", "length"}, ...]
    context_chunks: list[dict[str, Any]] = field(default_factory=list)

    model: Optional[str] = None
    timing: Optional[dict[str, float]] = None

    rag_retrieval_triggered: bool = False
    rag_retrieval_failed: bool = False
    rag_retrieval_error: Optional[str] = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)
