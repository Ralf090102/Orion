"""
Per-request correlation ID, shared across backend/*.py and src/*.py via a
contextvar rather than threading an explicit parameter through every
logging call site.

One ID is minted per chat request (see backend/api/chat.py and
backend/websockets/chat.py) and bound here; backend/logging_setup.py's
logging.Filter reads it back so every log line for that request -- no
matter which module emits it -- is tagged with the same value, and Rust's
orion-shell.log and Python's orion-backend.log can be grepped together by
this same ID for the same chat turn.

This is the "blanket log-line tagging" half of the correlation-ID feature.
The other half -- an explicit request_id/trace parameter threaded through
generate_chat_response() -> OrionRetriever.query() -> HybridSearcher.search()
-- exists separately, for building the structured per-query retrieval trace
(see src/generation/trace.py). contextvars is the right tool for "tag every
log line" (implicit, ambient); an explicit parameter is the right tool for
"assemble one structured record" (needs a concrete object to write into).
"""

import contextvars
import uuid

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)


def new_request_id() -> str:
    """Mint a correlation ID and bind it to the current context so every
    log line on this call stack -- including work offloaded via
    asyncio.to_thread(), which copies the current contextvars.Context into
    the executor thread -- is tagged with it. Each request is its own
    asyncio Task with an isolated context, so concurrent requests never
    bleed into each other's ID."""
    rid = str(uuid.uuid4())
    request_id_var.set(rid)
    return rid
