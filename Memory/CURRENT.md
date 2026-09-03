---
name: current-session-context
description: Rolling "where we left off" checkpoint; check first each session to resume instantly
metadata:
  type: project
---

## Last active: 2026-09-01

## What we just did
- **Backend startup latency fixed (~5min → seconds).** Root cause: `backend/app.py` eagerly imported the entire ML stack (torch, sentence-transformers, chromadb, langchain) at module load, before Uvicorn could bind the port — not eager model loading (already lazy), despite a stale comment in `backend.rs` claiming otherwise. Fixed by making those imports lazy/function-local across `src/retrieval/*.py` and `src/core/ingest.py`, and disabling Uvicorn's dev-only `reload=True` in packaged builds. Added a background retriever warm-up thread (`warm_up_retriever_background()` in `backend/dependencies.py`) right after, so the app stays instantly interactive while the ML stack loads off the critical path instead of either blocking launch or making the first query pay the full cost.
- **Two more Roadmap bugs fixed.** (1) Force-killing `app.exe` orphaned its Python backend child — fixed with a Windows Job Object (`win32job` crate, `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) tying the child's lifetime to the parent at the OS level, verified with a standalone repro. (2) `MMRSearcher.search()` crashed on a raw `numpy.ndarray` embedding (`if embedding:` ambiguous truth-value error), silently skipping MMR diversification rather than falling back to keyword search as previously assumed — fixed at the source (normalize to `list` in `SemanticSearcher.search()`) and the crash site, with a new regression test.
- **v0.1.2 built, installed, and confirmed working in the real app** — version bumped across all 4 declaration points, `npx tauri build` succeeded, and after a real install snag (an *old* orphaned v0.1.1 backend process file-locking the installer, then a retry landing at a different default install path than expected) the app is confirmed genuinely fast and correct via a real RAG round-trip with citations. Scoped as **local rebuild/reinstall only** — no git tag, no signed installer, no public GitHub release yet.
- All of today's code changes are committed and pushed to `main` (`cb76e74`, `6bae3fd`, `1d836b3`). Full narrative, root-cause evidence, and the install saga: see Eru's [[01-Projects/Orion/Sessions/4-Backend-Startup-Fix-2026-09-01-Recap]] and [[01-Projects/Orion/Orion-Roadmap]].
- Designed (not started) a separate, related project: **EruVoice** — a voice assistant for Eru (STT → `claude -p` scoped to the vault with the `eru` MCP tool → TTS via Kokoro-82M). Not part of Orion's own codebase; captured in Eru's [[01-Projects/EruVoice/EruVoice-Overview]] for a future session.

## What's next
- Two new open items added to the Roadmap this session, not yet investigated: (1) audit whether Orion's RAG answers are actually grounded in the ingested knowledge base vs. the LLM's own general knowledge — distinct from the citation-*metadata* wiring already fixed in the `polishing` branch; (2) explore `/` slash commands in the chat UI (e.g. `/kb_list`) — not yet scoped (frontend vs. backend parsing, initial command set).
- If/when ready for a full public v0.1.2 release: tag, signed installer (user runs the signing step themselves, same pattern as v0.1.1), GitHub Release, auto-updater manifest verification.
- See [[01-Projects/Orion/Orion-Roadmap]]'s `## Open` section for the full current checklist.
