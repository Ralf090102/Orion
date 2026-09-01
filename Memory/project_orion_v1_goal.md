---
name: project-orion-v1-goal
description: Orion's v1 goal is dual-purpose — ship a working local RAG assistant AND use the build process as an AI-engineering teaching vehicle for the user
metadata:
  type: project
---

Orion v1 has two simultaneous goals, not one: (1) ship a genuinely usable, demoable local RAG assistant, and (2) use the process of building it to teach the user AI engineering concepts as we go.

**Status as of 2026-07-30:** v1 shipped 2026-07-29 (Tauri desktop app merged to `main`), but that session was reactive debugging, not deliberate teaching — goal (2) hadn't really happened yet. 2026-07-30 picked this up explicitly: worked the roadmap's known-debt items (empty `test/`, stub `/api/metrics`, open CORS) one at a time as a teaching exercise on a `hardening` branch, narrating the reasoning behind each (test pyramid layering, middleware-based metrics collection, CORS allowlisting for a desktop app). See Eru's `01-Projects/Orion/Overview.md` for the technical specifics of what shipped.

**Why:** User resumed the project 2026-07-29 after ~5 months stale (last commit 2026-03-03). They explicitly want the collaboration itself to be a teaching context, not just a delivery pipeline — decisions and tradeoffs should be explained, not just executed silently.

**How to apply:** When working on Orion, don't just implement — surface the *why* behind architectural/library choices as part of the normal workflow (concise, not lecture-mode, matching the terse-response preference). With v1 shipped and the known-debt roadmap items being worked as of 2026-07-30, treat future Orion sessions as general hardening/polish work rather than gated on a single blocker. Checkpoint progress here in `Memory/` (this project's own, self-contained) and in Eru's `01-Projects/Orion/Overview.md`/`Sessions/` for the full project record — see `~/.claude/CLAUDE.md` (global) for the `eru` MCP tool convention used to write there.
