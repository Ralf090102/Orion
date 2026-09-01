---
name: current-session-context
description: Rolling "where we left off" checkpoint; check first each session to resume instantly
metadata:
  type: project
---

## Last active: 2026-08-26

## What we just did
- **Per-project Erudition hooks verified end-to-end (2026-08-26).** Closes the deferred item from 2026-08-25. Found and fixed a real bug: Eru's own SessionStart/Stop hooks (pointing at `D:\GitHub\Eru`) had been duplicated into the **global** `~/.claude/settings.json`, in addition to correctly living in `D:\GitHub\Eru\.claude\settings.json`. Because global hooks fire in every project, this is what was actually reaching Orion sessions — confirmed live: this session's own SessionStart hook output was Eru's checkpoint content, not Orion's, even though Orion's own project-scoped hook (verified correct in isolation via manual replication) never surfaced its output at all. Fixed by removing the duplicate `hooks` block from the global settings.json — Eru's hooks keep working from its own project-scoped copy. No data-loss exposure existed: the Stop hook's `local_memory_dir` was always project-scoped correctly in both configs, so `sync_memory`'s mirror-and-delete behavior never had a cross-vault target to clobber. Separately found a stray Windows **User-level env var `ERUDITION_VAULT_PATH=D:\GitHub\Eru`**, unrelated to the hook bug — confirmed unused (every real consumer, all project hook commands and the `eru` MCP server's own `env` block in `.claude.json`, sets `ERUDITION_VAULT_PATH` explicitly rather than relying on ambient env) and deleted.
- **Orion's own Erudition instance set up (2026-08-25).** Previously, Orion's project memory only existed as a manually-mirrored, drift-prone copy inside Eru's vault (`Memory/Orion/`) — that special-casing has been removed. Orion now has its own self-contained memory here, synced via its own `.claude/settings.json` hooks pointed at this repo, exactly like Eru manages its own. See `01-Projects/Orion/Overview.md` in Eru for full project history predating this — not re-narrated here.

## What's next
- Continue from wherever `01-Projects/Orion/Overview.md` (Eru) and this project's own git log leave off.
- Confirm on the next fresh Orion session that SessionStart now correctly injects this file's content (not Eru's) — couldn't verify in the session that made the fix, since it was already running with the old hook output cached.
