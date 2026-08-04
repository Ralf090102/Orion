# CLAUDE.md

This file provides guidance to Claude Code when working in this repository.

## Eru Sync

This project is tracked in [Eru](https://github.com/ralf090102/eru) (an Obsidian vault, second brain) at `01-Projects/Orion/Overview.md`. Eru is a separate git repo (`D:\GitHub\Eru`) reachable from this project via the `eru` MCP server (`mcp__eru__read_note` / `write_note` / `list_notes` / `search_notes`), registered project-scoped for this directory.

**At natural breakpoints** — a feature lands, a real decision gets made, or a session wraps — checkpoint progress back into Eru:
- Update `01-Projects/Orion/Overview.md` (Status/Tasks/Notes) via `mcp__eru__write_note` with `overwrite: true`
- Append a line to that day's `Daily/YYYY-MM-DD.md` log in Eru
- Update the rolling `Memory/CURRENT.md` "where we left off" note so the next session (in any project) resumes instantly

Don't checkpoint after every small edit — batch it to meaningful chunks of work, matching how the study notes in Eru's `02-Areas/AI-Engineering/` were checkpointed.

**Ownership goes beyond checkpointing.** Eru is the user's second brain — Claude owns creating, editing, and organizing content there freely, not just checkpointing at breakpoints. The user's only manual touchpoint is `00-Inbox/Dump.md`; check it when relevant for things they jotted down that need filing. This applies to all projects, not just Orion — see `Memory/Orion/reference_eru_workflow.md` in the vault.

**Memory mirroring:** when creating/updating local `.claude` memory in this project, also mirror it to `Memory/Orion/<name>.md` in the Eru vault (per Eru's own `CLAUDE.md` Memory Sync convention, extended per-project). Cross-project facts (e.g. user profile) live canonically in Eru's `Memory/` and get mirrored down locally.

If the `mcp__eru__*` tools aren't available in this session, the MCP server registration may need a session restart to load — check before assuming Eru sync is broken.
