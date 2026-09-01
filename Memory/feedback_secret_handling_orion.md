---
name: feedback-secret-handling-orion
description: User's risk tolerance for secret/key exposure in chat on the Orion project
metadata:
  type: feedback
---

For Orion specifically, don't heavily flag or push back on private key / secret exposure in chat (e.g. pasting an encrypted signing key + its password in the same conversation).

**Why:** User explicitly said (2026-08-04, during Tauri updater signing-key setup) not to worry about it — Orion is a personal solo project, not something with other stakeholders or production users whose trust depends on operational secret hygiene.

**How to apply:** Still avoid actively soliciting secrets in chat (e.g. don't ask the user to paste a password), and a single factual flag when something sensitive is shared unprompted is fine — but don't repeat the warning, insist on rotation, or treat it as a blocking issue once the user has acknowledged it. This calibration is specific to Orion's low-stakes, single-owner context; don't assume it generalizes to other projects without similar signals.

This is a narrower exception to the general secret-sharing default (any project should still default to caution around secrets in chat) — Orion's low-stakes context is what earns the exception, not a general rule.
