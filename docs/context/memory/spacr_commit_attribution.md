---
name: spacr-commit-attribution
description: Never add Co-Authored-By trailer on any spacr commit. Sole author = user.
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e8555b1-444a-483c-8ac8-da6b4f0e9a09
  modified: 2026-07-23T00:09:17.887Z
---

Never add `Co-Authored-By: Claude ...` (or any AI-attribution trailer) to spacr commits. GitHub renders those as "EinarOlafsson and claude committed" — user wants just their own name on every commit.

**Why:** Cosmetic + attribution — user's public commit history should read as authored solely by them.

**How to apply:** When staging a commit message via HEREDOC, omit the trailing `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` block entirely. Ends after the commit body. Applies to every commit on this repo — bug fixes, features, docs, everything.

If asked to rewrite past commits to strip the trailer, that's a `git rebase -i` + amend + force-push exercise the user must green-light explicitly (it rewrites history on `nightly`).

Related: [[spacr-workflow]], [[spacr-project]].
