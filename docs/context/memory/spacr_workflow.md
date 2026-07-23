---
name: spacr-workflow
description: "Ground rules for working on the user's spacr Python package in /mnt/firecuda2/Claude/repo/spacr"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e8555b1-444a-483c-8ac8-da6b4f0e9a09
  modified: 2026-07-20T01:00:41.393Z
---

Rules for spacr work in [[spacr-project]]:

1. **Never `git push` without explicit instruction.** Local commits are fine; anything remote requires the user to say "push".
2. **Multiple properly-named commits over one giant commit.** Group related changes into a single logical commit each (e.g., "fix typos in X module", "remove duplicate _v1 functions in Y"). Don't bundle unrelated fixes.
3. **Always test after each change.** Test in the spacr context — import the affected module, run relevant CLI/notebook path, or run the test suite. Never claim done without verification.
4. **Never push to `main`.** When the user does authorize pushing, options are: a private fork, a new branch, or `nightly` — never `main`.

**Why:** The user is iterating on their own scientific Python package; they want to review and steer, not receive a fait-accompli refactor. Broken code shipped upstream would disrupt their pipeline and other users of `EinarOlafsson/spacr`.

**How to apply:** Every time working under [[spacr-project]], make TodoWrite tasks for each logical change, test between commits, and stop for approval before any remote-affecting git operation.
