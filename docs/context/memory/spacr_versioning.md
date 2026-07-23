---
name: spacr-versioning
description: "Bump spacr's PATCH version periodically after batches of commits — smallest increment, never per-commit"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e8555b1-444a-483c-8ac8-da6b4f0e9a09
  modified: 2026-07-23T14:57:35.143Z
---

Bump `VERSION` in `/mnt/firecuda2/Claude/repo/spacr/setup.py` by the SMALLEST possible increment (PATCH: `X.Y.Z → X.Y.(Z+1)`) after batches of feature/fix work, not on every commit.

**Why:** The user wants version numbers to move forward as work accumulates, but not on every commit — that would be noisy. The rule is "many updates pushed → single small bump."

**How to apply:**
- Do NOT bump on every commit. Batches of related work land at one version.
- **Use 4-part versioning for small polish batches**: `X.Y.Z.W`. When 3-8 commits of polish / fixes / small features have accumulated, bump the 4th part (`1.4.1 → 1.4.1.1 → 1.4.1.2 → …`).
- Bump PATCH (3rd part, `1.4.1 → 1.4.2`) when a bigger batch lands (10+ commits with substantial new features), and reset the 4th part.
- Bump MINOR (`X.(Y+1).0`) only on user request or public-API deprecation.
- Never MAJOR without explicit sign-off.
- Do the bump as its own tiny commit right before the feature commit so history is easy to read.

**Examples:**
- Ship 3 small polish commits → `1.4.1 → 1.4.1.1`
- Ship 5 more polish commits → `1.4.1.1 → 1.4.1.2`
- Ship a big feature batch (v2 pipeline, DnD system, etc.) → `1.4.1.2 → 1.4.2`

**Where the version lives:**
- `setup.py`: `VERSION = "X.Y.Z"` (top-level constant, ~line 85)
- Automatically flows to `spacr.__version__` via `importlib.metadata`.

Related: [[spacr-workflow]], [[spacr-project]].
