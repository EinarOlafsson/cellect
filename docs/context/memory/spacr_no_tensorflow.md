---
name: spacr-no-tensorflow
description: "The user's spacr package must never depend on TensorFlow — remove any TF-backed code (e.g. stardist) rather than gating it"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7e8555b1-444a-483c-8ac8-da6b4f0e9a09
  modified: 2026-07-20T11:54:07.528Z
---

**Rule:** No TensorFlow dependency in [[spacr-project]]. Anything that pulls in `tensorflow` / `tensorflow-macos` / `keras`, transitively or directly, must be removed rather than gated behind an optional import. This includes `stardist` (which imports TF at module load time).

**Why:** The user is standardizing on the PyTorch/scikit-image/cellpose stack. TensorFlow adds heavy install cost (GPU driver friction, incompatible with common cellpose/torch pins) and duplicates model surface area for no benefit. This was communicated on 2026-07-20 when a stardist code path was flagged during the test-suite work.

**How to apply:**
- Before adding any new dependency, grep for `tensorflow`/`keras` in its transitive graph.
- When removing TF-adjacent code, also drop it from `_validate_organelle_settings`, any settings dictionaries, docstrings, help text, and existing tests.
- Existing spacr modules to sweep for TF/stardist references: `object.py`, `settings.py`, `submodules.py`, `deep_spacr.py`. Not exhaustive — grep every time.
