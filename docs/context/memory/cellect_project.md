---
name: cellect-project
description: "Cellect — iOS-first SwiftUI mobile annotation app for cell biology / citizen science, at /mnt/firecuda2/Claude/toxoplasma_projects/cellect"
metadata: 
  node_type: memory
  type: project
  originSessionId: 55abe256-2763-460f-97cc-d18fa400e2b0
  modified: 2026-07-23T15:22:34.050Z
---

**Cellect** (working name, easily renamed) is a native iOS mobile app the user is building at
`/mnt/firecuda2/Claude/toxoplasma_projects/cellect` for microscopy image annotation and citizen science. Authored on
this Linux box; user builds on their **Mac** with Xcode (iOS binaries need macOS).

Decisions locked (2026-07-23): **Native SwiftUI** (iOS 17+), Android later; project generated via
**XcodeGen** (`project.yml` → `xcodegen generate`, no checked-in .xcodeproj); on-device ML must use
**Core ML / ONNX, never TensorFlow** (see [[spacr-no-tensorflow]]); **the image folder's CSV is the
source of truth** (portable, readable by pandas/spacr), local cache only for resume/speed.

Feature status (as of 2026-07-23, on branch `main`, 6 commits, ~3.4k LOC Swift):
- ✅ **Feature 2 swipe annotation** — card-stack gestures → folder CSV (built first).
- ✅ **Feature 3 touch segmentation** — MaskBitmap paint engine (brush/erase, per-stroke undo,
  zero-copy overlay), 1-finger paint / 2-finger zoom-pan, exports `<img>_mask.png` (px=class id)
  + classes.json.
- ✅ **Feature 1 camera counting** — AVFoundation capture → CellCounter tier ladder
  (ClassicalCellCounter = Otsu+CCL floor; Core ML tiers stubbed via CoreMLCounterSpec.bundled=[])
  → review → uploads image + 16-bit instance mask + per-object/summary CSVs. User wants tiered
  models by device (best/high/medium/low + classical); classical is the working floor now.
- 🚧 **Feature 4 community** — NOT built. Needs Google Drive OAuth (user must create a GCP OAuth
  client) + real GoogleDriveProvider. Design: per-contributor submission subfolders keyed by a
  stable account/device hash. See ARCHITECTURE §7.
Later: on-device model training. GitHub private repo push still PENDING (user creates empty repo
`EinarOlafsson/cellect`, then `git remote add origin … && git push -u origin main` over SSH).

Reference desktop apps the UX mirrors:
`/mnt/firecuda2/Claude/toxoplasma_projects/plaque_assay_model/tools/plaque_annotate.py` (L/R click
= instance label) and `curate_masks_qt.py` (brush/erase/draw/wand mask painting).

Storage abstraction: `StorageProvider` protocol; `LocalFolderProvider` (Files + iCloud via
security-scoped bookmarks) done; `GoogleDriveProvider` stubbed. See `docs/ARCHITECTURE.md`.
