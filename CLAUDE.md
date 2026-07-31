# CLAUDE.md — Cellect

Orientation for any Claude Code session working on this repo (especially a **Mac** session that
can actually build). Read this first.

> **Resuming a handed-off session?** Read **`docs/HANDOFF.md`** next — it has the full vision,
> every decision made, current state, and prioritized next steps. Carried-over working preferences
> and project memory are in **`docs/context/memory/`** (re-seed them into your memory if useful).

## What this is

Cellect is a **native iOS SwiftUI app** (iOS 17+) for microscopy image annotation and citizen
science — for cell biologists. Working name; renamable (see `docs/ARCHITECTURE.md` §9).
Authored partly on a Linux box (can't compile iOS); **built on macOS**.

## Build & run (macOS + Xcode 15+)

There is **no checked-in `.xcodeproj`** — it's generated from `project.yml` with XcodeGen.

```sh
brew install xcodegen          # once
xcodegen generate              # → Cellect.xcodeproj (git-ignored)
open Cellect.xcodeproj          # set your signing team, then ⌘R
# or headless:
xcodebuild -project Cellect.xcodeproj -scheme Cellect \
  -destination 'generic/platform=iOS' build
```

Camera counting needs a **real device** (no camera in the simulator). Folder features work with
iCloud Drive / On My iPhone folders via the system picker. Full detail: `docs/BUILDING.md`.

**If a build fails:** the Linux author couldn't compile, so first-compile errors are expected and
welcome — fix them, keep changes minimal and idiomatic, and note anything non-obvious here.

## Architecture (see `docs/ARCHITECTURE.md` for the full spec)

- Layers: `App/` (entry + home), `Features/<Feature>/` (self-contained View+ViewModel),
  `Models/` (Codable value types, no SwiftUI), `Services/` (Storage/CSV/Persistence).
  **Rule:** `Models` and `Services` never import SwiftUI.
- **The folder is the source of truth.** Annotations are CSVs / mask PNGs written *into the image
  folder* (portable to pandas/spacr). Local stores hold only project metadata + resume state.
- **StorageProvider** protocol abstracts the folder. `LocalFolderProvider` (Files + iCloud via
  security-scoped bookmarks) is done; `GoogleDriveProvider` is stubbed behind the same protocol.
- On-device ML uses **Core ML / ONNX — never TensorFlow**.

## Feature status

- ✅ **Swipe annotation** (`Features/SwipeAnnotation/`) — feature 2.
- ✅ **Touch segmentation** (`Features/SemanticSegment/`) — feature 3. `MaskBitmap` is the paint
  engine; masks export as `<img>_mask.png` (pixel = class id) + `cellect_classes.json`.
- ✅ **Camera counting** (`Features/CameraCount/`) — feature 1. `ClassicalCellCounter` is the
  deterministic floor; nine workstation-trained foreground/contact-boundary models are registered
  for Core ML. Saves the image, 16-bit instance mask, objects CSV, exact settings, and diagnostics.
- ✅ **Model comparison** (`Features/ModelComparison/`) — serial model execution, bounded settings
  sweeps, lazy overlay review, chosen-result application, and complete mask/manifest export.
- ✅ **Workstation pipeline** (`WorkstationTrainingBundle/`, `WorkstationResults/`) — dataset
  adapters, scientific splits, training/evaluation, deployment parity, and Core ML conversion.
- 🚧 **Community** (feature 4) — not built. Backend decision: **Google Drive + OAuth**. Needs a
  GCP OAuth client ID + Drive API enabled (user action). Plan in `docs/ARCHITECTURE.md` §7.

## Conventions

- Commits: small, named, imperative subject + bulleted body. **Do not add a `Co-Authored-By`
  trailer** — the user is the sole author. Don't push without the user's go-ahead.
- SwiftUI: iOS 17 `@Observable` (not `ObservableObject`). Keep the `Models`/`Services` layers
  UIKit/SwiftUI-free where they already are.
- Match the surrounding code's style; keep comments at the existing density (explain *why*).

## Known gaps / next work

- Validate selected Core ML packages and comparison performance on representative physical iPhones.
- Publish selected model packages as versioned release assets or Git LFS objects; generated model
  packages are intentionally excluded from ordinary Git.
- Google Drive provider + OAuth (feature 4 foundation).
- Evaluate Cellpose-SAM conversion after workstation training completes.
