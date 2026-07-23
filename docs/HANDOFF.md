# Cellect — Session Handoff

This captures everything from the session that created Cellect, so a fresh Claude Code session
(on the Mac, where it can compile) can continue seamlessly. **A new session does not inherit the
old chat or the Linux box's memory files — this document + `CLAUDE.md` + the copied memory in
`docs/context/memory/` are how the context travels.**

> **To resume:** open this repo in Claude Code and say *"Read CLAUDE.md and docs/HANDOFF.md, then
> continue."* Optionally have it re-save the notes in `docs/context/memory/` into its own memory.

---

## 1. The vision (verbatim intent from the user)

Cellect is a mobile app (**iOS first, Android later**) for cell biologists and citizen science.
The user is a scientist who does a lot of image analysis — instance + semantic segmentation. Four
capabilities, plus a longer-term goal:

1. **Camera cell counting** — photograph cells in a dish through a microscope (DIC / BF / Phase),
   count them on-device, and upload the image, a mask, and a CSV (object count + size) to a linked
   Google Drive or iCloud folder.
2. **Swipe annotation** — point at a folder (iOS/iCloud/Drive); swipe left = annotation 1, right =
   annotation 2, other gestures for more classes. Writes a CSV in the folder with image filename +
   annotation; the user names the annotation column; multiple rounds add multiple columns.
3. **Touch semantic segmentation** — point at a folder; draw/paint object masks by touch (modelled
   on the desktop `curate_masks_qt.py` plaque tool).
4. **Community annotation** — the maintainer shares Drive folders; opted-in users annotate (swipe
   instance or touch semantic); their data saves to a per-contributor folder keyed by a stable
   device/account id, so repeat rounds accrete in the same place, for the maintainer to review.

**Goal:** a personal annotation tool + a way to give others the app + a community annotation
platform. **Eventually:** let users train their own classification / instance-segmentation models
for their science. Audience: cell biologists + citizen science.

## 2. Decisions locked this session

- **Native SwiftUI** (iOS 17+), Android later. (User has a Mac to build on.)
- Project generated with **XcodeGen** — no checked-in `.xcodeproj`.
- On-device ML: **Core ML / ONNX, never TensorFlow.**
- **The image folder is the source of truth** — annotations are CSV / mask PNG written *into the
  folder* (portable to pandas/spacr). Local stores hold only project metadata + resume state.
- Build order chosen by user: swipe (2) → touch seg (3) → camera count (1) → community (4).
- Community backend: **Google Drive + OAuth** (feature 4).
- **Counting models are tiered by device** (user's explicit ask): a `best` (full DL, no CV) / `high`
  / `medium` / `low` ladder for newer→older phones, with **classical CV as the always-available
  floor**. On "mini Cellpose-SAM": no official pocket version exists; realistic mobile options are
  small Cellpose or distilled SAMs (MobileSAM / EdgeSAM / EfficientSAM / FastSAM) — verify against
  current releases when actually converting one.

## 3. What's built (features 1, 2, 3 — committed)

7 commits on `main`, ~3,400 lines of Swift, 51 files. **Not yet compiled** — authored on Linux
(no iOS SDK), so the first Xcode build may surface small errors; that's expected. Most likely spots:
the AVFoundation camera code and `MaskBitmap`'s unsafe-pointer buffer.

- ✅ **Swipe annotation** — `Cellect/Features/SwipeAnnotation/`. Card-stack gestures → merged CSV
  (`cellect_annotations.csv`) in the folder. Debounced background flush; resumes at first unlabelled.
- ✅ **Touch segmentation** — `Cellect/Features/SemanticSegment/`. `MaskBitmap` paint engine
  (brush/erase, per-stroke undo/redo, region-scoped stamping, zero-copy overlay). One finger paints,
  two fingers zoom/pan. Exports `<img>_mask.png` (pixel = class id, 8-bit) + `cellect_classes.json`.
  Canvas capped at 2048px long side (`SegmentationConfig.maxDimension`).
- ✅ **Camera counting** — `Cellect/Features/CameraCount/`. AVFoundation capture → `CellCounter`
  tier registry → review (live polarity / min-area recount) → upload. `ClassicalCellCounter`
  (Otsu + 8-connected components) is the working floor. Core ML tiers plug in via
  `CoreMLCounterSpec.bundled` (currently `[]`). Uploads capped image PNG + 16-bit instance mask PNG
  + per-object CSV + appended summary CSV.
- Storage: `StorageProvider` protocol; `LocalFolderProvider` (Files + iCloud, security-scoped
  bookmarks) done; `GoogleDriveProvider` stubbed.

Full design: `docs/ARCHITECTURE.md`. Data formats there (CSV §4, mask §4b, counting §6b) are the
portable interop contract — keep them stable.

## 4. Next work (priority order)

1. **Build it.** `xcodegen generate && open Cellect.xcodeproj`, set signing team, run on a real
   iPhone (camera needs a device). Fix first-compile errors. See `docs/BUILDING.md`.
2. **Feature 4 (community) — needs the user to do this first:** create a **Google Cloud project**,
   enable the **Drive API**, make an **OAuth 2.0 client ID** (iOS type, bundle `com.cellect.app`).
   Then: implement `GoogleDriveProvider: StorageProvider` + Google Sign-In (add `GoogleSignIn` SPM
   dep in `project.yml`), a stable **contributor id** (hash of account+device, no PII in path), and
   the `submissions/<contributor-id>/…` layout (ARCHITECTURE §7). Because everything sits behind
   `StorageProvider`, Drive drops into all three existing features at once.
3. **Better counting:** distance-transform **watershed** to split touching cells in the classical
   counter; then convert a small Cellpose / SAM model → Core ML for the higher tiers.
4. **Full-resolution masks** — make the 2048px cap configurable.
5. On-device training (long-term).

## 5. Working conventions (from the user's memory — apply these)

- Commits: **small, named**, imperative. **Never add a `Co-Authored-By` trailer** — the user is the
  sole author. **Don't push without the user's go-ahead.**
- SwiftUI: iOS 17 `@Observable` (not `ObservableObject`). Keep `Models/` and `Services/` free of
  SwiftUI/UIKit where they already are.
- No TensorFlow anywhere.
- Match surrounding code style + comment density (comments explain *why*).

## 6. GitHub (not yet created)

The repo is local-only with full history. To put it on GitHub (user creates the empty **private**
repo `EinarOlafsson/cellect` first; SSH key already authorised as `EinarOlafsson`):

```sh
git remote add origin git@github.com:EinarOlafsson/cellect.git
git push -u origin main
```

Once on GitHub, prefer **git** (not Syncthing) as the sync channel between machines to avoid
`.git` sync-conflicts.

## 7. Environment notes

- This repo was authored under `/mnt/firecuda2/Claude/toxoplasma_projects/cellect` (a Syncthing
  folder) on a Linux box, then copied to the Mac.
- Reference desktop tools the UX mirrors (on the Linux box, **not** copied here):
  `toxoplasma_projects/plaque_assay_model/tools/plaque_annotate.py` (L/R click = instance label) and
  `.../curate_masks_qt.py` (brush/erase/draw/wand mask painting).
- The user's other project is **spacr** (image-analysis Python package); Cellect's CSV/mask formats
  are meant to be spacr/pandas-readable.
