# Cellect — Architecture

This document is the durable design reference. Code should match it; when they diverge, update
one to match the other deliberately.

## 1. Goals & constraints

- **iOS-first, Android-later.** All *data formats and sync semantics* are defined here in a
  platform-neutral way so an Android client can interoperate byte-for-byte. UI is native per
  platform.
- **No TensorFlow.** On-device ML uses **Core ML** (native) or **ONNX Runtime** — never
  TensorFlow Lite. Cellpose/spacr PyTorch models convert to Core ML / ONNX.
- **The folder is the source of truth.** Annotations live as a CSV *inside the image folder*
  (local, iCloud, or Drive), so results are portable and readable by desktop tools (pandas,
  spacr) with zero lock-in. A local cache exists only for speed and resume.
- **Offline-first.** Annotation must work with no network; sync reconciles later.

## 2. Layered structure

```
App/            SwiftUI entry, root navigation
Features/       One folder per user-facing feature (self-contained View + ViewModel)
  SwipeAnnotation/
  CameraCount/          (future)
  SemanticSegment/      (future)
Models/         Plain Codable value types — the domain vocabulary
Services/
  Storage/      StorageProvider protocol + Local/iCloud/Drive implementations
  CSV/          CSV read/write (portable, spec below)
  Persistence/  Local cache (resume, debounced flush)
Resources/      Assets, Info.plist
```

Rule: `Models` and `Services` never import SwiftUI. `Features` depend on them, not vice-versa.
This keeps the core reusable and testable, and portable to Android in spirit.

## 3. Domain model (`Models/`)

```
AnnotationClass        one label reachable by one gesture
  id: UUID
  label: String        # exact string written to the CSV cell, e.g. "mitotic"
  gesture: Gesture
  colorHex: String

Gesture (enum)         swipeLeft | swipeRight | swipeUp | swipeDown
                       (tap variants reserved for >4 classes later)

AnnotationColumn       one round of annotation = one CSV column
  id: UUID
  name: String         # user-defined CSV header, e.g. "phenotype"
  classes: [AnnotationClass]

AnnotationProject      a folder + its annotation config
  id: UUID
  displayName: String
  storageRef: StorageRef      # how to re-open the folder (bookmark / Drive id)
  columns: [AnnotationColumn] # multiple rounds
  csvFileName: String         # default "cellect_annotations.csv"

AnnotationRecord       one row = one image
  filename: String
  values: [String: String]    # columnName -> class label (or "")
```

Multiple annotation rounds = appending another `AnnotationColumn` and filling a new CSV column;
existing columns are preserved.

## 4. CSV contract (portable, the interop surface)

- UTF-8, `\n` line endings, comma-separated, RFC-4180 quoting (fields containing `,` `"` or
  newline are double-quoted; embedded `"` doubled).
- **First column is always `filename`** (basename incl. extension, e.g. `A01_f01.tif`).
- Each annotation round contributes **one column**, header = the column's user-defined `name`.
- A cell is the chosen class `label`, or empty string if not yet annotated / skipped.
- Reading an existing CSV merges: unknown columns are preserved untouched; the `filename`
  column is the join key; new images append new rows.
- Column-name collisions: if the user reuses an existing column name, we append a numeric
  suffix (`phenotype_2`) rather than overwrite — surfaced in the UI before it happens.

Example after two rounds (`phenotype`, then `quality`):

```csv
filename,phenotype,quality
A01_f01.tif,mitotic,good
A01_f02.tif,interphase,
A02_f01.tif,,blurry
```

## 5. Storage abstraction (`Services/Storage/`)

```swift
protocol StorageProvider {
    func listImages() async throws -> [StoredImage]
    func loadImageData(_ image: StoredImage) async throws -> Data
    func readFile(named: String) async throws -> Data?          // nil if absent
    func writeFile(named: String, data: Data) async throws
}
```

- `StoredImage` = stable id + display name + provider handle (a URL for local, a fileId for Drive).
- **LocalFolderProvider** covers *both* on-device Files and **iCloud Drive**, because iCloud
  folders are handed to us as ordinary security-scoped `file://` URLs by the system document
  picker. Implementation: `UIDocumentPicker` (folder mode) → `bookmarkData()` persisted in
  `StorageRef` → `startAccessingSecurityScopedResource()` on each session.
- **GoogleDriveProvider** (later) implements the same protocol over the Drive REST API + OAuth.
  Because everything upstream speaks `StorageProvider`, features don't change when we add it.

`StorageRef` is Codable and records *which* provider + the reopen token (bookmark blob or Drive
folder id), so projects survive app restarts.

## 6. Persistence & flush (`Services/Persistence/`)

Writing the CSV on every swipe would thrash iCloud/Drive. Mirroring the desktop
`plaque_annotate.py` worker-thread pattern:

- Swipes mutate an in-memory `[filename: AnnotationRecord]` immediately (UI is instant).
- A **debounced background flush** (≈2 s idle, or every N swipes, or on background/leave) writes
  the merged CSV via `StorageProvider.writeFile`.
- A local mirror (JSON now; SQLite if it grows) enables crash-safe resume before any flush lands.

## 7. Community backend (feature 4 — design only, not built)

- Maintainer shares a Drive folder (read for images, write for results).
- A user opts in → gets a stable **contributor id** = hash of (iCloud account or Google account) +
  device, so repeat sessions land in the *same* per-user subfolder:
  `.../submissions/<contributor-id>/<column>.csv` (+ masks for segmentation).
- Contributor id is stable per account so multiple rounds accrete; never PII in the path.
- Review tooling (desktop/spacr) reads `submissions/*` to accept/reject. No custom server
  required for v1 — Drive is the backend. A thin server is only needed if we later want
  auth-gated task assignment or leaderboards.

## 8. On-device training (future — design only)

- Export annotated crops + labels → a Create ML / Core ML training job (classification first,
  then instance seg via a Core ML-converted small U-Net/Cellpose). Runs on-device on newer
  Neural Engine hardware or off-loads to the maintainer's box.
- Out of scope until features 1–3 are solid.

## 9. Renaming from "Cellect"

Change in one place each: `project.yml` (`name:`, bundle id), `Info.plist` display name,
`App/CellectApp.swift` (struct name), and the `Cellect/` source folder name. No name is compiled
into data formats, so existing CSVs stay valid.
