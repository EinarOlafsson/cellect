# Cellect

A mobile image-annotation platform for cell biologists and citizen science.

> **Working name.** "Cellect" (cell + collect + select) is a placeholder — trivial to
> rename later. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the rename checklist.

Cellect turns a phone into a microscopy annotation station. It targets four capabilities,
built in slices:

1. **Camera cell counting** — photograph cells through a microscope (DIC / BF / Phase),
   segment + count them on-device, and upload the image, the mask, and a per-object CSV
   (count + size) to a linked Google Drive or iCloud folder.
2. **Swipe annotation** *(building first)* — point at a folder of images, swipe to assign
   classes, and write a CSV (`filename` + user-named columns) back into the folder. Multiple
   rounds add multiple columns.
3. **Touch semantic segmentation** — paint instance/semantic masks on images by hand, modelled
   on the desktop `curate_masks_qt.py` tool (brush / erase / draw / wand / divide).
4. **Community annotation** — the maintainer shares Drive folders; opted-in users annotate, and
   their contributions land in a per-user folder (stable device/account ID) for later review.

Later: on-device model training (classification + instance segmentation) so users can build
models for their own science.

## Platforms

- **iOS first** (SwiftUI, iOS 17+). This is the primary, best-supported target.
- **Android** later — the data contracts (CSV format, annotation schema, sync protocol in
  `docs/ARCHITECTURE.md`) are kept platform-neutral so a Kotlin UI can sit on top of the same
  formats. No Android code yet.

## Building (macOS + Xcode required)

iOS binaries can only be built on macOS. This repo is authored to be generated with
[XcodeGen](https://github.com/yonaskolb/XcodeGen) so there is no fragile checked-in
`.xcodeproj`.

```sh
brew install xcodegen        # once
cd cellect
xcodegen generate            # produces Cellect.xcodeproj from project.yml
open Cellect.xcodeproj        # build & run on a simulator or device (⌘R)
```

See [docs/BUILDING.md](docs/BUILDING.md) for signing, device deployment, and CI options.

## Status

| Slice | State |
|-------|-------|
| Project scaffold (XcodeGen, app shell) | ✅ initial |
| Swipe annotation (feature 2) | ✅ built |
| Local + iCloud folder access | ✅ built |
| Touch semantic segmentation (feature 3) | ✅ built |
| Camera cell counting (feature 1) | ✅ built (classical tier; Core ML tiers stubbed) |
| Google Drive provider | ⬜ stubbed (protocol in place) |
| Community backend (feature 4) | 🚧 next |
| On-device training | ⬜ future |
