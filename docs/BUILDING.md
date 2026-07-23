# Building Cellect (macOS)

## Prerequisites

- macOS with **Xcode 15+** (for iOS 17 SDK, Swift 5.9, the Observation framework).
- [XcodeGen](https://github.com/yonaskolb/XcodeGen): `brew install xcodegen`.

## Generate & open

```sh
cd cellect
xcodegen generate       # reads project.yml → Cellect.xcodeproj
open Cellect.xcodeproj
```

`Cellect.xcodeproj` is generated, so it's git-ignored — regenerate it any time sources change
structurally (adding/removing files is picked up automatically since sources are folder-based;
you only re-run `xcodegen` if it doesn't refresh).

## Run in the simulator

Select an iPhone simulator and press ⌘R. The simulator's Files app is limited, so to test the
folder picker copy some images into the simulator (drag files onto the simulator window, or use
`xcrun simctl`), or better — run on a real device with images in iCloud Drive / On My iPhone.

## Run on your iPhone

1. Plug in the device, select it as the run destination.
2. In the **Signing & Capabilities** tab of the `Cellect` target, pick your Apple ID team.
   (Or set `DEVELOPMENT_TEAM` in `project.yml` and regenerate.) A free Apple ID works for
   on-device testing.
3. ⌘R. Approve the developer certificate on the phone under
   Settings → General → VPN & Device Management the first time.

## iCloud Drive access

The folder picker (`.fileImporter` with `.folder`) can reach **On My iPhone** and **iCloud Drive**
folders out of the box — no extra entitlement needed for user-selected folders. To later show
the app's *own* iCloud container, add the iCloud capability + a container in Signing &
Capabilities.

## Adding Google Drive later

1. Add the `GoogleSignIn` (and Drive REST) Swift packages under `targets.Cellect.dependencies`
   in `project.yml`, then `xcodegen generate`.
2. Implement `GoogleDriveProvider: StorageProvider` and return it from
   `StorageProviderFactory.make(for:)` for `.googleDrive` refs. Nothing else changes.

## CI (build iOS without a local Mac)

If you ever want automated builds, the cheapest path is a macOS runner:

- **GitHub Actions** `macos-14` runner: `xcodegen generate` → `xcodebuild -scheme Cellect
  -destination 'generic/platform=iOS' build` (add signing secrets for archiving).
- **Codemagic** / **Bitrise** have XcodeGen steps and managed signing.
