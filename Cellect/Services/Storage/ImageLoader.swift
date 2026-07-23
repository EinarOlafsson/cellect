import CoreGraphics
import Foundation
import ImageIO

/// Decodes image bytes to a CGImage, applying EXIF orientation and capping the long side so
/// large microscopy images stay within a workable memory/perf budget. TIFF is supported via
/// ImageIO. The cap is deliberate and documented — see `SegmentationConfig`.
enum ImageLoader {
    static func decode(_ data: Data, maxDimension: Int) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,   // bake in orientation
            kCGImageSourceThumbnailMaxPixelSize: maxDimension   // caps, never upscales
        ]
        return CGImageSourceCreateThumbnailAtIndex(src, 0, options as CFDictionary)
    }

    /// Decode an existing mask PNG at full size (dimensions must match the decoded base).
    static func decodeExact(_ data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }
}

enum SegmentationConfig {
    /// Long-side cap for the editing canvas and the exported mask. Raise for full-res masks
    /// at the cost of memory; kept moderate for smooth painting on-device.
    static let maxDimension = 2048
}
