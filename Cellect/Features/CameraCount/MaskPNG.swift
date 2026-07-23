import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum MaskPNG {
    /// Encode a 16-bit grayscale PNG where each pixel == its instance label (0 = background).
    /// PNG is big-endian, so the buffer is byte-swapped and tagged accordingly.
    static func label16(_ labels: [UInt16], width w: Int, height h: Int) -> Data? {
        guard labels.count == w * h, w > 0, h > 0 else { return nil }
        var be = [UInt16](repeating: 0, count: labels.count)
        for i in 0..<labels.count { be[i] = labels[i].bigEndian }
        let bytes = be.withUnsafeBytes { Data($0) }
        guard let provider = CGDataProvider(data: bytes as CFData) else { return nil }

        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
            .union(.byteOrder16Big)
        guard let cg = CGImage(
            width: w, height: h,
            bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: w * 2,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: bitmapInfo,
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        ) else { return nil }

        return encodePNG(cg)
    }

    /// Encode an arbitrary CGImage as PNG.
    static func encodePNG(_ image: CGImage) -> Data? {
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }
}
