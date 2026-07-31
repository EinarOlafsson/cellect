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

    /// Decode a PNG (including a comparison overlay) without expanding it until requested.
    static func decodeImage(_ data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Restore the exact UInt16 instance IDs from a PNG produced by `label16`.
    static func decodeLabel16(_ data: Data) -> (labels: [UInt16], width: Int, height: Int)? {
        guard let image = decodeImage(data), image.width > 0, image.height > 0 else { return nil }
        let width = image.width
        let height = image.height
        var labels = [UInt16](repeating: 0, count: width * height)
        let rendered: Bool = labels.withUnsafeMutableBytes { bytes in
            guard let baseAddress = bytes.baseAddress,
                  let context = CGContext(
                    data: baseAddress,
                    width: width,
                    height: height,
                    bitsPerComponent: 16,
                    bytesPerRow: width * MemoryLayout<UInt16>.size,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGBitmapInfo.byteOrder16Little.rawValue
                        | CGImageAlphaInfo.none.rawValue
                  ) else { return false }
            context.interpolationQuality = .none
            context.setBlendMode(.copy)
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard rendered else { return nil }
        return (labels, width, height)
    }
}
