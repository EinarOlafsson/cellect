import CoreGraphics
import Foundation

/// Renders an instance label mask into a translucent colored overlay CGImage for review.
enum LabelOverlay {
    private static let palette: [(r: UInt8, g: UInt8, b: UInt8)] = [
        (18, 176, 110), (214, 69, 80), (58, 123, 213), (224, 168, 0),
        (142, 68, 173), (22, 160, 133), (231, 76, 60), (41, 128, 185)
    ]

    static func makeCGImage(labels: [UInt16], width w: Int, height h: Int, alpha: UInt8 = 130) -> CGImage? {
        guard labels.count == w * h, w > 0, h > 0 else { return nil }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<labels.count {
            let l = labels[i]
            if l == 0 { continue }
            let c = palette[Int(l) % palette.count]
            let o = i * 4
            rgba[o]     = UInt8(Int(c.r) * Int(alpha) / 255)
            rgba[o + 1] = UInt8(Int(c.g) * Int(alpha) / 255)
            rgba[o + 2] = UInt8(Int(c.b) * Int(alpha) / 255)
            rgba[o + 3] = alpha
        }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData) else { return nil }
        return CGImage(
            width: w, height: h,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }
}
