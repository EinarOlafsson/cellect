import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The paint engine. Holds two parallel buffers sized to the mask:
///   - `label`  : one UInt8 class id per pixel (0 = background). This is the scientific
///                output, exported as a grayscale PNG.
///   - `rgba`   : a premultiplied RGBA overlay derived from `label`, shown over the image.
///
/// All edits are **region-scoped** — a brush stamp only touches pixels inside its bounding
/// box — so painting cost is independent of image size. Undo/redo are per-stroke diffs.
final class MaskBitmap {
    let width: Int
    let height: Int

    private(set) var label: [UInt8]
    // RGBA premultiplied overlay in stable memory, so the display CGImage can wrap it with a
    // non-copying data provider (cheap to rebuild every touch-move). Owned here; freed in deinit.
    private let rgbaPtr: UnsafeMutablePointer<UInt8>
    private let rgbaCount: Int

    /// id -> premultiplied RGBA the overlay uses for that class (0 == transparent bg).
    private var palette: [UInt8: (r: UInt8, g: UInt8, b: UInt8, a: UInt8)] = [:]
    private let overlayAlpha: UInt8 = 140

    // Undo/redo: each entry maps pixelIndex -> (old, new) label values for one stroke.
    private var currentStroke: [Int: (old: UInt8, new: UInt8)] = [:]
    private var undoStack: [[Int: (old: UInt8, new: UInt8)]] = []
    private var redoStack: [[Int: (old: UInt8, new: UInt8)]] = []
    private let undoLimit = 30

    init(width: Int, height: Int, classes: [SegmentationClass]) {
        self.width = max(1, width)
        self.height = max(1, height)
        self.label = [UInt8](repeating: 0, count: self.width * self.height)
        self.rgbaCount = self.width * self.height * 4
        self.rgbaPtr = UnsafeMutablePointer<UInt8>.allocate(capacity: self.rgbaCount)
        self.rgbaPtr.initialize(repeating: 0, count: self.rgbaCount)
        setClasses(classes)
    }

    deinit {
        rgbaPtr.deinitialize(count: rgbaCount)
        rgbaPtr.deallocate()
    }

    /// Rebuild the color lookup (call if classes/colors change).
    func setClasses(_ classes: [SegmentationClass]) {
        palette.removeAll()
        for c in classes {
            let base = rgbFromHex(c.colorHex)
            let a = overlayAlpha
            // premultiply
            palette[c.id] = (
                r: UInt8(Int(base.r) * Int(a) / 255),
                g: UInt8(Int(base.g) * Int(a) / 255),
                b: UInt8(Int(base.b) * Int(a) / 255),
                a: a
            )
        }
    }

    // MARK: - Painting

    func beginStroke() { currentStroke.removeAll(keepingCapacity: true) }

    /// Paint a filled circle of `classId` (or erase when `classId == 0`) at pixel center.
    func stampCircle(cx: Int, cy: Int, radius: Int, classId: UInt8) {
        let r = max(1, radius)
        let r2 = r * r
        let minY = max(0, cy - r), maxY = min(height - 1, cy + r)
        let minX = max(0, cx - r), maxX = min(width - 1, cx + r)
        guard minY <= maxY, minX <= maxX else { return }
        for y in minY...maxY {
            let dy = y - cy
            let row = y * width
            for x in minX...maxX {
                let dx = x - cx
                if dx * dx + dy * dy > r2 { continue }
                let idx = row + x
                let old = label[idx]
                if old == classId { continue }
                if currentStroke[idx] == nil {
                    currentStroke[idx] = (old: old, new: classId)
                } else {
                    currentStroke[idx]!.new = classId
                }
                setPixel(idx: idx, id: classId)
            }
        }
    }

    /// Stamp along the segment a→b so fast finger moves leave a continuous line.
    func stampLine(from a: (x: Int, y: Int), to b: (x: Int, y: Int), radius: Int, classId: UInt8) {
        let dx = b.x - a.x, dy = b.y - a.y
        let dist = Int(Double(dx * dx + dy * dy).squareRoot())
        let step = max(1, radius / 2)
        let n = max(1, dist / step)
        for i in 0...n {
            let t = Double(i) / Double(n)
            let x = a.x + Int((Double(dx) * t).rounded())
            let y = a.y + Int((Double(dy) * t).rounded())
            stampCircle(cx: x, cy: y, radius: radius, classId: classId)
        }
    }

    func endStroke() {
        guard !currentStroke.isEmpty else { return }
        undoStack.append(currentStroke)
        if undoStack.count > undoLimit { undoStack.removeFirst() }
        redoStack.removeAll()
        currentStroke.removeAll(keepingCapacity: true)
    }

    private func setPixel(idx: Int, id: UInt8) {
        label[idx] = id
        let o = idx * 4
        if id == 0 {
            rgbaPtr[o] = 0; rgbaPtr[o + 1] = 0; rgbaPtr[o + 2] = 0; rgbaPtr[o + 3] = 0
        } else if let c = palette[id] {
            rgbaPtr[o] = c.r; rgbaPtr[o + 1] = c.g; rgbaPtr[o + 2] = c.b; rgbaPtr[o + 3] = c.a
        }
    }

    // MARK: - Undo / redo

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    func undo() {
        guard let stroke = undoStack.popLast() else { return }
        for (idx, v) in stroke { setPixel(idx: idx, id: v.old) }
        redoStack.append(stroke)
    }

    func redo() {
        guard let stroke = redoStack.popLast() else { return }
        for (idx, v) in stroke { setPixel(idx: idx, id: v.new) }
        undoStack.append(stroke)
    }

    func clearAll() {
        beginStroke()
        for idx in 0..<label.count where label[idx] != 0 {
            currentStroke[idx] = (old: label[idx], new: 0)
            setPixel(idx: idx, id: 0)
        }
        endStroke()
    }

    // MARK: - Overlay image (for display)

    /// Current overlay as a CGImage wrapping the shared rgba buffer with NO copy — cheap to
    /// call every touch-move. The returned image is only valid while this MaskBitmap lives.
    func makeOverlayCGImage() -> CGImage? {
        let bytesPerRow = width * 4
        guard let provider = CGDataProvider(
            dataInfo: nil, data: rgbaPtr, size: rgbaCount, releaseData: { _, _, _ in }
        ) else { return nil }
        return CGImage(
            width: width, height: height,
            bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent
        )
    }

    // MARK: - Load / export label mask (grayscale, pixel == class id)

    /// Populate `label` (and overlay) from an existing grayscale mask CGImage of matching size.
    func loadLabel(from image: CGImage) {
        guard image.width == width, image.height == height else { return }
        var gray = [UInt8](repeating: 0, count: width * height)
        gray.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(
                    data: base, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                  ) else { return }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        for idx in 0..<label.count where gray[idx] != label[idx] {
            setPixel(idx: idx, id: gray[idx])
        }
        // Loading is not an undoable stroke.
        currentStroke.removeAll()
        undoStack.removeAll(); redoStack.removeAll()
    }

    /// PNG bytes of the grayscale label mask, ready to write into the folder.
    func labelPNGData() -> Data? {
        var buffer = label   // mutable copy; the context's backing store must stay valid
        let cg: CGImage? = buffer.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(
                    data: base, width: width, height: height,
                    bitsPerComponent: 8, bytesPerRow: width,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                  ) else { return nil }
            return ctx.makeImage()   // snapshots the pixels, valid after the closure
        }
        guard let cg else { return nil }

        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil
        ) else { return nil }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return out as Data
    }

    var isEmpty: Bool { !label.contains { $0 != 0 } }

    // MARK: - Helpers

    private func rgbFromHex(_ hex: String) -> (r: UInt8, g: UInt8, b: UInt8) {
        let s = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        var v: UInt64 = 0
        Scanner(string: s).scanHexInt64(&v)
        return (UInt8((v >> 16) & 0xFF), UInt8((v >> 8) & 0xFF), UInt8(v & 0xFF))
    }
}
