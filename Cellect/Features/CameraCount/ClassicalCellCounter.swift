import CoreGraphics
import Foundation

/// The always-available counting floor: Otsu threshold → 8-connected components → size stats.
/// Works on the CGImage as given (cap resolution upstream, as the segmentation flow does, so the
/// returned label mask lines up with the image you save). No model, no network.
///
/// Known limitation: touching cells that share a blob are counted as one. Watershed splitting via
/// the distance transform is the planned next refinement (see docs/ARCHITECTURE.md §6b).
struct ClassicalCellCounter: CellCounter {
    let tier: CounterTier = .classical
    let displayName = "Classical (Otsu + CCL)"

    func count(in image: CGImage, options: CountOptions) async throws -> CountResult {
        let w = image.width, h = image.height
        guard w > 0, h > 0 else { throw CountError.decodeFailed }
        let gray = try grayscale(image, width: w, height: h)

        // Otsu threshold, then binarize by polarity.
        let t = otsuThreshold(gray)
        var fg = [Bool](repeating: false, count: w * h)
        switch options.polarity {
        case .darkObjects:   for i in 0..<gray.count { fg[i] = gray[i] < t }
        case .brightObjects: for i in 0..<gray.count { fg[i] = gray[i] > t }
        }

        // Connected components (8-connectivity).
        let (labels, n) = connectedComponents(fg, width: w, height: h)

        // Accumulate per-label stats.
        var area = [Int](repeating: 0, count: n + 1)
        var sumX = [Int](repeating: 0, count: n + 1)
        var sumY = [Int](repeating: 0, count: n + 1)
        var minX = [Int](repeating: Int.max, count: n + 1)
        var minY = [Int](repeating: Int.max, count: n + 1)
        var maxX = [Int](repeating: 0, count: n + 1)
        var maxY = [Int](repeating: 0, count: n + 1)
        for y in 0..<h {
            let row = y * w
            for x in 0..<w {
                let l = labels[row + x]
                if l == 0 { continue }
                area[l] += 1
                sumX[l] += x; sumY[l] += y
                if x < minX[l] { minX[l] = x }; if x > maxX[l] { maxX[l] = x }
                if y < minY[l] { minY[l] = y }; if y > maxY[l] { maxY[l] = y }
            }
        }

        // Keep labels above the min-area threshold; relabel sequentially 1..k.
        var remap = [Int](repeating: 0, count: n + 1)
        var objects: [DetectedObject] = []
        var nextId = 0
        for l in 1...max(1, n) where l <= n {
            guard area[l] >= options.minAreaPixels else { continue }
            nextId += 1
            remap[l] = nextId
            let a = area[l]
            objects.append(DetectedObject(
                id: nextId,
                areaPixels: a,
                centroid: CGPoint(x: Double(sumX[l]) / Double(a), y: Double(sumY[l]) / Double(a)),
                bbox: CGRect(x: minX[l], y: minY[l],
                             width: maxX[l] - minX[l] + 1, height: maxY[l] - minY[l] + 1)
            ))
        }

        // Build the final 16-bit instance label mask with the compacted ids.
        var labelMask = [UInt16](repeating: 0, count: w * h)
        for i in 0..<labelMask.count {
            let l = labels[i]
            if l != 0 { labelMask[i] = UInt16(truncatingIfNeeded: remap[l]) }
        }

        return CountResult(
            objects: objects,
            imageSize: CGSize(width: w, height: h),
            labelMask: labelMask,
            micronsPerPixel: options.micronsPerPixel
        )
    }

    // MARK: - Pixels

    private func grayscale(_ image: CGImage, width w: Int, height h: Int) throws -> [UInt8] {
        var gray = [UInt8](repeating: 0, count: w * h)
        let ok: Bool = gray.withUnsafeMutableBytes { raw in
            guard let base = raw.baseAddress,
                  let ctx = CGContext(
                    data: base, width: w, height: h,
                    bitsPerComponent: 8, bytesPerRow: w,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                  ) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw CountError.decodeFailed }
        return gray
    }

    private func otsuThreshold(_ gray: [UInt8]) -> UInt8 {
        var hist = [Int](repeating: 0, count: 256)
        for g in gray { hist[Int(g)] += 1 }
        let total = gray.count
        var sum = 0.0
        for i in 0..<256 { sum += Double(i) * Double(hist[i]) }
        var sumB = 0.0, wB = 0, best = 0.0
        var threshold = 127
        for i in 0..<256 {
            wB += hist[i]
            if wB == 0 { continue }
            let wF = total - wB
            if wF == 0 { break }
            sumB += Double(i) * Double(hist[i])
            let mB = sumB / Double(wB)
            let mF = (sum - sumB) / Double(wF)
            let between = Double(wB) * Double(wF) * (mB - mF) * (mB - mF)
            if between > best { best = between; threshold = i }
        }
        return UInt8(threshold)
    }

    // MARK: - Connected components (two-pass union-find, 8-connectivity)

    private func connectedComponents(_ fg: [Bool], width w: Int, height h: Int) -> ([Int], Int) {
        var parent = [0]                       // index 0 = background sentinel
        func find(_ x: Int) -> Int {
            var x = x
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x] }
            return x
        }
        func union(_ a: Int, _ b: Int) {
            let ra = find(a), rb = find(b)
            if ra != rb { parent[max(ra, rb)] = min(ra, rb) }
        }

        var labels = [Int](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = y * w
            for x in 0..<w {
                let i = row + x
                if !fg[i] { continue }
                var best = 0
                func consider(_ nx: Int, _ ny: Int) {
                    guard nx >= 0, nx < w, ny >= 0 else { return }
                    let l = labels[ny * w + nx]
                    guard l > 0 else { return }
                    if best == 0 { best = l } else if l != best { union(best, l); best = find(best) }
                }
                consider(x - 1, y)       // W
                consider(x - 1, y - 1)   // NW
                consider(x, y - 1)       // N
                consider(x + 1, y - 1)   // NE
                if best == 0 {
                    let newLabel = parent.count
                    parent.append(newLabel)
                    labels[i] = newLabel
                } else {
                    labels[i] = best
                }
            }
        }

        // Second pass: flatten to roots and compact to 1..n.
        var remap = [Int: Int]()
        var n = 0
        for i in 0..<labels.count {
            let l = labels[i]
            if l == 0 { continue }
            let r = find(l)
            if let id = remap[r] {
                labels[i] = id
            } else {
                n += 1; remap[r] = n; labels[i] = n
            }
        }
        return (labels, n)
    }
}
