import Foundation

/// Foreground and contact-boundary probabilities in one model-space grid.
struct SegmentationProbabilityMap {
    let width: Int
    let height: Int
    let foreground: [Float]
    let boundary: [Float]

    init(width: Int, height: Int, foreground: [Float], boundary: [Float]) {
        precondition(width > 0 && height > 0)
        precondition(foreground.count == width * height)
        precondition(boundary.count == width * height)
        self.width = width
        self.height = height
        self.foreground = foreground
        self.boundary = boundary
    }

    func resized(to size: Int) -> SegmentationProbabilityMap {
        guard width != size || height != size else { return self }
        return SegmentationProbabilityMap(
            width: size,
            height: size,
            foreground: Self.bilinear(
                foreground,
                sourceWidth: width,
                sourceHeight: height,
                targetWidth: size,
                targetHeight: size
            ),
            boundary: Self.bilinear(
                boundary,
                sourceWidth: width,
                sourceHeight: height,
                targetWidth: size,
                targetHeight: size
            )
        )
    }

    private static func bilinear(
        _ source: [Float],
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ) -> [Float] {
        var output = [Float](repeating: 0, count: targetWidth * targetHeight)
        let xScale = Double(sourceWidth) / Double(targetWidth)
        let yScale = Double(sourceHeight) / Double(targetHeight)
        for y in 0..<targetHeight {
            let sourceY = min(
                Double(sourceHeight - 1),
                max(0, (Double(y) + 0.5) * yScale - 0.5)
            )
            let y0 = Int(floor(sourceY))
            let y1 = min(sourceHeight - 1, y0 + 1)
            let yFraction = Float(sourceY - Double(y0))
            for x in 0..<targetWidth {
                let sourceX = min(
                    Double(sourceWidth - 1),
                    max(0, (Double(x) + 0.5) * xScale - 0.5)
                )
                let x0 = Int(floor(sourceX))
                let x1 = min(sourceWidth - 1, x0 + 1)
                let xFraction = Float(sourceX - Double(x0))
                let top = source[y0 * sourceWidth + x0] * (1 - xFraction)
                    + source[y0 * sourceWidth + x1] * xFraction
                let bottom = source[y1 * sourceWidth + x0] * (1 - xFraction)
                    + source[y1 * sourceWidth + x1] * xFraction
                output[y * targetWidth + x] = top * (1 - yFraction)
                    + bottom * yFraction
            }
        }
        return output
    }
}

enum SegmentationProbabilityFusion {
    static func merge(
        maps: [SegmentationProbabilityMap],
        weights: [Double],
        strategy: EnsembleMergeStrategy
    ) throws -> SegmentationProbabilityMap {
        guard !maps.isEmpty else {
            throw CountError.inferenceFailed("Select at least one trained model.")
        }
        let targetSize = maps.map(\.width).max() ?? maps[0].width
        let aligned = maps.map { $0.resized(to: targetSize) }
        switch strategy {
        case .weightedMean:
            return weightedMean(aligned, weights: weights)
        case .median:
            return orderStatistic(aligned, useMaximum: nil)
        case .maximum:
            return orderStatistic(aligned, useMaximum: true)
        case .minimum:
            return orderStatistic(aligned, useMaximum: false)
        }
    }

    private static func weightedMean(
        _ maps: [SegmentationProbabilityMap],
        weights suppliedWeights: [Double]
    ) -> SegmentationProbabilityMap {
        let count = maps[0].foreground.count
        var weights = maps.indices.map { index -> Double in
            guard index < suppliedWeights.count,
                  suppliedWeights[index].isFinite else { return 1 }
            return max(0, suppliedWeights[index])
        }
        var denominator = weights.reduce(0, +)
        if denominator <= 0 {
            weights = [Double](repeating: 1, count: maps.count)
            denominator = Double(maps.count)
        }
        var foreground = [Float](repeating: 0, count: count)
        var boundary = [Float](repeating: 0, count: count)
        for (modelIndex, map) in maps.enumerated() {
            let normalizedWeight = Float(weights[modelIndex] / denominator)
            for index in 0..<count {
                foreground[index] += map.foreground[index] * normalizedWeight
                boundary[index] += map.boundary[index] * normalizedWeight
            }
        }
        return SegmentationProbabilityMap(
            width: maps[0].width,
            height: maps[0].height,
            foreground: foreground,
            boundary: boundary
        )
    }

    /// Median when `useMaximum` is nil; otherwise the requested endpoint statistic.
    private static func orderStatistic(
        _ maps: [SegmentationProbabilityMap],
        useMaximum: Bool?
    ) -> SegmentationProbabilityMap {
        let count = maps[0].foreground.count
        var foreground = [Float](repeating: 0, count: count)
        var boundary = [Float](repeating: 0, count: count)
        var foregroundScratch = [Float]()
        var boundaryScratch = [Float]()
        foregroundScratch.reserveCapacity(maps.count)
        boundaryScratch.reserveCapacity(maps.count)
        for index in 0..<count {
            foregroundScratch.removeAll(keepingCapacity: true)
            boundaryScratch.removeAll(keepingCapacity: true)
            for map in maps {
                foregroundScratch.append(map.foreground[index])
                boundaryScratch.append(map.boundary[index])
            }
            if let useMaximum {
                foreground[index] = useMaximum
                    ? (foregroundScratch.max() ?? 0)
                    : (foregroundScratch.min() ?? 0)
                boundary[index] = useMaximum
                    ? (boundaryScratch.max() ?? 0)
                    : (boundaryScratch.min() ?? 0)
            } else {
                foregroundScratch.sort()
                boundaryScratch.sort()
                foreground[index] = median(foregroundScratch)
                boundary[index] = median(boundaryScratch)
            }
        }
        return SegmentationProbabilityMap(
            width: maps[0].width,
            height: maps[0].height,
            foreground: foreground,
            boundary: boundary
        )
    }

    private static func median(_ values: [Float]) -> Float {
        let middle = values.count / 2
        if values.count.isMultiple(of: 2) {
            return (values[middle - 1] + values[middle]) / 2
        }
        return values[middle]
    }
}

struct AutomaticBoundaryCutoffSelection {
    let threshold: Double
    let stabilityAUC: Double
    /// Nil means the heuristic had insufficient evidence and used the preferred fallback.
    let heuristicStabilityScore: Double?
    let usedFallback: Bool
    let fallbackReason: BoundaryCutoffFallbackReason?
    let candidates: [BoundaryCutoffCandidate]
}

/// Selects a boundary cutoff from an unlabeled image by partition stability and plausibility.
/// This is deliberately called stability AUC: true accuracy AUC requires ground-truth masks.
enum AutomaticBoundaryCutoffSelector {
    static let algorithmVersion = "boundary-stability-v2"

    static func select(
        foreground: [Bool],
        foregroundProbability: [Float],
        boundaryProbability: [Float],
        width: Int,
        height: Int,
        referenceThreshold: Double,
        sourcePixelsPerModelPixel: Double,
        minimumSourceArea: Int,
        maximumSourceArea: Int,
        labelsForThreshold: (Double) -> [Int]
    ) -> AutomaticBoundaryCutoffSelection {
        let thresholds = stride(from: 0.10, through: 0.9001, by: 0.05).map {
            (Double($0) * 100).rounded() / 100
        }
        var summaries = [(count: Int, quality: Double)]()
        summaries.reserveCapacity(thresholds.count)
        var pairStabilities = [Double]()
        pairStabilities.reserveCapacity(max(0, thresholds.count - 1))
        var previousLabels: [Int]?

        for threshold in thresholds {
            let rawLabels = labelsForThreshold(threshold)
            let labels = areaFiltered(
                rawLabels,
                sourcePixelsPerModelPixel: sourcePixelsPerModelPixel,
                minimumSourceArea: minimumSourceArea,
                maximumSourceArea: maximumSourceArea
            )
            let summary = summarize(
                labels: labels,
                foreground: foreground,
                foregroundProbability: foregroundProbability,
                boundaryProbability: boundaryProbability,
                width: width,
                height: height
            )
            summaries.append(summary)
            if let previousLabels {
                pairStabilities.append(partitionAgreement(previousLabels, labels))
            }
            previousLabels = labels
        }

        let stabilityAUC = pairStabilities.isEmpty
            ? 0
            : pairStabilities.reduce(0, +) / Double(pairStabilities.count)
        var candidates = [BoundaryCutoffCandidate]()
        candidates.reserveCapacity(thresholds.count)
        let reference = min(0.95, max(0.05, referenceThreshold))

        for index in thresholds.indices {
            let localStability: Double
            if pairStabilities.isEmpty {
                localStability = 0
            } else if index == 0 {
                localStability = pairStabilities[0]
            } else if index == thresholds.count - 1 {
                localStability = pairStabilities[pairStabilities.count - 1]
            } else {
                // The center of a plateau must be stable on both sides.
                localStability = min(
                    pairStabilities[index - 1],
                    pairStabilities[index]
                )
            }
            let prior = exp(-abs(thresholds[index] - reference) / 0.18)
            let hasObjects = summaries[index].count > 0
            let score = hasObjects
                ? 0.58 * localStability + 0.27 * summaries[index].quality + 0.15 * prior
                : 0
            candidates.append(BoundaryCutoffCandidate(
                threshold: thresholds[index],
                objectCount: summaries[index].count,
                localStability: localStability,
                qualityScore: summaries[index].quality,
                selectionScore: score
            ))
        }

        let choice = chooseStablePlateauCenter(
            candidates: candidates,
            reference: reference
        )
        let selectedIndex = choice.index
        let selected = selectedIndex.map { candidates[$0] }
        let partitionsUnchanged = !pairStabilities.isEmpty
            && pairStabilities.allSatisfy { $0 >= 0.999_999 }
        let heuristicStabilityScore = selected.map {
            min(1, max(0, 0.75 * $0.localStability + 0.25 * $0.qualityScore))
        } ?? 0
        // A single/unchanged partition contains no evidence for adapting the cutoff. Likewise,
        // an edge optimum or an unstable neighborhood is safer at the user's preferred prior.
        let fallbackReason: BoundaryCutoffFallbackReason?
        if selected == nil {
            fallbackReason = .noCandidate
        } else if selected?.objectCount ?? 0 <= 1 {
            fallbackReason = .insufficientObjects
        } else if partitionsUnchanged {
            fallbackReason = .unchangedPartition
        } else if selected?.localStability ?? 0 < 0.55 {
            fallbackReason = .unstablePartition
        } else if choice.plateauTouchesEdge
                    || selectedIndex == 0
                    || selectedIndex == candidates.count - 1 {
            fallbackReason = .edgeOptimum
        } else {
            fallbackReason = nil
        }
        let usedFallback = fallbackReason != nil
        return AutomaticBoundaryCutoffSelection(
            threshold: usedFallback ? reference : (selected?.threshold ?? reference),
            stabilityAUC: stabilityAUC,
            heuristicStabilityScore: usedFallback ? nil : heuristicStabilityScore,
            usedFallback: usedFallback,
            fallbackReason: fallbackReason,
            candidates: candidates
        )
    }

    /// Chooses the center of the widest contiguous near-best stable region. If there is no
    /// stable region, the scalar best candidate is returned so the fallback checks can explain
    /// why adaptation was rejected.
    private static func chooseStablePlateauCenter(
        candidates: [BoundaryCutoffCandidate],
        reference: Double
    ) -> (index: Int?, plateauTouchesEdge: Bool) {
        guard let scalarBest = candidates.indices.max(by: { left, right in
            let a = candidates[left]
            let b = candidates[right]
            if abs(a.selectionScore - b.selectionScore) > 0.000_001 {
                return a.selectionScore < b.selectionScore
            }
            return abs(a.threshold - reference) > abs(b.threshold - reference)
        }) else { return (nil, false) }

        let scoreFloor = max(0, candidates[scalarBest].selectionScore - 0.06)
        var plateaus = [ClosedRange<Int>]()
        var plateauStart: Int?
        for index in candidates.indices {
            let candidate = candidates[index]
            let eligible = candidate.objectCount > 1
                && candidate.localStability >= 0.55
                && candidate.selectionScore >= scoreFloor
            if eligible {
                if plateauStart == nil { plateauStart = index }
            } else if let start = plateauStart {
                plateaus.append(start...(index - 1))
                plateauStart = nil
            }
        }
        if let start = plateauStart, let last = candidates.indices.last {
            plateaus.append(start...last)
        }
        guard let plateau = plateaus.max(by: { left, right in
            let leftWidth = left.upperBound - left.lowerBound + 1
            let rightWidth = right.upperBound - right.lowerBound + 1
            if leftWidth != rightWidth { return leftWidth < rightWidth }
            let leftMean = left.reduce(0.0) {
                $0 + candidates[$1].selectionScore
            } / Double(leftWidth)
            let rightMean = right.reduce(0.0) {
                $0 + candidates[$1].selectionScore
            } / Double(rightWidth)
            if abs(leftMean - rightMean) > 0.000_001 { return leftMean < rightMean }
            let leftCenter = centerIndex(of: left, candidates: candidates, reference: reference)
            let rightCenter = centerIndex(of: right, candidates: candidates, reference: reference)
            return abs(candidates[leftCenter].threshold - reference)
                > abs(candidates[rightCenter].threshold - reference)
        }) else {
            return (scalarBest, false)
        }
        let center = centerIndex(
            of: plateau,
            candidates: candidates,
            reference: reference
        )
        return (
            center,
            plateau.lowerBound == candidates.indices.first
                || plateau.upperBound == candidates.indices.last
        )
    }

    private static func centerIndex(
        of plateau: ClosedRange<Int>,
        candidates: [BoundaryCutoffCandidate],
        reference: Double
    ) -> Int {
        let lowerCenter = (plateau.lowerBound + plateau.upperBound) / 2
        let upperCenter = (plateau.lowerBound + plateau.upperBound + 1) / 2
        return abs(candidates[lowerCenter].threshold - reference)
            <= abs(candidates[upperCenter].threshold - reference)
            ? lowerCenter
            : upperCenter
    }

    private static func areaFiltered(
        _ labels: [Int],
        sourcePixelsPerModelPixel: Double,
        minimumSourceArea: Int,
        maximumSourceArea: Int
    ) -> [Int] {
        let maximumLabel = labels.max() ?? 0
        guard maximumLabel > 0 else { return labels }
        var areas = [Int](repeating: 0, count: maximumLabel + 1)
        for label in labels where label > 0 { areas[label] += 1 }
        var remap = [Int](repeating: 0, count: maximumLabel + 1)
        var next = 0
        for label in 1...maximumLabel {
            let estimatedSourceArea = Double(areas[label]) * sourcePixelsPerModelPixel
            guard estimatedSourceArea >= Double(minimumSourceArea),
                  maximumSourceArea == 0
                    || estimatedSourceArea <= Double(maximumSourceArea) else { continue }
            next += 1
            remap[label] = next
        }
        return labels.map { $0 > 0 ? remap[$0] : 0 }
    }

    private static func summarize(
        labels: [Int],
        foreground: [Bool],
        foregroundProbability: [Float],
        boundaryProbability: [Float],
        width: Int,
        height: Int
    ) -> (count: Int, quality: Double) {
        let count = labels.max() ?? 0
        guard count > 0 else { return (0, 0) }
        var labeledPixels = 0
        var foregroundSum = 0.0
        var interfaceBoundarySum = 0.0
        var interfacePixels = 0
        var internalBoundarySum = 0.0
        var internalPixels = 0
        for index in labels.indices where labels[index] > 0 {
            labeledPixels += 1
            foregroundSum += Double(foregroundProbability[index])
            let label = labels[index]
            let x = index % width
            let y = index / width
            var touchesDifferentCell = false
            if x > 0, labels[index - 1] > 0, labels[index - 1] != label {
                touchesDifferentCell = true
            }
            if x + 1 < width, labels[index + 1] > 0, labels[index + 1] != label {
                touchesDifferentCell = true
            }
            if y > 0, labels[index - width] > 0, labels[index - width] != label {
                touchesDifferentCell = true
            }
            if y + 1 < height, labels[index + width] > 0, labels[index + width] != label {
                touchesDifferentCell = true
            }
            if touchesDifferentCell {
                interfacePixels += 1
                interfaceBoundarySum += Double(boundaryProbability[index])
            } else if x > 0, x + 1 < width, y > 0, y + 1 < height,
                      labels[index - 1] == label,
                      labels[index + 1] == label,
                      labels[index - width] == label,
                      labels[index + width] == label {
                internalPixels += 1
                internalBoundarySum += Double(boundaryProbability[index])
            }
        }
        let foregroundPixels = max(1, foreground.reduce(0) { $0 + ($1 ? 1 : 0) })
        let retention = min(1, Double(labeledPixels) / Double(foregroundPixels))
        let meanForeground = foregroundSum / Double(max(1, labeledPixels))
        let internalBoundary = internalBoundarySum / Double(max(1, internalPixels))
        let boundaryContrast: Double
        if interfacePixels > 0 {
            let interfaceBoundary = interfaceBoundarySum / Double(interfacePixels)
            boundaryContrast = min(
                1,
                max(0, 0.5 + 0.5 * (interfaceBoundary - internalBoundary))
            )
        } else {
            // Isolated cells provide no cutoff-specific split-ridge evidence.
            boundaryContrast = 0.5
        }
        let quality = min(1, max(0,
            0.45 * meanForeground + 0.35 * boundaryContrast + 0.20 * retention
        ))
        return (count, quality)
    }

    private static func partitionAgreement(_ first: [Int], _ second: [Int]) -> Double {
        guard first.count == second.count else { return 0 }
        let firstMaximum = first.max() ?? 0
        let secondMaximum = second.max() ?? 0
        if firstMaximum == 0 && secondMaximum == 0 { return 1 }
        if firstMaximum == 0 || secondMaximum == 0 { return 0 }

        var firstArea = [Int](repeating: 0, count: firstMaximum + 1)
        var secondArea = [Int](repeating: 0, count: secondMaximum + 1)
        var overlaps = [UInt64: Int]()
        for index in first.indices {
            let a = first[index]
            let b = second[index]
            if a > 0 { firstArea[a] += 1 }
            if b > 0 { secondArea[b] += 1 }
            if a > 0 && b > 0 {
                let key = (UInt64(UInt32(a)) << 32) | UInt64(UInt32(b))
                overlaps[key, default: 0] += 1
            }
        }

        var bestFirst = [Double](repeating: 0, count: firstMaximum + 1)
        var bestSecond = [Double](repeating: 0, count: secondMaximum + 1)
        for (key, overlap) in overlaps {
            let a = Int(UInt32(key >> 32))
            let b = Int(UInt32(key & 0xffff_ffff))
            let union = firstArea[a] + secondArea[b] - overlap
            guard union > 0 else { continue }
            let iou = Double(overlap) / Double(union)
            bestFirst[a] = max(bestFirst[a], iou)
            bestSecond[b] = max(bestSecond[b], iou)
        }
        let firstTotal = max(1, firstArea.reduce(0, +))
        let secondTotal = max(1, secondArea.reduce(0, +))
        let firstMatch = zip(firstArea, bestFirst).reduce(0.0) {
            $0 + Double($1.0) * $1.1
        } / Double(firstTotal)
        let secondMatch = zip(secondArea, bestSecond).reduce(0.0) {
            $0 + Double($1.0) * $1.1
        } / Double(secondTotal)
        let labelCountAgreement = 1 - Double(abs(firstMaximum - secondMaximum))
            / Double(max(firstMaximum, secondMaximum))
        return min(1, max(0, 0.85 * (firstMatch + secondMatch) / 2
            + 0.15 * labelCountAgreement))
    }
}
