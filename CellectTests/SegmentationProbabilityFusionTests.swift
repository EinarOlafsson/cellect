import XCTest
@testable import Cellect

final class SegmentationProbabilityFusionTests: XCTestCase {
    func testWeightedMeanNormalizesWeights() throws {
        let low = constantMap(foreground: 0.2, boundary: 0.8)
        let high = constantMap(foreground: 0.4, boundary: 0.2)

        let result = try SegmentationProbabilityFusion.merge(
            maps: [low, high],
            weights: [1, 3],
            strategy: .weightedMean
        )

        XCTAssertEqual(result.foreground[0], 0.35, accuracy: 0.0001)
        XCTAssertEqual(result.boundary[0], 0.35, accuracy: 0.0001)
    }

    func testMedianMaximumAndMinimum() throws {
        let low = constantMap(foreground: 0.2, boundary: 0.8)
        let high = constantMap(foreground: 0.4, boundary: 0.2)

        let median = try SegmentationProbabilityFusion.merge(
            maps: [low, high], weights: [], strategy: .median
        )
        let maximum = try SegmentationProbabilityFusion.merge(
            maps: [low, high], weights: [], strategy: .maximum
        )
        let minimum = try SegmentationProbabilityFusion.merge(
            maps: [low, high], weights: [], strategy: .minimum
        )

        XCTAssertEqual(median.foreground[0], 0.3, accuracy: 0.0001)
        XCTAssertEqual(maximum.foreground[0], 0.4, accuracy: 0.0001)
        XCTAssertEqual(maximum.boundary[0], 0.8, accuracy: 0.0001)
        XCTAssertEqual(minimum.foreground[0], 0.2, accuracy: 0.0001)
        XCTAssertEqual(minimum.boundary[0], 0.2, accuracy: 0.0001)
    }

    func testFlatPartitionUsesFallbackWithoutConfidence() {
        let labels = [
            1, 1, 0,
            1, 1, 0,
            0, 0, 2,
        ]

        let selection = select(
            labelsForThreshold: { _ in labels },
            pixelCount: labels.count,
            width: 3,
            height: 3
        )

        XCTAssertTrue(selection.usedFallback)
        XCTAssertEqual(selection.fallbackReason, .unchangedPartition)
        XCTAssertNil(selection.heuristicStabilityScore)
        XCTAssertEqual(selection.threshold, 0.45, accuracy: 0.0001)
        XCTAssertEqual(selection.candidates.count, 17)
    }

    func testStableTwoObjectPlateauAdapts() {
        let twoCells = [
            1, 1, 2, 2,
            1, 1, 2, 2,
            1, 1, 2, 2,
            1, 1, 2, 2,
        ]
        let fragmented = [
            1, 1, 2, 2,
            1, 1, 2, 2,
            3, 3, 4, 4,
            3, 3, 4, 4,
        ]
        let merged = [Int](repeating: 1, count: 16)

        let selection = select(
            labelsForThreshold: { threshold in
                if threshold < 0.2 { return fragmented }
                if threshold > 0.7 { return merged }
                return twoCells
            },
            pixelCount: 16,
            width: 4,
            height: 4
        )

        XCTAssertFalse(selection.usedFallback)
        XCTAssertNotNil(selection.heuristicStabilityScore)
        let selectedCandidate = selection.candidates.first {
            abs($0.threshold - selection.threshold) < 0.0001
        }
        XCTAssertEqual(selectedCandidate?.objectCount, 2)
    }

    func testEqualCountsDoNotImplyUnchangedPartition() {
        let vertical = [
            1, 1, 2, 2,
            1, 1, 2, 2,
            1, 1, 2, 2,
            1, 1, 2, 2,
        ]
        let horizontal = [
            1, 1, 1, 1,
            1, 1, 1, 1,
            2, 2, 2, 2,
            2, 2, 2, 2,
        ]

        let selection = select(
            labelsForThreshold: { $0 < 0.5 ? vertical : horizontal },
            pixelCount: 16,
            width: 4,
            height: 4
        )

        XCTAssertNotEqual(selection.fallbackReason, .unchangedPartition)
    }

    private func constantMap(
        foreground: Float,
        boundary: Float
    ) -> SegmentationProbabilityMap {
        SegmentationProbabilityMap(
            width: 2,
            height: 2,
            foreground: [Float](repeating: foreground, count: 4),
            boundary: [Float](repeating: boundary, count: 4)
        )
    }

    private func select(
        labelsForThreshold: @escaping (Double) -> [Int],
        pixelCount: Int,
        width: Int,
        height: Int
    ) -> AutomaticBoundaryCutoffSelection {
        AutomaticBoundaryCutoffSelector.select(
            foreground: [Bool](repeating: true, count: pixelCount),
            foregroundProbability: [Float](repeating: 0.9, count: pixelCount),
            boundaryProbability: [Float](repeating: 0.5, count: pixelCount),
            width: width,
            height: height,
            referenceThreshold: 0.45,
            sourcePixelsPerModelPixel: 1,
            minimumSourceArea: 0,
            maximumSourceArea: 0,
            labelsForThreshold: labelsForThreshold
        )
    }
}
