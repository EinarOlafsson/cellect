import CoreGraphics
import CoreML
import Foundation

/// Runs one workstation-trained foreground/boundary model.
struct CoreMLCellCounter: CellCounter {
    let spec: CoreMLCounterSpec
    var identifier: String { spec.id }
    var tier: CounterTier { spec.tier }
    var displayName: String { spec.displayName }

    func count(in image: CGImage, options: CountOptions) async throws -> CountResult {
        try await Task.detached(priority: .userInitiated) {
            try CoreMLSegmentation.countSingle(in: image, spec: spec, options: options)
        }.value
    }
}

/// Runs selected models sequentially, fuses probability maps, then reconstructs instances once.
struct CoreMLEnsembleCellCounter: CellCounter {
    let specs: [CoreMLCounterSpec]
    let modelWeights: [String: Double]
    let mergeStrategy: EnsembleMergeStrategy

    var identifier: String {
        let members = specs.map(\.id).sorted().joined(separator: "+")
        return "ensemble:\(mergeStrategy.rawValue):\(members)"
    }

    var tier: CounterTier { specs.map(\.tier).max() ?? .classical }
    var displayName: String { "Ensemble · \(specs.count) models · \(mergeStrategy.title)" }

    func count(in image: CGImage, options: CountOptions) async throws -> CountResult {
        try await Task.detached(priority: .userInitiated) {
            try CoreMLSegmentation.countEnsemble(
                in: image,
                specs: specs,
                modelWeights: modelWeights,
                mergeStrategy: mergeStrategy,
                options: options
            )
        }.value
    }
}

enum CoreMLSegmentation {
    private static let cache = CoreMLModelCache()

    /// Runs the expensive neural-network stage once for a model-comparison group. The returned
    /// probabilities can be postprocessed repeatedly without loading or predicting again.
    static func predictForComparison(
        in image: CGImage,
        spec: CoreMLCounterSpec,
        options: CountOptions
    ) throws -> SegmentationProbabilityMap {
        try predict(in: image, spec: spec, options: options)
    }

    /// Reconstructs one exact mask from a probability map retained only for the current model.
    static func postprocessForComparison(
        _ probabilities: SegmentationProbabilityMap,
        sourceWidth: Int,
        sourceHeight: Int,
        spec: CoreMLCounterSpec,
        options: CountOptions
    ) -> CountResult {
        postprocess(
            probabilities,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            modelIdentifiers: [spec.id],
            modelExportVersions: [spec.id: spec.exportVersion],
            modelSourceHashes: [spec.id: spec.sourceTorchScriptSHA256],
            requestedWeights: [spec.id: 1],
            normalizedWeights: [spec.id: 1],
            mergeStrategy: nil,
            options: options
        )
    }

    /// Comparison processes models serially; evicting between them avoids a transient peak where
    /// two large Core ML networks are resident at once.
    static func evictComparisonModel() {
        cache.evict()
    }

    static func countSingle(
        in image: CGImage,
        spec: CoreMLCounterSpec,
        options: CountOptions
    ) throws -> CountResult {
        let probabilities = try predict(in: image, spec: spec, options: options)
        return postprocess(
            probabilities,
            sourceWidth: image.width,
            sourceHeight: image.height,
            modelIdentifiers: [spec.id],
            modelExportVersions: [spec.id: spec.exportVersion],
            modelSourceHashes: [spec.id: spec.sourceTorchScriptSHA256],
            requestedWeights: [spec.id: 1],
            normalizedWeights: [spec.id: 1],
            mergeStrategy: nil,
            options: options
        )
    }

    static func countEnsemble(
        in image: CGImage,
        specs: [CoreMLCounterSpec],
        modelWeights: [String: Double],
        mergeStrategy: EnsembleMergeStrategy,
        options: CountOptions
    ) throws -> CountResult {
        guard specs.count >= 2 else {
            throw CountError.inferenceFailed("Select at least two trained models for an ensemble.")
        }
        defer { cache.evict() }
        var maps = [SegmentationProbabilityMap]()
        maps.reserveCapacity(specs.count)
        var orderedWeights = [Double]()
        orderedWeights.reserveCapacity(specs.count)
        for spec in specs {
            try Task.checkCancellation()
            // Release the previous large model before loading the next one.
            cache.evict()
            let map: SegmentationProbabilityMap = try autoreleasepool {
                try predict(in: image, spec: spec, options: options)
            }
            maps.append(map)
            orderedWeights.append(max(0, modelWeights[spec.id] ?? 1))
        }
        let fused = try SegmentationProbabilityFusion.merge(
            maps: maps,
            weights: orderedWeights,
            strategy: mergeStrategy
        )
        let requestedWeights = mergeStrategy.usesModelWeights
            ? Dictionary(uniqueKeysWithValues: specs.map {
                ($0.id, max(0, modelWeights[$0.id] ?? 1))
            })
            : [:]
        let normalizedWeights = resolvedWeights(
            identifiers: specs.map(\.id),
            requested: requestedWeights,
            strategy: mergeStrategy
        )
        return postprocess(
            fused,
            sourceWidth: image.width,
            sourceHeight: image.height,
            modelIdentifiers: specs.map(\.id),
            modelExportVersions: Dictionary(uniqueKeysWithValues: specs.map {
                ($0.id, $0.exportVersion)
            }),
            modelSourceHashes: Dictionary(uniqueKeysWithValues: specs.map {
                ($0.id, $0.sourceTorchScriptSHA256)
            }),
            requestedWeights: requestedWeights,
            normalizedWeights: normalizedWeights,
            mergeStrategy: mergeStrategy,
            options: options
        )
    }

    private static func resolvedWeights(
        identifiers: [String],
        requested: [String: Double],
        strategy: EnsembleMergeStrategy
    ) -> [String: Double] {
        guard !identifiers.isEmpty else { return [:] }
        if !strategy.usesModelWeights { return [:] }
        let denominator = identifiers.reduce(0) { partial, identifier in
            partial + max(0, requested[identifier] ?? 1)
        }
        if denominator <= 0 {
            let uniform = 1 / Double(identifiers.count)
            return Dictionary(uniqueKeysWithValues: identifiers.map { ($0, uniform) })
        }
        return Dictionary(uniqueKeysWithValues: identifiers.map {
            ($0, max(0, requested[$0] ?? 1) / denominator)
        })
    }

    // MARK: - Prediction

    private static func predict(
        in image: CGImage,
        spec: CoreMLCounterSpec,
        options: CountOptions
    ) throws -> SegmentationProbabilityMap {
        guard image.width > 0, image.height > 0 else { throw CountError.decodeFailed }
        let input = try makeInput(
            image: image,
            size: spec.inputSize,
            inverted: options.invertModelInput
        )
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(multiArray: input)
        ])
        let prediction: MLFeatureProvider
        do {
            prediction = try cache.model(
                for: spec,
                computeMode: options.computeMode
            ).prediction(from: provider)
        } catch {
            throw CountError.inferenceFailed(error.localizedDescription)
        }
        guard let logits = prediction.featureValue(
            for: "segmentation_logits"
        )?.multiArrayValue else {
            throw CountError.inferenceFailed("The model returned no segmentation logits.")
        }
        let modelSize = spec.inputSize
        guard logits.shape.map(\.intValue) == [1, 2, modelSize, modelSize] else {
            throw CountError.inferenceFailed(
                "Unexpected model output shape \(logits.shape.map(\.intValue))."
            )
        }
        let values = MultiArrayReader(logits)
        var foregroundProbability = [Float](repeating: 0, count: modelSize * modelSize)
        var boundaryProbability = [Float](repeating: 0, count: modelSize * modelSize)
        for y in 0..<modelSize {
            for x in 0..<modelSize {
                let index = y * modelSize + x
                foregroundProbability[index] = probability(
                    values.value(channel: 0, y: y, x: x)
                )
                boundaryProbability[index] = probability(
                    values.value(channel: 1, y: y, x: x)
                )
            }
        }
        return SegmentationProbabilityMap(
            width: modelSize,
            height: modelSize,
            foreground: foregroundProbability,
            boundary: boundaryProbability
        )
    }

    // MARK: - Shared postprocessing

    private static func postprocess(
        _ probabilities: SegmentationProbabilityMap,
        sourceWidth: Int,
        sourceHeight: Int,
        modelIdentifiers: [String],
        modelExportVersions: [String: String],
        modelSourceHashes: [String: String],
        requestedWeights: [String: Double],
        normalizedWeights: [String: Double],
        mergeStrategy: EnsembleMergeStrategy?,
        options: CountOptions
    ) -> CountResult {
        let width = probabilities.width
        let height = probabilities.height
        var foreground = probabilities.foreground.map {
            $0 >= Float(options.foregroundProbabilityThreshold)
        }
        for _ in 0..<max(0, options.openingIterations) {
            foreground = open(foreground, width: width, height: height)
        }
        for _ in 0..<max(0, options.closingIterations) {
            foreground = close(foreground, width: width, height: height)
        }

        var selectedThreshold: Double?
        var stabilityAUC: Double?
        var automaticHeuristicStabilityScore: Double?
        var automaticFallback = false
        var automaticFallbackReason: BoundaryCutoffFallbackReason?
        var candidates = [BoundaryCutoffCandidate]()
        let labels: [Int]
        if options.separateTouchingCells {
            let rawLabelsForThreshold: (Double) -> [Int] = { threshold in
                separatedLabels(
                    foreground: foreground,
                    boundaryProbability: probabilities.boundary,
                    threshold: threshold,
                    width: width,
                    height: height
                )
            }
            if options.automaticBoundaryThreshold {
                let selection = AutomaticBoundaryCutoffSelector.select(
                    foreground: foreground,
                    foregroundProbability: probabilities.foreground,
                    boundaryProbability: probabilities.boundary,
                    width: width,
                    height: height,
                    referenceThreshold: options.boundaryProbabilityThreshold,
                    sourcePixelsPerModelPixel:
                        Double(sourceWidth * sourceHeight) / Double(width * height),
                    minimumSourceArea: options.minAreaPixels,
                    maximumSourceArea: options.maxAreaPixels,
                    labelsForThreshold: rawLabelsForThreshold
                )
                selectedThreshold = selection.threshold
                stabilityAUC = selection.stabilityAUC
                automaticHeuristicStabilityScore = selection.heuristicStabilityScore
                automaticFallback = selection.usedFallback
                automaticFallbackReason = selection.fallbackReason
                candidates = selection.candidates
                labels = filterByModelConfidence(
                    labels: rawLabelsForThreshold(selection.threshold),
                    foregroundProbability: probabilities.foreground,
                    boundaryProbability: probabilities.boundary,
                    width: width,
                    height: height,
                    options: options
                )
            } else {
                selectedThreshold = options.boundaryProbabilityThreshold
                labels = filterByModelConfidence(
                    labels: rawLabelsForThreshold(options.boundaryProbabilityThreshold),
                    foregroundProbability: probabilities.foreground,
                    boundaryProbability: probabilities.boundary,
                    width: width,
                    height: height,
                    options: options
                )
            }
        } else {
            let components = connectedComponents(
                foreground,
                width: width,
                height: height
            ).0
            labels = filterByModelConfidence(
                labels: components,
                foregroundProbability: probabilities.foreground,
                boundaryProbability: probabilities.boundary,
                width: width,
                height: height,
                options: options
            )
        }

        let diagnostics = CountDiagnostics(
            modelIdentifiers: modelIdentifiers,
            modelExportVersions: modelExportVersions,
            modelSourceTorchScriptSHA256: modelSourceHashes,
            requestedModelWeights: requestedWeights,
            normalizedModelWeights: normalizedWeights,
            ensembleMergeStrategy: mergeStrategy,
            probabilityGridWidth: width,
            probabilityGridHeight: height,
            probabilityResamplingMethod: mergeStrategy == nil
                ? "none"
                : "bilinear-pixel-center-v1",
            postprocessingAlgorithmVersion: "coreml-instance-fifo-v1",
            automaticBoundarySelectionUsed:
                options.separateTouchingCells && options.automaticBoundaryThreshold,
            boundarySelectionAlgorithmVersion:
                options.separateTouchingCells && options.automaticBoundaryThreshold
                    ? AutomaticBoundaryCutoffSelector.algorithmVersion
                    : nil,
            automaticBoundaryFallbackUsed: automaticFallback,
            automaticBoundaryFallbackReason: automaticFallbackReason,
            automaticBoundaryHeuristicStabilityScore: automaticHeuristicStabilityScore,
            selectedBoundaryProbabilityThreshold: selectedThreshold,
            boundaryStabilityAUC: stabilityAUC,
            boundaryCandidates: candidates
        )
        return measureAtSourceResolution(
            labels: labels,
            modelWidth: width,
            modelHeight: height,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            diagnostics: diagnostics,
            options: options
        )
    }

    // MARK: - Model input

    private static func makeInput(
        image: CGImage,
        size: Int,
        inverted: Bool
    ) throws -> MLMultiArray {
        var gray = [UInt8](repeating: 0, count: size * size)
        let drewImage: Bool = gray.withUnsafeMutableBytes { bytes in
            guard let base = bytes.baseAddress,
                  let context = CGContext(
                    data: base,
                    width: size,
                    height: size,
                    bitsPerComponent: 8,
                    bytesPerRow: size,
                    space: CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue
                  ) else { return false }
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))
            return true
        }
        guard drewImage else { throw CountError.decodeFailed }

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: size), NSNumber(value: size)],
            dataType: .float32
        )
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let channelStride = array.strides[1].intValue
        let rowStride = array.strides[2].intValue
        let columnStride = array.strides[3].intValue
        for y in 0..<size {
            for x in 0..<size {
                let normalized = Float(gray[y * size + x]) / 255.0
                let value = inverted ? 1 - normalized : normalized
                for channel in 0..<3 {
                    pointer[channel * channelStride + y * rowStride + x * columnStride] = value
                }
            }
        }
        return array
    }

    private static func probability(_ logit: Float) -> Float {
        if logit >= 0 { return 1 / (1 + exp(-logit)) }
        let exponential = exp(logit)
        return exponential / (1 + exponential)
    }

    // MARK: - Instance separation

    private static func separatedLabels(
        foreground: [Bool],
        boundaryProbability: [Float],
        threshold: Double,
        width: Int,
        height: Int
    ) -> [Int] {
        var interiors = [Bool](repeating: false, count: foreground.count)
        for index in interiors.indices {
            interiors[index] = foreground[index]
                && boundaryProbability[index] < Float(threshold)
        }
        var (labels, markerCount) = connectedComponents(
            interiors,
            width: width,
            height: height
        )
        if markerCount == 0 {
            return connectedComponents(foreground, width: width, height: height).0
        }
        expandMarkers(
            labels: &labels,
            markerCount: &markerCount,
            through: foreground,
            width: width,
            height: height
        )
        return labels
    }

    private static func open(_ pixels: [Bool], width: Int, height: Int) -> [Bool] {
        dilate(erode(pixels, width: width, height: height), width: width, height: height)
    }

    private static func close(_ pixels: [Bool], width: Int, height: Int) -> [Bool] {
        erode(dilate(pixels, width: width, height: height), width: width, height: height)
    }

    private static func erode(_ pixels: [Bool], width: Int, height: Int) -> [Bool] {
        guard width >= 3, height >= 3 else { return pixels }
        var output = [Bool](repeating: false, count: pixels.count)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var keep = true
                for dy in -1...1 {
                    for dx in -1...1 where !pixels[(y + dy) * width + x + dx] {
                        keep = false
                    }
                }
                output[y * width + x] = keep
            }
        }
        return output
    }

    private static func dilate(_ pixels: [Bool], width: Int, height: Int) -> [Bool] {
        guard width >= 3, height >= 3 else { return pixels }
        var output = [Bool](repeating: false, count: pixels.count)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                var value = false
                for dy in -1...1 {
                    for dx in -1...1 where pixels[(y + dy) * width + x + dx] {
                        value = true
                    }
                }
                output[y * width + x] = value
            }
        }
        return output
    }

    private static func connectedComponents(
        _ pixels: [Bool],
        width: Int,
        height: Int
    ) -> ([Int], Int) {
        var labels = [Int](repeating: 0, count: pixels.count)
        var queue = [Int]()
        queue.reserveCapacity(pixels.count / 4)
        var nextLabel = 0
        for start in pixels.indices where pixels[start] && labels[start] == 0 {
            nextLabel += 1
            labels[start] = nextLabel
            queue.removeAll(keepingCapacity: true)
            queue.append(start)
            var head = 0
            while head < queue.count {
                let index = queue[head]
                head += 1
                let x = index % width
                let y = index / width
                for dy in -1...1 {
                    for dx in -1...1 where dx != 0 || dy != 0 {
                        let nx = x + dx
                        let ny = y + dy
                        guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                        let neighbor = ny * width + nx
                        if pixels[neighbor] && labels[neighbor] == 0 {
                            labels[neighbor] = nextLabel
                            queue.append(neighbor)
                        }
                    }
                }
            }
        }
        return (labels, nextLabel)
    }

    /// Multi-source geodesic expansion used by the current on-device postprocessor.
    private static func expandMarkers(
        labels: inout [Int],
        markerCount: inout Int,
        through foreground: [Bool],
        width: Int,
        height: Int
    ) {
        var queue = labels.indices.filter { labels[$0] > 0 }
        queue.reserveCapacity(foreground.count)
        var head = 0
        while head < queue.count {
            let index = queue[head]
            head += 1
            let x = index % width
            let y = index / width
            for dy in -1...1 {
                for dx in -1...1 where dx != 0 || dy != 0 {
                    let nx = x + dx
                    let ny = y + dy
                    guard nx >= 0, nx < width, ny >= 0, ny < height else { continue }
                    let neighbor = ny * width + nx
                    if foreground[neighbor] && labels[neighbor] == 0 {
                        labels[neighbor] = labels[index]
                        queue.append(neighbor)
                    }
                }
            }
        }
        let orphanMask = labels.indices.map { foreground[$0] && labels[$0] == 0 }
        let (orphans, orphanCount) = connectedComponents(
            orphanMask,
            width: width,
            height: height
        )
        guard orphanCount > 0 else { return }
        for index in labels.indices where orphans[index] > 0 {
            labels[index] = markerCount + orphans[index]
        }
        markerCount += orphanCount
    }

    /// Applies fixed, user-visible object quality gates. It does not remove per-image low-tail
    /// outliers automatically because a genuinely faint population could otherwise be erased.
    private static func filterByModelConfidence(
        labels: [Int],
        foregroundProbability: [Float],
        boundaryProbability: [Float],
        width: Int,
        height: Int,
        options: CountOptions
    ) -> [Int] {
        let maximumLabel = labels.max() ?? 0
        guard maximumLabel > 0,
              (options.minimumMeanCellProbability > 0
                || options.minimumCoreFraction > 0
                || options.minimumBoundarySupport > 0) else { return labels }

        var area = [Int](repeating: 0, count: maximumLabel + 1)
        var foregroundSum = [Double](repeating: 0, count: maximumLabel + 1)
        var coreCount = [Int](repeating: 0, count: maximumLabel + 1)
        var perimeterCount = [Int](repeating: 0, count: maximumLabel + 1)
        var boundarySum = [Double](repeating: 0, count: maximumLabel + 1)
        for index in labels.indices {
            let label = labels[index]
            guard label > 0 else { continue }
            area[label] += 1
            foregroundSum[label] += Double(foregroundProbability[index])
            if foregroundProbability[index] >= Float(options.coreProbabilityThreshold) {
                coreCount[label] += 1
            }
            let x = index % width
            let y = index / width
            var isPerimeter = x == 0 || x + 1 == width || y == 0 || y + 1 == height
            if !isPerimeter {
                isPerimeter = labels[index - 1] != label
                    || labels[index + 1] != label
                    || labels[index - width] != label
                    || labels[index + width] != label
            }
            if isPerimeter {
                perimeterCount[label] += 1
                boundarySum[label] += Double(boundaryProbability[index])
            }
        }

        var keep = [Bool](repeating: false, count: maximumLabel + 1)
        for label in 1...maximumLabel where area[label] > 0 {
            let meanForeground = foregroundSum[label] / Double(area[label])
            let coreFraction = Double(coreCount[label]) / Double(area[label])
            let boundarySupport = perimeterCount[label] > 0
                ? boundarySum[label] / Double(perimeterCount[label])
                : 0
            keep[label] = meanForeground >= options.minimumMeanCellProbability
                && coreFraction >= options.minimumCoreFraction
                && boundarySupport >= options.minimumBoundarySupport
        }

        var remap = [Int](repeating: 0, count: maximumLabel + 1)
        var nextLabel = 0
        for label in 1...maximumLabel where keep[label] {
            nextLabel += 1
            remap[label] = nextLabel
        }
        return labels.map { $0 > 0 ? remap[$0] : 0 }
    }

    // MARK: - Source-resolution result

    private static func measureAtSourceResolution(
        labels: [Int],
        modelWidth: Int,
        modelHeight: Int,
        sourceWidth: Int,
        sourceHeight: Int,
        diagnostics: CountDiagnostics,
        options: CountOptions
    ) -> CountResult {
        let maximumLabel = labels.max() ?? 0
        var rawMask = [Int](repeating: 0, count: sourceWidth * sourceHeight)
        var area = [Int](repeating: 0, count: maximumLabel + 1)
        var sumX = [Int](repeating: 0, count: maximumLabel + 1)
        var sumY = [Int](repeating: 0, count: maximumLabel + 1)
        var minX = [Int](repeating: Int.max, count: maximumLabel + 1)
        var minY = [Int](repeating: Int.max, count: maximumLabel + 1)
        var maxX = [Int](repeating: 0, count: maximumLabel + 1)
        var maxY = [Int](repeating: 0, count: maximumLabel + 1)

        for y in 0..<sourceHeight {
            let modelY = min(modelHeight - 1, y * modelHeight / sourceHeight)
            for x in 0..<sourceWidth {
                let modelX = min(modelWidth - 1, x * modelWidth / sourceWidth)
                let label = labels[modelY * modelWidth + modelX]
                let index = y * sourceWidth + x
                rawMask[index] = label
                guard label > 0 else { continue }
                area[label] += 1
                sumX[label] += x
                sumY[label] += y
                minX[label] = min(minX[label], x)
                minY[label] = min(minY[label], y)
                maxX[label] = max(maxX[label], x)
                maxY[label] = max(maxY[label], y)
            }
        }

        var remap = [Int](repeating: 0, count: maximumLabel + 1)
        var objects = [DetectedObject]()
        if maximumLabel > 0 {
            for label in 1...maximumLabel
            where area[label] >= options.minAreaPixels
                && (options.maxAreaPixels == 0 || area[label] <= options.maxAreaPixels)
                && objects.count < Int(UInt16.max) {
                let identifier = objects.count + 1
                remap[label] = identifier
                let objectArea = area[label]
                objects.append(DetectedObject(
                    id: identifier,
                    areaPixels: objectArea,
                    centroid: CGPoint(
                        x: Double(sumX[label]) / Double(objectArea),
                        y: Double(sumY[label]) / Double(objectArea)
                    ),
                    bbox: CGRect(
                        x: minX[label],
                        y: minY[label],
                        width: maxX[label] - minX[label] + 1,
                        height: maxY[label] - minY[label] + 1
                    )
                ))
            }
        }

        var labelMask = [UInt16](repeating: 0, count: rawMask.count)
        for index in labelMask.indices {
            let label = rawMask[index]
            if label > 0 { labelMask[index] = UInt16(remap[label]) }
        }
        return CountResult(
            objects: objects,
            imageSize: CGSize(width: sourceWidth, height: sourceHeight),
            labelMask: labelMask,
            micronsPerPixel: options.micronsPerPixel,
            diagnostics: diagnostics
        )
    }
}

private final class CoreMLModelCache {
    private let lock = NSLock()
    private var cachedKey: String?
    private var cachedModel: MLModel?

    func model(
        for spec: CoreMLCounterSpec,
        computeMode: ModelComputeMode
    ) throws -> MLModel {
        lock.lock()
        defer { lock.unlock() }
        let cacheKey = "\(spec.modelResourceName):\(computeMode.rawValue)"
        if cachedKey == cacheKey, let cachedModel { return cachedModel }
        // Drop the old model before loading a new one to avoid a transient two-model peak.
        cachedKey = nil
        cachedModel = nil
        guard let url = Bundle.main.url(
            forResource: spec.modelResourceName,
            withExtension: "mlmodelc"
        ) else {
            throw CountError.modelUnavailable
        }
        let configuration = MLModelConfiguration()
        switch computeMode {
        case .automatic:             configuration.computeUnits = .all
        case .cpuAndNeuralEngine:    configuration.computeUnits = .cpuAndNeuralEngine
        case .cpuAndGPU:             configuration.computeUnits = .cpuAndGPU
        case .cpuOnly:               configuration.computeUnits = .cpuOnly
        }
        let model = try MLModel(contentsOf: url, configuration: configuration)
        cachedKey = cacheKey
        cachedModel = model
        return model
    }

    func evict() {
        lock.lock()
        cachedKey = nil
        cachedModel = nil
        lock.unlock()
    }
}

private struct MultiArrayReader {
    let array: MLMultiArray
    let channelStride: Int
    let rowStride: Int
    let columnStride: Int

    init(_ array: MLMultiArray) {
        self.array = array
        channelStride = array.strides[1].intValue
        rowStride = array.strides[2].intValue
        columnStride = array.strides[3].intValue
    }

    func value(channel: Int, y: Int, x: Int) -> Float {
        let offset = channel * channelStride + y * rowStride + x * columnStride
        switch array.dataType {
        case .float16:
            let pointer = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
            return Float(Float16(bitPattern: pointer[offset]))
        case .float32:
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
            return pointer[offset]
        case .double:
            let pointer = array.dataPointer.bindMemory(to: Double.self, capacity: array.count)
            return Float(pointer[offset])
        default:
            return array[offset].floatValue
        }
    }
}
