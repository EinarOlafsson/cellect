import CoreGraphics
import Foundation

struct ModelComparisonProgress: Equatable {
    let completed: Int
    let total: Int
    let modelName: String
    let variantName: String
}

struct ModelComparisonResult: Identifiable {
    let id: UUID
    let candidate: ModelComparisonCandidate
    let maskFileURL: URL
    let overlayFileURL: URL
    let objects: [DetectedObject]
    let imageSize: CGSize
    let micronsPerPixel: Double?
    let diagnostics: CountDiagnostics?
    let inferenceStartedAt: Date
    let inferenceCompletedAt: Date

    var objectCount: Int { objects.count }

    var meanDiameterPixels: Double {
        guard !objects.isEmpty else { return 0 }
        return objects.map(\.equivalentDiameterPixels).reduce(0, +) / Double(objects.count)
    }

    var elapsed: TimeInterval {
        inferenceCompletedAt.timeIntervalSince(inferenceStartedAt)
    }
}

struct ModelComparisonFailure: Identifiable, Codable {
    let id: UUID
    let candidateID: String
    let modelIdentifier: String
    let modelName: String
    let variantName: String
    let message: String
}

struct ModelComparisonOutput {
    let sessionID: UUID
    let artifactsDirectory: URL
    let results: [ModelComparisonResult]
    let failures: [ModelComparisonFailure]
}

enum ModelComparisonRunner {
    /// Serial execution is intentional. Several bundled models are hundreds of megabytes, and
    /// parallel model loading can terminate the app on a physical iPhone.
    static func run(
        image: CGImage,
        candidates: [ModelComparisonCandidate],
        modelSpecs: [CoreMLCounterSpec],
        progress: @escaping @MainActor @Sendable (ModelComparisonProgress) -> Void
    ) async throws -> ModelComparisonOutput {
        let sessionID = UUID()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("CellectModelComparisons", isDirectory: true)
            .appendingPathComponent(sessionID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let worker = Task.detached(priority: .userInitiated) {
            () async throws -> ModelComparisonOutput in
            var completed = 0
            var results = [ModelComparisonResult]()
            var failures = [ModelComparisonFailure]()
            var index = 0

            while index < candidates.count {
                try Task.checkCancellation()
                let candidate = candidates[index]

                if candidate.modelIdentifier == "classical" {
                    let startedAt = Date()
                    do {
                        let countResult = try await ClassicalCellCounter().count(
                            in: image,
                            options: candidate.options
                        )
                        try Task.checkCancellation()
                        results.append(try persist(
                            countResult,
                            candidate: candidate,
                            directory: directory,
                            startedAt: startedAt,
                            completedAt: Date()
                        ))
                    } catch is CancellationError {
                        throw CancellationError()
                    } catch {
                        failures.append(failure(for: candidate, error: error))
                    }
                    completed += 1
                    await progress(ModelComparisonProgress(
                        completed: completed,
                        total: candidates.count,
                        modelName: candidate.modelDisplayName,
                        variantName: candidate.variantName
                    ))
                    index += 1
                    continue
                }

                guard let spec = modelSpecs.first(where: {
                    $0.id == candidate.modelIdentifier
                }) else {
                    failures.append(ModelComparisonFailure(
                        id: UUID(),
                        candidateID: candidate.id,
                        modelIdentifier: candidate.modelIdentifier,
                        modelName: candidate.modelDisplayName,
                        variantName: candidate.variantName,
                        message: "The compiled model is no longer available."
                    ))
                    completed += 1
                    await progress(ModelComparisonProgress(
                        completed: completed,
                        total: candidates.count,
                        modelName: candidate.modelDisplayName,
                        variantName: candidate.variantName
                    ))
                    index += 1
                    continue
                }

                // Adjacent candidates with the same input transform and compute mode share one
                // probability prediction. Only their inexpensive postprocessing differs.
                let groupStart = index
                let invert = candidate.options.invertModelInput
                let computeMode = candidate.options.computeMode.rawValue
                while index < candidates.count {
                    let next = candidates[index]
                    guard next.modelIdentifier == candidate.modelIdentifier,
                          next.options.invertModelInput == invert,
                          next.options.computeMode.rawValue == computeMode else { break }
                    index += 1
                }
                let group = Array(candidates[groupStart..<index])
                let predictionStartedAt = Date()

                do {
                    CoreMLSegmentation.evictComparisonModel()
                    let probabilities = try autoreleasepool {
                        try CoreMLSegmentation.predictForComparison(
                            in: image,
                            spec: spec,
                            options: candidate.options
                        )
                    }
                    for member in group {
                        try Task.checkCancellation()
                        do {
                            let countResult = autoreleasepool {
                                CoreMLSegmentation.postprocessForComparison(
                                    probabilities,
                                    sourceWidth: image.width,
                                    sourceHeight: image.height,
                                    spec: spec,
                                    options: member.options
                                )
                            }
                            try Task.checkCancellation()
                            results.append(try persist(
                                countResult,
                                candidate: member,
                                directory: directory,
                                startedAt: predictionStartedAt,
                                completedAt: Date()
                            ))
                        } catch is CancellationError {
                            throw CancellationError()
                        } catch {
                            failures.append(failure(for: member, error: error))
                        }
                        completed += 1
                        await progress(ModelComparisonProgress(
                            completed: completed,
                            total: candidates.count,
                            modelName: member.modelDisplayName,
                            variantName: member.variantName
                        ))
                    }
                } catch is CancellationError {
                    CoreMLSegmentation.evictComparisonModel()
                    throw CancellationError()
                } catch {
                    // A prediction failure applies to every postprocessing variant in the group.
                    for member in group {
                        failures.append(failure(for: member, error: error))
                        completed += 1
                        await progress(ModelComparisonProgress(
                            completed: completed,
                            total: candidates.count,
                            modelName: member.modelDisplayName,
                            variantName: member.variantName
                        ))
                    }
                }
                CoreMLSegmentation.evictComparisonModel()
            }

            return ModelComparisonOutput(
                sessionID: sessionID,
                artifactsDirectory: directory,
                results: results.sorted { $0.candidate.ordinal < $1.candidate.ordinal },
                failures: failures
            )
        }

        do {
            return try await withTaskCancellationHandler {
                try await worker.value
            } onCancel: {
                worker.cancel()
            }
        } catch {
            try? FileManager.default.removeItem(at: directory)
            throw error
        }
    }

    private static func persist(
        _ countResult: CountResult,
        candidate: ModelComparisonCandidate,
        directory: URL,
        startedAt: Date,
        completedAt: Date
    ) throws -> ModelComparisonResult {
        guard let maskData = MaskPNG.label16(
            countResult.labelMask,
            width: Int(countResult.imageSize.width),
            height: Int(countResult.imageSize.height)
        ) else {
            throw ModelComparisonError.maskEncodingFailed
        }
        guard let overlay = LabelOverlay.makeCGImage(
            labels: countResult.labelMask,
            width: Int(countResult.imageSize.width),
            height: Int(countResult.imageSize.height)
        ), let overlayData = MaskPNG.encodePNG(overlay) else {
            throw ModelComparisonError.overlayEncodingFailed
        }

        let stem = String(format: "%04d_%@", candidate.ordinal + 1, safeFilename(candidate.modelIdentifier))
        let maskURL = directory.appendingPathComponent("\(stem)_mask.png")
        let overlayURL = directory.appendingPathComponent("\(stem)_overlay.png")
        try maskData.write(to: maskURL, options: .atomic)
        try overlayData.write(to: overlayURL, options: .atomic)

        return ModelComparisonResult(
            id: UUID(),
            candidate: candidate,
            maskFileURL: maskURL,
            overlayFileURL: overlayURL,
            objects: countResult.objects,
            imageSize: countResult.imageSize,
            micronsPerPixel: countResult.micronsPerPixel,
            diagnostics: countResult.diagnostics,
            inferenceStartedAt: startedAt,
            inferenceCompletedAt: completedAt
        )
    }

    private static func failure(
        for candidate: ModelComparisonCandidate,
        error: Error
    ) -> ModelComparisonFailure {
        ModelComparisonFailure(
            id: UUID(),
            candidateID: candidate.id,
            modelIdentifier: candidate.modelIdentifier,
            modelName: candidate.modelDisplayName,
            variantName: candidate.variantName,
            message: error.localizedDescription
        )
    }

    private static func safeFilename(_ text: String) -> String {
        text.lowercased().map {
            $0.isLetter || $0.isNumber ? $0 : "_"
        }.reduce(into: "") { $0.append($1) }
    }
}

private enum ModelComparisonError: LocalizedError {
    case maskEncodingFailed
    case overlayEncodingFailed

    var errorDescription: String? {
        switch self {
        case .maskEncodingFailed:
            return "A 16-bit comparison mask could not be encoded."
        case .overlayEncodingFailed:
            return "A comparison overlay could not be encoded."
        }
    }
}
