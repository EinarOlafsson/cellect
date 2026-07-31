import CoreGraphics
import Foundation

/// Exports one source image, every 16-bit candidate mask, and a reproducibility manifest into
/// the capture project's chosen folder.
struct ModelComparisonResultWriter {
    let provider: StorageProvider
    let project: CaptureProject

    @discardableResult
    func save(image: CGImage, output: ModelComparisonOutput) async throws -> String {
        let exportedAt = Date()
        let stem = "comparison_\(Self.stamp(exportedAt))_\(output.sessionID.uuidString.prefix(8).lowercased())"
        let sourceName = "\(stem)_source.png"
        guard let sourcePNG = MaskPNG.encodePNG(image) else {
            throw ModelComparisonWriteError.imageEncodingFailed
        }
        try await provider.writeFile(named: sourceName, data: sourcePNG)

        var manifestResults = [ManifestResult]()
        for result in output.results.sorted(by: {
            $0.candidate.ordinal < $1.candidate.ordinal
        }) {
            let maskName = String(
                format: "%@_%04d_%@_mask.png",
                stem,
                result.candidate.ordinal + 1,
                Self.safeFilename(result.candidate.modelIdentifier)
            )
            let maskData = try Data(contentsOf: result.maskFileURL)
            try await provider.writeFile(named: maskName, data: maskData)
            manifestResults.append(ManifestResult(
                runID: result.id,
                candidate: result.candidate,
                maskFilename: maskName,
                detectedObjectCount: result.objectCount,
                meanDiameterPixels: result.meanDiameterPixels,
                imageWidth: Int(result.imageSize.width),
                imageHeight: Int(result.imageSize.height),
                inferenceStartedAt: result.inferenceStartedAt,
                inferenceCompletedAt: result.inferenceCompletedAt,
                diagnostics: result.diagnostics
            ))
        }

        let manifest = Manifest(
            schemaVersion: 1,
            sessionID: output.sessionID,
            projectID: project.id,
            projectName: project.displayName,
            exportedAt: exportedAt,
            sourceImageFilename: sourceName,
            results: manifestResults,
            failures: output.failures
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try await provider.writeFile(
            named: "\(stem)_manifest.json",
            data: try encoder.encode(manifest)
        )
        return stem
    }

    private static func stamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return formatter.string(from: date)
    }

    private static func safeFilename(_ text: String) -> String {
        text.lowercased().map {
            $0.isLetter || $0.isNumber ? $0 : "_"
        }.reduce(into: "") { $0.append($1) }
    }
}

private struct Manifest: Codable {
    let schemaVersion: Int
    let sessionID: UUID
    let projectID: UUID
    let projectName: String
    let exportedAt: Date
    let sourceImageFilename: String
    let results: [ManifestResult]
    let failures: [ModelComparisonFailure]
}

private struct ManifestResult: Codable {
    let runID: UUID
    let candidate: ModelComparisonCandidate
    let maskFilename: String
    let detectedObjectCount: Int
    let meanDiameterPixels: Double
    let imageWidth: Int
    let imageHeight: Int
    let inferenceStartedAt: Date
    let inferenceCompletedAt: Date
    let diagnostics: CountDiagnostics?
}

private enum ModelComparisonWriteError: LocalizedError {
    case imageEncodingFailed

    var errorDescription: String? {
        "The comparison source image could not be encoded as PNG."
    }
}
