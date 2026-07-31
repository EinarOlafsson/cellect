import CoreGraphics
import Foundation

/// Writes a capture's outputs into the project folder: the image, a 16-bit instance mask, a
/// per-object CSV, and one appended row in the project's rolling summary CSV.
struct CountResultWriter {
    let provider: StorageProvider
    let project: CaptureProject

    /// Returns the stem used for the written files (e.g. "capture_20260723_101530_a1b2c3d4").
    @discardableResult
    func save(
        image: CGImage,
        result: CountResult,
        counterIdentifier: String,
        counterName: String,
        options: CountOptions,
        runID: UUID,
        inferenceStartedAt: Date,
        inferenceCompletedAt: Date,
        timestamp: Date
    ) async throws -> String {
        let shortRunID = runID.uuidString.prefix(8).lowercased()
        let stem = "capture_\(Self.stamp(inferenceStartedAt))_\(shortRunID)"

        guard let png = MaskPNG.encodePNG(image) else {
            throw CountWriteError.imageEncodingFailed
        }
        try await provider.writeFile(named: "\(stem).png", data: png)
        guard let maskPNG = MaskPNG.label16(
            result.labelMask,
            width: Int(result.imageSize.width),
            height: Int(result.imageSize.height)
        ) else {
            throw CountWriteError.maskEncodingFailed
        }
        try await provider.writeFile(named: "\(stem)_mask.png", data: maskPNG)
        try await writeObjectsCSV(stem: stem, result: result)
        try await writeSettings(
            stem: stem,
            result: result,
            counterIdentifier: counterIdentifier,
            counterName: counterName,
            options: options,
            runID: runID,
            inferenceStartedAt: inferenceStartedAt,
            inferenceCompletedAt: inferenceCompletedAt,
            timestamp: timestamp
        )
        try await appendSummary(stem: stem, result: result, counterName: counterName, timestamp: timestamp)
        return stem
    }

    // MARK: - CSVs

    private func writeObjectsCSV(stem: String, result: CountResult) async throws {
        let calibrated = result.micronsPerPixel != nil
        var header = ["object_id", "area_px", "equiv_diameter_px",
                      "centroid_x", "centroid_y", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
        if calibrated { header.insert("equiv_diameter_um", at: 3) }

        let rows: [[String]] = result.objects.map { o in
            var r = [
                String(o.id),
                String(o.areaPixels),
                fmt(o.equivalentDiameterPixels),
                fmt(Double(o.centroid.x)), fmt(Double(o.centroid.y)),
                String(Int(o.bbox.minX)), String(Int(o.bbox.minY)),
                String(Int(o.bbox.width)), String(Int(o.bbox.height))
            ]
            if calibrated, let um = result.equivalentDiameterMicrons(o) {
                r.insert(fmt(um), at: 3)
            }
            return r
        }
        let text = CSV.serialize(header: header, rows: rows)
        try await provider.writeFile(named: "\(stem)_objects.csv", data: Data(text.utf8))
    }

    private func writeSettings(
        stem: String,
        result: CountResult,
        counterIdentifier: String,
        counterName: String,
        options: CountOptions,
        runID: UUID,
        inferenceStartedAt: Date,
        inferenceCompletedAt: Date,
        timestamp: Date
    ) async throws {
        let record = RunSettingsRecord(
            schemaVersion: 2,
            runID: runID,
            modelIdentifier: counterIdentifier,
            modelName: counterName,
            appVersion: Bundle.main.object(
                forInfoDictionaryKey: "CFBundleShortVersionString"
            ) as? String,
            buildNumber: Bundle.main.object(
                forInfoDictionaryKey: "CFBundleVersion"
            ) as? String,
            inferenceStartedAt: inferenceStartedAt,
            inferenceCompletedAt: inferenceCompletedAt,
            timestamp: timestamp,
            imageWidth: Int(result.imageSize.width),
            imageHeight: Int(result.imageSize.height),
            detectedObjectCount: result.count,
            options: options,
            diagnostics: result.diagnostics
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try await provider.writeFile(
            named: "\(stem)_settings.json",
            data: try encoder.encode(record)
        )
    }

    private func appendSummary(stem: String, result: CountResult, counterName: String, timestamp: Date) async throws {
        let header = ["filename", "count", "mean_diameter_px", "mean_diameter_um", "timestamp", "counter"]
        let meanPx = result.objects.isEmpty ? 0 :
            result.objects.map(\.equivalentDiameterPixels).reduce(0, +) / Double(result.objects.count)
        let meanUm = result.micronsPerPixel.map { meanPx * $0 }
        let row = [
            "\(stem).png",
            String(result.count),
            fmt(meanPx),
            meanUm.map(fmt) ?? "",
            ISO8601DateFormatter().string(from: timestamp),
            counterName
        ]

        // Read-modify-write so the summary accumulates across captures.
        var rows: [[String]] = []
        if let data = try await provider.readFile(named: project.summaryCSVName),
           let text = String(data: data, encoding: .utf8) {
            let (existingHeader, existingRows) = CSV.parse(text)
            if existingHeader == header { rows = existingRows }
        }
        rows.append(row)
        let text = CSV.serialize(header: header, rows: rows)
        try await provider.writeFile(named: project.summaryCSVName, data: Data(text.utf8))
    }

    // MARK: - Helpers

    private func fmt(_ v: Double) -> String { String(format: "%.2f", v) }

    private static func stamp(_ date: Date) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd_HHmmss"
        return f.string(from: date)
    }
}

private enum CountWriteError: LocalizedError {
    case imageEncodingFailed
    case maskEncodingFailed

    var errorDescription: String? {
        switch self {
        case .imageEncodingFailed: return "The captured image could not be encoded as PNG."
        case .maskEncodingFailed:  return "The instance mask could not be encoded as PNG."
        }
    }
}

private struct RunSettingsRecord: Codable {
    let schemaVersion: Int
    let runID: UUID
    let modelIdentifier: String
    let modelName: String
    let appVersion: String?
    let buildNumber: String?
    let inferenceStartedAt: Date
    let inferenceCompletedAt: Date
    let timestamp: Date
    let imageWidth: Int
    let imageHeight: Int
    let detectedObjectCount: Int
    let options: CountOptions
    let diagnostics: CountDiagnostics?
}
