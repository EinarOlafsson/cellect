import Foundation

/// A destination folder + counting settings for camera capture sessions.
struct CaptureProject: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var displayName: String
    var storageRef: StorageRef
    var polarity: CountPolarity = .darkObjects
    var minAreaPixels: Int = 40
    /// Optional calibration; enables micron sizes in the CSV.
    var micronsPerPixel: Double?
    /// CSV that accumulates one summary row per capture.
    var summaryCSVName: String = "captures_summary.csv"
}
