import CoreGraphics
import Foundation

/// One detected cell/object.
struct DetectedObject: Identifiable, Hashable {
    let id: Int                 // instance label (1-based) in the label mask
    let areaPixels: Int
    let centroid: CGPoint       // in image pixel coordinates
    let bbox: CGRect            // in image pixel coordinates

    /// Diameter of a circle with the same area — a robust size proxy.
    var equivalentDiameterPixels: Double {
        2.0 * (Double(areaPixels) / .pi).squareRoot()
    }
}

/// The outcome of counting one frame: per-object stats + a 16-bit instance label mask.
struct CountResult {
    let objects: [DetectedObject]
    let imageSize: CGSize
    /// Instance labels, one UInt16 per pixel (0 = background). Row-major, width*height.
    let labelMask: [UInt16]
    /// Microns per pixel if the user calibrated; nil means sizes are reported in pixels only.
    let micronsPerPixel: Double?

    var count: Int { objects.count }

    /// Per-object size in microns if calibrated, else nil.
    func equivalentDiameterMicrons(_ o: DetectedObject) -> Double? {
        guard let mpp = micronsPerPixel else { return nil }
        return o.equivalentDiameterPixels * mpp
    }
}

/// Image polarity: are objects darker than background (typical brightfield cells) or brighter?
enum CountPolarity: String, Codable, CaseIterable, Identifiable {
    case darkObjects, brightObjects
    var id: String { rawValue }
    var title: String { self == .darkObjects ? "Dark cells" : "Bright cells" }
}

/// Tunables for a counting run.
struct CountOptions {
    var polarity: CountPolarity = .darkObjects
    var minAreaPixels: Int = 40          // drop specks
    var micronsPerPixel: Double? = nil   // optional calibration
}

/// A pluggable counter. Classical CV is the floor; Core ML tiers slot in above it.
protocol CellCounter {
    var tier: CounterTier { get }
    var displayName: String { get }
    /// Segment + measure objects in a frame. May run off the main actor.
    func count(in image: CGImage, options: CountOptions) async throws -> CountResult
}

enum CountError: LocalizedError {
    case decodeFailed
    case modelUnavailable
    var errorDescription: String? {
        switch self {
        case .decodeFailed:    return "Couldn't read the image pixels."
        case .modelUnavailable: return "This segmentation model isn't available on this device."
        }
    }
}
