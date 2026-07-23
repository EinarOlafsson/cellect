import CoreML
import Foundation

/// Quality/heaviness tiers. Higher = more accurate + more compute. `classical` is the floor.
enum CounterTier: Int, Comparable, CaseIterable, Codable {
    case classical, low, medium, high, best
    static func < (a: CounterTier, b: CounterTier) -> Bool { a.rawValue < b.rawValue }

    var title: String {
        switch self {
        case .classical: return "Classical CV"
        case .low:       return "Light model"
        case .medium:    return "Balanced model"
        case .high:      return "Accurate model"
        case .best:      return "Best (full DL)"
        }
    }
}

/// Rough device-capability gate: which tier ceiling this phone can reasonably run.
/// Classical is always available regardless of this.
enum DeviceCapability {
    static var physicalMemoryGB: Double {
        Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824.0
    }

    /// The heaviest model tier we'll attempt on this device.
    static var tierCeiling: CounterTier {
        let gb = physicalMemoryGB
        if gb >= 7.5 { return .best }     // e.g. Pro / newest
        if gb >= 5.5 { return .high }
        if gb >= 3.5 { return .medium }
        return .low
    }
}

/// Describes a Core ML segmentation model we might ship for a tier. None are bundled yet —
/// when you convert a Cellpose/SAM model to Core ML, add its spec here and drop the
/// `.mlmodelc` into the app bundle; the registry starts offering it automatically.
struct CoreMLCounterSpec {
    let tier: CounterTier
    let modelResourceName: String      // bundle resource, compiled `.mlmodelc`
    let displayName: String

    /// Populate as models are added, e.g.:
    /// CoreMLCounterSpec(tier: .high, modelResourceName: "CellposeSAM", displayName: "Cellpose-SAM")
    static let bundled: [CoreMLCounterSpec] = []

    var isAvailable: Bool {
        Bundle.main.url(forResource: modelResourceName, withExtension: "mlmodelc") != nil
    }
}

/// A Core ML–backed counter (slot). Inference/output parsing is model-specific and wired when a
/// real model is added; until then it reports itself unavailable so the registry falls back.
struct CoreMLCellCounter: CellCounter {
    let spec: CoreMLCounterSpec
    var tier: CounterTier { spec.tier }
    var displayName: String { spec.displayName }

    func count(in image: CGImage, options: CountOptions) async throws -> CountResult {
        // TODO: load spec model, run segmentation, convert output → instance labels → stats.
        // Kept behind `CoreMLCounterSpec.bundled` (empty) so this path is never taken yet.
        throw CountError.modelUnavailable
    }
}

/// Chooses counters for this device: classical floor + any bundled model tiers the device allows.
enum CounterRegistry {
    static func availableCounters() -> [CellCounter] {
        var counters: [CellCounter] = [ClassicalCellCounter()]
        let ceiling = DeviceCapability.tierCeiling
        for spec in CoreMLCounterSpec.bundled where spec.tier <= ceiling && spec.isAvailable {
            counters.append(CoreMLCellCounter(spec: spec))
        }
        return counters.sorted { $0.tier > $1.tier }   // best first
    }

    /// The best counter this device can run right now.
    static func best() -> CellCounter {
        availableCounters().first ?? ClassicalCellCounter()
    }
}
