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

/// How multiple trained models combine their foreground and boundary probabilities.
/// Fusion happens before instance reconstruction so labels from different models never collide.
enum EnsembleMergeStrategy: String, Codable, CaseIterable, Identifiable {
    case weightedMean
    case median
    case maximum
    case minimum

    var id: String { rawValue }

    var title: String {
        switch self {
        case .weightedMean: return "Weighted average"
        case .median:       return "Median"
        case .maximum:      return "Maximum / permissive"
        case .minimum:      return "Minimum / conservative"
        }
    }

    var explanation: String {
        switch self {
        case .weightedMean:
            return "Averages the models' probability maps using the weights below."
        case .median:
            return "Uses the middle probability at each pixel and resists one outlier model."
        case .maximum:
            return "Accepts evidence from any model; usually finds more cells and more splits."
        case .minimum:
            return "Requires every model to agree; usually produces fewer, more conservative cells."
        }
    }

    var usesModelWeights: Bool { self == .weightedMean }
}

/// One point from the automatic boundary-cutoff stability sweep.
struct BoundaryCutoffCandidate: Codable, Hashable, Identifiable {
    var threshold: Double
    var objectCount: Int
    var localStability: Double
    var qualityScore: Double
    var selectionScore: Double

    var id: Double { threshold }
}

enum BoundaryCutoffFallbackReason: String, Codable, Hashable {
    case noCandidate
    case insufficientObjects
    case unchangedPartition
    case unstablePartition
    case edgeOptimum

    var title: String {
        switch self {
        case .noCandidate:         return "no usable candidate"
        case .insufficientObjects: return "too few separated objects"
        case .unchangedPartition:  return "all cutoffs produced the same partition"
        case .unstablePartition:   return "the candidate was unstable"
        case .edgeOptimum:         return "the candidate was at the sweep edge"
        }
    }
}

/// Reproducibility information generated during trained-model inference.
struct CountDiagnostics: Codable, Hashable {
    var modelIdentifiers: [String]
    var modelExportVersions: [String: String]
    var modelSourceTorchScriptSHA256: [String: String]
    var requestedModelWeights: [String: Double]
    var normalizedModelWeights: [String: Double]
    var ensembleMergeStrategy: EnsembleMergeStrategy?
    var probabilityGridWidth: Int
    var probabilityGridHeight: Int
    var probabilityResamplingMethod: String
    var postprocessingAlgorithmVersion: String
    var automaticBoundarySelectionUsed: Bool
    var boundarySelectionAlgorithmVersion: String?
    var automaticBoundaryFallbackUsed: Bool
    var automaticBoundaryFallbackReason: BoundaryCutoffFallbackReason?
    var automaticBoundaryHeuristicStabilityScore: Double?
    var selectedBoundaryProbabilityThreshold: Double?
    var boundaryStabilityAUC: Double?
    var boundaryCandidates: [BoundaryCutoffCandidate]
}

/// The outcome of counting one frame: per-object stats + a 16-bit instance label mask.
struct CountResult {
    let objects: [DetectedObject]
    let imageSize: CGSize
    /// Instance labels, one UInt16 per pixel (0 = background). Row-major, width*height.
    let labelMask: [UInt16]
    /// Microns per pixel if the user calibrated; nil means sizes are reported in pixels only.
    let micronsPerPixel: Double?
    /// Model composition and automatic cutoff search, when a trained model was used.
    let diagnostics: CountDiagnostics?

    init(
        objects: [DetectedObject],
        imageSize: CGSize,
        labelMask: [UInt16],
        micronsPerPixel: Double?,
        diagnostics: CountDiagnostics? = nil
    ) {
        self.objects = objects
        self.imageSize = imageSize
        self.labelMask = labelMask
        self.micronsPerPixel = micronsPerPixel
        self.diagnostics = diagnostics
    }

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

/// Core ML execution hardware. This changes where inference runs, not the trained weights.
enum ModelComputeMode: String, Codable, CaseIterable, Identifiable {
    case automatic
    case cpuAndNeuralEngine
    case cpuAndGPU
    case cpuOnly

    var id: String { rawValue }

    var title: String {
        switch self {
        case .automatic: return "Automatic"
        case .cpuAndNeuralEngine: return "CPU + Neural Engine"
        case .cpuAndGPU: return "CPU + GPU"
        case .cpuOnly: return "CPU only"
        }
    }
}

/// Tunables for a counting run.
struct CountOptions: Codable, Equatable {
    var polarity: CountPolarity = .darkObjects
    var minAreaPixels: Int = 40              // drop specks
    /// Zero disables the upper-size filter.
    var maxAreaPixels: Int = 0
    var micronsPerPixel: Double? = nil       // optional calibration

    // Classical CV preprocessing and thresholding.
    var classicalDenoiseEnabled = true
    var classicalThresholdOffset = 0         // added to Otsu, in 8-bit intensity units

    // Shared binary-mask cleanup.
    var openingIterations = 1
    var closingIterations = 1

    // Trained foreground/boundary network post-processing.
    var foregroundProbabilityThreshold = 0.5
    var boundaryProbabilityThreshold = 0.45
    /// When enabled, sweep boundary cutoffs and choose a stable, plausible partition near
    /// the user's preferred cutoff. This experimental heuristic has no per-image ground truth.
    var automaticBoundaryThreshold = false
    /// Zero disables each object-level confidence filter.
    var minimumMeanCellProbability = 0.0
    var coreProbabilityThreshold = 0.70
    var minimumCoreFraction = 0.0
    var minimumBoundarySupport = 0.0
    var separateTouchingCells = true
    var invertModelInput = false
    var computeMode: ModelComputeMode = .automatic

    // Ensemble composition. A single-model counter leaves the identifier list empty.
    var ensembleModelIdentifiers: [String] = []
    var ensembleModelWeights: [String: Double] = [:]
    var ensembleMergeStrategy: EnsembleMergeStrategy = .weightedMean
}

/// A pluggable counter. Classical CV is the floor; Core ML tiers slot in above it.
protocol CellCounter {
    var identifier: String { get }
    var tier: CounterTier { get }
    var displayName: String { get }
    /// Segment + measure objects in a frame. May run off the main actor.
    func count(in image: CGImage, options: CountOptions) async throws -> CountResult
}

enum CountError: LocalizedError {
    case decodeFailed
    case modelUnavailable
    case inferenceFailed(String)
    var errorDescription: String? {
        switch self {
        case .decodeFailed:    return "Couldn't read the image pixels."
        case .modelUnavailable: return "This segmentation model isn't available on this device."
        case .inferenceFailed(let message): return "Model inference failed: \(message)"
        }
    }
}
