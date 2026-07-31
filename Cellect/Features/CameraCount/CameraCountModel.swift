import CoreGraphics
import Foundation
import Observation

/// Orchestrates one capture project: live camera → capture → count → review → save.
@MainActor
@Observable
final class CameraCountModel {
    enum Phase: Equatable { case preview, working, review }

    let project: CaptureProject
    let camera = CameraController()

    private(set) var phase: Phase = .preview
    private(set) var captured: CGImage?
    private(set) var overlay: CGImage?
    private(set) var result: CountResult?
    private(set) var counterName: String = ""
    private(set) var lastCounterIdentifier: String?
    private(set) var lastCountOptions: CountOptions?
    private(set) var lastCountDiagnostics: CountDiagnostics?
    private(set) var currentRunID: UUID?
    private(set) var currentInferenceStartedAt: Date?
    private(set) var currentInferenceCompletedAt: Date?
    var isSaving = false
    var errorMessage: String?
    var lastSavedStem: String?

    // Editable review settings (seeded from the project, let the user re-count).
    var polarity: CountPolarity
    var minAreaPixels: Int
    var maxAreaPixels = 0
    var selectedCounterID: String
    var classicalDenoiseEnabled = true
    var classicalThresholdOffset = 0
    var openingIterations = 1
    var closingIterations = 1
    var foregroundProbabilityThreshold = 0.5
    var boundaryProbabilityThreshold = 0.45
    var automaticBoundaryThreshold = false
    var minimumMeanCellProbability = 0.0
    var coreProbabilityThreshold = 0.70
    var minimumCoreFraction = 0.0
    var minimumBoundarySupport = 0.0
    var separateTouchingCells = true
    var invertModelInput = false
    var computeMode: ModelComputeMode = .automatic
    var ensembleEnabled = false
    var selectedEnsembleModelIDs = Set<String>()
    var ensembleModelWeights = [String: Double]()
    var ensembleMergeStrategy: EnsembleMergeStrategy = .weightedMean

    private let provider: StorageProvider
    /// Prevents an older, slower model run from overwriting a newer recount.
    private var runGeneration = 0

    init(project: CaptureProject) throws {
        let recommendedCounter = CounterRegistry.best()
        self.project = project
        self.provider = try StorageProviderFactory.make(for: project.storageRef)
        self.polarity = project.polarity
        self.minAreaPixels = project.minAreaPixels
        self.selectedCounterID = recommendedCounter.identifier
        let availableModels = CounterRegistry.availableModelSpecs()
        self.selectedEnsembleModelIDs = Set(availableModels.prefix(2).map(\.id))
        self.ensembleModelWeights = Dictionary(
            uniqueKeysWithValues: availableModels.map { ($0.id, 1) }
        )
    }

    func onAppear() { camera.onAppear() }
    func onDisappear() { camera.onDisappear() }

    var summary: String {
        guard let result else { return "" }
        let mean = result.objects.isEmpty ? 0 :
            result.objects.map(\.equivalentDiameterPixels).reduce(0, +) / Double(result.objects.count)
        if let mpp = result.micronsPerPixel {
            return "\(result.count) cells · mean ⌀ \(String(format: "%.1f", mean * mpp)) µm"
        }
        return "\(result.count) cells · mean ⌀ \(String(format: "%.0f", mean)) px"
    }

    var availableCounterIDs: [String] {
        CounterRegistry.availableCounters().map(\.identifier)
    }

    func counterDisplayName(for identifier: String) -> String {
        CounterRegistry.counter(for: identifier)?.displayName ?? identifier
    }

    var selectedModelSpec: CoreMLCounterSpec? {
        CounterRegistry.spec(for: selectedCounterID)
    }

    var selectedCounterIsTrainedModel: Bool {
        ensembleEnabled || selectedModelSpec != nil
    }

    var availableModelSpecs: [CoreMLCounterSpec] {
        CounterRegistry.availableModelSpecs()
    }

    /// Immutable settings used to build a comparison plan. The planner removes ensemble fields
    /// because each candidate intentionally represents one model.
    var comparisonOptionsSnapshot: CountOptions {
        makeCountOptions()
    }

    var selectedEnsembleSpecs: [CoreMLCounterSpec] {
        availableModelSpecs.filter { selectedEnsembleModelIDs.contains($0.id) }
    }

    var canEnableEnsemble: Bool { availableModelSpecs.count >= 2 }

    var ensembleSelectionSummary: String {
        "\(selectedEnsembleSpecs.count) models · \(ensembleMergeStrategy.title)"
    }

    /// True when the visible controls no longer describe the mask currently on screen.
    var hasUnappliedChanges: Bool {
        guard result != nil,
              let lastCounterIdentifier,
              let lastCountOptions else { return false }
        guard let intendedIdentifier = intendedCounterIdentifier else { return true }
        return lastCounterIdentifier != intendedIdentifier
            || lastCountOptions != makeCountOptions()
    }

    func isEnsembleModelSelected(_ identifier: String) -> Bool {
        selectedEnsembleModelIDs.contains(identifier)
    }

    func canDeselectEnsembleModel(_ identifier: String) -> Bool {
        !selectedEnsembleModelIDs.contains(identifier)
            || selectedEnsembleModelIDs.count > 2
    }

    func setEnsembleEnabled(_ enabled: Bool) {
        guard !enabled || canEnableEnsemble else { return }
        if enabled && selectedEnsembleModelIDs.count < 2 {
            selectedEnsembleModelIDs = Set(availableModelSpecs.prefix(2).map(\.id))
        }
        ensembleEnabled = enabled
    }

    func setEnsembleModel(_ identifier: String, selected: Bool) {
        guard availableModelSpecs.contains(where: { $0.id == identifier }) else { return }
        if selected {
            selectedEnsembleModelIDs.insert(identifier)
            if ensembleModelWeights[identifier] == nil {
                ensembleModelWeights[identifier] = 1
            }
        } else if canDeselectEnsembleModel(identifier) {
            selectedEnsembleModelIDs.remove(identifier)
        }
    }

    func ensembleWeight(for identifier: String) -> Double {
        ensembleModelWeights[identifier] ?? 1
    }

    func setEnsembleWeight(_ weight: Double, for identifier: String) {
        ensembleModelWeights[identifier] = min(3, max(0.1, weight))
    }

    func normalizedEnsembleWeight(for identifier: String) -> Double {
        guard selectedEnsembleModelIDs.contains(identifier) else { return 0 }
        let denominator = selectedEnsembleSpecs.reduce(0) {
            $0 + max(0, ensembleWeight(for: $1.id))
        }
        guard denominator > 0 else { return 1 / Double(max(1, selectedEnsembleSpecs.count)) }
        return max(0, ensembleWeight(for: identifier)) / denominator
    }

    func resetAdjustableSettings() {
        polarity = project.polarity
        minAreaPixels = project.minAreaPixels
        maxAreaPixels = 0
        classicalDenoiseEnabled = true
        classicalThresholdOffset = 0
        openingIterations = 1
        closingIterations = 1
        foregroundProbabilityThreshold = 0.5
        boundaryProbabilityThreshold = 0.45
        automaticBoundaryThreshold = false
        minimumMeanCellProbability = 0.0
        coreProbabilityThreshold = 0.70
        minimumCoreFraction = 0.0
        minimumBoundarySupport = 0.0
        separateTouchingCells = true
        invertModelInput = false
        computeMode = .automatic
        ensembleMergeStrategy = .weightedMean
        for spec in availableModelSpecs {
            ensembleModelWeights[spec.id] = 1
        }
    }

    func capture() async {
        guard phase != .working else { return }
        errorMessage = nil
        phase = .working
        do {
            let image = try await camera.capture()
            captured = image
            await runCount()
        } catch {
            errorMessage = error.localizedDescription
            phase = .preview
        }
    }

    /// Run the same counter on a microscope image chosen from Files.
    func importImage(from url: URL) async {
        guard phase != .working else { return }
        errorMessage = nil
        phase = .working
        let secured = url.startAccessingSecurityScopedResource()
        defer {
            if secured { url.stopAccessingSecurityScopedResource() }
        }
        do {
            let data = try Data(contentsOf: url)
            guard let image = ImageLoader.decode(
                data,
                maxDimension: CountConfig.maxDimension
            ) else {
                throw CountError.decodeFailed
            }
            captured = image
            await runCount()
        } catch {
            errorMessage = error.localizedDescription
            phase = .preview
        }
    }

    /// Re-run counting on the already-captured frame (after tweaking polarity / min area).
    func recount() async {
        guard captured != nil, phase != .working else { return }
        phase = .working
        await runCount()
    }

    private func runCount() async {
        guard let image = captured else { phase = .preview; return }
        runGeneration &+= 1
        let generation = runGeneration
        let runID = UUID()
        let inferenceStartedAt = Date()
        let options = makeCountOptions()
        let counter: CellCounter
        if ensembleEnabled {
            let identifiers = options.ensembleModelIdentifiers
            guard identifiers.count >= 2,
                  let ensemble = CounterRegistry.ensembleCounter(
                    identifiers: identifiers,
                    weights: options.ensembleModelWeights,
                    strategy: options.ensembleMergeStrategy
                  ) else {
                errorMessage = "Select at least two available trained models for an ensemble."
                phase = .review
                return
            }
            counter = ensemble
        } else {
            let selected = CounterRegistry.counter(for: selectedCounterID)
                ?? CounterRegistry.best()
            selectedCounterID = selected.identifier
            counter = selected
        }
        let attemptedCounterName = counter.displayName

        do {
            let res = try await counter.count(in: image, options: options)
            guard generation == runGeneration else { return }
            counterName = attemptedCounterName
            currentRunID = runID
            currentInferenceStartedAt = inferenceStartedAt
            currentInferenceCompletedAt = Date()
            lastCounterIdentifier = counter.identifier
            lastCountOptions = options
            lastCountDiagnostics = res.diagnostics
            result = res
            overlay = LabelOverlay.makeCGImage(
                labels: res.labelMask,
                width: Int(res.imageSize.width),
                height: Int(res.imageSize.height)
            )
            lastSavedStem = nil
            phase = .review
        } catch {
            guard generation == runGeneration else { return }
            errorMessage = error.localizedDescription
            phase = .review
        }
    }

    private var intendedCounterIdentifier: String? {
        if ensembleEnabled {
            return CounterRegistry.ensembleCounter(
                identifiers: selectedEnsembleSpecs.map(\.id),
                weights: selectedRunWeights,
                strategy: ensembleMergeStrategy
            )?.identifier
        }
        return (CounterRegistry.counter(for: selectedCounterID) ?? CounterRegistry.best()).identifier
    }

    private var selectedRunWeights: [String: Double] {
        guard ensembleEnabled && ensembleMergeStrategy.usesModelWeights else { return [:] }
        return Dictionary(uniqueKeysWithValues: selectedEnsembleSpecs.map {
            ($0.id, ensembleWeight(for: $0.id))
        })
    }

    private func makeCountOptions() -> CountOptions {
        CountOptions(
            polarity: polarity,
            minAreaPixels: minAreaPixels,
            maxAreaPixels: maxAreaPixels,
            micronsPerPixel: project.micronsPerPixel,
            classicalDenoiseEnabled: classicalDenoiseEnabled,
            classicalThresholdOffset: classicalThresholdOffset,
            openingIterations: openingIterations,
            closingIterations: closingIterations,
            foregroundProbabilityThreshold: foregroundProbabilityThreshold,
            boundaryProbabilityThreshold: boundaryProbabilityThreshold,
            automaticBoundaryThreshold: automaticBoundaryThreshold,
            minimumMeanCellProbability: minimumMeanCellProbability,
            coreProbabilityThreshold: coreProbabilityThreshold,
            minimumCoreFraction: minimumCoreFraction,
            minimumBoundarySupport: minimumBoundarySupport,
            separateTouchingCells: separateTouchingCells,
            invertModelInput: invertModelInput,
            computeMode: computeMode,
            ensembleModelIdentifiers: ensembleEnabled ? selectedEnsembleSpecs.map(\.id) : [],
            ensembleModelWeights: selectedRunWeights,
            ensembleMergeStrategy: ensembleMergeStrategy
        )
    }

    /// Installs an already-generated comparison mask without repeating Core ML inference. If a
    /// temporary artifact cannot be decoded, the same chosen settings are rerun as a safe fallback.
    func applyComparisonResult(_ comparison: ModelComparisonResult) async {
        guard captured != nil else { return }
        runGeneration &+= 1
        phase = .working
        errorMessage = nil
        apply(options: comparison.candidate.options)
        ensembleEnabled = false
        selectedCounterID = comparison.candidate.modelIdentifier

        do {
            let loaded = try await Task.detached(priority: .userInitiated) {
                guard let decoded = MaskPNG.decodeLabel16(
                    try Data(contentsOf: comparison.maskFileURL)
                ), decoded.width == Int(comparison.imageSize.width),
                   decoded.height == Int(comparison.imageSize.height) else {
                    throw CountError.decodeFailed
                }
                let overlayData = try Data(contentsOf: comparison.overlayFileURL)
                guard let overlay = MaskPNG.decodeImage(overlayData) else {
                    throw CountError.decodeFailed
                }
                let result = CountResult(
                    objects: comparison.objects,
                    imageSize: comparison.imageSize,
                    labelMask: decoded.labels,
                    micronsPerPixel: comparison.micronsPerPixel,
                    diagnostics: comparison.diagnostics
                )
                return (result, overlay)
            }.value

            counterName = comparison.candidate.modelDisplayName
            currentRunID = comparison.id
            currentInferenceStartedAt = comparison.inferenceStartedAt
            currentInferenceCompletedAt = comparison.inferenceCompletedAt
            lastCounterIdentifier = comparison.candidate.modelIdentifier
            lastCountOptions = comparison.candidate.options
            lastCountDiagnostics = comparison.diagnostics
            result = loaded.0
            overlay = loaded.1
            lastSavedStem = nil
            phase = .review
        } catch {
            // The comparison settings remain installed, so this recreates exactly the selected
            // result if iOS purged a temporary file before the user tapped Use.
            await runCount()
        }
    }

    /// Writes every comparison mask and its exact settings manifest to the selected project
    /// folder. Returns the shared filename stem for display in the comparison screen.
    func exportComparison(_ output: ModelComparisonOutput) async -> String? {
        guard let image = captured else { return nil }
        isSaving = true
        defer { isSaving = false }
        do {
            return try await ModelComparisonResultWriter(
                provider: provider,
                project: project
            ).save(image: image, output: output)
        } catch {
            errorMessage = "Couldn't export comparison: \(error.localizedDescription)"
            return nil
        }
    }

    private func apply(options: CountOptions) {
        polarity = options.polarity
        minAreaPixels = options.minAreaPixels
        maxAreaPixels = options.maxAreaPixels
        classicalDenoiseEnabled = options.classicalDenoiseEnabled
        classicalThresholdOffset = options.classicalThresholdOffset
        openingIterations = options.openingIterations
        closingIterations = options.closingIterations
        foregroundProbabilityThreshold = options.foregroundProbabilityThreshold
        boundaryProbabilityThreshold = options.boundaryProbabilityThreshold
        automaticBoundaryThreshold = options.automaticBoundaryThreshold
        minimumMeanCellProbability = options.minimumMeanCellProbability
        coreProbabilityThreshold = options.coreProbabilityThreshold
        minimumCoreFraction = options.minimumCoreFraction
        minimumBoundarySupport = options.minimumBoundarySupport
        separateTouchingCells = options.separateTouchingCells
        invertModelInput = options.invertModelInput
        computeMode = options.computeMode
    }

    func retake() {
        runGeneration &+= 1
        captured = nil
        overlay = nil
        result = nil
        lastCounterIdentifier = nil
        lastCountOptions = nil
        lastCountDiagnostics = nil
        currentRunID = nil
        currentInferenceStartedAt = nil
        currentInferenceCompletedAt = nil
        lastSavedStem = nil
        phase = .preview
    }

    func save() async {
        guard let image = captured,
              let result,
              let lastCounterIdentifier,
              let lastCountOptions,
              let currentRunID,
              let currentInferenceStartedAt,
              let currentInferenceCompletedAt else { return }
        isSaving = true
        defer { isSaving = false }
        let writer = CountResultWriter(provider: provider, project: project)
        do {
            let stem = try await writer.save(
                image: image, result: result,
                counterIdentifier: lastCounterIdentifier,
                counterName: counterName,
                options: lastCountOptions,
                runID: currentRunID,
                inferenceStartedAt: currentInferenceStartedAt,
                inferenceCompletedAt: currentInferenceCompletedAt,
                timestamp: Date()
            )
            if self.currentRunID == currentRunID {
                lastSavedStem = stem
            }
        } catch {
            errorMessage = "Couldn't save: \(error.localizedDescription)"
        }
    }
}
