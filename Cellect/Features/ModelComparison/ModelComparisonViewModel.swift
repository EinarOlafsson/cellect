import CoreGraphics
import Foundation
import Observation

@MainActor
@Observable
final class ModelComparisonViewModel {
    enum Phase: Equatable {
        case setup
        case running
        case results
    }

    let image: CGImage
    let baseline: CountOptions
    let modelChoices: [ComparisonModelChoice]

    var phase: Phase = .setup
    var selectedModelIDs: Set<String>
    var sweepPreset: ComparisonSweepPreset = .quick
    var includeInvertedInput = false
    private(set) var progress: ModelComparisonProgress?
    private(set) var results = [ModelComparisonResult]()
    private(set) var failures = [ModelComparisonFailure]()
    private(set) var selectedIndex = 0
    private(set) var selectedOverlay: CGImage?
    private(set) var errorMessage: String?
    private(set) var output: ModelComparisonOutput?

    private let modelSpecs: [CoreMLCounterSpec]
    private var runTask: Task<Void, Never>?
    private var overlayTask: Task<Void, Never>?
    private var selectionGeneration = 0

    init(
        image: CGImage,
        baseline: CountOptions,
        modelSpecs: [CoreMLCounterSpec]
    ) {
        self.image = image
        self.baseline = baseline
        self.modelSpecs = modelSpecs
        self.modelChoices = modelSpecs.map {
            ComparisonModelChoice(
                id: $0.id,
                displayName: $0.displayName,
                isTrainedModel: true
            )
        } + [ComparisonModelChoice(
            id: "classical",
            displayName: "Classical CV (denoise + Otsu)",
            isTrainedModel: false
        )]
        self.selectedModelIDs = Set(modelSpecs.map(\.id) + ["classical"])
    }

    var selectedChoices: [ComparisonModelChoice] {
        modelChoices.filter { selectedModelIDs.contains($0.id) }
    }

    var plan: [ModelComparisonCandidate] {
        ModelComparisonPlanner.makePlan(
            models: selectedChoices,
            baseline: baseline,
            preset: sweepPreset,
            includeInvertedInput: includeInvertedInput
        )
    }

    var canStart: Bool { !plan.isEmpty && phase != .running }

    var selectedResult: ModelComparisonResult? {
        guard results.indices.contains(selectedIndex) else { return nil }
        return results[selectedIndex]
    }

    var selectionValue: Double { Double(selectedIndex) }

    var progressFraction: Double {
        guard let progress, progress.total > 0 else { return 0 }
        return Double(progress.completed) / Double(progress.total)
    }

    func setModel(_ identifier: String, selected: Bool) {
        if selected {
            selectedModelIDs.insert(identifier)
        } else {
            selectedModelIDs.remove(identifier)
        }
    }

    func selectAllModels() {
        selectedModelIDs = Set(modelChoices.map(\.id))
    }

    func clearModelSelection() {
        selectedModelIDs.removeAll()
    }

    func start() {
        guard canStart else { return }
        runTask?.cancel()
        overlayTask?.cancel()
        cleanOutputArtifacts()
        let candidates = plan
        phase = .running
        progress = ModelComparisonProgress(
            completed: 0,
            total: candidates.count,
            modelName: candidates[0].modelDisplayName,
            variantName: candidates[0].variantName
        )
        results = []
        failures = []
        selectedOverlay = nil
        errorMessage = nil

        runTask = Task { [weak self] in
            guard let self else { return }
            do {
                let output = try await ModelComparisonRunner.run(
                    image: image,
                    candidates: candidates,
                    modelSpecs: modelSpecs
                ) { [weak self] progress in
                    guard let self, self.phase == .running else { return }
                    self.progress = progress
                }
                guard !Task.isCancelled, phase == .running else { return }
                self.output = output
                results = output.results
                failures = output.failures
                guard !results.isEmpty else {
                    errorMessage = failures.first?.message
                        ?? "No comparison masks could be generated."
                    phase = .setup
                    return
                }
                selectedIndex = 0
                phase = .results
                loadOverlay(for: 0)
            } catch is CancellationError {
                if phase == .running { phase = .setup }
            } catch {
                errorMessage = error.localizedDescription
                phase = .setup
            }
        }
    }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        progress = nil
        if phase == .running { phase = .setup }
    }

    func selectResult(at proposedIndex: Int) {
        guard !results.isEmpty else { return }
        let index = min(results.count - 1, max(0, proposedIndex))
        guard index != selectedIndex || selectedOverlay == nil else { return }
        selectedIndex = index
        loadOverlay(for: index)
    }

    func selectResult(sliderValue: Double) {
        selectResult(at: Int(sliderValue.rounded()))
    }

    func close() {
        cancel()
        overlayTask?.cancel()
        overlayTask = nil
        cleanOutputArtifacts()
    }

    private func loadOverlay(for index: Int) {
        guard results.indices.contains(index) else { return }
        selectionGeneration &+= 1
        let generation = selectionGeneration
        let url = results[index].overlayFileURL
        selectedOverlay = nil
        overlayTask?.cancel()
        overlayTask = Task {
            // Debounce fast slider scrubbing so intermediate positions do not all decode a
            // source-resolution PNG. Cancellation also propagates to the active decode worker.
            try? await Task.sleep(nanoseconds: 80_000_000)
            guard !Task.isCancelled else { return }
            let worker = Task.detached(priority: .userInitiated) {
                guard !Task.isCancelled,
                      let data = try? Data(contentsOf: url) else { return nil as CGImage? }
                return MaskPNG.decodeImage(data)
            }
            let image = await withTaskCancellationHandler {
                await worker.value
            } onCancel: {
                worker.cancel()
            }
            guard !Task.isCancelled,
                  generation == selectionGeneration,
                  selectedIndex == index else { return }
            selectedOverlay = image
        }
    }

    private func cleanOutputArtifacts() {
        guard let directory = output?.artifactsDirectory else { return }
        output = nil
        try? FileManager.default.removeItem(at: directory)
    }
}
