import Foundation

/// A bounded, reproducible sweep. Continuous settings cannot literally be tested at every
/// possible value, so the app exposes the exact discrete plan before a run starts.
enum ComparisonSweepPreset: String, CaseIterable, Codable, Identifiable {
    case quick
    case thorough

    var id: String { rawValue }

    var title: String {
        switch self {
        case .quick: return "Quick"
        case .thorough: return "Thorough"
        }
    }

    var explanation: String {
        switch self {
        case .quick:
            return "Tests the baseline plus permissive and strict foreground, boundary, and confidence choices."
        case .thorough:
            return "Adds a foreground/boundary grid and one-at-a-time tests of every mask-changing filter."
        }
    }
}

/// A model choice independent of Bundle/Core ML, which keeps plan generation testable.
struct ComparisonModelChoice: Identifiable, Equatable {
    let id: String
    let displayName: String
    let isTrainedModel: Bool
}

struct ComparisonSettingChange: Codable, Equatable, Identifiable {
    let id: String
    let title: String
    let value: String
}

/// One exact model/settings combination in a comparison run.
struct ModelComparisonCandidate: Identifiable, Codable, Equatable {
    let id: String
    let ordinal: Int
    let modelIdentifier: String
    let modelDisplayName: String
    let variantName: String
    let options: CountOptions
    let changes: [ComparisonSettingChange]

    var summary: String {
        changes.isEmpty
            ? "Current settings"
            : changes.map { "\($0.title): \($0.value)" }.joined(separator: " · ")
    }
}

enum ModelComparisonPlanner {
    /// Builds candidates in the supplied model order. Duplicate option sets are removed while
    /// retaining their first human-readable variant name.
    static func makePlan(
        models: [ComparisonModelChoice],
        baseline suppliedBaseline: CountOptions,
        preset: ComparisonSweepPreset,
        includeInvertedInput: Bool
    ) -> [ModelComparisonCandidate] {
        var baseline = suppliedBaseline
        baseline.ensembleModelIdentifiers = []
        baseline.ensembleModelWeights = [:]

        var drafts = [Draft]()
        for model in models {
            let modelDrafts = model.isTrainedModel
                ? trainedDrafts(
                    baseline: baseline,
                    preset: preset,
                    includeInvertedInput: includeInvertedInput
                )
                : classicalDrafts(baseline: baseline, preset: preset)

            var uniqueOptions = [CountOptions]()
            for draft in modelDrafts where !uniqueOptions.contains(draft.options) {
                uniqueOptions.append(draft.options)
                drafts.append(Draft(
                    model: model,
                    variantName: draft.variantName,
                    options: draft.options,
                    changes: changes(from: baseline, to: draft.options, trained: model.isTrainedModel)
                ))
            }
        }

        return drafts.enumerated().map { index, draft in
            ModelComparisonCandidate(
                id: String(format: "%04d-%@-%@", index + 1, safeID(draft.model.id), safeID(draft.variantName)),
                ordinal: index,
                modelIdentifier: draft.model.id,
                modelDisplayName: draft.model.displayName,
                variantName: draft.variantName,
                options: draft.options,
                changes: draft.changes
            )
        }
    }

    private struct Draft {
        let model: ComparisonModelChoice
        let variantName: String
        let options: CountOptions
        let changes: [ComparisonSettingChange]

        init(
            model: ComparisonModelChoice,
            variantName: String,
            options: CountOptions,
            changes: [ComparisonSettingChange]
        ) {
            self.model = model
            self.variantName = variantName
            self.options = options
            self.changes = changes
        }

        init(variantName: String, options: CountOptions) {
            self.model = ComparisonModelChoice(id: "", displayName: "", isTrainedModel: true)
            self.variantName = variantName
            self.options = options
            self.changes = []
        }
    }

    private static func trainedDrafts(
        baseline: CountOptions,
        preset: ComparisonSweepPreset,
        includeInvertedInput: Bool
    ) -> [Draft] {
        var drafts = [Draft(variantName: "Baseline", options: baseline)]

        if preset == .quick {
            append(&drafts, baseline: baseline, name: "Permissive foreground") {
                $0.foregroundProbabilityThreshold = 0.35
            }
            append(&drafts, baseline: baseline, name: "Strict foreground") {
                $0.foregroundProbabilityThreshold = 0.65
            }
            append(&drafts, baseline: baseline, name: "More boundary splits") {
                $0.separateTouchingCells = true
                $0.automaticBoundaryThreshold = false
                $0.boundaryProbabilityThreshold = 0.25
            }
            append(&drafts, baseline: baseline, name: "Fewer boundary splits") {
                $0.separateTouchingCells = true
                $0.automaticBoundaryThreshold = false
                $0.boundaryProbabilityThreshold = 0.65
            }
            append(&drafts, baseline: baseline, name: "Confidence filter") {
                $0.minimumMeanCellProbability = 0.55
            }
            append(&drafts, baseline: baseline, name: "Adaptive boundary") {
                $0.separateTouchingCells = true
                $0.automaticBoundaryThreshold = true
            }
        } else {
            // The interaction most likely to change masks is sampled jointly. Other controls
            // are varied one at a time so the user can interpret why a mask changed.
            for foreground in [0.35, 0.50, 0.65] {
                for boundary in [0.25, 0.45, 0.65] {
                    append(
                        &drafts,
                        baseline: baseline,
                        name: "Foreground/boundary grid"
                    ) {
                        $0.foregroundProbabilityThreshold = foreground
                        $0.separateTouchingCells = true
                        $0.automaticBoundaryThreshold = false
                        $0.boundaryProbabilityThreshold = boundary
                    }
                }
            }

            for preferred in [0.30, 0.45, 0.60] {
                append(&drafts, baseline: baseline, name: "Adaptive boundary") {
                    $0.separateTouchingCells = true
                    $0.automaticBoundaryThreshold = true
                    $0.boundaryProbabilityThreshold = preferred
                }
            }
            for minimum in [0.25, 0.50, 0.70] {
                append(&drafts, baseline: baseline, name: "Mean confidence filter") {
                    $0.minimumMeanCellProbability = minimum
                }
            }
            for coreLevel in [0.60, 0.80] {
                for coreFraction in [0.20, 0.50] {
                    append(&drafts, baseline: baseline, name: "Core confidence filter") {
                        $0.coreProbabilityThreshold = coreLevel
                        $0.minimumCoreFraction = coreFraction
                    }
                }
            }
            for support in [0.15, 0.35] {
                append(&drafts, baseline: baseline, name: "Perimeter support filter") {
                    $0.minimumBoundarySupport = support
                }
            }
            append(&drafts, baseline: baseline, name: "Do not split touching cells") {
                $0.separateTouchingCells = false
                $0.automaticBoundaryThreshold = false
            }
            append(&drafts, baseline: baseline, name: "No opening") {
                $0.openingIterations = 0
            }
            append(&drafts, baseline: baseline, name: "Stronger opening") {
                $0.openingIterations = 2
            }
            append(&drafts, baseline: baseline, name: "No closing") {
                $0.closingIterations = 0
            }
            append(&drafts, baseline: baseline, name: "Stronger closing") {
                $0.closingIterations = 2
            }
            append(&drafts, baseline: baseline, name: "Keep smaller objects") {
                $0.minAreaPixels = max(0, baseline.minAreaPixels / 2)
            }
            append(&drafts, baseline: baseline, name: "Remove more small objects") {
                $0.minAreaPixels = max(baseline.minAreaPixels + 1, baseline.minAreaPixels * 2)
            }
            if baseline.maxAreaPixels > 0 {
                append(&drafts, baseline: baseline, name: "More permissive maximum area") {
                    $0.maxAreaPixels = max(baseline.maxAreaPixels + 1, baseline.maxAreaPixels * 2)
                }
                append(&drafts, baseline: baseline, name: "Stricter maximum area") {
                    $0.maxAreaPixels = max($0.minAreaPixels, baseline.maxAreaPixels / 2)
                }
            }
        }

        if includeInvertedInput {
            append(&drafts, baseline: baseline, name: "Inverted grayscale input") {
                $0.invertModelInput.toggle()
            }
        }
        return drafts
    }

    private static func classicalDrafts(
        baseline: CountOptions,
        preset: ComparisonSweepPreset
    ) -> [Draft] {
        var drafts = [Draft(variantName: "Baseline", options: baseline)]
        append(&drafts, baseline: baseline, name: "Opposite polarity") {
            $0.polarity = $0.polarity == .darkObjects ? .brightObjects : .darkObjects
        }
        for offset in (preset == .quick ? [-20, 20] : [-40, -20, 0, 20, 40]) {
            append(&drafts, baseline: baseline, name: "Otsu threshold offset") {
                $0.classicalThresholdOffset = offset
            }
        }
        append(&drafts, baseline: baseline, name: "Toggle denoising") {
            $0.classicalDenoiseEnabled.toggle()
        }
        if preset == .thorough {
            for opening in [0, 2] {
                append(&drafts, baseline: baseline, name: "Opening cleanup") {
                    $0.openingIterations = opening
                }
            }
            for closing in [0, 2] {
                append(&drafts, baseline: baseline, name: "Closing cleanup") {
                    $0.closingIterations = closing
                }
            }
            append(&drafts, baseline: baseline, name: "Keep smaller objects") {
                $0.minAreaPixels = max(0, baseline.minAreaPixels / 2)
            }
            append(&drafts, baseline: baseline, name: "Remove more small objects") {
                $0.minAreaPixels = max(baseline.minAreaPixels + 1, baseline.minAreaPixels * 2)
            }
        }
        return drafts
    }

    private static func append(
        _ drafts: inout [Draft],
        baseline: CountOptions,
        name: String,
        mutation: (inout CountOptions) -> Void
    ) {
        var options = baseline
        mutation(&options)
        drafts.append(Draft(variantName: name, options: options))
    }

    private static func changes(
        from baseline: CountOptions,
        to options: CountOptions,
        trained: Bool
    ) -> [ComparisonSettingChange] {
        var output = [ComparisonSettingChange]()
        func add(_ id: String, _ title: String, _ value: String) {
            output.append(ComparisonSettingChange(id: id, title: title, value: value))
        }
        func percent(_ value: Double) -> String { "\(Int((value * 100).rounded()))%" }

        if options.minAreaPixels != baseline.minAreaPixels {
            add("minimumArea", "Minimum area", "\(options.minAreaPixels) px")
        }
        if options.maxAreaPixels != baseline.maxAreaPixels {
            add(
                "maximumArea",
                "Maximum area",
                options.maxAreaPixels == 0 ? "Off" : "\(options.maxAreaPixels) px"
            )
        }
        if options.openingIterations != baseline.openingIterations {
            add("opening", "Opening", "\(options.openingIterations)")
        }
        if options.closingIterations != baseline.closingIterations {
            add("closing", "Closing", "\(options.closingIterations)")
        }

        if trained {
            if options.foregroundProbabilityThreshold != baseline.foregroundProbabilityThreshold {
                add("foregroundConfidence", "Foreground", percent(options.foregroundProbabilityThreshold))
            }
            if options.separateTouchingCells != baseline.separateTouchingCells {
                add("separateTouchingCells", "Separate touching cells", options.separateTouchingCells ? "On" : "Off")
            }
            if options.automaticBoundaryThreshold != baseline.automaticBoundaryThreshold {
                add("automaticBoundary", "Adaptive boundary", options.automaticBoundaryThreshold ? "On" : "Off")
            }
            if options.boundaryProbabilityThreshold != baseline.boundaryProbabilityThreshold {
                add("boundaryCutoff", "Boundary cutoff", percent(options.boundaryProbabilityThreshold))
            }
            if options.minimumMeanCellProbability != baseline.minimumMeanCellProbability {
                add("meanCellConfidence", "Mean cell confidence", percent(options.minimumMeanCellProbability))
            }
            if options.coreProbabilityThreshold != baseline.coreProbabilityThreshold {
                add("coreLevel", "Core level", percent(options.coreProbabilityThreshold))
            }
            if options.minimumCoreFraction != baseline.minimumCoreFraction {
                add("coreFraction", "Minimum core", percent(options.minimumCoreFraction))
            }
            if options.minimumBoundarySupport != baseline.minimumBoundarySupport {
                add("boundarySupport", "Perimeter support", percent(options.minimumBoundarySupport))
            }
            if options.invertModelInput != baseline.invertModelInput {
                add("invertInput", "Invert input", options.invertModelInput ? "On" : "Off")
            }
        } else {
            if options.polarity != baseline.polarity {
                add("polarity", "Cell polarity", options.polarity.title)
            }
            if options.classicalDenoiseEnabled != baseline.classicalDenoiseEnabled {
                add("denoise", "Median denoise", options.classicalDenoiseEnabled ? "On" : "Off")
            }
            if options.classicalThresholdOffset != baseline.classicalThresholdOffset {
                add("thresholdOffset", "Otsu offset", "\(options.classicalThresholdOffset)")
            }
        }
        return output
    }

    private static func safeID(_ text: String) -> String {
        let allowed = CharacterSet.alphanumerics
        return text.lowercased().unicodeScalars.map {
            allowed.contains($0) ? Character(String($0)) : "-"
        }.reduce(into: "") { $0.append($1) }
            .replacingOccurrences(of: "--", with: "-")
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
    }
}
