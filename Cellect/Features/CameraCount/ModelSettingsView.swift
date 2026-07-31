import SwiftUI

/// Runtime and post-processing controls for every selectable counter.
/// Architecture, weights, and tensor dimensions are intentionally read-only because changing
/// those values would require retraining or exporting a different Core ML model.
struct ModelSettingsView: View {
    @Bindable var model: CameraCountModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                modelSection
                objectFilterSection

                if model.selectedCounterIsTrainedModel {
                    trainedModelSection
                } else {
                    classicalSection
                }

                maskCleanupSection

                Section {
                    Button("Reset adjustable settings", role: .destructive) {
                        model.resetAdjustableSettings()
                    }
                }
            }
            .navigationTitle("Model settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .safeAreaInset(edge: .bottom) {
                Button {
                    Task {
                        await model.recount()
                        dismiss()
                    }
                } label: {
                    if model.phase == .working {
                        ProgressView("Counting…")
                            .frame(maxWidth: .infinity)
                    } else {
                        Label("Apply and recount", systemImage: "arrow.clockwise")
                            .frame(maxWidth: .infinity)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    model.captured == nil
                        || model.phase == .working
                        || (model.ensembleEnabled && model.selectedEnsembleSpecs.count < 2)
                )
                .padding()
                .background(.regularMaterial)
            }
        }
    }

    private var modelSection: some View {
        Section {
            Toggle(
                "Combine multiple trained models",
                isOn: Binding(
                    get: { model.ensembleEnabled },
                    set: { model.setEnsembleEnabled($0) }
                )
            )
            .disabled(!model.canEnableEnsemble)
            settingHelp(.ensembleEnabled)

            if model.ensembleEnabled {
                Picker("Merge probabilities", selection: $model.ensembleMergeStrategy) {
                    ForEach(EnsembleMergeStrategy.allCases) { strategy in
                        Text(strategy.title).tag(strategy)
                    }
                }

                Text(model.ensembleMergeStrategy.explanation)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                settingHelp(.ensembleMergeStrategy)

                ForEach(model.availableModelSpecs) { spec in
                    ensembleModelRow(spec)
                }
                settingHelp(.ensembleModelSelection)

                if model.ensembleMergeStrategy.usesModelWeights {
                    settingHelp(.ensembleWeight)
                }

                LabeledContent(
                    "Selected",
                    value: "\(model.selectedEnsembleSpecs.count) models"
                )
            } else {
                Picker("Model", selection: $model.selectedCounterID) {
                    ForEach(model.availableCounterIDs, id: \.self) { identifier in
                        Text(model.counterDisplayName(for: identifier)).tag(identifier)
                    }
                }
                settingHelp(.modelSelection)

                if let spec = model.selectedModelSpec {
                    if spec.isRecommended {
                        Label("Recommended from the workstation evaluation", systemImage: "star.fill")
                            .foregroundStyle(.green)
                    }
                    modelDetails(spec)
                } else {
                    Text("Deterministic median filtering, thresholding, morphology, and connected components. No trained weights.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        } header: {
            Text("Counting model")
        } footer: {
            if model.ensembleEnabled {
                Text("Models run one at a time to limit iPhone memory use. Their foreground and boundary probabilities are merged before a single shared mask reconstruction. Select another model before removing either of the final two.")
            } else if model.selectedCounterIsTrainedModel {
                Text("Weights, architecture, and input dimensions are fixed by training. Every setting that can affect on-device inference or mask post-processing is available below.")
            }
        }
    }

    @ViewBuilder
    private func ensembleModelRow(_ spec: CoreMLCounterSpec) -> some View {
        let selected = model.isEnsembleModelSelected(spec.id)
        VStack(alignment: .leading, spacing: 8) {
            Toggle(
                isOn: Binding(
                    get: { model.isEnsembleModelSelected(spec.id) },
                    set: { model.setEnsembleModel(spec.id, selected: $0) }
                )
            ) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(spec.displayName)
                    Text("\(spec.architecture) · \(spec.inputSize)² · Dice \(String(format: "%.3f", spec.testDice))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .disabled(selected && !model.canDeselectEnsembleModel(spec.id))

            if selected && model.ensembleMergeStrategy.usesModelWeights {
                HStack {
                    Text("Weight")
                        .font(.caption)
                    Slider(
                        value: Binding(
                            get: { model.ensembleWeight(for: spec.id) },
                            set: { model.setEnsembleWeight($0, for: spec.id) }
                        ),
                        in: 0.1...3,
                        step: 0.1
                    )
                    Text(String(format: "%.1f", model.ensembleWeight(for: spec.id)))
                        .font(.caption.monospacedDigit())
                        .frame(width: 28, alignment: .trailing)
                    Text("(\(percent(model.normalizedEnsembleWeight(for: spec.id))))")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .frame(width: 42, alignment: .trailing)
                }
            }
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private func modelDetails(_ spec: CoreMLCounterSpec) -> some View {
        LabeledContent("Architecture", value: spec.architecture)
        LabeledContent("Fixed input", value: "\(spec.inputSize) × \(spec.inputSize)")
        LabeledContent("Parameters", value: parameterText(spec.parameterCount))
        LabeledContent("Core ML precision", value: spec.precision)
        LabeledContent("Export version", value: spec.exportVersion)
        LabeledContent(
            "Source weights SHA-256",
            value: "\(spec.sourceTorchScriptSHA256.prefix(12))…"
        )
        LabeledContent("Held-out Dice", value: String(format: "%.4f", spec.testDice))
        LabeledContent("Held-out IoU", value: String(format: "%.4f", spec.testIoU))
        LabeledContent("Count MAE", value: String(format: "%.1f", spec.countMAE))
    }

    private var objectFilterSection: some View {
        Section {
            Stepper(
                "Minimum area: \(model.minAreaPixels) px",
                value: $model.minAreaPixels,
                in: 0...1_000_000,
                step: 10
            )
            settingHelp(.minimumArea)

            Toggle(
                "Limit maximum area",
                isOn: Binding(
                    get: { model.maxAreaPixels > 0 },
                    set: { enabled in
                        model.maxAreaPixels = enabled
                            ? max(model.minAreaPixels + 10, 10_000)
                            : 0
                    }
                )
            )

            if model.maxAreaPixels > 0 {
                Stepper(
                    "Maximum area: \(model.maxAreaPixels) px",
                    value: $model.maxAreaPixels,
                    in: max(model.minAreaPixels, 10)...10_000_000,
                    step: 100
                )
            }
            settingHelp(.maximumArea)
        } header: {
            Text("Object-size filter")
        } footer: {
            Text("Area is measured after the model mask is mapped back to the source image.")
        }
    }

    private var trainedModelSection: some View {
        Section {
            VStack(alignment: .leading) {
                LabeledContent(
                    "Foreground confidence",
                    value: percent(model.foregroundProbabilityThreshold)
                )
                Slider(
                    value: $model.foregroundProbabilityThreshold,
                    in: 0.05...0.95,
                    step: 0.01
                )
            }
            settingHelp(.foregroundThreshold)

            Toggle("Separate touching cells", isOn: $model.separateTouchingCells)
            settingHelp(.touchingSeparation)

            if model.separateTouchingCells {
                Toggle(
                    "Adaptive boundary cutoff (experimental)",
                    isOn: $model.automaticBoundaryThreshold
                )
                settingHelp(.adaptiveBoundary)

                VStack(alignment: .leading) {
                    LabeledContent(
                        model.automaticBoundaryThreshold
                            ? "Preferred cutoff"
                            : "Boundary cutoff",
                        value: percent(model.boundaryProbabilityThreshold)
                    )
                    Slider(
                        value: $model.boundaryProbabilityThreshold,
                        in: 0.05...0.95,
                        step: 0.01
                    )
                }
                settingHelp(.boundaryCutoff)

                if model.automaticBoundaryThreshold {
                    Text("The preferred cutoff is a user-controlled prior and safe fallback. Adaptive mode compares instance partitions across a 10–90% sweep; it measures heuristic stability, not accuracy against ground truth.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let diagnostics = model.lastCountDiagnostics,
                   diagnostics.automaticBoundarySelectionUsed,
                   let chosen = diagnostics.selectedBoundaryProbabilityThreshold {
                    adaptiveBoundaryResult(diagnostics, chosen: chosen)
                }
            }

            VStack(alignment: .leading) {
                LabeledContent(
                    "Minimum mean cell confidence",
                    value: disabledPercent(model.minimumMeanCellProbability)
                )
                Slider(
                    value: $model.minimumMeanCellProbability,
                    in: 0...0.95,
                    step: 0.01
                )
            }
            settingHelp(.meanCellConfidence)

            VStack(alignment: .leading) {
                LabeledContent(
                    "High-confidence core level",
                    value: percent(model.coreProbabilityThreshold)
                )
                Slider(
                    value: $model.coreProbabilityThreshold,
                    in: 0.50...0.95,
                    step: 0.01
                )
            }
            settingHelp(.coreProbabilityLevel)

            VStack(alignment: .leading) {
                LabeledContent(
                    "Minimum high-confidence core",
                    value: disabledPercent(model.minimumCoreFraction)
                )
                Slider(
                    value: $model.minimumCoreFraction,
                    in: 0...0.95,
                    step: 0.01
                )
            }
            settingHelp(.minimumCoreFraction)

            VStack(alignment: .leading) {
                LabeledContent(
                    "Minimum perimeter support",
                    value: disabledPercent(model.minimumBoundarySupport)
                )
                Slider(
                    value: $model.minimumBoundarySupport,
                    in: 0...0.95,
                    step: 0.01
                )
            }
            settingHelp(.minimumPerimeterSupport)

            Toggle("Invert grayscale input", isOn: $model.invertModelInput)
            settingHelp(.inputInversion)

            Picker("Compute hardware", selection: $model.computeMode) {
                ForEach(ModelComputeMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            settingHelp(.computeHardware)
        } header: {
            Text("Trained-model inference")
        } footer: {
            Text("Boundary cutoff controls how touching foreground regions are divided; it does not remove foreground false positives. The separate mean-confidence, core, perimeter, and area filters remove weak or implausible regions. A zero minimum disables that filter. Automatic compute lets Core ML choose CPU, GPU, or Neural Engine.")
        }
    }

    @ViewBuilder
    private func adaptiveBoundaryResult(
        _ diagnostics: CountDiagnostics,
        chosen: Double
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(
                diagnostics.automaticBoundaryFallbackUsed
                    ? "Last result used preferred fallback: \(percent(chosen))"
                    : "Last result chose: \(percent(chosen))",
                systemImage: diagnostics.automaticBoundaryFallbackUsed
                    ? "arrow.uturn.backward.circle"
                    : "checkmark.circle"
            )
            .font(.subheadline)

            if let reason = diagnostics.automaticBoundaryFallbackReason {
                Text("Fallback reason: \(reason.title).")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let auc = diagnostics.boundaryStabilityAUC {
                LabeledContent("Sweep stability AUC", value: percent(auc))
                    .font(.caption)
            }
            if let score = diagnostics.automaticBoundaryHeuristicStabilityScore {
                LabeledContent("Heuristic stability score", value: percent(score))
                    .font(.caption)
            }

            Button("Use \(percent(chosen)) manually") {
                model.boundaryProbabilityThreshold = chosen
                model.automaticBoundaryThreshold = false
            }
            .font(.caption)

            if !diagnostics.boundaryCandidates.isEmpty {
                DisclosureGroup("Inspect boundary sweep") {
                    ForEach(diagnostics.boundaryCandidates) { candidate in
                        HStack(spacing: 8) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text("\(percent(candidate.threshold)) · \(candidate.objectCount) candidate objects")
                                    .font(.caption.monospacedDigit())
                                Text("local stability \(percent(candidate.localStability)) · score \(percent(candidate.selectionScore))")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Button("Use") {
                                model.boundaryProbabilityThreshold = candidate.threshold
                                model.automaticBoundaryThreshold = false
                            }
                            .font(.caption)
                        }
                        .padding(.vertical, 2)
                    }
                }
                .font(.caption)

                Text("Candidate-object counts are measured before the separate weak-object confidence filters are applied.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }

    private var classicalSection: some View {
        Section {
            Picker("Cell appearance", selection: $model.polarity) {
                ForEach(CountPolarity.allCases) { polarity in
                    Text(polarity.title).tag(polarity)
                }
            }
            .pickerStyle(.segmented)
            settingHelp(.classicalPolarity)

            Toggle("3 × 3 median denoise", isOn: $model.classicalDenoiseEnabled)
            settingHelp(.medianDenoise)

            VStack(alignment: .leading) {
                LabeledContent(
                    "Otsu threshold offset",
                    value: signed(model.classicalThresholdOffset)
                )
                Slider(
                    value: Binding(
                        get: { Double(model.classicalThresholdOffset) },
                        set: { model.classicalThresholdOffset = Int($0.rounded()) }
                    ),
                    in: -100...100,
                    step: 1
                )
            }
            settingHelp(.otsuOffset)
        } header: {
            Text("Classical CV")
        } footer: {
            Text("The offset shifts the automatically calculated 8-bit Otsu threshold.")
        }
    }

    private var maskCleanupSection: some View {
        Section {
            Stepper(
                "Opening iterations: \(model.openingIterations)",
                value: $model.openingIterations,
                in: 0...4
            )
            settingHelp(.openingIterations)
            Stepper(
                "Closing iterations: \(model.closingIterations)",
                value: $model.closingIterations,
                in: 0...4
            )
            settingHelp(.closingIterations)
        } header: {
            Text("Mask cleanup")
        } footer: {
            Text("Opening removes isolated pixels; closing fills small gaps. Zero disables either operation.")
        }
    }

    private func parameterText(_ count: Int) -> String {
        String(format: "%.2f million", Double(count) / 1_000_000)
    }

    private func percent(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private func disabledPercent(_ value: Double) -> String {
        value == 0 ? "Off" : percent(value)
    }

    private func signed(_ value: Int) -> String {
        value > 0 ? "+\(value)" : "\(value)"
    }

    private func settingHelp(_ key: ModelSettingKey) -> some View {
        Text(ModelSettingDescriptionCatalog.helpText(for: key))
            .font(.caption2)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
    }
}
