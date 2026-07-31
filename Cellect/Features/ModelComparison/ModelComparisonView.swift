import SwiftUI

struct ModelComparisonView: View {
    @Bindable var cameraModel: CameraCountModel
    @State private var comparison: ModelComparisonViewModel
    @Environment(\.dismiss) private var dismiss

    @State private var isApplying = false
    @State private var isExporting = false
    @State private var exportedStem: String?
    @State private var zoomScale: CGFloat = 1
    @State private var committedZoomScale: CGFloat = 1
    @State private var zoomOffset: CGSize = .zero
    @State private var committedZoomOffset: CGSize = .zero

    init(
        cameraModel: CameraCountModel,
        image: CGImage,
        baseline: CountOptions,
        modelSpecs: [CoreMLCounterSpec]
    ) {
        self.cameraModel = cameraModel
        _comparison = State(initialValue: ModelComparisonViewModel(
            image: image,
            baseline: baseline,
            modelSpecs: modelSpecs
        ))
    }

    var body: some View {
        NavigationStack {
            Group {
                switch comparison.phase {
                case .setup:
                    setupView
                case .running:
                    runningView
                case .results:
                    resultsView
                }
            }
            .navigationTitle("Compare models")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        comparison.close()
                        dismiss()
                    }
                    .disabled(isApplying || isExporting)
                }
            }
        }
        .interactiveDismissDisabled(comparison.phase == .running || isApplying || isExporting)
        .onDisappear { comparison.close() }
    }

    // MARK: - Setup

    private var setupView: some View {
        Form {
            Section {
                HStack {
                    Button("Select all") { comparison.selectAllModels() }
                    Spacer()
                    Button("Clear") { comparison.clearModelSelection() }
                }

                ForEach(comparison.modelChoices) { choice in
                    Toggle(
                        isOn: Binding(
                            get: { comparison.selectedModelIDs.contains(choice.id) },
                            set: { comparison.setModel(choice.id, selected: $0) }
                        )
                    ) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(choice.displayName)
                            Text(choice.isTrainedModel
                                 ? "Core ML foreground + contact-boundary model"
                                 : "Threshold-based reference method")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
            } header: {
                Text("Models")
            } footer: {
                Text("All available models are selected initially and run one at a time to protect iPhone memory.")
            }

            Section {
                Picker("Sweep", selection: $comparison.sweepPreset) {
                    ForEach(ComparisonSweepPreset.allCases) { preset in
                        Text(preset.title).tag(preset)
                    }
                }
                .pickerStyle(.segmented)

                Text(comparison.sweepPreset.explanation)
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Toggle("Also test inverted input", isOn: $comparison.includeInvertedInput)
                Text(ModelSettingDescriptionCatalog.helpText(for: .inputInversion))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                LabeledContent("Masks to generate", value: "\(comparison.plan.count)")
                    .font(.headline)
            } header: {
                Text("Settings sweep")
            } footer: {
                Text("Continuous sliders have infinitely many possible values. Cellect uses the finite, reproducible grid shown by the preset; every exact value is saved with its mask.")
            }

            if comparison.plan.count >= 150 {
                Section {
                    Label(
                        "This is a large run. Keep Cellect in the foreground and connect the phone to power.",
                        systemImage: "clock.badge.exclamationmark"
                    )
                    .foregroundStyle(.orange)
                }
            }

            if let error = comparison.errorMessage {
                Section {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.red)
                }
            }

            Section {
                Button {
                    comparison.start()
                } label: {
                    Label(
                        "Generate \(comparison.plan.count) masks",
                        systemImage: "square.stack.3d.up"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!comparison.canStart)
            }
        }
    }

    // MARK: - Running

    private var runningView: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            Image(decorative: comparison.image, scale: 1)
                .resizable()
                .scaledToFit()
                .opacity(0.32)

            VStack(spacing: 18) {
                ProgressView(value: comparison.progressFraction)
                    .progressViewStyle(.linear)
                    .tint(.mint)
                    .frame(maxWidth: 320)

                if let progress = comparison.progress {
                    Text("\(progress.completed) of \(progress.total) masks")
                        .font(.title3.bold())
                        .monospacedDigit()
                    Text(progress.modelName)
                        .font(.headline)
                    Text(progress.variantName)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Preparing comparison…")
                }

                Text("Core ML inference is reused across post-processing settings for each model.")
                    .font(.caption)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: 320)

                Button("Cancel", role: .cancel) { comparison.cancel() }
                    .buttonStyle(.bordered)
            }
            .padding(24)
            .foregroundStyle(.white)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 20))
            .padding()
        }
    }

    // MARK: - Results

    private var resultsView: some View {
        VStack(spacing: 0) {
            comparisonCanvas
                .frame(minHeight: 260)
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    sliderControls
                    if let selected = comparison.selectedResult {
                        resultSummary(selected)
                        changedSettings(selected)
                        exactSettings(selected)
                        resultActions(selected)
                    }
                    failureSummary
                }
                .padding()
            }
            .background(.regularMaterial)
        }
        .background(Color.black)
    }

    private var comparisonCanvas: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black
                Image(decorative: comparison.image, scale: 1)
                    .resizable()
                    .scaledToFit()
                if let overlay = comparison.selectedOverlay {
                    Image(decorative: overlay, scale: 1)
                        .resizable()
                        .scaledToFit()
                } else {
                    ProgressView().tint(.white)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .scaleEffect(zoomScale)
            .offset(zoomOffset)
            .contentShape(Rectangle())
            .gesture(
                MagnifyGesture()
                    .onChanged { value in
                        zoomScale = min(8, max(1, committedZoomScale * value.magnification))
                        zoomOffset = clampedOffset(
                            committedZoomOffset,
                            scale: zoomScale,
                            viewport: geometry.size
                        )
                    }
                    .onEnded { _ in
                        committedZoomScale = zoomScale
                        zoomOffset = clampedOffset(
                            zoomOffset,
                            scale: zoomScale,
                            viewport: geometry.size
                        )
                        committedZoomOffset = zoomOffset
                    }
                    .simultaneously(with:
                        DragGesture(minimumDistance: 5)
                            .onChanged { value in
                                guard zoomScale > 1 else { return }
                                zoomOffset = clampedOffset(
                                    CGSize(
                                        width: committedZoomOffset.width + value.translation.width,
                                        height: committedZoomOffset.height + value.translation.height
                                    ),
                                    scale: zoomScale,
                                    viewport: geometry.size
                                )
                            }
                            .onEnded { _ in committedZoomOffset = zoomOffset }
                    )
            )
            .onTapGesture(count: 2) { resetZoom() }
            .clipped()
            .overlay(alignment: .top) {
                Text("Pinch to inspect boundaries · double-tap to reset")
                    .font(.caption2)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.top, 6)
            }
        }
    }

    private var sliderControls: some View {
        VStack(spacing: 8) {
            HStack {
                Button {
                    comparison.selectResult(at: comparison.selectedIndex - 1)
                } label: {
                    Image(systemName: "chevron.left")
                }
                .disabled(comparison.selectedIndex == 0)

                Slider(
                    value: Binding(
                        get: { comparison.selectionValue },
                        set: { comparison.selectResult(sliderValue: $0) }
                    ),
                    in: 0...Double(max(1, comparison.results.count - 1)),
                    step: 1
                )

                Button {
                    comparison.selectResult(at: comparison.selectedIndex + 1)
                } label: {
                    Image(systemName: "chevron.right")
                }
                .disabled(comparison.selectedIndex + 1 >= comparison.results.count)
            }

            Text("Mask \(comparison.selectedIndex + 1) of \(comparison.results.count)")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private func resultSummary(_ result: ModelComparisonResult) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(result.candidate.modelDisplayName)
                .font(.headline)
            Text(result.candidate.variantName)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            HStack {
                Label("\(result.objectCount) cells", systemImage: "circle.grid.3x3.fill")
                Spacer()
                Text("mean ⌀ \(String(format: "%.0f", result.meanDiameterPixels)) px")
                    .monospacedDigit()
            }
            .font(.subheadline)
        }
    }

    @ViewBuilder
    private func changedSettings(_ result: ModelComparisonResult) -> some View {
        GroupBox("What changed") {
            if result.candidate.changes.isEmpty {
                Text("This mask uses the settings that were active when comparison began.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 12) {
                    ForEach(result.candidate.changes) { change in
                        VStack(alignment: .leading, spacing: 3) {
                            LabeledContent(change.title, value: change.value)
                                .font(.subheadline.bold())
                            if let key = settingKey(for: change.id) {
                                Text(ModelSettingDescriptionCatalog.helpText(for: key))
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private func exactSettings(_ result: ModelComparisonResult) -> some View {
        DisclosureGroup("Exact settings and explanations") {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(settingPresentations(for: result)) { item in
                    VStack(alignment: .leading, spacing: 3) {
                        LabeledContent(item.guidance.title, value: item.value)
                            .font(.subheadline.bold())
                        Text(item.guidance.helpText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(.top, 8)
        }
    }

    private func resultActions(_ result: ModelComparisonResult) -> some View {
        VStack(spacing: 10) {
            Button {
                guard let output = comparison.output else { return }
                isExporting = true
                Task {
                    exportedStem = await cameraModel.exportComparison(output)
                    isExporting = false
                }
            } label: {
                if isExporting {
                    ProgressView("Exporting all masks…")
                        .frame(maxWidth: .infinity)
                } else {
                    Label("Export all masks", systemImage: "square.and.arrow.up.on.square")
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.bordered)
            .disabled(isExporting || isApplying || comparison.output == nil)

            if let exportedStem {
                Label("Exported as \(exportedStem)…", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            }

            Button {
                isApplying = true
                Task {
                    await cameraModel.applyComparisonResult(result)
                    isApplying = false
                    comparison.close()
                    dismiss()
                }
            } label: {
                if isApplying {
                    ProgressView("Applying mask…")
                        .frame(maxWidth: .infinity)
                } else {
                    Label("Use this model and settings", systemImage: "checkmark.circle.fill")
                        .frame(maxWidth: .infinity)
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isApplying || isExporting)
        }
    }

    @ViewBuilder
    private var failureSummary: some View {
        if !comparison.failures.isEmpty {
            DisclosureGroup("\(comparison.failures.count) failed iterations") {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(comparison.failures) { failure in
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(failure.modelName) · \(failure.variantName)")
                                .font(.caption.bold())
                            Text(failure.message)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(.top, 6)
            }
            .foregroundStyle(.orange)
        }
    }

    // MARK: - Setting presentation

    private struct SettingPresentation: Identifiable {
        let key: ModelSettingKey
        let value: String
        var id: String { key.rawValue }
        var guidance: ModelSettingGuidance {
            ModelSettingDescriptionCatalog.guidance(for: key)
        }
    }

    private func settingPresentations(
        for result: ModelComparisonResult
    ) -> [SettingPresentation] {
        let options = result.candidate.options
        var items = [
            SettingPresentation(key: .minimumArea, value: "\(options.minAreaPixels) px"),
            SettingPresentation(
                key: .maximumArea,
                value: options.maxAreaPixels == 0 ? "Off" : "\(options.maxAreaPixels) px"
            ),
            SettingPresentation(key: .openingIterations, value: "\(options.openingIterations)"),
            SettingPresentation(key: .closingIterations, value: "\(options.closingIterations)"),
        ]
        if result.candidate.modelIdentifier == "classical" {
            items.insert(contentsOf: [
                SettingPresentation(key: .classicalPolarity, value: options.polarity.title),
                SettingPresentation(key: .medianDenoise, value: options.classicalDenoiseEnabled ? "On" : "Off"),
                SettingPresentation(key: .otsuOffset, value: "\(options.classicalThresholdOffset)"),
            ], at: 0)
        } else {
            items.insert(contentsOf: [
                SettingPresentation(key: .foregroundThreshold, value: percent(options.foregroundProbabilityThreshold)),
                SettingPresentation(key: .touchingSeparation, value: options.separateTouchingCells ? "On" : "Off"),
                SettingPresentation(key: .adaptiveBoundary, value: options.automaticBoundaryThreshold ? "On" : "Off"),
                SettingPresentation(key: .boundaryCutoff, value: percent(options.boundaryProbabilityThreshold)),
                SettingPresentation(key: .meanCellConfidence, value: disabledPercent(options.minimumMeanCellProbability)),
                SettingPresentation(key: .coreProbabilityLevel, value: percent(options.coreProbabilityThreshold)),
                SettingPresentation(key: .minimumCoreFraction, value: disabledPercent(options.minimumCoreFraction)),
                SettingPresentation(key: .minimumPerimeterSupport, value: disabledPercent(options.minimumBoundarySupport)),
                SettingPresentation(key: .inputInversion, value: options.invertModelInput ? "On" : "Off"),
                SettingPresentation(key: .computeHardware, value: options.computeMode.title),
            ], at: 0)
        }
        return items
    }

    private func settingKey(for changeID: String) -> ModelSettingKey? {
        switch changeID {
        case "minimumArea": return .minimumArea
        case "maximumArea": return .maximumArea
        case "opening": return .openingIterations
        case "closing": return .closingIterations
        case "foregroundConfidence": return .foregroundThreshold
        case "separateTouchingCells": return .touchingSeparation
        case "automaticBoundary": return .adaptiveBoundary
        case "boundaryCutoff": return .boundaryCutoff
        case "meanCellConfidence": return .meanCellConfidence
        case "coreLevel": return .coreProbabilityLevel
        case "coreFraction": return .minimumCoreFraction
        case "boundarySupport": return .minimumPerimeterSupport
        case "invertInput": return .inputInversion
        case "polarity": return .classicalPolarity
        case "denoise": return .medianDenoise
        case "thresholdOffset": return .otsuOffset
        default: return nil
        }
    }

    private func percent(_ value: Double) -> String {
        "\(Int((value * 100).rounded()))%"
    }

    private func disabledPercent(_ value: Double) -> String {
        value == 0 ? "Off" : percent(value)
    }

    private func clampedOffset(
        _ proposed: CGSize,
        scale: CGFloat,
        viewport: CGSize
    ) -> CGSize {
        guard scale > 1 else { return .zero }
        let maximumX = viewport.width * (scale - 1) / 2
        let maximumY = viewport.height * (scale - 1) / 2
        return CGSize(
            width: min(max(proposed.width, -maximumX), maximumX),
            height: min(max(proposed.height, -maximumY), maximumY)
        )
    }

    private func resetZoom() {
        withAnimation(.spring(response: 0.3, dampingFraction: 0.82)) {
            zoomScale = 1
            committedZoomScale = 1
            zoomOffset = .zero
            committedZoomOffset = .zero
        }
    }
}
