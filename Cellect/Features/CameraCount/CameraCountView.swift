import SwiftUI
import UniformTypeIdentifiers

/// Capture → count → review screen for one capture project.
struct CameraCountView: View {
    @Bindable var model: CameraCountModel
    @State private var showingImageImporter = false
    @State private var showingModelSettings = false
    @State private var showingModelComparison = false
    @State private var zoomScale: CGFloat = 1
    @State private var committedZoomScale: CGFloat = 1
    @State private var zoomOffset: CGSize = .zero
    @State private var committedZoomOffset: CGSize = .zero

    private let maximumZoomScale: CGFloat = 8

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            switch model.phase {
            case .preview: previewLayer
            case .working: workingLayer
            case .review:  reviewLayer
            }
        }
        .navigationTitle(model.project.displayName)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Button {
                    showingModelSettings = true
                } label: {
                    Label("Model settings", systemImage: "slider.horizontal.3")
                }
                .disabled(model.phase == .working || model.isSaving)
            }
        }
        .onAppear { model.onAppear() }
        .onDisappear { model.onDisappear() }
        .sheet(isPresented: $showingModelSettings) {
            ModelSettingsView(model: model)
        }
        .fullScreenCover(isPresented: $showingModelComparison) {
            if let image = model.captured {
                ModelComparisonView(
                    cameraModel: model,
                    image: image,
                    baseline: model.comparisonOptionsSnapshot,
                    modelSpecs: model.availableModelSpecs
                )
            } else {
                ContentUnavailableView(
                    "No image to compare",
                    systemImage: "photo.badge.exclamationmark"
                )
            }
        }
        .alert("Camera", isPresented: .constant(model.errorMessage != nil)) {
            Button("OK") { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
        .fileImporter(
            isPresented: $showingImageImporter,
            allowedContentTypes: [.image],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                Task { await model.importImage(from: url) }
            case .failure(let error):
                model.errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Preview

    private var previewLayer: some View {
        ZStack {
            if model.camera.authorized {
                CameraPreviewView(session: model.camera.session).ignoresSafeArea()
            } else {
                ContentUnavailableView("Camera unavailable", systemImage: "camera.fill",
                                       description: Text("Grant camera access in Settings to capture."))
                    .foregroundStyle(.white)
            }
            VStack {
                Spacer()
                HStack(spacing: 42) {
                    Button { showingImageImporter = true } label: {
                        Label("Test image", systemImage: "photo")
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(.ultraThinMaterial, in: Capsule())
                    }
                    .foregroundStyle(.white)

                    Button { Task { await model.capture() } } label: {
                        Circle().fill(.white).frame(width: 74, height: 74)
                            .overlay(
                                Circle()
                                    .stroke(.white.opacity(0.6), lineWidth: 4)
                                    .frame(width: 88, height: 88)
                            )
                    }
                    .disabled(!model.camera.authorized)
                }
                .padding(.bottom, 28)
            }
        }
    }

    private var workingLayer: some View {
        ZStack {
            if let img = model.captured {
                Image(decorative: img, scale: 1).resizable().scaledToFit().opacity(0.5)
            }
            ProgressView("Counting…").tint(.white).foregroundStyle(.white)
        }
    }

    // MARK: - Review

    private var reviewLayer: some View {
        VStack(spacing: 0) {
            GeometryReader { geometry in
                ZStack {
                    if let img = model.captured {
                        Image(decorative: img, scale: 1).resizable().scaledToFit()
                    }
                    if let ov = model.overlay {
                        Image(decorative: ov, scale: 1).resizable().scaledToFit()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .scaleEffect(zoomScale)
                .offset(zoomOffset)
                .contentShape(Rectangle())
                .gesture(
                    magnifyGesture(in: geometry.size)
                        .simultaneously(with: panGesture(in: geometry.size))
                )
                .onTapGesture(count: 2) {
                    resetZoom(animated: true)
                }
                .accessibilityAction(named: "Reset zoom") {
                    resetZoom(animated: true)
                }
                .overlay(alignment: .top) {
                    Text("Pinch to zoom · drag to pan · double-tap to reset")
                        .font(.caption2)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.top, 6)
                }
                .clipped()
            }
            reviewControls
        }
    }

    private var reviewControls: some View {
        VStack(spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.summary).font(.headline)
                    Text(model.counterName).font(.caption2).foregroundStyle(.secondary)
                    if let diagnostics = model.lastCountDiagnostics,
                       diagnostics.automaticBoundarySelectionUsed,
                       let cutoff = diagnostics.selectedBoundaryProbabilityThreshold {
                        Text(adaptiveBoundarySummary(diagnostics, cutoff: cutoff))
                            .font(.caption2)
                            .foregroundStyle(
                                diagnostics.automaticBoundaryFallbackUsed
                                    ? .orange
                                    : .secondary
                            )
                    }
                }
                Spacer()
                if let stem = model.lastSavedStem {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.subheadline)
                        .accessibilityLabel("Saved \(stem)")
                }
            }

            if model.hasUnappliedChanges {
                Label(
                    "Settings changed — recount before saving",
                    systemImage: "exclamationmark.arrow.triangle.2.circlepath"
                )
                .font(.caption)
                .foregroundStyle(.orange)
                .frame(maxWidth: .infinity, alignment: .leading)
            }

            HStack {
                Label(
                    model.hasUnappliedChanges ? "Next recount" : "Counting model",
                    systemImage: "cpu"
                )
                Spacer()
                if model.ensembleEnabled {
                    Button {
                        showingModelSettings = true
                    } label: {
                        Text(model.ensembleSelectionSummary)
                    }
                    .buttonStyle(.bordered)
                } else {
                    Picker("Counting model", selection: Binding(
                        get: { model.selectedCounterID },
                        set: { model.selectedCounterID = $0 }
                    )) {
                        ForEach(model.availableCounterIDs, id: \.self) { identifier in
                            Text(model.counterDisplayName(for: identifier)).tag(identifier)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.menu)
                }
            }

            HStack {
                Label(
                    "Min \(model.minAreaPixels) px"
                        + (model.maxAreaPixels > 0 ? " · Max \(model.maxAreaPixels) px" : ""),
                    systemImage: "scope"
                )
                .font(.caption)
                Spacer()
                Button {
                    showingModelSettings = true
                } label: {
                    Label("All settings", systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.bordered)
            }

            Button {
                showingModelComparison = true
            } label: {
                Label("Compare all models and settings", systemImage: "square.stack.3d.up")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .tint(.mint)
            .disabled(model.captured == nil || model.isSaving)

            HStack {
                Button { Task { await model.recount() } } label: {
                    Label("Recount", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                .disabled(model.isSaving)
                Spacer()
                Button {
                    resetZoom(animated: false)
                    model.retake()
                } label: {
                    Label("Retake", systemImage: "camera")
                }
                    .buttonStyle(.bordered)
                    .disabled(model.isSaving)
                Button { Task { await model.save() } } label: {
                    if model.isSaving { ProgressView() } else { Label("Save", systemImage: "square.and.arrow.up") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(
                    model.isSaving
                        || model.result == nil
                        || model.hasUnappliedChanges
                )
            }
        }
        .padding()
        .background(.regularMaterial)
    }

    // MARK: - Review zoom

    private func adaptiveBoundarySummary(
        _ diagnostics: CountDiagnostics,
        cutoff: Double
    ) -> String {
        let cutoffText = "\(Int((cutoff * 100).rounded()))%"
        if diagnostics.automaticBoundaryFallbackUsed {
            if let reason = diagnostics.automaticBoundaryFallbackReason {
                return "Adaptive boundary fallback \(cutoffText) · \(reason.title)"
            }
            return "Adaptive boundary fallback · \(cutoffText)"
        }
        if let stability = diagnostics.boundaryStabilityAUC {
            return "Adaptive boundary \(cutoffText) · stability AUC \(Int((stability * 100).rounded()))%"
        }
        return "Adaptive boundary \(cutoffText)"
    }

    private func magnifyGesture(in viewport: CGSize) -> some SwiftUI.Gesture {
        MagnifyGesture()
            .onChanged { value in
                zoomScale = min(
                    maximumZoomScale,
                    max(1, committedZoomScale * value.magnification)
                )
                zoomOffset = clampedOffset(
                    committedZoomOffset,
                    scale: zoomScale,
                    viewport: viewport
                )
            }
            .onEnded { _ in
                if zoomScale <= 1 {
                    resetZoom(animated: true)
                } else {
                    committedZoomScale = zoomScale
                    zoomOffset = clampedOffset(
                        zoomOffset,
                        scale: zoomScale,
                        viewport: viewport
                    )
                    committedZoomOffset = zoomOffset
                }
            }
    }

    private func panGesture(in viewport: CGSize) -> some SwiftUI.Gesture {
        DragGesture(minimumDistance: 5)
            .onChanged { value in
                guard zoomScale > 1 else { return }
                let proposed = CGSize(
                    width: committedZoomOffset.width + value.translation.width,
                    height: committedZoomOffset.height + value.translation.height
                )
                zoomOffset = clampedOffset(
                    proposed,
                    scale: zoomScale,
                    viewport: viewport
                )
            }
            .onEnded { _ in
                guard zoomScale > 1 else { return }
                committedZoomOffset = zoomOffset
            }
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

    private func resetZoom(animated: Bool) {
        let update = {
            zoomScale = 1
            committedZoomScale = 1
            zoomOffset = .zero
            committedZoomOffset = .zero
        }
        if animated {
            withAnimation(.spring(response: 0.3, dampingFraction: 0.82), update)
        } else {
            update()
        }
    }
}
