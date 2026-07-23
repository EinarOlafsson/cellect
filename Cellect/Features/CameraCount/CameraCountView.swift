import SwiftUI

/// Capture → count → review screen for one capture project.
struct CameraCountView: View {
    @Bindable var model: CameraCountModel

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
        .onAppear { model.onAppear() }
        .onDisappear { model.onDisappear() }
        .alert("Camera", isPresented: .constant(model.errorMessage != nil)) {
            Button("OK") { model.errorMessage = nil }
        } message: { Text(model.errorMessage ?? "") }
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
                Button { Task { await model.capture() } } label: {
                    Circle().fill(.white).frame(width: 74, height: 74)
                        .overlay(Circle().stroke(.white.opacity(0.6), lineWidth: 4).frame(width: 88, height: 88))
                }
                .disabled(!model.camera.authorized)
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
            GeometryReader { _ in
                ZStack {
                    if let img = model.captured {
                        Image(decorative: img, scale: 1).resizable().scaledToFit()
                    }
                    if let ov = model.overlay {
                        Image(decorative: ov, scale: 1).resizable().scaledToFit()
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                }
                Spacer()
                if let stem = model.lastSavedStem {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .foregroundStyle(.green).font(.subheadline)
                        .accessibilityLabel("Saved \(stem)")
                }
            }

            HStack(spacing: 12) {
                Picker("Polarity", selection: Binding(
                    get: { model.polarity }, set: { model.polarity = $0 }
                )) {
                    ForEach(CountPolarity.allCases) { p in Text(p.title).tag(p) }
                }
                .pickerStyle(.segmented)

                Stepper("min \(model.minAreaPixels)px",
                        value: Binding(get: { model.minAreaPixels },
                                       set: { model.minAreaPixels = $0 }),
                        in: 5...2000, step: 5)
                    .fixedSize()
            }

            HStack {
                Button { Task { await model.recount() } } label: {
                    Label("Recount", systemImage: "arrow.clockwise")
                }
                .buttonStyle(.bordered)
                Spacer()
                Button { model.retake() } label: { Label("Retake", systemImage: "camera") }
                    .buttonStyle(.bordered)
                Button { Task { await model.save() } } label: {
                    if model.isSaving { ProgressView() } else { Label("Save", systemImage: "square.and.arrow.up") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.isSaving || model.result == nil)
            }
        }
        .padding()
        .background(.regularMaterial)
    }
}
