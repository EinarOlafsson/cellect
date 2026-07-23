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
    var isSaving = false
    var errorMessage: String?
    var lastSavedStem: String?

    // Editable review settings (seeded from the project, let the user re-count).
    var polarity: CountPolarity
    var minAreaPixels: Int

    private let provider: StorageProvider

    init(project: CaptureProject) throws {
        self.project = project
        self.provider = try StorageProviderFactory.make(for: project.storageRef)
        self.polarity = project.polarity
        self.minAreaPixels = project.minAreaPixels
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

    func capture() async {
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

    /// Re-run counting on the already-captured frame (after tweaking polarity / min area).
    func recount() async {
        guard captured != nil else { return }
        phase = .working
        await runCount()
    }

    private func runCount() async {
        guard let image = captured else { phase = .preview; return }
        let counter = CounterRegistry.best()
        counterName = counter.displayName
        let options = CountOptions(
            polarity: polarity,
            minAreaPixels: minAreaPixels,
            micronsPerPixel: project.micronsPerPixel
        )
        do {
            let res = try await counter.count(in: image, options: options)
            result = res
            overlay = LabelOverlay.makeCGImage(
                labels: res.labelMask,
                width: Int(res.imageSize.width),
                height: Int(res.imageSize.height)
            )
            phase = .review
        } catch {
            errorMessage = error.localizedDescription
            phase = .review
        }
    }

    func retake() {
        captured = nil
        overlay = nil
        result = nil
        lastSavedStem = nil
        phase = .preview
    }

    func save() async {
        guard let image = captured, let result else { return }
        isSaving = true
        defer { isSaving = false }
        let writer = CountResultWriter(provider: provider, project: project)
        do {
            let stem = try await writer.save(
                image: image, result: result,
                counterName: counterName, timestamp: Date()
            )
            lastSavedStem = stem
        } catch {
            errorMessage = "Couldn't save: \(error.localizedDescription)"
        }
    }
}
