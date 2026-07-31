import CoreGraphics
import Foundation
import Observation

/// Drives a segmentation session over a folder: lists images, loads the current image + any
/// existing mask into the controller, and writes masks (+ a classes sidecar) back to the folder.
@MainActor
@Observable
final class SegmentationSession {
    enum State: Equatable { case loading, ready, empty, failed(String) }

    let project: SegmentationProject
    let controller: SegmentationController
    private let provider: StorageProvider

    private(set) var state: State = .loading
    private(set) var images: [StoredImage] = []
    private(set) var index = 0
    var isSaving = false

    init(project: SegmentationProject) throws {
        self.project = project
        self.provider = try StorageProviderFactory.make(for: project.storageRef)
        self.controller = SegmentationController(classes: project.classes)
    }

    var current: StoredImage? { images.indices.contains(index) ? images[index] : nil }
    var progressText: String {
        images.isEmpty ? "" : "\(index + 1) / \(images.count) · \(current?.filename ?? "")"
    }

    func load() async {
        state = .loading
        do {
            images = try await provider.listImages()
            try? await writeClassesSidecarIfNeeded()
            guard !images.isEmpty else { state = .empty; return }
            await loadCurrent()
            state = .ready
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    private func loadCurrent() async {
        guard let img = current else { return }
        do {
            let data = try await provider.loadImageData(img)
            guard let base = ImageLoader.decode(data, maxDimension: SegmentationConfig.maxDimension) else {
                state = .failed("Couldn't decode \(img.filename)")
                return
            }
            let maskName = project.maskFilename(forImage: img.filename)
            var existing: CGImage?
            if let maskData = try await provider.readFile(named: maskName) {
                existing = ImageLoader.decodeExact(maskData)
            }
            controller.loadImage(base: base, existingMask: existing)
            controller.canvas?.reloadFromController()
        } catch {
            state = .failed(error.localizedDescription)
        }
    }

    func saveCurrent() async {
        guard let img = current, controller.maskDirty, let mask = controller.mask,
              let png = mask.labelPNGData() else { return }
        isSaving = true
        defer { isSaving = false }
        let maskName = project.maskFilename(forImage: img.filename)
        do {
            try await provider.writeFile(named: maskName, data: png)
            controller.maskDirty = false
        } catch {
            state = .failed("Couldn't save mask: \(error.localizedDescription)")
        }
    }

    func goNext() async { await step(+1) }
    func goPrev() async { await step(-1) }

    private func step(_ delta: Int) async {
        await saveCurrent()
        let new = index + delta
        guard images.indices.contains(new) else { return }
        index = new
        await loadCurrent()
    }

    /// Write a small JSON describing id → label → color so downstream tools can interpret masks.
    private func writeClassesSidecarIfNeeded() async throws {
        if try await provider.readFile(named: project.classesFileName) != nil { return }
        struct ClassInfo: Codable { let id: UInt8; let label: String; let color: String }
        let infos = project.classes.map { ClassInfo(id: $0.id, label: $0.label, color: $0.colorHex) }
        let data = try JSONEncoder().encode(infos)
        try await provider.writeFile(named: project.classesFileName, data: data)
    }
}
