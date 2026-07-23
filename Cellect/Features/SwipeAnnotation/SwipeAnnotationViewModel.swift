import SwiftUI
import Observation

/// Drives one annotation session: loads images + existing CSV, records swipes into an
/// in-memory table, and debounce-flushes the merged CSV back to the folder.
@MainActor
@Observable
final class SwipeAnnotationViewModel {
    enum LoadState: Equatable { case loading, ready, empty, failed(String) }

    private(set) var loadState: LoadState = .loading
    private(set) var images: [StoredImage] = []
    private(set) var index: Int = 0
    private(set) var records: [String: AnnotationRecord] = [:]
    private(set) var isFlushing = false

    let project: AnnotationProject
    private let provider: StorageProvider
    private let csvStore: AnnotationCSVStore

    /// Cache of decoded image data for the current + upcoming cards (prefetch window).
    private var imageDataCache: [String: Data] = [:]
    private var flushTask: Task<Void, Never>?
    private var dirty = false

    init(project: AnnotationProject) throws {
        self.project = project
        self.provider = try StorageProviderFactory.make(for: project.storageRef)
        self.csvStore = AnnotationCSVStore(
            provider: provider,
            csvFileName: project.csvFileName
        )
    }

    var activeColumn: AnnotationColumn? { project.activeColumn }

    var currentImage: StoredImage? {
        images.indices.contains(index) ? images[index] : nil
    }

    var progressText: String {
        guard !images.isEmpty else { return "" }
        let done = annotatedCount
        return "\(min(index + 1, images.count)) / \(images.count)  ·  \(done) labelled"
    }

    private var annotatedCount: Int {
        guard let col = activeColumn else { return 0 }
        return records.values.filter { ($0.values[col.name] ?? "").isEmpty == false }.count
    }

    // MARK: - Loading

    func load() async {
        loadState = .loading
        do {
            let listed = try await provider.listImages()
            let (existing, _) = try await csvStore.load()
            records = existing
            images = listed
            // Resume at the first unlabelled image for the active column.
            index = firstUnlabelledIndex()
            loadState = listed.isEmpty ? .empty : .ready
            await prefetchAround(index)
        } catch {
            loadState = .failed(error.localizedDescription)
        }
    }

    private func firstUnlabelledIndex() -> Int {
        guard let col = activeColumn else { return 0 }
        for (i, img) in images.enumerated() {
            if (records[img.filename]?.values[col.name] ?? "").isEmpty { return i }
        }
        return images.count  // all labelled → land on the completion screen
    }

    // MARK: - Image data

    func imageData(for image: StoredImage) -> Data? { imageDataCache[image.filename] }

    private func prefetchAround(_ center: Int) async {
        let window = max(0, center - 1)...min(images.count - 1, center + 2)
        guard images.indices.contains(center) else { return }
        for i in window where images.indices.contains(i) {
            let img = images[i]
            if imageDataCache[img.filename] != nil { continue }
            if let data = try? await provider.loadImageData(img) {
                imageDataCache[img.filename] = data
            }
        }
    }

    // MARK: - Annotating

    /// Apply the class bound to `gesture` for the current image, then advance.
    func apply(_ gesture: Gesture) {
        guard let col = activeColumn,
              let cls = col.classFor(gesture),
              let img = currentImage else { return }
        var rec = records[img.filename] ?? AnnotationRecord(filename: img.filename)
        rec.values[col.name] = cls.label
        records[img.filename] = rec
        dirty = true
        advance()
        scheduleFlush()
    }

    /// Skip the current image without labelling.
    func skip() { advance() }

    /// Step back one card (does not clear the label; user can re-swipe to overwrite).
    func goBack() {
        guard index > 0 else { return }
        index -= 1
        Task { await prefetchAround(index) }
    }

    func classFor(_ gesture: Gesture) -> AnnotationClass? {
        activeColumn?.classFor(gesture)
    }

    private func advance() {
        if index < images.count - 1 {
            index += 1
            Task { await prefetchAround(index) }
        } else {
            index = images.count  // one past the end → "done" state
        }
    }

    var isComplete: Bool { !images.isEmpty && index >= images.count }

    // MARK: - Flushing (debounced)

    private func scheduleFlush() {
        flushTask?.cancel()
        flushTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled else { return }
            await self?.flush()
        }
    }

    /// Force an immediate write (call on leaving the screen / backgrounding).
    func flush() async {
        guard dirty else { return }
        isFlushing = true
        defer { isFlushing = false }
        let order = images.map(\.filename)
        let columns = project.columns.map(\.name)
        do {
            try await csvStore.save(
                records: records,
                orderedFilenames: order,
                columnOrder: columns
            )
            dirty = false
        } catch {
            // Keep dirty=true so a later flush retries; surface via loadState if persistent.
            loadState = .failed("Couldn't save CSV: \(error.localizedDescription)")
        }
    }
}
