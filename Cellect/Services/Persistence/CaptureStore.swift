import Foundation
import Observation

/// Owns and persists capture projects (destination folder + counting settings).
@Observable
final class CaptureStore {
    private(set) var projects: [CaptureProject] = []

    @ObservationIgnored
    private let fileURL: URL = {
        let dir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("cellect_capture_projects.json")
    }()

    init() { load() }

    func add(_ p: CaptureProject) { projects.append(p); save() }
    func update(_ p: CaptureProject) {
        guard let i = projects.firstIndex(where: { $0.id == p.id }) else { return }
        projects[i] = p; save()
    }
    func delete(_ p: CaptureProject) { projects.removeAll { $0.id == p.id }; save() }
    func project(id: UUID) -> CaptureProject? { projects.first { $0.id == id } }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([CaptureProject].self, from: data) else { return }
        projects = decoded
    }
    private func save() {
        guard let data = try? JSONEncoder().encode(projects) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
