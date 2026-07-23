import Foundation
import Observation

/// Owns and persists segmentation projects (metadata + class config only; masks live in the
/// image folder). Mirrors `ProjectStore`.
@Observable
final class SegmentationStore {
    private(set) var projects: [SegmentationProject] = []

    @ObservationIgnored
    private let fileURL: URL = {
        let dir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("cellect_segmentation_projects.json")
    }()

    init() { load() }

    func add(_ project: SegmentationProject) { projects.append(project); save() }

    func update(_ project: SegmentationProject) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id }) else { return }
        projects[idx] = project
        save()
    }

    func delete(_ project: SegmentationProject) {
        projects.removeAll { $0.id == project.id }
        save()
    }

    func project(id: UUID) -> SegmentationProject? { projects.first { $0.id == id } }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([SegmentationProject].self, from: data)
        else { return }
        projects = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(projects) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
