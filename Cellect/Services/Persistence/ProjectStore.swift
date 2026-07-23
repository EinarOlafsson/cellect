import Foundation
import Observation

/// Owns the list of annotation projects and persists them to Application Support as JSON.
/// This is metadata only (folder bookmarks + column config); the annotations themselves
/// live in each folder's CSV.
@Observable
final class ProjectStore {
    private(set) var projects: [AnnotationProject] = []

    @ObservationIgnored
    private let fileURL: URL = {
        let dir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("cellect_projects.json")
    }()

    init() { load() }

    func add(_ project: AnnotationProject) {
        projects.append(project)
        save()
    }

    func update(_ project: AnnotationProject) {
        guard let idx = projects.firstIndex(where: { $0.id == project.id }) else { return }
        projects[idx] = project
        save()
    }

    func delete(_ project: AnnotationProject) {
        projects.removeAll { $0.id == project.id }
        save()
    }

    func project(id: UUID) -> AnnotationProject? {
        projects.first { $0.id == id }
    }

    // MARK: - Persistence

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([AnnotationProject].self, from: data)
        else { return }
        projects = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(projects) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}

/// Resolves a `StorageRef` to a live provider. Central seam for adding Google Drive later.
enum StorageProviderFactory {
    static func make(for ref: StorageRef) throws -> StorageProvider {
        switch ref.kind {
        case .local:
            return try LocalFolderProvider(ref: ref)
        case .googleDrive:
            throw StorageError.cannotAccessFolder  // GoogleDriveProvider: not yet implemented
        }
    }
}
