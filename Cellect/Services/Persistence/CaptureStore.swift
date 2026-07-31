import Foundation
import Observation

/// Owns and persists capture projects (destination folder + counting settings).
@Observable
final class CaptureStore {
    private(set) var projects: [CaptureProject] = []
    private(set) var fixtureImportMessage: String?

    private static let fixtureNames = [
        "LIVECell_0_A172.jpg",
        "LIVECell_1_A172_dense.jpg",
        "LIVECell_2_BT474.jpg",
        "LIVECell_3_BV2.jpg",
        "LIVECell_4_Huh7.jpg",
        "LIVECell_5_MCF7.jpg",
        "LIVECell_6_SkBr3.jpg",
        "LIVECell_7_SHSY5Y.jpg",
        "LIVECell_8_SKOV3.jpg",
        "LIVECell_9_SKOV3_variant.jpg",
    ]

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

    /// Copy the small, licensed development fixture set into each selected capture folder.
    /// Existing files are preserved, so this is safe to run after every development install.
    func installBundledTestImages() async {
        guard !projects.isEmpty else { return }
        var added = 0
        var destinations = 0
        do {
            for project in projects {
                let provider = try StorageProviderFactory.make(for: project.storageRef)
                destinations += 1
                for name in Self.fixtureNames {
                    if try await provider.readFile(named: name) != nil { continue }
                    guard let source = bundledFixture(named: name) else {
                        throw CocoaError(.fileNoSuchFile)
                    }
                    try await provider.writeFile(
                        named: name,
                        data: try Data(contentsOf: source)
                    )
                    added += 1
                }
                let licenseName = "LIVECell_LICENSE.txt"
                if try await provider.readFile(named: licenseName) == nil,
                   let source = bundledFixture(named: licenseName) {
                    try await provider.writeFile(
                        named: licenseName,
                        data: try Data(contentsOf: source)
                    )
                }
            }
            fixtureImportMessage = added > 0
                ? "Added \(added) LIVECell test images to \(destinations) project folder."
                : "LIVECell test images are already in the selected folder."
        } catch {
            fixtureImportMessage = "Couldn't add test images: \(error.localizedDescription)"
        }
    }

    private func bundledFixture(named name: String) -> URL? {
        let path = name as NSString
        let base = path.deletingPathExtension
        let ext = path.pathExtension
        return Bundle.main.url(
            forResource: base,
            withExtension: ext,
            subdirectory: "DevelopmentTestImages"
        ) ?? Bundle.main.url(forResource: base, withExtension: ext)
    }

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
