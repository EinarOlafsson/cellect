import Foundation

/// Reads/writes a user-picked folder on the device or in iCloud Drive.
///
/// iCloud folders arrive from the system document picker as ordinary security-scoped
/// `file://` URLs, so this one provider covers both "on device" and "iCloud".
final class LocalFolderProvider: StorageProvider {
    private let folderURL: URL
    /// True while we hold a security-scoped access claim we must balance with a `stop`.
    private let didStartAccess: Bool

    /// Resolve a persisted bookmark back into an accessible folder URL.
    init(ref: StorageRef) throws {
        guard ref.kind == .local, let bookmark = ref.bookmark else {
            throw StorageError.cannotAccessFolder
        }
        var stale = false
        let url = try URL(
            resolvingBookmarkData: bookmark,
            options: [],
            relativeTo: nil,
            bookmarkDataIsStale: &stale
        )
        if stale { throw StorageError.bookmarkStale }
        guard url.hasDirectoryPath else { throw StorageError.notADirectory }
        self.folderURL = url
        self.didStartAccess = url.startAccessingSecurityScopedResource()
    }

    deinit {
        if didStartAccess { folderURL.stopAccessingSecurityScopedResource() }
    }

    /// Build a persistable ref from a freshly picked folder URL (picker already granted access).
    static func makeRef(from pickedURL: URL) throws -> StorageRef {
        let needsStop = pickedURL.startAccessingSecurityScopedResource()
        defer { if needsStop { pickedURL.stopAccessingSecurityScopedResource() } }
        let bookmark = try pickedURL.bookmarkData(
            options: [],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
        return StorageRef(
            kind: .local,
            bookmark: bookmark,
            driveFolderID: nil,
            displayPath: pickedURL.lastPathComponent
        )
    }

    func listImages() async throws -> [StoredImage] {
        let keys: [URLResourceKey] = [.isRegularFileKey, .nameKey]
        let contents = try FileManager.default.contentsOfDirectory(
            at: folderURL,
            includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles]
        )
        return contents
            .filter { supportedImageExtensions.contains($0.pathExtension.lowercased()) }
            .map { StoredImage(id: $0.absoluteString, filename: $0.lastPathComponent) }
            .sorted { $0.filename.localizedStandardCompare($1.filename) == .orderedAscending }
    }

    func loadImageData(_ image: StoredImage) async throws -> Data {
        guard let url = URL(string: image.id) else { throw StorageError.cannotAccessFolder }
        return try Data(contentsOf: url)
    }

    func readFile(named name: String) async throws -> Data? {
        let url = folderURL.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        return try Data(contentsOf: url)
    }

    func writeFile(named name: String, data: Data) async throws {
        let url = folderURL.appendingPathComponent(name)
        try data.write(to: url, options: .atomic)
    }
}
