import Foundation

/// Identifies how to re-open a storage folder across app launches.
/// Local/iCloud use a security-scoped bookmark blob; Drive (future) uses a folder id.
struct StorageRef: Codable, Hashable {
    enum Kind: String, Codable { case local, googleDrive }
    var kind: Kind
    /// Security-scoped bookmark data for `.local`; nil for other kinds.
    var bookmark: Data?
    /// Drive folder id for `.googleDrive`; nil for local.
    var driveFolderID: String?
    /// Human-readable path/name for display only.
    var displayPath: String
}

/// A folder plus its annotation configuration. Persisted by `ProjectStore`.
struct AnnotationProject: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var displayName: String
    var storageRef: StorageRef
    var columns: [AnnotationColumn] = []
    var csvFileName: String = "cellect_annotations.csv"
    /// Index into `columns` currently being annotated.
    var activeColumnIndex: Int = 0

    var activeColumn: AnnotationColumn? {
        columns.indices.contains(activeColumnIndex) ? columns[activeColumnIndex] : nil
    }
}
