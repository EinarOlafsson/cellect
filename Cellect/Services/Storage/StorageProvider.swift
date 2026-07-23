import Foundation

/// An image discovered inside a storage folder.
struct StoredImage: Identifiable, Hashable {
    /// Stable within a session; for local this is the file URL string.
    var id: String
    /// Basename incl. extension, e.g. "A01_f01.tif". This is the CSV join key.
    var filename: String
}

/// Abstracts a folder of images + sidecar files (the CSV) across Local/iCloud/Drive.
/// Features depend only on this, so adding Google Drive later changes nothing upstream.
protocol StorageProvider {
    /// Image files in the folder, sorted by filename.
    func listImages() async throws -> [StoredImage]
    /// Raw bytes of an image (decoded by the UI layer).
    func loadImageData(_ image: StoredImage) async throws -> Data
    /// Sidecar file bytes, or nil if it doesn't exist yet.
    func readFile(named name: String) async throws -> Data?
    /// Create/overwrite a sidecar file.
    func writeFile(named name: String, data: Data) async throws
}

enum StorageError: LocalizedError {
    case cannotAccessFolder
    case bookmarkStale
    case notADirectory

    var errorDescription: String? {
        switch self {
        case .cannotAccessFolder: return "Cellect can't access that folder. Try selecting it again."
        case .bookmarkStale:      return "The saved folder link expired. Please re-select the folder."
        case .notADirectory:      return "That item isn't a folder."
        }
    }
}

/// Image extensions we surface. TIFF included because microscopy uses it heavily; the UI
/// falls back gracefully if iOS can't decode an exotic variant.
let supportedImageExtensions: Set<String> = [
    "png", "jpg", "jpeg", "heic", "heif", "tif", "tiff", "bmp", "gif", "webp"
]
