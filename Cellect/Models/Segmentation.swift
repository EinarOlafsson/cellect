import Foundation

/// A paintable class for semantic segmentation. `id` is the pixel value written into the
/// exported grayscale mask PNG; `0` is reserved for background and never a class.
struct SegmentationClass: Identifiable, Codable, Hashable {
    var id: UInt8            // 1...255; the mask pixel value
    var label: String
    var colorHex: String
}

/// A folder + its segmentation configuration. Masks are written back into the folder as
/// `<stem>_mask.png` (grayscale, pixel == class id), with a sidecar `cellect_classes.json`.
struct SegmentationProject: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var displayName: String
    var storageRef: StorageRef
    var classes: [SegmentationClass]
    /// Suffix appended to an image stem to form its mask filename. "_mask" → A01_mask.png.
    var maskSuffix: String = "_mask"
    var classesFileName: String = "cellect_classes.json"

    func maskFilename(forImage filename: String) -> String {
        let stem = (filename as NSString).deletingPathExtension
        return "\(stem)\(maskSuffix).png"
    }
}
