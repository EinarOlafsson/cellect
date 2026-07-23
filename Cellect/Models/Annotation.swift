import Foundation

/// A directional gesture that assigns a class. Four cardinal swipes cover the common case;
/// tap-based inputs are reserved for datasets needing more than four classes.
enum Gesture: String, Codable, CaseIterable, Identifiable {
    case swipeLeft, swipeRight, swipeUp, swipeDown
    var id: String { rawValue }

    var label: String {
        switch self {
        case .swipeLeft:  return "Swipe Left"
        case .swipeRight: return "Swipe Right"
        case .swipeUp:    return "Swipe Up"
        case .swipeDown:  return "Swipe Down"
        }
    }

    var systemImage: String {
        switch self {
        case .swipeLeft:  return "arrow.left"
        case .swipeRight: return "arrow.right"
        case .swipeUp:    return "arrow.up"
        case .swipeDown:  return "arrow.down"
        }
    }
}

/// One label reachable by one gesture. `label` is written verbatim into the CSV cell.
struct AnnotationClass: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    var label: String
    var gesture: Gesture
    /// Hex like "#12b06e" used for the card edge/flash when this class is chosen.
    var colorHex: String
}

/// One round of annotation == one CSV column. Multiple rounds add multiple columns.
struct AnnotationColumn: Identifiable, Codable, Hashable {
    var id: UUID = UUID()
    /// User-defined CSV header, e.g. "phenotype".
    var name: String
    var classes: [AnnotationClass]

    func classFor(_ gesture: Gesture) -> AnnotationClass? {
        classes.first { $0.gesture == gesture }
    }
}

/// One image's annotations across all rounds. Keyed by column name in the CSV.
struct AnnotationRecord: Codable, Hashable {
    var filename: String
    var values: [String: String] = [:]
}
