import SwiftUI

extension Color {
    /// Create a Color from "#rrggbb" (alpha optional as "#aarrggbb"). Falls back to gray.
    init(hex: String) {
        let s = hex.trimmingCharacters(in: CharacterSet(charactersIn: "#")).uppercased()
        var value: UInt64 = 0
        Scanner(string: s).scanHexInt64(&value)
        let r, g, b, a: Double
        switch s.count {
        case 6:
            r = Double((value >> 16) & 0xFF) / 255
            g = Double((value >> 8) & 0xFF) / 255
            b = Double(value & 0xFF) / 255
            a = 1
        case 8:
            a = Double((value >> 24) & 0xFF) / 255
            r = Double((value >> 16) & 0xFF) / 255
            g = Double((value >> 8) & 0xFF) / 255
            b = Double(value & 0xFF) / 255
        default:
            r = 0.5; g = 0.5; b = 0.5; a = 1
        }
        self.init(.sRGB, red: r, green: g, blue: b, opacity: a)
    }
}

/// Default class palette (green/red first, matching the desktop plaque tool's 1=green/2=red).
enum ClassPalette {
    static let hexes = ["#12b06e", "#d64550", "#3a7bd5", "#e0a800", "#8e44ad", "#16a085"]
    static func hex(_ index: Int) -> String { hexes[index % hexes.count] }
}
