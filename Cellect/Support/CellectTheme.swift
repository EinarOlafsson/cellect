import SwiftUI

enum CellectAppearance: String, CaseIterable, Identifiable {
    case system
    case light
    case dark

    var id: String { rawValue }

    var title: String {
        switch self {
        case .system: "System"
        case .light: "Light"
        case .dark: "Dark"
        }
    }

    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
        }
    }
}

enum CellectTheme {
    static let mint = Color(red: 0.32, green: 0.82, blue: 0.67)
    static let cyan = Color(red: 0.39, green: 0.75, blue: 0.84)
    static let blue = Color(red: 0.35, green: 0.55, blue: 0.90)
    static let green = Color(red: 0.31, green: 0.76, blue: 0.46)
    static let violet = Color(red: 0.51, green: 0.50, blue: 0.89)

    static let wordmark = LinearGradient(
        colors: [mint, cyan, blue],
        startPoint: .leading,
        endPoint: .trailing
    )
}

/// The softly illuminated microscopy palette used by the original Cellect app.
struct CellectThemeBackground: View {
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        GeometryReader { proxy in
            let size = proxy.size

            ZStack {
                baseColor

                glow(CellectTheme.mint, opacity: colorScheme == .dark ? 0.42 : 0.34)
                    .frame(width: size.width * 1.05, height: size.height * 0.34)
                    .position(x: size.width * 0.06, y: size.height * 0.25)

                glow(CellectTheme.blue, opacity: colorScheme == .dark ? 0.34 : 0.24)
                    .frame(width: size.width * 0.96, height: size.height * 0.32)
                    .position(x: size.width * 0.96, y: size.height * 0.29)

                glow(CellectTheme.violet, opacity: colorScheme == .dark ? 0.28 : 0.19)
                    .frame(width: size.width * 0.84, height: size.height * 0.27)
                    .position(x: size.width * 0.49, y: size.height * 0.55)

                glow(CellectTheme.green, opacity: colorScheme == .dark ? 0.36 : 0.27)
                    .frame(width: size.width * 1.02, height: size.height * 0.36)
                    .position(x: size.width * 0.82, y: size.height * 0.72)

                glow(CellectTheme.cyan, opacity: colorScheme == .dark ? 0.28 : 0.20)
                    .frame(width: size.width * 0.88, height: size.height * 0.28)
                    .position(x: size.width * 0.12, y: size.height * 0.84)
            }
        }
        .ignoresSafeArea()
        .accessibilityHidden(true)
    }

    private var baseColor: Color {
        colorScheme == .dark
            ? Color(red: 0.025, green: 0.075, blue: 0.085)
            : Color(red: 0.965, green: 0.985, blue: 0.982)
    }

    private func glow(_ color: Color, opacity: Double) -> some View {
        Ellipse()
            .fill(
                RadialGradient(
                    colors: [color.opacity(opacity), color.opacity(opacity * 0.35), .clear],
                    center: .center,
                    startRadius: 0,
                    endRadius: 210
                )
            )
    }
}
