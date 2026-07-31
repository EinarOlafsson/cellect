import SwiftUI

/// Home screen, recreated from the original Cellect theme recovered from the old app.
struct RootView: View {
    @State private var showingSettings = false

    var body: some View {
        NavigationStack {
            ZStack {
                CellectThemeBackground()

                ScrollView {
                    VStack(alignment: .leading, spacing: 0) {
                        header
                            .padding(.bottom, 30)

                        FeatureSection(title: "Annotate") {
                            FeatureNavigationCard(
                                systemImage: "hand.draw",
                                title: "Swipe Annotation",
                                subtitle: "Label a folder of images by swiping"
                            ) {
                                SwipeProjectListView()
                            }

                            FeatureNavigationCard(
                                systemImage: "scribble.variable",
                                title: "Touch Segmentation",
                                subtitle: "Paint semantic masks on images"
                            ) {
                                SegmentationProjectListView()
                            }
                        }

                        FeatureSection(title: "Capture") {
                            FeatureNavigationCard(
                                systemImage: "camera.viewfinder",
                                title: "Camera Cell Count",
                                subtitle: "Photograph & count cells to a folder"
                            ) {
                                CaptureProjectListView()
                            }
                        }

                        FeatureSection(title: "Community") {
                            FeatureCardLabel(
                                systemImage: "person.3",
                                title: "Shared Tasks",
                                subtitle: "Annotate shared datasets",
                                enabled: false
                            )
                            .accessibilityHint("Coming soon")
                        }
                    }
                    .frame(maxWidth: 720)
                    .padding(.horizontal, 20)
                    .padding(.top, 22)
                    .padding(.bottom, 40)
                    .frame(maxWidth: .infinity)
                }
            }
            .toolbar(.hidden, for: .navigationBar)
            .sheet(isPresented: $showingSettings) {
                CellectSettingsView()
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .center) {
                Text("Cellect")
                    .font(.system(size: 44, weight: .bold, design: .rounded))
                    .foregroundStyle(CellectTheme.wordmark)
                    .accessibilityAddTraits(.isHeader)

                Spacer()

                Button {
                    showingSettings = true
                } label: {
                    Image(systemName: "gearshape.fill")
                        .font(.system(size: 25, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 44, height: 44)
                        .contentShape(Rectangle())
                }
                .accessibilityLabel("Settings")
            }

            Text("Microscopy annotation, in your pocket")
                .font(.system(size: 17))
                .foregroundStyle(.secondary)
        }
    }
}

private struct FeatureSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content

    init(title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title.uppercased())
                .font(.system(size: 14, weight: .semibold))
                .tracking(1.15)
                .foregroundStyle(.secondary)
                .padding(.leading, 4)

            VStack(spacing: 12) {
                content
            }
        }
        .padding(.bottom, 26)
    }
}

private struct FeatureNavigationCard<Destination: View>: View {
    let systemImage: String
    let title: String
    let subtitle: String
    @ViewBuilder let destination: Destination

    init(
        systemImage: String,
        title: String,
        subtitle: String,
        @ViewBuilder destination: () -> Destination
    ) {
        self.systemImage = systemImage
        self.title = title
        self.subtitle = subtitle
        self.destination = destination()
    }

    var body: some View {
        NavigationLink {
            destination
                .toolbar(.visible, for: .navigationBar)
        } label: {
            FeatureCardLabel(
                systemImage: systemImage,
                title: title,
                subtitle: subtitle,
                enabled: true
            )
        }
        .buttonStyle(.plain)
    }
}

private struct FeatureCardLabel: View {
    @Environment(\.colorScheme) private var colorScheme

    let systemImage: String
    let title: String
    let subtitle: String
    let enabled: Bool

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: systemImage)
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(CellectTheme.mint)
                .frame(width: 50, height: 50)
                .background(
                    CellectTheme.mint.opacity(colorScheme == .dark ? 0.16 : 0.14),
                    in: RoundedRectangle(cornerRadius: 15, style: .continuous)
                )

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.primary)
                Text(subtitle)
                    .font(.system(size: 15))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 16)
        .frame(minHeight: 80)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 23, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 23, style: .continuous)
                .strokeBorder(
                    colorScheme == .dark
                        ? Color.white.opacity(0.14)
                        : Color.white.opacity(0.48),
                    lineWidth: 1
                )
        }
        .opacity(enabled ? 1 : 0.72)
    }
}

private struct CellectSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage("cellect.appearance") private var appearanceRawValue = CellectAppearance.system.rawValue
    @AppStorage("cellect.theme.selectedID") private var selectedThemeID = "cellect"

    private var appearance: Binding<CellectAppearance> {
        Binding(
            get: { CellectAppearance(rawValue: appearanceRawValue) ?? .system },
            set: { appearanceRawValue = $0.rawValue }
        )
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Theme") {
                    LabeledContent("Selected", value: "Cellect")
                    Text("The original mint, blue, violet, and green microscopy palette.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Appearance") {
                    Picker("Appearance", selection: appearance) {
                        ForEach(CellectAppearance.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                }
            }
            .scrollContentBackground(.hidden)
            .background(CellectThemeBackground())
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear {
                selectedThemeID = "cellect"
            }
        }
    }
}

#Preview {
    RootView()
        .environment(ProjectStore())
        .environment(SegmentationStore())
        .environment(CaptureStore())
}
