import SwiftUI

/// Home screen. Lists the four capabilities; only Swipe Annotation is wired up so far.
struct RootView: View {
    @Environment(ProjectStore.self) private var projectStore

    var body: some View {
        NavigationStack {
            List {
                Section("Annotate") {
                    NavigationLink {
                        SwipeProjectListView()
                    } label: {
                        FeatureRow(
                            systemImage: "hand.draw",
                            title: "Swipe Annotation",
                            subtitle: "Label a folder of images by swiping",
                            enabled: true
                        )
                    }

                    NavigationLink {
                        SegmentationProjectListView()
                    } label: {
                        FeatureRow(
                            systemImage: "scribble.variable",
                            title: "Touch Segmentation",
                            subtitle: "Paint semantic masks on images",
                            enabled: true
                        )
                    }
                }

                Section("Capture") {
                    NavigationLink {
                        CaptureProjectListView()
                    } label: {
                        FeatureRow(
                            systemImage: "camera.viewfinder",
                            title: "Camera Cell Count",
                            subtitle: "Photograph & count cells to a folder",
                            enabled: true
                        )
                    }
                }

                Section("Community") {
                    FeatureRow(
                        systemImage: "person.3",
                        title: "Shared Tasks",
                        subtitle: "Annotate shared datasets — coming soon",
                        enabled: false
                    )
                }
            }
            .navigationTitle("Cellect")
        }
    }
}

private struct FeatureRow: View {
    let systemImage: String
    let title: String
    let subtitle: String
    let enabled: Bool

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: systemImage)
                .font(.title2)
                .frame(width: 34)
                .foregroundStyle(enabled ? Color.accentColor : .secondary)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding(.vertical, 4)
        .opacity(enabled ? 1 : 0.55)
    }
}

#Preview {
    RootView()
        .environment(ProjectStore())
        .environment(SegmentationStore())
        .environment(CaptureStore())
}
