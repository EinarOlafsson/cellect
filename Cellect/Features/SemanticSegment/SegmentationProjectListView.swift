import SwiftUI

/// Lists segmentation projects and starts new ones.
struct SegmentationProjectListView: View {
    @Environment(SegmentationStore.self) private var store
    @State private var showingSetup = false

    var body: some View {
        List {
            if store.projects.isEmpty {
                ContentUnavailableView(
                    "No projects yet",
                    systemImage: "scribble.variable",
                    description: Text("Create a project to paint masks on a folder of images.")
                )
            }
            ForEach(store.projects) { project in
                NavigationLink {
                    SegmentationContainer(projectID: project.id)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(project.displayName).font(.headline)
                        Text("\(project.storageRef.displayPath) · \(project.classes.count) classes")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { $0.map { store.projects[$0] }.forEach(store.delete) }
        }
        .navigationTitle("Segmentation")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button { showingSetup = true } label: { Label("New Project", systemImage: "plus") }
            }
        }
        .sheet(isPresented: $showingSetup) {
            NavigationStack { SegmentationSetupView() }
        }
    }
}
