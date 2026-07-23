import SwiftUI

/// Lists capture projects and starts new ones.
struct CaptureProjectListView: View {
    @Environment(CaptureStore.self) private var store
    @State private var showingSetup = false

    var body: some View {
        List {
            if store.projects.isEmpty {
                ContentUnavailableView(
                    "No capture projects",
                    systemImage: "camera.viewfinder",
                    description: Text("Create one to photograph and count cells into a folder.")
                )
            }
            ForEach(store.projects) { project in
                NavigationLink {
                    CameraCountContainer(projectID: project.id)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(project.displayName).font(.headline)
                        Text("\(project.storageRef.displayPath) · \(project.polarity.title)")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { $0.map { store.projects[$0] }.forEach(store.delete) }
        }
        .navigationTitle("Camera Count")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button { showingSetup = true } label: { Label("New Project", systemImage: "plus") }
            }
        }
        .sheet(isPresented: $showingSetup) {
            NavigationStack { CaptureSetupView() }
        }
    }
}
