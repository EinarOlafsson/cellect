import SwiftUI

/// Lists existing swipe-annotation projects and starts new ones.
struct SwipeProjectListView: View {
    @Environment(ProjectStore.self) private var store
    @State private var showingSetup = false

    var body: some View {
        List {
            if store.projects.isEmpty {
                ContentUnavailableView(
                    "No projects yet",
                    systemImage: "hand.draw",
                    description: Text("Create a project to point Cellect at a folder of images.")
                )
            }
            ForEach(store.projects) { project in
                NavigationLink {
                    SwipeAnnotationContainer(projectID: project.id)
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(project.displayName).font(.headline)
                        Text(columnSummary(project))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .onDelete { indexSet in
                indexSet.map { store.projects[$0] }.forEach(store.delete)
            }
        }
        .navigationTitle("Swipe Projects")
        .toolbar {
            ToolbarItem(placement: .primaryAction) {
                Button {
                    showingSetup = true
                } label: {
                    Label("New Project", systemImage: "plus")
                }
            }
        }
        .sheet(isPresented: $showingSetup) {
            NavigationStack {
                ProjectSetupView()
            }
        }
    }

    private func columnSummary(_ project: AnnotationProject) -> String {
        let cols = project.columns.map(\.name).joined(separator: ", ")
        let folder = project.storageRef.displayPath
        return cols.isEmpty ? folder : "\(folder) · \(cols)"
    }
}
