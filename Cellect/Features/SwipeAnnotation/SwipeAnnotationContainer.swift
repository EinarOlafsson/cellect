import SwiftUI

/// Builds the view model (which can fail if the folder can't be reopened) and hosts the UI.
struct SwipeAnnotationContainer: View {
    @Environment(ProjectStore.self) private var store
    let projectID: UUID

    @State private var viewModel: SwipeAnnotationViewModel?
    @State private var initError: String?

    var body: some View {
        Group {
            if let viewModel {
                SwipeAnnotationView(viewModel: viewModel)
            } else if let initError {
                ContentUnavailableView {
                    Label("Can't open project", systemImage: "exclamationmark.triangle")
                } description: {
                    Text(initError)
                }
            } else {
                ProgressView()
            }
        }
        .task(id: projectID) {
            guard viewModel == nil, let project = store.project(id: projectID) else { return }
            do {
                let vm = try SwipeAnnotationViewModel(project: project)
                viewModel = vm
                await vm.load()
            } catch {
                initError = error.localizedDescription
            }
        }
    }
}
