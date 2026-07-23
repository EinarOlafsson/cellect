import SwiftUI

/// Builds the capture model (can fail to reopen the destination folder) and hosts the camera UI.
struct CameraCountContainer: View {
    @Environment(CaptureStore.self) private var store
    let projectID: UUID

    @State private var model: CameraCountModel?
    @State private var initError: String?

    var body: some View {
        Group {
            if let model {
                CameraCountView(model: model)
            } else if let initError {
                ContentUnavailableView {
                    Label("Can't open project", systemImage: "exclamationmark.triangle")
                } description: { Text(initError) }
            } else {
                ProgressView()
            }
        }
        .task(id: projectID) {
            guard model == nil, let project = store.project(id: projectID) else { return }
            do { model = try CameraCountModel(project: project) }
            catch { initError = error.localizedDescription }
        }
    }
}
