import SwiftUI

/// Builds the segmentation session (can fail to reopen the folder) and hosts the editor.
struct SegmentationContainer: View {
    @Environment(SegmentationStore.self) private var store
    let projectID: UUID

    @State private var session: SegmentationSession?
    @State private var initError: String?

    var body: some View {
        Group {
            if let session {
                SegmentationEditorView(session: session)
            } else if let initError {
                ContentUnavailableView {
                    Label("Can't open project", systemImage: "exclamationmark.triangle")
                } description: { Text(initError) }
            } else {
                ProgressView()
            }
        }
        .task(id: projectID) {
            guard session == nil, let project = store.project(id: projectID) else { return }
            do {
                let s = try SegmentationSession(project: project)
                session = s
                await s.load()
            } catch {
                initError = error.localizedDescription
            }
        }
    }
}
