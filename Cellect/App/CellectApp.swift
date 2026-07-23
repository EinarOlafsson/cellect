import SwiftUI

@main
struct CellectApp: App {
    /// App-wide store of swipe-annotation projects, persisted locally.
    @State private var projectStore = ProjectStore()
    /// App-wide store of segmentation projects.
    @State private var segmentationStore = SegmentationStore()
    /// App-wide store of camera capture projects.
    @State private var captureStore = CaptureStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(projectStore)
                .environment(segmentationStore)
                .environment(captureStore)
                .tint(.accentColor)
        }
    }
}
