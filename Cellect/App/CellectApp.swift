import SwiftUI

@main
struct CellectApp: App {
    /// App-wide store of annotation projects, persisted locally.
    @State private var projectStore = ProjectStore()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(projectStore)
                .tint(.accentColor)
        }
    }
}
