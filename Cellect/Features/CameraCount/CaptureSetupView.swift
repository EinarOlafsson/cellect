import SwiftUI
import UniformTypeIdentifiers

/// New capture project: pick the destination folder + counting defaults.
struct CaptureSetupView: View {
    @Environment(CaptureStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var projectName = ""
    @State private var pickedRef: StorageRef?
    @State private var showingFolderPicker = false
    @State private var polarity: CountPolarity = .darkObjects
    @State private var minArea = 40
    @State private var calibrate = false
    @State private var micronsPerPixel = 1.0
    @State private var errorMessage: String?

    var body: some View {
        Form {
            Section("Destination folder") {
                Button { showingFolderPicker = true } label: {
                    HStack {
                        Image(systemName: "folder")
                        Text(pickedRef?.displayPath ?? "Choose a folder…")
                        Spacer()
                        if pickedRef != nil { Image(systemName: "checkmark").foregroundStyle(.green) }
                    }
                }
                Text("Captured image, 16-bit instance mask, and CSVs are written here.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Project") {
                TextField("Project name", text: $projectName)
            }

            Section("Counting defaults") {
                Picker("Cells appear", selection: $polarity) {
                    ForEach(CountPolarity.allCases) { p in Text(p.title).tag(p) }
                }
                Stepper("Min object size: \(minArea) px", value: $minArea, in: 5...2000, step: 5)
                Toggle("Calibrate size (µm/px)", isOn: $calibrate)
                if calibrate {
                    HStack {
                        Text("µm per pixel")
                        Spacer()
                        TextField("µm/px", value: $micronsPerPixel, format: .number)
                            .keyboardType(.decimalPad)
                            .multilineTextAlignment(.trailing)
                            .frame(width: 100)
                    }
                }
            }

            if let errorMessage {
                Section { Text(errorMessage).foregroundStyle(.red) }
            }
        }
        .navigationTitle("New Capture Project")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Create") { create() }.disabled(pickedRef == nil)
            }
        }
        .fileImporter(
            isPresented: $showingFolderPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                guard let url = urls.first else { return }
                do {
                    let ref = try LocalFolderProvider.makeRef(from: url)
                    pickedRef = ref
                    if projectName.isEmpty { projectName = ref.displayPath }
                    errorMessage = nil
                } catch { errorMessage = error.localizedDescription }
            case .failure(let error):
                errorMessage = error.localizedDescription
            }
        }
    }

    private func create() {
        guard let ref = pickedRef else { return }
        let project = CaptureProject(
            displayName: projectName.isEmpty ? ref.displayPath : projectName,
            storageRef: ref,
            polarity: polarity,
            minAreaPixels: minArea,
            micronsPerPixel: calibrate ? micronsPerPixel : nil
        )
        store.add(project)
        dismiss()
    }
}
