import SwiftUI
import UniformTypeIdentifiers

/// New segmentation project: pick a folder and define the classes to paint.
struct SegmentationSetupView: View {
    @Environment(SegmentationStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var projectName = ""
    @State private var pickedRef: StorageRef?
    @State private var showingFolderPicker = false
    @State private var errorMessage: String?
    @State private var classRows: [ClassRow] = [
        ClassRow(label: "cell", colorHex: ClassPalette.hex(0)),
        ClassRow(label: "background_object", colorHex: ClassPalette.hex(1))
    ]

    private struct ClassRow: Identifiable {
        let id = UUID()
        var label: String
        var colorHex: String
    }

    var body: some View {
        Form {
            Section("Folder") {
                Button { showingFolderPicker = true } label: {
                    HStack {
                        Image(systemName: "folder")
                        Text(pickedRef?.displayPath ?? "Choose a folder…")
                        Spacer()
                        if pickedRef != nil { Image(systemName: "checkmark").foregroundStyle(.green) }
                    }
                }
                Text("Masks are written as <image>_mask.png (pixel value = class id), plus a classes JSON.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Project") {
                TextField("Project name", text: $projectName)
            }

            Section("Classes (mask id in paint order)") {
                ForEach($classRows) { $row in
                    HStack {
                        Circle().fill(Color(hex: row.colorHex)).frame(width: 18, height: 18)
                        TextField("class label", text: $row.label)
                            .textInputAutocapitalization(.never)
                    }
                }
                .onDelete { classRows.remove(atOffsets: $0) }

                Button {
                    let idx = classRows.count
                    classRows.append(ClassRow(label: "class_\(idx + 1)", colorHex: ClassPalette.hex(idx)))
                } label: { Label("Add class", systemImage: "plus") }
                .disabled(classRows.count >= 255)
            }

            if let errorMessage {
                Section { Text(errorMessage).foregroundStyle(.red) }
            }
        }
        .navigationTitle("New Segmentation")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            ToolbarItem(placement: .confirmationAction) {
                Button("Create") { create() }.disabled(!isValid)
            }
        }
        .fileImporter(
            isPresented: $showingFolderPicker,
            allowedContentTypes: [.folder],
            allowsMultipleSelection: false
        ) { result in
            handleFolderPick(result)
        }
    }

    private var validClasses: [SegmentationClass] {
        classRows.enumerated().compactMap { idx, row in
            let label = row.label.trimmingCharacters(in: .whitespaces)
            guard !label.isEmpty else { return nil }
            return SegmentationClass(id: UInt8(idx + 1), label: label, colorHex: row.colorHex)
        }
    }

    private var isValid: Bool { pickedRef != nil && !validClasses.isEmpty }

    private func handleFolderPick(_ result: Result<[URL], Error>) {
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

    private func create() {
        guard let ref = pickedRef else { return }
        let project = SegmentationProject(
            displayName: projectName.isEmpty ? ref.displayPath : projectName,
            storageRef: ref,
            classes: validClasses
        )
        store.add(project)
        dismiss()
    }
}
