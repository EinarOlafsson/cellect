import SwiftUI
import UniformTypeIdentifiers

/// New-project setup: pick a folder, name the annotation column, and map gestures to classes.
struct ProjectSetupView: View {
    @Environment(ProjectStore.self) private var store
    @Environment(\.dismiss) private var dismiss

    @State private var projectName = ""
    @State private var columnName = ""
    @State private var gestureLabels: [Gesture: String] = [
        .swipeLeft: "", .swipeRight: "", .swipeUp: "", .swipeDown: ""
    ]
    @State private var pickedRef: StorageRef?
    @State private var showingFolderPicker = false
    @State private var errorMessage: String?

    private let orderedGestures: [Gesture] = [.swipeRight, .swipeLeft, .swipeUp, .swipeDown]

    var body: some View {
        Form {
            Section("Folder") {
                Button {
                    showingFolderPicker = true
                } label: {
                    HStack {
                        Image(systemName: "folder")
                        Text(pickedRef?.displayPath ?? "Choose a folder…")
                        Spacer()
                        if pickedRef != nil { Image(systemName: "checkmark").foregroundStyle(.green) }
                    }
                }
                Text("Local or iCloud Drive folder of images. A CSV is written back into it.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            Section("Project") {
                TextField("Project name", text: $projectName)
                TextField("Column name (e.g. phenotype)", text: $columnName)
                    .textInputAutocapitalization(.never)
            }

            Section("Gestures → classes") {
                ForEach(orderedGestures) { gesture in
                    HStack {
                        Image(systemName: gesture.systemImage)
                            .frame(width: 22)
                            .foregroundStyle(.secondary)
                        Text(gesture.label).frame(width: 96, alignment: .leading)
                        TextField("class label", text: binding(for: gesture))
                            .multilineTextAlignment(.trailing)
                            .textInputAutocapitalization(.never)
                    }
                }
                Text("Fill at least two. Empty gestures are ignored.")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if let errorMessage {
                Section { Text(errorMessage).foregroundStyle(.red) }
            }
        }
        .navigationTitle("New Project")
        .toolbar {
            ToolbarItem(placement: .cancellationAction) {
                Button("Cancel") { dismiss() }
            }
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

    private func binding(for gesture: Gesture) -> Binding<String> {
        Binding(
            get: { gestureLabels[gesture] ?? "" },
            set: { gestureLabels[gesture] = $0 }
        )
    }

    private var filledClasses: [AnnotationClass] {
        orderedGestures.enumerated().compactMap { idx, gesture in
            let label = (gestureLabels[gesture] ?? "").trimmingCharacters(in: .whitespaces)
            guard !label.isEmpty else { return nil }
            return AnnotationClass(label: label, gesture: gesture, colorHex: ClassPalette.hex(idx))
        }
    }

    private var isValid: Bool {
        pickedRef != nil
            && !columnName.trimmingCharacters(in: .whitespaces).isEmpty
            && filledClasses.count >= 2
    }

    private func handleFolderPick(_ result: Result<[URL], Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            do {
                let ref = try LocalFolderProvider.makeRef(from: url)
                pickedRef = ref
                if projectName.isEmpty { projectName = ref.displayPath }
                errorMessage = nil
            } catch {
                errorMessage = error.localizedDescription
            }
        case .failure(let error):
            errorMessage = error.localizedDescription
        }
    }

    private func create() {
        guard let ref = pickedRef else { return }
        let column = AnnotationColumn(
            name: columnName.trimmingCharacters(in: .whitespaces),
            classes: filledClasses
        )
        let project = AnnotationProject(
            displayName: projectName.isEmpty ? ref.displayPath : projectName,
            storageRef: ref,
            columns: [column]
        )
        store.add(project)
        dismiss()
    }
}
