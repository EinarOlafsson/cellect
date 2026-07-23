import SwiftUI

/// The painting screen: canvas + tool bar (tool, class, brush size, undo/redo) + image nav.
struct SegmentationEditorView: View {
    @Bindable var session: SegmentationSession
    @Environment(\.scenePhase) private var scenePhase

    private var controller: SegmentationController { session.controller }

    var body: some View {
        VStack(spacing: 0) {
            switch session.state {
            case .loading:
                Spacer(); ProgressView("Loading…"); Spacer()
            case .failed(let msg):
                Spacer()
                ContentUnavailableView("Problem", systemImage: "exclamationmark.triangle",
                                       description: Text(msg))
                Spacer()
            case .empty:
                Spacer()
                ContentUnavailableView("No images", systemImage: "photo.on.rectangle",
                                       description: Text("This folder has no supported images."))
                Spacer()
            case .ready:
                canvas
                controls
            }
        }
        .navigationTitle("Segment")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if session.isSaving {
                    ProgressView().controlSize(.small)
                } else {
                    Button { Task { await session.saveCurrent() } } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .disabled(!controller.maskDirty)
                    .accessibilityLabel("Save mask")
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { Task { await session.saveCurrent() } }
        }
        .onDisappear { Task { await session.saveCurrent() } }
    }

    private var canvas: some View {
        SegmentationCanvasRepresentable(controller: controller)
            .background(Color(.systemGray6))
            .overlay(alignment: .top) {
                Text("One finger paints · two fingers zoom & pan")
                    .font(.caption2)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(.top, 6)
            }
    }

    // MARK: - Controls

    private var controls: some View {
        VStack(spacing: 10) {
            classPicker
            HStack(spacing: 12) {
                toolPicker
                Spacer()
                undoRedo
            }
            brushRow
            navRow
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var classPicker: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(controller.classes) { cls in
                    let selected = controller.activeClassId == cls.id && controller.tool != .erase
                    Button {
                        controller.activeClassId = cls.id
                        if controller.tool == .erase { controller.tool = .brush }
                    } label: {
                        HStack(spacing: 6) {
                            Circle().fill(Color(hex: cls.colorHex)).frame(width: 14, height: 14)
                            Text(cls.label).font(.subheadline)
                        }
                        .padding(.horizontal, 12).padding(.vertical, 7)
                        .background(
                            Color(hex: cls.colorHex).opacity(selected ? 0.28 : 0.12),
                            in: Capsule()
                        )
                        .overlay(
                            Capsule().stroke(Color(hex: cls.colorHex),
                                             lineWidth: selected ? 2 : 0)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var toolPicker: some View {
        Picker("Tool", selection: Binding(
            get: { controller.tool },
            set: { controller.tool = $0 }
        )) {
            ForEach(SegTool.allCases) { tool in
                Image(systemName: tool.systemImage).tag(tool)
            }
        }
        .pickerStyle(.segmented)
        .frame(width: 160)
    }

    private var undoRedo: some View {
        HStack(spacing: 6) {
            Button { controller.undo() } label: { Image(systemName: "arrow.uturn.backward") }
                .disabled(!controller.undoAvailable)
            Button { controller.redo() } label: { Image(systemName: "arrow.uturn.forward") }
                .disabled(!controller.redoAvailable)
            Menu {
                Button(role: .destructive) { controller.clear() } label: {
                    Label("Clear mask", systemImage: "trash")
                }
            } label: { Image(systemName: "ellipsis.circle") }
        }
        .buttonStyle(.bordered)
    }

    private var brushRow: some View {
        HStack {
            Image(systemName: "circle.fill").font(.system(size: 10))
            Slider(
                value: Binding(
                    get: { Double(controller.brushRadius) },
                    set: { controller.brushRadius = Int($0) }
                ),
                in: 3...200
            )
            Image(systemName: "circle.fill").font(.system(size: 22))
            Text("\(controller.brushRadius)px").font(.caption.monospacedDigit())
                .frame(width: 48, alignment: .trailing)
        }
        .foregroundStyle(.secondary)
    }

    private var navRow: some View {
        HStack {
            Button { Task { await session.goPrev() } } label: {
                Label("Prev", systemImage: "chevron.left")
            }
            .disabled(session.index == 0)
            Spacer()
            Text(session.progressText).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
            Button { Task { await session.goNext() } } label: {
                Label("Next", systemImage: "chevron.right")
            }
            .disabled(session.index >= session.images.count - 1)
        }
        .buttonStyle(.bordered)
        .font(.subheadline)
    }
}
