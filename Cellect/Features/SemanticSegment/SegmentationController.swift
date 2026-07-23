import CoreGraphics
import Observation
import UIKit

enum SegTool: String, CaseIterable, Identifiable {
    case brush, erase, pan
    var id: String { rawValue }
    var systemImage: String {
        switch self {
        case .brush: return "paintbrush.pointed"
        case .erase: return "eraser"
        case .pan:   return "hand.draw"
        }
    }
    var title: String { rawValue.capitalized }
}

/// Editable state for one image's segmentation, shared between the SwiftUI toolbar and the
/// UIKit canvas. The canvas paints into `mask`; toolbar buttons drive undo/redo/save.
@MainActor
@Observable
final class SegmentationController {
    var tool: SegTool = .brush
    /// Brush radius in image pixels.
    var brushRadius: Int = 24
    var activeClassId: UInt8 = 1

    private(set) var mask: MaskBitmap?
    private(set) var baseImage: CGImage?
    private(set) var imageSize: CGSize = .zero

    var undoAvailable = false
    var redoAvailable = false
    var maskDirty = false

    let classes: [SegmentationClass]

    /// Set by the canvas so controller actions can force a redraw.
    @ObservationIgnored weak var canvas: SegmentationCanvasView?

    init(classes: [SegmentationClass]) {
        self.classes = classes
        self.activeClassId = classes.first?.id ?? 1
    }

    /// Load a base image + optional existing mask, building a fresh MaskBitmap.
    func loadImage(base: CGImage, existingMask: CGImage?) {
        // Detach the old overlay CGImage from the layer BEFORE the previous MaskBitmap is
        // released — that image wraps the old mask's buffer with a non-copying provider.
        canvas?.prepareForNewImage()
        baseImage = base
        imageSize = CGSize(width: base.width, height: base.height)
        let m = MaskBitmap(width: base.width, height: base.height, classes: classes)
        if let existingMask { m.loadLabel(from: existingMask) }
        mask = m
        maskDirty = false
        refreshEditState()
    }

    func color(for id: UInt8) -> String {
        classes.first { $0.id == id }?.colorHex ?? "#888888"
    }

    // MARK: - Actions (driven from the toolbar)

    func undo() { mask?.undo(); maskDirty = true; canvas?.refreshOverlay(); refreshEditState() }
    func redo() { mask?.redo(); maskDirty = true; canvas?.refreshOverlay(); refreshEditState() }
    func clear() { mask?.clearAll(); maskDirty = true; canvas?.refreshOverlay(); refreshEditState() }

    /// Called by the canvas at the end of every stroke.
    func strokeCommitted() { maskDirty = true; refreshEditState() }

    func refreshEditState() {
        undoAvailable = mask?.canUndo ?? false
        redoAvailable = mask?.canRedo ?? false
    }
}
