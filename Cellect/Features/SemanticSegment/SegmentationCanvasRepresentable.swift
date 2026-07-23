import SwiftUI

/// Bridges the UIKit painting canvas into SwiftUI. The canvas reads tool/brush/class live from
/// the controller, so there's no per-change UIView update needed here.
struct SegmentationCanvasRepresentable: UIViewRepresentable {
    let controller: SegmentationController

    func makeUIView(context: Context) -> SegmentationCanvasView {
        SegmentationCanvasView(controller: controller)
    }

    func updateUIView(_ uiView: SegmentationCanvasView, context: Context) {
        // No-op: painting state is pulled from the controller at touch time; image swaps are
        // pushed via controller.canvas?.reloadFromController().
    }
}
