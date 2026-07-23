import UIKit

/// Renders the base image + colored mask overlay and handles painting (one finger) and
/// zoom/pan (two fingers). Coordinates: an image pixel (ix,iy) maps to view point
/// `pan + (ix*zoom, iy*zoom)`; the inverse maps touches back to pixels.
final class SegmentationCanvasView: UIView {
    private let controller: SegmentationController

    private let containerLayer = CALayer()
    private let baseLayer = CALayer()
    private let overlayLayer = CALayer()

    private var zoom: CGFloat = 1
    private var pan: CGPoint = .zero
    private var didInitialFit = false

    private let minZoom: CGFloat = 0.05
    private let maxZoom: CGFloat = 40

    // Painting
    private var isPainting = false
    private var lastPixel: (x: Int, y: Int)?
    // Single-finger pan (only in .pan tool)
    private var isFingerPanning = false
    private var lastPanPoint: CGPoint?

    init(controller: SegmentationController) {
        self.controller = controller
        super.init(frame: .zero)
        backgroundColor = UIColor.systemGray6
        clipsToBounds = true
        controller.canvas = self

        baseLayer.magnificationFilter = .nearest
        baseLayer.minificationFilter = .linear
        overlayLayer.magnificationFilter = .nearest
        overlayLayer.minificationFilter = .nearest
        containerLayer.addSublayer(baseLayer)
        containerLayer.addSublayer(overlayLayer)
        layer.addSublayer(containerLayer)

        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(handlePinch))
        pinch.delegate = self
        addGestureRecognizer(pinch)
        let twoFingerPan = UIPanGestureRecognizer(target: self, action: #selector(handleTwoFingerPan))
        twoFingerPan.minimumNumberOfTouches = 2
        twoFingerPan.maximumNumberOfTouches = 2
        twoFingerPan.delegate = self
        addGestureRecognizer(twoFingerPan)

        isMultipleTouchEnabled = true
        reloadFromController()
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) not used") }

    /// Pull the current base image into the layers (call after controller.loadImage).
    func reloadFromController() {
        guard let base = controller.baseImage else { return }
        baseLayer.contents = base
        didInitialFit = false
        setNeedsLayout()
        refreshOverlay()
    }

    func refreshOverlay() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        overlayLayer.contents = controller.mask?.makeOverlayCGImage()
        CATransaction.commit()
    }

    /// Drop references to the current image's CGImages so the backing MaskBitmap can be freed
    /// safely (its overlay image uses a non-copying provider). Called before a new image loads.
    func prepareForNewImage() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        baseLayer.contents = nil
        overlayLayer.contents = nil
        CATransaction.commit()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        guard controller.imageSize.width > 0 else { return }
        if !didInitialFit, bounds.width > 0 {
            fitToView()
            didInitialFit = true
        }
        applyTransform()
    }

    private func fitToView() {
        let img = controller.imageSize
        let fit = min(bounds.width / img.width, bounds.height / img.height)
        zoom = fit > 0 ? fit : 1
        pan = CGPoint(
            x: (bounds.width - img.width * zoom) / 2,
            y: (bounds.height - img.height * zoom) / 2
        )
    }

    private func applyTransform() {
        let img = controller.imageSize
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        containerLayer.frame = CGRect(x: pan.x, y: pan.y,
                                      width: img.width * zoom, height: img.height * zoom)
        baseLayer.frame = containerLayer.bounds
        overlayLayer.frame = containerLayer.bounds
        CATransaction.commit()
    }

    // MARK: - Coordinate mapping

    private func pixel(from viewPoint: CGPoint) -> (x: Int, y: Int) {
        let ix = Int(((viewPoint.x - pan.x) / zoom).rounded())
        let iy = Int(((viewPoint.y - pan.y) / zoom).rounded())
        let w = Int(controller.imageSize.width), h = Int(controller.imageSize.height)
        return (min(max(0, ix), w - 1), min(max(0, iy), h - 1))
    }

    // MARK: - Painting (single finger)

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        if (event?.allTouches?.count ?? 1) > 1 { cancelStroke(); return }
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)

        if controller.tool == .pan {
            isFingerPanning = true
            lastPanPoint = p
            return
        }
        startStroke(at: p)
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        if (event?.allTouches?.count ?? 1) > 1 { cancelStroke(); return }
        guard let touch = touches.first else { return }
        let p = touch.location(in: self)

        if isFingerPanning {
            if let last = lastPanPoint {
                pan.x += p.x - last.x
                pan.y += p.y - last.y
                lastPanPoint = p
                applyTransform()
            }
            return
        }
        guard isPainting, let mask = controller.mask else { return }
        let px = pixel(from: p)
        let id: UInt8 = controller.tool == .erase ? 0 : controller.activeClassId
        mask.stampLine(from: lastPixel ?? px, to: px, radius: controller.brushRadius, classId: id)
        lastPixel = px
        refreshOverlay()
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) { endStroke() }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) { cancelStroke() }

    private func startStroke(at point: CGPoint) {
        guard let mask = controller.mask else { return }
        mask.beginStroke()
        let px = pixel(from: point)
        lastPixel = px
        let id: UInt8 = controller.tool == .erase ? 0 : controller.activeClassId
        mask.stampCircle(cx: px.x, cy: px.y, radius: controller.brushRadius, classId: id)
        isPainting = true
        refreshOverlay()
    }

    private func endStroke() {
        if isFingerPanning { isFingerPanning = false; lastPanPoint = nil; return }
        guard isPainting, let mask = controller.mask else { return }
        mask.endStroke()
        isPainting = false
        lastPixel = nil
        controller.strokeCommitted()
    }

    private func cancelStroke() {
        // A second finger arrived mid-stroke: finalize what's painted so it's undoable, then
        // hand off to the zoom/pan recognizers.
        if isPainting { controller.mask?.endStroke(); controller.strokeCommitted() }
        isPainting = false
        isFingerPanning = false
        lastPixel = nil
        lastPanPoint = nil
    }

    // MARK: - Zoom / pan (two fingers)

    @objc private func handlePinch(_ g: UIPinchGestureRecognizer) {
        guard g.state == .changed else { return }
        let focal = g.location(in: self)
        let ix = (focal.x - pan.x) / zoom
        let iy = (focal.y - pan.y) / zoom
        zoom = min(max(minZoom, zoom * g.scale), maxZoom)
        pan.x = focal.x - ix * zoom
        pan.y = focal.y - iy * zoom
        g.scale = 1
        applyTransform()
    }

    @objc private func handleTwoFingerPan(_ g: UIPanGestureRecognizer) {
        let t = g.translation(in: self)
        pan.x += t.x
        pan.y += t.y
        g.setTranslation(.zero, in: self)
        applyTransform()
    }
}

extension SegmentationCanvasView: UIGestureRecognizerDelegate {
    func gestureRecognizer(_ g: UIGestureRecognizer,
                           shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer) -> Bool {
        true   // pinch + two-finger pan together
    }
}
