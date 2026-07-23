import AVFoundation
import CoreGraphics
import Foundation
import Observation

enum CameraError: LocalizedError {
    case notReady, noCamera
    var errorDescription: String? {
        switch self {
        case .notReady: return "The camera isn't ready yet."
        case .noCamera: return "No usable camera was found."
        }
    }
}

enum CountConfig {
    /// Long-side cap for captured frames fed to the counter (perf + memory).
    static let maxDimension = 2048
}

/// Wraps an AVFoundation photo session: permission, live preview, single-shot capture → CGImage.
/// Session work runs on a private queue; UI-observed state is mutated on the main thread. The
/// AVFoundation delegate lives in a separate object so this stays a plain `@Observable`.
@Observable
final class CameraController {
    let session = AVCaptureSession()

    var authorized = false
    var running = false
    var errorMessage: String?

    @ObservationIgnored private let photoOutput = AVCapturePhotoOutput()
    @ObservationIgnored private let sessionQueue = DispatchQueue(label: "cellect.camera.session")
    @ObservationIgnored private var configured = false
    @ObservationIgnored private var activeDelegate: PhotoCaptureDelegate?

    // MARK: - Lifecycle

    func onAppear() {
        Task { [weak self] in
            guard let self else { return }
            await self.requestAccess()
            if self.authorized {
                self.configureIfNeeded()
                self.start()
            }
        }
    }

    func onDisappear() { stop() }

    private func requestAccess() async {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setAuthorized(true)
        case .notDetermined:
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            setAuthorized(granted)
        default:
            setAuthorized(false)
        }
    }

    private func configureIfNeeded() {
        sessionQueue.async { [weak self] in
            guard let self, !self.configured else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .photo
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
                ?? AVCaptureDevice.default(for: .video)
            guard let device,
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input) else {
                self.session.commitConfiguration()
                self.report(CameraError.noCamera.localizedDescription)
                return
            }
            self.session.addInput(input)
            if self.session.canAddOutput(self.photoOutput) {
                self.session.addOutput(self.photoOutput)
            }
            self.session.commitConfiguration()
            self.configured = true
        }
    }

    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.session.isRunning { self.session.startRunning() }
            self.setRunning(self.session.isRunning)
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning { self.session.stopRunning() }
            self.setRunning(false)
        }
    }

    // MARK: - Capture

    func capture() async throws -> CGImage {
        try await withCheckedThrowingContinuation { cont in
            sessionQueue.async { [weak self] in
                guard let self else { cont.resume(throwing: CameraError.notReady); return }
                guard self.configured, self.session.isRunning else {
                    cont.resume(throwing: CameraError.notReady); return
                }
                let delegate = PhotoCaptureDelegate { [weak self] result in
                    self?.activeDelegate = nil     // release after the callback
                    cont.resume(with: result)
                }
                self.activeDelegate = delegate     // AVFoundation doesn't retain the delegate
                self.photoOutput.capturePhoto(with: AVCapturePhotoSettings(), delegate: delegate)
            }
        }
    }

    // MARK: - Main-thread state helpers

    private func setAuthorized(_ v: Bool) {
        DispatchQueue.main.async {
            self.authorized = v
            if !v { self.errorMessage = "Camera access denied. Enable it in Settings." }
        }
    }
    private func setRunning(_ v: Bool) { DispatchQueue.main.async { self.running = v } }
    private func report(_ m: String) { DispatchQueue.main.async { self.errorMessage = m } }
}

/// Owns one photo-capture callback. Kept alive by `CameraController.activeDelegate` until done.
private final class PhotoCaptureDelegate: NSObject, AVCapturePhotoCaptureDelegate {
    private let completion: (Result<CGImage, Error>) -> Void
    init(completion: @escaping (Result<CGImage, Error>) -> Void) { self.completion = completion }

    func photoOutput(_ output: AVCapturePhotoOutput,
                     didFinishProcessingPhoto photo: AVCapturePhoto,
                     error: Error?) {
        if let error { completion(.failure(error)); return }
        guard let data = photo.fileDataRepresentation(),
              let cg = ImageLoader.decode(data, maxDimension: CountConfig.maxDimension) else {
            completion(.failure(CountError.decodeFailed)); return
        }
        completion(.success(cg))
    }
}
