import SwiftUI
import UIKit

/// The swipe surface: a card showing the current image, dragged toward a gesture to assign
/// its class. A legend maps directions to labels; a progress line tracks completion.
struct SwipeAnnotationView: View {
    @Bindable var viewModel: SwipeAnnotationViewModel
    @Environment(\.scenePhase) private var scenePhase

    @State private var dragOffset: CGSize = .zero
    /// Set when we commit a swipe, to animate the card fully off-screen before advancing.
    @State private var flyAway: CGSize? = nil

    private let commitDistance: CGFloat = 110

    var body: some View {
        VStack(spacing: 0) {
            content
        }
        .navigationTitle(viewModel.activeColumn?.name ?? "Annotate")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                if viewModel.isFlushing {
                    ProgressView().controlSize(.small)
                } else {
                    Button { Task { await viewModel.flush() } } label: {
                        Image(systemName: "square.and.arrow.down")
                    }
                    .accessibilityLabel("Save now")
                }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { Task { await viewModel.flush() } }
        }
        .onDisappear { Task { await viewModel.flush() } }
    }

    @ViewBuilder
    private var content: some View {
        switch viewModel.loadState {
        case .loading:
            Spacer(); ProgressView("Loading images…"); Spacer()
        case .failed(let message):
            Spacer()
            ContentUnavailableView("Problem", systemImage: "exclamationmark.triangle",
                                   description: Text(message))
            Spacer()
        case .empty:
            Spacer()
            ContentUnavailableView("No images", systemImage: "photo.on.rectangle",
                                   description: Text("This folder has no supported images."))
            Spacer()
        case .ready:
            if viewModel.isComplete {
                completionView
            } else {
                annotationSurface
            }
        }
    }

    // MARK: - Annotation surface

    private var annotationSurface: some View {
        VStack(spacing: 12) {
            legend
            cardStack
            controls
            Text(viewModel.progressText)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .padding(.bottom, 8)
        }
        .padding(.horizontal)
    }

    private var cardStack: some View {
        GeometryReader { geo in
            ZStack {
                // Next card peeking underneath for continuity.
                if let next = peekImage {
                    card(for: next, data: viewModel.imageData(for: next))
                        .scaleEffect(0.96)
                        .opacity(0.5)
                }
                if let current = viewModel.currentImage {
                    card(for: current, data: viewModel.imageData(for: current))
                        .offset(x: (flyAway?.width ?? dragOffset.width),
                                y: (flyAway?.height ?? dragOffset.height))
                        .rotationEffect(.degrees(Double((flyAway?.width ?? dragOffset.width) / 22)))
                        .overlay(alignment: .center) { directionHint }
                        .gesture(dragGesture(in: geo.size))
                        .animation(.spring(duration: 0.28), value: flyAway)
                }
            }
            .frame(width: geo.size.width, height: geo.size.height)
        }
        .frame(maxHeight: .infinity)
    }

    private var peekImage: StoredImage? {
        // The image after current, if any (index+1); read from the VM's ordered list.
        guard let current = viewModel.currentImage,
              let idx = viewModel.images.firstIndex(of: current),
              viewModel.images.indices.contains(idx + 1) else { return nil }
        return viewModel.images[idx + 1]
    }

    private func card(for image: StoredImage, data: Data?) -> some View {
        ZStack {
            RoundedRectangle(cornerRadius: 18)
                .fill(Color(.secondarySystemBackground))
                .shadow(radius: 6, y: 3)
            if let data, let uiImage = UIImage(data: data) {
                Image(uiImage: uiImage)
                    .resizable()
                    .scaledToFit()
                    .padding(10)
            } else {
                VStack(spacing: 8) {
                    ProgressView()
                    Text(image.filename).font(.caption).foregroundStyle(.secondary)
                }
            }
            VStack {
                Spacer()
                Text(image.filename)
                    .font(.caption2)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(.ultraThinMaterial, in: Capsule())
                    .padding(8)
            }
        }
    }

    /// A colored badge showing which class the current drag would assign.
    @ViewBuilder
    private var directionHint: some View {
        if let gesture = gestureForCurrentDrag(), let cls = viewModel.classFor(gesture) {
            Text(cls.label.uppercased())
                .font(.title2.bold())
                .padding(.horizontal, 16).padding(.vertical, 10)
                .background(Color(hex: cls.colorHex).opacity(0.9), in: RoundedRectangle(cornerRadius: 12))
                .foregroundStyle(.white)
                .rotationEffect(.degrees(-8))
                .opacity(min(1, dragMagnitude / commitDistance))
        }
    }

    // MARK: - Legend & controls

    private var legend: some View {
        let classes = viewModel.activeColumn?.classes ?? []
        return HStack(spacing: 10) {
            ForEach(classes) { cls in
                HStack(spacing: 5) {
                    Image(systemName: cls.gesture.systemImage)
                    Text(cls.label).lineLimit(1)
                }
                .font(.caption.bold())
                .padding(.horizontal, 9).padding(.vertical, 6)
                .background(Color(hex: cls.colorHex).opacity(0.18), in: Capsule())
                .foregroundStyle(Color(hex: cls.colorHex))
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var controls: some View {
        HStack {
            Button { viewModel.goBack(); resetCard() } label: {
                Label("Back", systemImage: "arrow.uturn.backward")
            }
            Spacer()
            Button { viewModel.skip(); resetCard() } label: {
                Label("Skip", systemImage: "forward")
            }
        }
        .buttonStyle(.bordered)
        .font(.subheadline)
    }

    private var completionView: some View {
        VStack(spacing: 16) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 56)).foregroundStyle(.green)
            Text("Round complete").font(.title2.bold())
            Text(viewModel.progressText).foregroundStyle(.secondary)
            Button {
                Task { await viewModel.flush() }
            } label: {
                Label("Save CSV", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
    }

    // MARK: - Gesture handling

    private var dragMagnitude: CGFloat {
        max(abs(dragOffset.width), abs(dragOffset.height))
    }

    private func dragGesture(in size: CGSize) -> some SwiftUI.Gesture {
        DragGesture()
            .onChanged { value in
                if flyAway == nil { dragOffset = value.translation }
            }
            .onEnded { value in
                dragOffset = value.translation
                if let gesture = gestureForCurrentDrag(), viewModel.classFor(gesture) != nil {
                    commit(gesture, in: size)
                } else {
                    withAnimation(.spring) { dragOffset = .zero }
                }
            }
    }

    /// Which gesture the current drag points at — dominant axis past a small deadzone.
    private func gestureForCurrentDrag() -> Gesture? {
        let dx = dragOffset.width, dy = dragOffset.height
        guard max(abs(dx), abs(dy)) > 24 else { return nil }
        if abs(dx) >= abs(dy) {
            return dx < 0 ? .swipeLeft : .swipeRight
        } else {
            return dy < 0 ? .swipeUp : .swipeDown
        }
    }

    private func commit(_ gesture: Gesture, in size: CGSize) {
        // Fling the card off-screen in the drag direction, then apply + reset.
        let target: CGSize
        switch gesture {
        case .swipeLeft:  target = CGSize(width: -size.width * 1.4, height: dragOffset.height)
        case .swipeRight: target = CGSize(width: size.width * 1.4, height: dragOffset.height)
        case .swipeUp:    target = CGSize(width: dragOffset.width, height: -size.height * 1.4)
        case .swipeDown:  target = CGSize(width: dragOffset.width, height: size.height * 1.4)
        }
        flyAway = target
        let haptic = UIImpactFeedbackGenerator(style: .light)
        haptic.impactOccurred()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.24) {
            viewModel.apply(gesture)
            resetCard()
        }
    }

    private func resetCard() {
        flyAway = nil
        dragOffset = .zero
    }
}
