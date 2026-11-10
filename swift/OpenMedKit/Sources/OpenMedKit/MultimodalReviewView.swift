import SwiftUI

/// Reusable local review. Replacing the case or source creates a fresh unseen presentation.
@MainActor
public struct MultimodalReviewView: View {
    private let review: MultimodalReviewCase
    private let source: any LocalMultimodalReviewSource
    private let onDecision: (MultimodalReviewDecision) -> Void

    public init(
        review: MultimodalReviewCase, source: any LocalMultimodalReviewSource,
        onDecision: @escaping (MultimodalReviewDecision) -> Void
    ) {
        self.review = review
        self.source = source
        self.onDecision = onDecision
    }

    public var body: some View {
        ReviewPanel(review: review, source: source, onDecision: onDecision)
            .id(ReviewIdentity(digest: review.digest, source: ObjectIdentifier(source)))
    }
}

private struct ReviewIdentity: Hashable {
    let digest: String
    let source: ObjectIdentifier
}

private struct ReviewExposure: Equatable {
    let identity: UUID
    let frame: CGRect
}

private struct ReviewExposureKey: PreferenceKey {
    static let defaultValue: ReviewExposure? = nil
    static func reduce(value: inout ReviewExposure?, nextValue: () -> ReviewExposure?) { value = nextValue() ?? value }
}

private struct ReviewViewportKey: PreferenceKey {
    static let defaultValue: CGSize = .zero
    static func reduce(value: inout CGSize, nextValue: () -> CGSize) {
        let next = nextValue()
        if next.width > 0 && next.height > 0 { value = next }
    }
}

private struct ReviewVisibility: Equatable {
    let viewport: CGSize
    let exposure: ReviewExposure?
}

@MainActor
private struct ReviewPanel: View {
    @State private var session: MultimodalReviewSession
    @Environment(\.scenePhase) private var scenePhase
    @FocusState private var focusedControl: String?
    @State private var viewportSize: CGSize = .zero
    @State private var exposure: ReviewExposure?
    private let onDecision: (MultimodalReviewDecision) -> Void

    init(
        review: MultimodalReviewCase, source: any LocalMultimodalReviewSource,
        onDecision: @escaping (MultimodalReviewDecision) -> Void
    ) {
        _session = State(initialValue: MultimodalReviewSession(review: review, source: source))
        self.onDecision = onDecision
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                if session.completed {
                    Label("Choice returned for review", systemImage: "checkmark.circle")
                        .accessibilityIdentifier("review-completed")
                } else if let content = session.content {
                    let shownIdentity = session.presentationID
                    protectedContent(content, identity: shownIdentity)
                        .id(shownIdentity)
                    alternatives(content)
                    actions
                } else {
                    hold
                }
            }
            .padding(20)
            .frame(maxWidth: 760, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .coordinateSpace(name: "multimodal-review-viewport")
        .background(GeometryReader { proxy in Color.clear.preference(key: ReviewViewportKey.self, value: proxy.size) })
        .onPreferenceChange(ReviewViewportKey.self) { viewportSize = $0 }
        .onPreferenceChange(ReviewExposureKey.self) { exposure = $0 }
        .onChange(of: ReviewVisibility(viewport: viewportSize, exposure: exposure)) { _, visibility in markVisible(visibility) }
        .privacySensitive()
        .task { if scenePhase == .active { session.load() } }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { session.load() } else { session.hide() }
        }
        .onDisappear { session.hide() }
    }

    private func markVisible(_ visibility: ReviewVisibility) {
        guard scenePhase == .active, let exposure = visibility.exposure, exposure.frame.width > 0, exposure.frame.height > 0 else { return }
        let visible = CGRect(origin: .zero, size: visibility.viewport).intersection(exposure.frame)
        guard !visible.isNull, visible.width >= min(64, exposure.frame.width), visible.height >= min(64, exposure.frame.height) else { return }
        session.markPresented(exposure.identity)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.title2.bold())
            Text("Compare the source and alternatives before choosing.")
                .foregroundStyle(.secondary)
            Text(provenance).font(.caption).foregroundStyle(.secondary)
            Text("Revision \(session.review.revision.prefix(12))")
                .font(.caption.monospaced()).foregroundStyle(.secondary)
            ForEach(session.review.uncertainty, id: \.self) { reason in
                Label(reasonLabel(reason), systemImage: "exclamationmark.triangle")
                    .font(.callout)
            }
            if !session.review.canAccept {
                Text("This transcript is provisional. Accept is unavailable until it is finalized.")
                    .font(.callout)
            }
        }
    }

    private var title: String {
        switch session.review.kind {
        case .visualConflict: return "Review conflicting values"
        case .handwriting: return "Review handwriting"
        case .speakerRole: return "Review speaker attribution"
        }
    }

    private var provenance: String {
        switch session.review.evidence.projection {
        case .visual(let visual):
            return "Source region · Page \(visual.page + 1) · \(visual.unit.rawValue) coordinates"
        case .transcript(let transcript):
            guard let span = transcript.currentScalarSpan else { return "Transcript citation unavailable" }
            return "Transcript revision \(transcript.revision) · Unicode scalars [\(span.start), \(span.end))"
        default: return "Unsupported source"
        }
    }

    private func reasonLabel(_ reason: MultimodalReviewUncertainty) -> String {
        switch reason {
        case .conflictingValues: return "The source readers disagree."
        case .ambiguousHandwriting: return "The handwriting has more than one interpretation."
        case .uncertainSpeaker: return "The speaker role is unconfirmed."
        case .provisionalTranscript: return "The transcript may still change."
        }
    }

    private func protectedContent(_ content: LocalMultimodalReviewContent, identity: UUID) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Source evidence").font(.headline)
            VStack(alignment: .leading, spacing: 12) {
                if let image = content.image {
                    Image(image, scale: 1, label: Text("Protected source region. Compare it with the alternatives."))
                        .resizable().interpolation(.none).scaledToFit()
                        .frame(maxHeight: 320)
                        .accessibilityLabel("Protected source region. Compare it with the alternatives.")
                        .accessibilityIdentifier("review-source-image")
                }
                if let text = content.text {
                    Text(verbatim: text)
                        .font(.body)
                        .accessibilityIdentifier("review-source-text")
                }
            }
            .background(
                GeometryReader { proxy in
                    Color.clear.preference(
                        key: ReviewExposureKey.self,
                        value: ReviewExposure(identity: identity, frame: proxy.frame(in: .named("multimodal-review-viewport"))))
                })
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .contain)
    }

    private func alternatives(_ content: LocalMultimodalReviewContent) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Alternatives").font(.headline)
            ForEach(Array(content.alternatives.enumerated()), id: \.element.reference) { index, alternative in
                Button {
                    session.select(alternative.reference)
                } label: {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: session.selected == alternative.reference ? "checkmark.circle.fill" : "circle")
                        Text(verbatim: alternative.value).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("review-alternative-\(index + 1)")
                .focused($focusedControl, equals: "alternative-\(index + 1)")
                .accessibilityLabel(Text(verbatim: "Alternative \(index + 1): \(alternative.value)"))
                .accessibilityValue(session.selected == alternative.reference ? "Selected" : "Not selected")
                .disabled(!session.canAct)
            }
        }
    }

    private var actions: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { decisionButtons }
            VStack(alignment: .leading, spacing: 12) { decisionButtons }
        }
    }

    @ViewBuilder
    private var decisionButtons: some View {
        Button("Accept selected") { decide(.accept) }
            .buttonStyle(.borderedProminent)
            .focused($focusedControl, equals: "accept")
            .reviewShortcut("a", enabled: focusedControl != nil)
            .accessibilityIdentifier("review-accept")
            .disabled(!session.canAccept)
        Button("Reject alternatives") { decide(.reject) }
            .buttonStyle(.bordered)
            .focused($focusedControl, equals: "reject")
            .reviewShortcut("r", enabled: focusedControl != nil)
            .accessibilityIdentifier("review-reject")
            .disabled(!session.canAct)
        Button("Defer") { decide(.defer) }
            .buttonStyle(.bordered)
            .focused($focusedControl, equals: "defer")
            .reviewShortcut("d", enabled: focusedControl != nil)
            .accessibilityIdentifier("review-defer")
            .disabled(!session.canAct)
    }

    private func decide(_ choice: MultimodalReviewChoice) {
        guard scenePhase == .active, let decision = session.decide(choice) else { return }
        onDecision(decision)
    }

    private var hold: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(holdMessage, systemImage: "hand.raised")
                .accessibilityIdentifier("review-hold")
            Button("Reload local evidence") { session.load() }
                .buttonStyle(.bordered)
                .accessibilityIdentifier("review-reload")
            Text("No choice can be approved until the current source is displayed.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var holdMessage: String {
        switch session.refusal {
        case .changed: return "The evidence changed. Request its current revision."
        case .missing: return "The protected source is unavailable."
        case .invalid, .unsupported: return "This source cannot be safely displayed."
        case .unavailable: return "Local source access is unavailable."
        default: return "Evidence is hidden until this view is active."
        }
    }
}

extension View {
    @ViewBuilder
    fileprivate func reviewShortcut(_ key: Character, enabled: Bool) -> some View {
        #if os(watchOS)
            self
        #else
            self.keyboardShortcut(enabled ? KeyboardShortcut(KeyEquivalent(key), modifiers: [.command, .shift]) : nil)
        #endif
    }
}
