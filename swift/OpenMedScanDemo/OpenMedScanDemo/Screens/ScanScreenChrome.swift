import SwiftUI

/// Shared chrome for each stage screen — the top logo strip, the workflow
/// indicator, and the sticky bottom action bar. Screens provide their own
/// content + primary action via the trailing content closures.
public struct ScanScreenChrome<Content: View, ActionBar: View>: View {
    public let currentStage: ScanStage
    public let furthestReached: ScanStage
    public let onJump: (ScanStage) -> Void
    public let content: () -> Content
    public let actionBar: () -> ActionBar

    public init(
        currentStage: ScanStage,
        furthestReached: ScanStage,
        onJump: @escaping (ScanStage) -> Void,
        @ViewBuilder content: @escaping () -> Content,
        @ViewBuilder actionBar: @escaping () -> ActionBar
    ) {
        self.currentStage = currentStage
        self.furthestReached = furthestReached
        self.onJump = onJump
        self.content = content
        self.actionBar = actionBar
    }

    public var body: some View {
        ZStack(alignment: .top) {
            Color.omPaper.ignoresSafeArea()

            VStack(spacing: 0) {
                topBar
                OMRule()
                ScrollView {
                    VStack(spacing: OM.Space.s5) {
                        content()
                    }
                    .padding(.horizontal, OM.Space.s5)
                    .padding(.top, OM.Space.s4)
                    .padding(.bottom, 120)
                }
                .scrollDismissesKeyboard(.interactively)
            }

            VStack(spacing: 0) {
                Spacer(minLength: 0)
                OMRule()
                actionBar()
                    .padding(.horizontal, OM.Space.s5)
                    .padding(.vertical, OM.Space.s3)
                    .background(.thinMaterial)
            }
            .ignoresSafeArea(edges: .bottom)
        }
    }

    private var topBar: some View {
        HStack(alignment: .center, spacing: OM.Space.s4) {
            OMBrandLockup(compact: true)
            Spacer()
            OMWorkflowIndicator(
                currentStage: currentStage,
                furthestReached: furthestReached,
                onJump: onJump
            )
            .frame(maxWidth: 240)
        }
        .padding(.horizontal, OM.Space.s5)
        .padding(.vertical, OM.Space.s3)
        .background(Color.omPaper)
    }
}

/// A header block for every stage screen: eyebrow + serif display headline
/// + optional subhead. Factored out of the individual screens.
public struct ScanStageHeader: View {
    public let eyebrow: String
    public let spans: [OMDisplayHeadline.Span]
    public let subhead: String?
    public var scale: OMDisplayHeadline.Scale = .md

    public init(
        eyebrow: String,
        spans: [OMDisplayHeadline.Span],
        subhead: String? = nil,
        scale: OMDisplayHeadline.Scale = .md
    ) {
        self.eyebrow = eyebrow
        self.spans = spans
        self.subhead = subhead
        self.scale = scale
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s3) {
            Text(eyebrow).omEyebrow()
            OMDisplayHeadline(spans, scale: scale)
            if let subhead {
                Text(subhead)
                    .font(.om.body(17))
                    .foregroundStyle(Color.omFgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
