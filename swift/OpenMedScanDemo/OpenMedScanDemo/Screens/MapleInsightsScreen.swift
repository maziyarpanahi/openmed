import SwiftUI

/// Relationship, reasoning, and document-grounded chat workspace powered by
/// the same local Maple model used earlier in the scan flow.
public struct MapleInsightsScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager
    @FocusState private var isChatFocused: Bool

    public init(flow: ScanFlowViewModel, downloads: ModelDownloadManager) {
        self.flow = flow
        self.downloads = downloads
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.insights.eyebrow,
                spans: [.plain("Connect the "), .accent("clinical story"), .plain(".")],
                subhead: "Inspect entity relationships, ask document-grounded questions, and keep every token on this device.",
                scale: .lg
            )

            localRuntimeBanner
            relationSection
            briefSection
            chatSection
            safetyNote
        }
    }

    private var localRuntimeBanner: some View {
        HStack(alignment: .center, spacing: OM.Space.s3) {
            ZStack {
                Circle()
                    .fill(Color.omTealAccent)
                    .frame(width: 44, height: 44)
                Image(systemName: "leaf.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Color.omPaper)
            }

            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text("MAPLE PREVIEW").omEyebrow()
                    OMBadge("2-bit MLX", tone: .ink)
                }
                Text("Sparse 20B-A1B · exact vocabulary head")
                    .font(.om.body(13, weight: .medium))
                    .foregroundStyle(Color.omInk)
                Text("No cloud fallback · no telemetry · de-identified context only")
                    .font(.om.mono(10))
                    .foregroundStyle(Color.omFgMuted)
            }
            Spacer(minLength: 0)
            Image(systemName: "checkmark.shield.fill")
                .font(.system(size: 20))
                .foregroundStyle(Color.omTealAccent)
        }
        .padding(OM.Space.s4)
        .background(
            LinearGradient(
                colors: [Color.omTealSoft, Color.omBgElevated],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: OM.Radius.lg, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: OM.Radius.lg, style: .continuous)
                .strokeBorder(Color.omTealAccent.opacity(0.35), lineWidth: OM.Stroke.hairline)
        )
    }

    @ViewBuilder
    private var relationSection: some View {
        let relations = flow.clinicalOutput?.relations ?? []
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s4) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("RELATION MAP").omEyebrow()
                        Text("How extracted concepts connect")
                            .font(.om.heading(19, weight: .semibold))
                            .foregroundStyle(Color.omInk)
                    }
                    Spacer()
                    OMBadge(
                        "\(relations.count) links",
                        tone: relations.isEmpty ? .neutral : .accent
                    )
                }

                if relations.isEmpty {
                    HStack(alignment: .top, spacing: OM.Space.s3) {
                        Image(systemName: "point.3.connected.trianglepath.dotted")
                            .font(.system(size: 24))
                            .foregroundStyle(Color.omFgSubtle)
                        Text("No supported relation was found. Maple omits uncertain links instead of filling gaps.")
                            .font(.om.body(14))
                            .foregroundStyle(Color.omFgMuted)
                    }
                    .padding(.vertical, OM.Space.s2)
                } else {
                    ForEach(Array(relations.prefix(12).enumerated()), id: \.element.id) {
                        index, relation in
                        relationRow(relation)
                        if index < min(relations.count, 12) - 1 { OMRule() }
                    }
                }
            }
        }
    }

    private func relationRow(_ relation: DetectedRelation) -> some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            HStack(spacing: OM.Space.s2) {
                conceptPill(relation.head, icon: "circle.fill")
                Image(systemName: "arrow.right")
                    .font(.system(size: 11, weight: .bold))
                    .foregroundStyle(Color.omTealAccent)
                conceptPill(relation.tail, icon: "diamond.fill")
            }
            HStack(spacing: 6) {
                Text(relation.label.uppercased())
                    .font(.om.mono(10, weight: .semibold))
                    .foregroundStyle(Color.omTealHover)
                if let confidence = relation.confidence {
                    Text("· \(String(format: "%.0f%%", confidence * 100))")
                        .font(.om.mono(10))
                        .foregroundStyle(Color.omFgSubtle)
                } else {
                    Text("· MAPLE GENERATED")
                        .font(.om.mono(10))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
        }
        .padding(.vertical, 3)
    }

    private func conceptPill(_ text: String, icon: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 7))
                .foregroundStyle(Color.omTealAccent)
            Text(text)
                .font(.om.body(13, weight: .semibold))
                .foregroundStyle(Color.omInk)
                .lineLimit(2)
        }
        .padding(.vertical, 7)
        .padding(.horizontal, 9)
        .background(Color.omTealSoft, in: RoundedRectangle(cornerRadius: OM.Radius.md))
    }

    private var briefSection: some View {
        OMCard(elevation: .raised) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("MAPLE BRIEF").omEyebrow()
                    Spacer()
                    Image(systemName: "sparkles")
                        .foregroundStyle(Color.omTealAccent)
                }

                if let brief = flow.mapleBrief {
                    Text(brief)
                        .font(.om.body(16))
                        .foregroundStyle(Color.omInk)
                        .textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("Generate a concise synthesis of clinical facts, extracted relationships, uncertainty, and follow-up evidence.")
                        .font(.om.body(15))
                        .foregroundStyle(Color.omFgMuted)
                    Text("Use the action below to create the brief.")
                        .font(.om.mono(10))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
        }
    }

    private var chatSection: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s4) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text("ASK MAPLE").omEyebrow()
                        Text("Chat with this de-identified note")
                            .font(.om.heading(19, weight: .semibold))
                            .foregroundStyle(Color.omInk)
                    }
                    Spacer()
                    OMBadge("Local", tone: .positive, systemImage: "lock.fill")
                }

                if flow.mapleChatTurns.isEmpty {
                    starterQuestions
                } else {
                    VStack(spacing: OM.Space.s3) {
                        ForEach(flow.mapleChatTurns) { turn in
                            MapleChatBubble(turn: turn)
                        }
                    }
                }

                HStack(alignment: .bottom, spacing: OM.Space.s2) {
                    TextField(
                        "Ask about evidence, medication, or follow-up…",
                        text: $flow.mapleChatDraft,
                        axis: .vertical
                    )
                    .font(.om.body(15))
                    .lineLimit(1...4)
                    .focused($isChatFocused)
                    .padding(.vertical, 10)
                    .padding(.horizontal, 12)
                    .background(
                        Color.omPaper2,
                        in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                            .strokeBorder(
                                isChatFocused ? Color.omTealAccent : Color.omBorder,
                                lineWidth: isChatFocused
                                    ? OM.Stroke.focusRing
                                    : OM.Stroke.hairline
                            )
                    )
                    .onSubmit { submitQuestion() }

                    Button(action: submitQuestion) {
                        Image(systemName: flow.isWorking ? "ellipsis" : "arrow.up")
                            .font(.system(size: 14, weight: .bold))
                            .foregroundStyle(Color.omPaper)
                            .frame(width: 42, height: 42)
                            .background(Color.omInk, in: Circle())
                    }
                    .buttonStyle(.plain)
                    .disabled(
                        flow.mapleChatDraft.trimmingCharacters(
                            in: .whitespacesAndNewlines
                        ).isEmpty || flow.isWorking
                    )
                    .accessibilityLabel("Ask Maple")
                }
            }
        }
    }

    private var starterQuestions: some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            Text("TRY A QUESTION").omMonoTag(size: 10)
                .foregroundStyle(Color.omFgSubtle)
            ForEach(
                [
                    "Which findings are linked to medications?",
                    "What follow-up evidence is documented?",
                    "What important uncertainty remains?",
                ], id: \.self
            ) { question in
                Button {
                    flow.mapleChatDraft = question
                    submitQuestion()
                } label: {
                    HStack {
                        Text(question)
                            .font(.om.body(13, weight: .medium))
                            .multilineTextAlignment(.leading)
                        Spacer()
                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 10, weight: .semibold))
                    }
                    .foregroundStyle(Color.omInk)
                    .padding(OM.Space.s3)
                    .background(
                        Color.omPaper2,
                        in: RoundedRectangle(
                            cornerRadius: OM.Radius.md,
                            style: .continuous
                        )
                    )
                }
                .buttonStyle(.plain)
            }
        }
    }

    private var safetyNote: some View {
        HStack(alignment: .top, spacing: OM.Space.s2) {
            Image(systemName: "cross.case.fill")
                .foregroundStyle(Color.omSignal)
            Text("For clinician review only. Maple can miss context or generate incorrect text; verify against the source before any clinical decision. This demo does not diagnose or recommend treatment.")
                .font(.om.body(12))
                .foregroundStyle(Color.omFgMuted)
        }
        .padding(.horizontal, OM.Space.s2)
    }

    private func submitQuestion() {
        isChatFocused = false
        Task { await flow.askMaple() }
    }
}

/// Shared chat presentation used by the live workspace and the simulator reel.
struct MapleChatBubble: View {
    let turn: ScanFlowViewModel.MapleChatTurn

    var body: some View {
        HStack {
            if turn.role == .user { Spacer(minLength: 36) }
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    Text(turn.role == .user ? "YOU" : "MAPLE")
                        .font(.om.mono(9, weight: .semibold))
                        .kerning(1)
                        .foregroundStyle(
                            turn.role == .user
                                ? Color.omPaper.opacity(0.72)
                                : Color.omTealHover
                        )
                    if turn.isStreaming, !turn.content.isEmpty {
                        Circle()
                            .fill(Color.omTealAccent)
                            .frame(width: 5, height: 5)
                            .accessibilityHidden(true)
                        Text("STREAMING")
                            .font(.om.mono(8, weight: .semibold))
                            .kerning(0.7)
                            .foregroundStyle(Color.omTealAccent)
                    }
                }
                if turn.isStreaming, turn.content.isEmpty {
                    HStack(spacing: OM.Space.s2) {
                        ProgressView()
                            .controlSize(.small)
                            .tint(Color.omTealAccent)
                        Text("Reasoning privately…")
                            .font(.om.body(13, italic: true))
                            .foregroundStyle(Color.omFgMuted)
                    }
                    .accessibilityLabel("Maple is reasoning privately")
                } else {
                    HStack(alignment: .lastTextBaseline, spacing: 3) {
                        Text(turn.content)
                            .font(.om.body(14))
                            .foregroundStyle(turn.role == .user ? Color.omPaper : Color.omInk)
                            .textSelection(.enabled)
                        if turn.isStreaming {
                            RoundedRectangle(cornerRadius: 1)
                                .fill(Color.omTealAccent)
                                .frame(width: 2, height: 15)
                                .accessibilityHidden(true)
                        }
                    }
                }
            }
            .padding(OM.Space.s3)
            .background(
                turn.role == .user ? Color.omInk : Color.omTealSoft,
                in: RoundedRectangle(cornerRadius: OM.Radius.lg, style: .continuous)
            )
            if turn.role == .assistant { Spacer(minLength: 36) }
        }
    }
}
