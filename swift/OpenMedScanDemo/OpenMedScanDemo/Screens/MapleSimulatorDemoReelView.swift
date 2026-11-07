import SwiftUI

/// A deterministic simulator-only tour built from the shipping scan-demo
/// chrome and components. It never initializes MLX and labels all generated
/// content as synthetic; physical-device runs use the same streaming bubble.
struct MapleSimulatorDemoReelView: View {
    @State private var phase = 0
    @State private var playbackID = 0
    @State private var isPlaying = false
    @State private var downloadProgress = 0.0
    @State private var chatResponse = ""
    @State private var isChatStreaming = false

    var body: some View {
        ScanScreenChrome(
            currentStage: currentStage,
            furthestReached: currentStage,
            onJump: { _ in },
            content: {
                VStack(alignment: .leading, spacing: OM.Space.s4) {
                    demoDisclosure
                    Group {
                        if isPlaying {
                            phaseContent
                                .id(phase)
                                .transition(
                                    .asymmetric(
                                        insertion: .opacity.combined(
                                            with: .move(edge: .trailing)
                                        ),
                                        removal: .opacity.combined(
                                            with: .move(edge: .leading)
                                        )
                                    )
                                )
                        } else {
                            readyContent
                                .transition(.opacity)
                        }
                    }
                    .animation(.easeInOut(duration: 0.38), value: phase)
                    .animation(.easeInOut(duration: 0.25), value: isPlaying)
                }
            },
            actionBar: { actionBar }
        )
        .preferredColorScheme(.light)
        .task(id: playbackID) {
            guard isPlaying else { return }
            await runPlayback()
        }
    }

    private var currentStage: ScanStage {
        guard isPlaying else { return .input }
        switch phase {
        case 0: return .input
        case 1: return .deidentify
        case 2: return .clinical
        default: return .insights
        }
    }

    private var demoDisclosure: some View {
        HStack(spacing: OM.Space.s2) {
            Image(systemName: "iphone.gen3")
                .foregroundStyle(Color.omSignal)
            Text("SIMULATOR PREVIEW")
                .font(.om.mono(9, weight: .semibold))
                .kerning(0.9)
                .foregroundStyle(Color.omInk)
            Spacer()
            OMBadge("Synthetic results", tone: .signal)
        }
        .padding(.horizontal, OM.Space.s3)
        .padding(.vertical, 9)
        .background(
            Color.omSignalSoft,
            in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
        )
        .overlay {
            RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                .strokeBorder(Color.omSignal.opacity(0.22), lineWidth: OM.Stroke.hairline)
        }
    }

    private var readyContent: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: "15-SECOND IPHONE TOUR",
                spans: [.plain("Scan privately with "), .accent("Maple"), .plain(".")],
                subhead: "A realistic pass through download, PII redaction, prompted extraction, relations, and streaming chat.",
                scale: .lg
            )

            OMCard(elevation: .raised) {
                VStack(alignment: .leading, spacing: OM.Space.s4) {
                    HStack(spacing: OM.Space.s3) {
                        ZStack {
                            Circle()
                                .fill(Color.omTealAccent)
                                .frame(width: 44, height: 44)
                            Image(systemName: "leaf.fill")
                                .foregroundStyle(Color.omPaper)
                        }
                        VStack(alignment: .leading, spacing: 3) {
                            Text("MAPLE PREVIEW · 2-BIT").omEyebrow()
                            Text("One local clinical workspace")
                                .font(.om.heading(18, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                        }
                    }

                    OMRule()

                    HStack(spacing: 0) {
                        tourStep("arrow.down", "Download")
                        tourArrow
                        tourStep("eye.slash", "De-ID")
                        tourArrow
                        tourStep("text.quote", "Extract")
                        tourArrow
                        tourStep("text.bubble", "Chat")
                    }

                    Label(
                        "Real Maple inference requires a recent physical iPhone.",
                        systemImage: "checkmark.shield.fill"
                    )
                    .font(.om.body(12))
                    .foregroundStyle(Color.omFgMuted)
                }
            }
        }
    }

    private func tourStep(_ icon: String, _ title: String) -> some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(Color.omTealAccent)
                .frame(width: 32, height: 32)
                .background(Color.omTealSoft, in: Circle())
            Text(title)
                .font(.om.mono(8, weight: .semibold))
                .foregroundStyle(Color.omFgMuted)
        }
        .frame(maxWidth: .infinity)
    }

    private var tourArrow: some View {
        Image(systemName: "chevron.right")
            .font(.system(size: 8, weight: .bold))
            .foregroundStyle(Color.omBorderStrong)
    }

    @ViewBuilder
    private var phaseContent: some View {
        switch phase {
        case 0: downloadPhase
        case 1: piiPhase
        case 2: extractionPhase
        default: chatPhase
        }
    }

    private var downloadPhase: some View {
        VStack(alignment: .leading, spacing: OM.Space.s4) {
            ScanStageHeader(
                eyebrow: ScanStage.input.eyebrow,
                spans: [.plain("Bring your clinical "), .accent("text"), .plain(".")],
                subhead: "Cache Maple once for private, offline inference.",
                scale: .md
            )

            OMDownloadRow(
                modelID: .maplePreview,
                entry: demoDownloadEntry,
                onStart: {},
                onCancel: {}
            )

            OMCard(padding: OM.Space.s4) {
                HStack(spacing: OM.Space.s3) {
                    Image(systemName: "doc.viewfinder.fill")
                        .font(.system(size: 18))
                        .foregroundStyle(Color.omTealAccent)
                    VStack(alignment: .leading, spacing: 2) {
                        Text("SYNTHETIC DISCHARGE NOTE").omEyebrow()
                        Text("1 page · ready for on-device OCR")
                            .font(.om.body(13, weight: .medium))
                            .foregroundStyle(Color.omInk)
                    }
                    Spacer()
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(Color.omTealAccent)
                }
            }
        }
    }

    private var demoDownloadEntry: ModelDownloadManager.Entry {
        let total = ScanModelID.maplePreview.conservativeTotalBytes
        let bytes = Int64(Double(total) * downloadProgress)
        return ModelDownloadManager.Entry(
            id: .maplePreview,
            state: .downloading(
                bytesDownloaded: bytes,
                bytesExpected: total,
                bytesPerSecond: 48 * 1_024 * 1_024
            ),
            bytesOnDisk: bytes,
            bytesEstimatedTotal: total
        )
    }

    private var piiPhase: some View {
        VStack(alignment: .leading, spacing: OM.Space.s4) {
            ScanStageHeader(
                eyebrow: ScanStage.deidentify.eyebrow,
                spans: [.plain("Redact, then "), .accent("verify"), .plain(".")],
                subhead: "Maple proposes exact spans; OpenMedKit applies the mask.",
                scale: .md
            )

            OMCard(elevation: .raised) {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        Text("DEEPGROVE MAPLE · 2-BIT").omEyebrow()
                        Spacer()
                        OMBadge("3 PII spans", tone: .accent)
                    }

                    OMEntityHighlight(
                        text: Self.sourceNote,
                        entities: piiEntities,
                        bodyFont: .om.body(15),
                        showsLabels: true
                    )

                    OMRule()

                    VStack(alignment: .leading, spacing: 6) {
                        Text("MASKED NOTE").omEyebrow()
                        Text(Self.maskedNote)
                            .font(.om.body(14, weight: .medium))
                            .foregroundStyle(Color.omInk)
                            .lineSpacing(3)
                    }

                    Label(
                        "Raw identifiers are removed before extraction or chat.",
                        systemImage: "lock.fill"
                    )
                    .font(.om.body(11))
                    .foregroundStyle(Color.omFgMuted)
                }
            }
        }
    }

    private var extractionPhase: some View {
        VStack(alignment: .leading, spacing: OM.Space.s4) {
            ScanStageHeader(
                eyebrow: ScanStage.clinical.eyebrow,
                spans: [.plain("Prompt what to "), .accent("extract"), .plain(".")],
                subhead: "Allowed labels go into Maple's local prompt; output must match the source.",
                scale: .md
            )

            OMCard(elevation: .raised, padding: OM.Space.s4) {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        HStack(spacing: 6) {
                            Image(systemName: "text.quote")
                            Text("PROMPTED EXTRACTION")
                        }
                        .font(.om.eyebrow())
                        .textCase(.uppercase)
                        .kerning(1.4)
                        .foregroundStyle(Color.omTealAccent)
                        Spacer()
                        OMBadge("Validated JSON", tone: .positive)
                    }

                    Text("“Extract only condition, medication, dosage, and follow-up. Preserve exact source spans and return supported relations.”")
                        .font(.om.mono(11))
                        .foregroundStyle(Color.omInk)
                        .lineSpacing(4)
                        .padding(OM.Space.s3)
                        .background(
                            Color.omPaper2,
                            in: RoundedRectangle(
                                cornerRadius: OM.Radius.md,
                                style: .continuous
                            )
                        )

                    HStack(spacing: 6) {
                        OMChip("condition", tone: .signal)
                        OMChip("medication", tone: .accent)
                        OMChip("dosage", tone: .ink)
                    }

                    OMRule()
                    extractionRow("CONDITION", "chronic migraine", .omSignal)
                    extractionRow("MEDICATION", "sumatriptan", .omTealAccent)
                    extractionRow("FOLLOW-UP", "within 48 hours", .omInk2)

                    HStack(spacing: 6) {
                        relationPill("chronic migraine")
                        Image(systemName: "arrow.right")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(Color.omSignal)
                        Text("TREATED WITH")
                            .font(.om.mono(8, weight: .semibold))
                            .foregroundStyle(Color.omSignal)
                        Image(systemName: "arrow.right")
                            .font(.system(size: 9, weight: .bold))
                            .foregroundStyle(Color.omSignal)
                        relationPill("sumatriptan")
                    }
                }
            }
        }
    }

    private func extractionRow(_ label: String, _ value: String, _ tone: Color) -> some View {
        HStack(spacing: OM.Space.s3) {
            Circle()
                .fill(tone)
                .frame(width: 7, height: 7)
            Text(label)
                .font(.om.mono(8, weight: .semibold))
                .kerning(0.7)
                .foregroundStyle(Color.omFgSubtle)
                .frame(width: 78, alignment: .leading)
            Text(value)
                .font(.om.body(13, weight: .semibold))
                .foregroundStyle(Color.omInk)
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 11))
                .foregroundStyle(Color.omTealAccent)
        }
    }

    private func relationPill(_ text: String) -> some View {
        Text(text)
            .font(.om.body(9, weight: .semibold))
            .foregroundStyle(Color.omInk)
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .padding(.horizontal, 7)
            .padding(.vertical, 6)
            .background(Color.omTealSoft, in: Capsule())
    }

    private var chatPhase: some View {
        VStack(alignment: .leading, spacing: OM.Space.s4) {
            ScanStageHeader(
                eyebrow: ScanStage.insights.eyebrow,
                spans: [.plain("Ask the "), .accent("masked note"), .plain(".")],
                subhead: "Private reasoning stays hidden; the final answer streams as it is generated.",
                scale: .md
            )

            OMCard(elevation: .raised) {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("ASK MAPLE").omEyebrow()
                            Text("Document-grounded chat")
                                .font(.om.heading(18, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                        }
                        Spacer()
                        OMBadge("On device", tone: .positive, systemImage: "lock.fill")
                    }

                    MapleChatBubble(
                        turn: ScanFlowViewModel.MapleChatTurn(
                            role: .user,
                            content: "What follow-up is documented?"
                        )
                    )
                    MapleChatBubble(
                        turn: ScanFlowViewModel.MapleChatTurn(
                            role: .assistant,
                            content: chatResponse,
                            isStreaming: isChatStreaming
                        )
                    )

                    HStack(spacing: OM.Space.s2) {
                        Image(systemName: "eye.slash.fill")
                            .foregroundStyle(Color.omTealAccent)
                        Text("Hidden reasoning is never rendered")
                            .font(.om.body(11, weight: .medium))
                            .foregroundStyle(Color.omFgMuted)
                        Spacer()
                        Text("MLX STREAM")
                            .font(.om.mono(8, weight: .semibold))
                            .kerning(0.7)
                            .foregroundStyle(Color.omTealAccent)
                    }
                }
            }

            Text("Simulator text is scripted. On a physical iPhone, these same bubbles receive real Maple generation chunks.")
                .font(.om.body(11))
                .foregroundStyle(Color.omFgSubtle)
                .padding(.horizontal, OM.Space.s2)
        }
    }

    @ViewBuilder
    private var actionBar: some View {
        if isPlaying {
            VStack(spacing: 8) {
                HStack {
                    Text(actionTitle)
                        .font(.om.body(13, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                    Spacer()
                    Text("\(phase + 1) / 4")
                        .font(.om.mono(10, weight: .semibold))
                        .foregroundStyle(Color.omFgMuted)
                }
                HStack(spacing: 5) {
                    ForEach(0..<4, id: \.self) { index in
                        Capsule()
                            .fill(index <= phase ? Color.omTealAccent : Color.omStone200)
                            .frame(height: 4)
                    }
                }
            }
        } else {
            Button(action: beginPlayback) {
                HStack {
                    Image(systemName: "play.fill")
                    Text("Play realistic demo")
                    Spacer()
                    Text("15 SEC")
                        .font(.om.mono(10, weight: .semibold))
                        .kerning(0.8)
                }
            }
            .buttonStyle(.omPrimary(.md))
            .accessibilityIdentifier("maple-demo-play")
        }
    }

    private var actionTitle: String {
        switch phase {
        case 0: return "Caching Maple locally"
        case 1: return "PII spans validated"
        case 2: return "Prompt response validated"
        default: return isChatStreaming ? "Final answer streaming" : "Grounded answer complete"
        }
    }

    private var piiEntities: [DetectedEntity] {
        [
            demoEntity(label: "patient name", text: "Jordan Whitfield"),
            demoEntity(label: "visit date", text: "June 1, 2026"),
            demoEntity(label: "phone", text: "(720) 555-0148"),
        ]
    }

    private func demoEntity(label: String, text: String) -> DetectedEntity {
        let range = Self.sourceNote.range(of: text)!
        let start = Self.sourceNote.distance(from: Self.sourceNote.startIndex, to: range.lowerBound)
        let end = Self.sourceNote.distance(from: Self.sourceNote.startIndex, to: range.upperBound)
        return DetectedEntity(
            label: label,
            text: text,
            confidence: nil,
            start: start,
            end: end
        )
    }

    private func beginPlayback() {
        phase = 0
        downloadProgress = 0
        chatResponse = ""
        isChatStreaming = false
        isPlaying = true
        playbackID += 1
    }

    @MainActor
    private func runPlayback() async {
        try? await Task.sleep(for: .milliseconds(150))
        withAnimation(.linear(duration: 2.7)) {
            downloadProgress = 1
        }

        try? await Task.sleep(for: .milliseconds(3_200))
        guard !Task.isCancelled else { return }
        withAnimation { phase = 1 }

        try? await Task.sleep(for: .milliseconds(3_150))
        guard !Task.isCancelled else { return }
        withAnimation { phase = 2 }

        try? await Task.sleep(for: .milliseconds(3_150))
        guard !Task.isCancelled else { return }
        chatResponse = ""
        isChatStreaming = true
        withAnimation { phase = 3 }

        try? await Task.sleep(for: .milliseconds(650))
        let chunks = [
            "PCP ",
            "follow-up ",
            "within ",
            "48 hours, ",
            "with ",
            "neurology ",
            "follow-up ",
            "within ",
            "2 weeks. ",
            "Verify ",
            "against the source note.",
        ]
        for chunk in chunks {
            guard !Task.isCancelled else { return }
            chatResponse.append(chunk)
            try? await Task.sleep(for: .milliseconds(220))
        }
        isChatStreaming = false

        try? await Task.sleep(for: .seconds(3))
        guard !Task.isCancelled else { return }
        withAnimation { isPlaying = false }
    }

    private static let sourceNote =
        "Jordan Whitfield was seen on June 1, 2026. Call (720) 555-0148. Chronic migraine treated with sumatriptan 50 mg; PCP follow-up within 48 hours."

    private static let maskedNote =
        "[PATIENT_NAME] was seen on [VISIT_DATE]. Call [PHONE]. Chronic migraine treated with sumatriptan 50 mg; PCP follow-up within 48 hours."
}
