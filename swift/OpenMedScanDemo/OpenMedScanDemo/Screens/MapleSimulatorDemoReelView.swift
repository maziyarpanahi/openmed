import SwiftUI

/// A deterministic simulator-only reel for recording the native workflow.
///
/// The reel never initializes an MLX runtime and labels every generated result
/// as synthetic. Real Maple inference remains a physical-device workflow.
struct MapleSimulatorDemoReelView: View {
    @State private var phase = 0
    @State private var playbackID = 0
    @State private var isPlaying = false
    @State private var downloadProgress = 0.0

    var body: some View {
        ZStack {
            Color.omPaper.ignoresSafeArea()

            VStack(spacing: 0) {
                brandBar

                ZStack {
                    if isPlaying {
                        phaseContent
                            .id(phase)
                            .transition(
                                .asymmetric(
                                    insertion: .opacity.combined(with: .move(edge: .trailing)),
                                    removal: .opacity.combined(with: .move(edge: .leading))
                                )
                            )
                    } else {
                        readyCard
                            .transition(.opacity.combined(with: .scale(scale: 0.98)))
                    }
                }
                .animation(.easeInOut(duration: 0.42), value: phase)
                .animation(.easeInOut(duration: 0.3), value: isPlaying)
                .frame(maxWidth: .infinity, maxHeight: .infinity)

                timelineBar
            }
        }
        .preferredColorScheme(.light)
        .task(id: playbackID) {
            guard isPlaying else { return }
            await runPlayback()
        }
    }

    private var brandBar: some View {
        HStack {
            OMBrandLockup(compact: true)
            Spacer()
            VStack(alignment: .trailing, spacing: 3) {
                Text("SIMULATOR PREVIEW")
                    .font(.om.mono(9, weight: .semibold))
                    .kerning(1.1)
                    .foregroundStyle(Color.omSignal)
                Text("SYNTHETIC RESULTS")
                    .font(.om.mono(8, weight: .medium))
                    .kerning(0.8)
                    .foregroundStyle(Color.omFgSubtle)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 13)
        .background(Color.omPaper2)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Color.omBorder).frame(height: 1)
        }
    }

    private var readyCard: some View {
        VStack(alignment: .leading, spacing: 22) {
            Spacer()

            VStack(alignment: .leading, spacing: 10) {
                Text("15-SECOND PRODUCT TOUR").omEyebrow()
                Text("Meet Maple\non iPhone.")
                    .font(.om.display(45, weight: .medium))
                    .foregroundStyle(Color.omInk)
                    .lineSpacing(-5)
                Text("Download, redact PII, extract clinical structure, then chat over the masked note.")
                    .font(.om.body(16))
                    .foregroundStyle(Color.omFgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button(action: beginPlayback) {
                HStack {
                    Image(systemName: "play.fill")
                    Text("Play demo")
                    Spacer()
                    Text("15 SEC")
                        .font(.om.mono(10, weight: .semibold))
                        .kerning(1)
                }
                .font(.om.body(16, weight: .semibold))
                .foregroundStyle(Color.omPaper)
                .padding(.horizontal, 18)
                .frame(height: 54)
                .background(Color.omInk, in: RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(.plain)
            .accessibilityIdentifier("maple-demo-play")

            Label("UI preview only · real inference requires a recent physical iPhone", systemImage: "iphone.gen3")
                .font(.om.body(12))
                .foregroundStyle(Color.omFgSubtle)

            Spacer()
        }
        .padding(.horizontal, 22)
    }

    @ViewBuilder
    private var phaseContent: some View {
        switch phase {
        case 0:
            downloadPhase
        case 1:
            piiPhase
        case 2:
            entityPhase
        default:
            chatPhase
        }
    }

    private var downloadPhase: some View {
        reelPage(
            step: "01 · DOWNLOAD",
            title: "Maple, ready\nfor local work.",
            subtitle: "A sparse clinical model cached once for private, offline inference."
        ) {
            VStack(alignment: .leading, spacing: 16) {
                HStack(spacing: 14) {
                    ZStack {
                        RoundedRectangle(cornerRadius: 13)
                            .fill(Color.omTealSoft)
                        Image(systemName: "leaf.fill")
                            .font(.system(size: 25, weight: .semibold))
                            .foregroundStyle(Color.omTealAccent)
                    }
                    .frame(width: 58, height: 58)

                    VStack(alignment: .leading, spacing: 4) {
                        Text("DEEPGROVE · 2-BIT")
                            .font(.om.mono(9, weight: .semibold))
                            .kerning(1)
                            .foregroundStyle(Color.omTealAccent)
                        Text("Maple Preview")
                            .font(.om.heading(20))
                            .foregroundStyle(Color.omInk)
                        Text("20B-A1B sparse · MLX")
                            .font(.om.body(12))
                            .foregroundStyle(Color.omFgMuted)
                    }
                    Spacer()
                }

                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color.omStone200)
                        Capsule()
                            .fill(Color.omTealAccent)
                            .frame(width: geometry.size.width * downloadProgress)
                    }
                }
                .frame(height: 8)

                HStack {
                    Text("DOWNLOADING MODEL")
                        .font(.om.mono(10, weight: .semibold))
                        .kerning(0.8)
                        .foregroundStyle(Color.omFgMuted)
                    Spacer()
                    Text("≈ 5.0 GB")
                        .font(.om.mono(12, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                }
            }
            .padding(18)
            .background(Color.omPaper2, in: RoundedRectangle(cornerRadius: 16))
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.omBorder, lineWidth: 1)
            }
        }
    }

    private var piiPhase: some View {
        reelPage(
            step: "02 · PII REDACTION",
            title: "Private by\ndefault.",
            subtitle: "Maple proposes spans. OpenMedKit validates offsets and applies deterministic masking."
        ) {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    Text("MASKED NOTE").omEyebrow()
                    Spacer()
                    Label("4 spans", systemImage: "checkmark.shield.fill")
                        .font(.om.body(11, weight: .semibold))
                        .foregroundStyle(Color.omTealAccent)
                }

                Text("[PATIENT] called on [DATE]. Reach them at [PHONE]. Follow-up at [LOCATION].")
                    .font(.om.body(17, weight: .medium))
                    .foregroundStyle(Color.omInk)
                    .lineSpacing(6)

                HStack(spacing: 7) {
                    demoChip("NAME")
                    demoChip("DATE")
                    demoChip("PHONE")
                    demoChip("LOCATION")
                }

                Divider()

                Label("Raw identifiers never enter chat context", systemImage: "lock.fill")
                    .font(.om.body(12))
                    .foregroundStyle(Color.omFgMuted)
            }
            .padding(18)
            .background(Color.omHighlight.opacity(0.27), in: RoundedRectangle(cornerRadius: 16))
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.omBorder, lineWidth: 1)
            }
        }
    }

    private var entityPhase: some View {
        reelPage(
            step: "03 · ENTITIES + RELATIONS",
            title: "From note\nto structure.",
            subtitle: "Validated entities stay anchored to the de-identified source."
        ) {
            VStack(alignment: .leading, spacing: 13) {
                entityRow(icon: "lungs.fill", label: "CONDITION", value: "Asthma", tone: .omSignal)
                entityRow(icon: "pills.fill", label: "MEDICATION", value: "Albuterol", tone: .omTealAccent)
                entityRow(icon: "number", label: "DOSAGE", value: "2 puffs", tone: .omInk2)

                HStack(spacing: 8) {
                    relationNode("Albuterol")
                    Image(systemName: "arrow.right")
                        .foregroundStyle(Color.omSignal)
                    Text("TREATS")
                        .font(.om.mono(9, weight: .semibold))
                        .kerning(0.8)
                        .foregroundStyle(Color.omSignal)
                    Image(systemName: "arrow.right")
                        .foregroundStyle(Color.omSignal)
                    relationNode("Asthma")
                }
                .padding(.top, 4)
            }
            .padding(18)
            .background(Color.omPaper2, in: RoundedRectangle(cornerRadius: 16))
            .overlay {
                RoundedRectangle(cornerRadius: 16)
                    .stroke(Color.omBorder, lineWidth: 1)
            }
        }
    }

    private var chatPhase: some View {
        reelPage(
            step: "04 · GROUNDED CHAT",
            title: "Ask the\nmasked note.",
            subtitle: "Only de-identified evidence reaches Maple. Private reasoning stays hidden."
        ) {
            VStack(spacing: 12) {
                HStack {
                    Spacer(minLength: 42)
                    VStack(alignment: .leading, spacing: 5) {
                        Text("YOU")
                            .font(.om.mono(9, weight: .semibold))
                            .foregroundStyle(Color.omPaper.opacity(0.72))
                        Text("What needs follow-up?")
                            .font(.om.body(15, weight: .medium))
                            .foregroundStyle(Color.omPaper)
                    }
                    .padding(14)
                    .background(Color.omInk, in: RoundedRectangle(cornerRadius: 15))
                }

                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("MAPLE")
                                .font(.om.mono(9, weight: .semibold))
                                .foregroundStyle(Color.omTealAccent)
                            Spacer()
                            Image(systemName: "checkmark.seal.fill")
                                .foregroundStyle(Color.omTealAccent)
                        }
                        Text("Pulmonology follow-up in 2 weeks. Verify inhaler use and symptom trend against the source note.")
                            .font(.om.body(15))
                            .foregroundStyle(Color.omInk)
                            .lineSpacing(3)
                    }
                    .padding(14)
                    .background(Color.omTealSoft, in: RoundedRectangle(cornerRadius: 15))
                    Spacer(minLength: 24)
                }

                HStack(spacing: 6) {
                    Label("Masked context", systemImage: "eye.slash.fill")
                    Spacer()
                    Label("On device", systemImage: "iphone.gen3")
                }
                .font(.om.body(11, weight: .semibold))
                .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    private func reelPage<Content: View>(
        step: String,
        title: String,
        subtitle: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            VStack(alignment: .leading, spacing: 8) {
                Text(step).omEyebrow()
                Text(title)
                    .font(.om.display(39, weight: .medium))
                    .foregroundStyle(Color.omInk)
                    .lineSpacing(-5)
                Text(subtitle)
                    .font(.om.body(14))
                    .foregroundStyle(Color.omFgMuted)
                    .fixedSize(horizontal: false, vertical: true)
            }
            content()
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 22)
        .padding(.top, 20)
        .padding(.bottom, 10)
    }

    private var timelineBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 5) {
                ForEach(0..<4, id: \.self) { index in
                    Capsule()
                        .fill(index <= phase && isPlaying ? Color.omTealAccent : Color.omStone200)
                        .frame(height: 4)
                }
            }
            HStack {
                Text(isPlaying ? "MAPLE · OPENMEDKIT" : "PHYSICAL DEVICE FOR REAL MLX")
                Spacer()
                Text(isPlaying ? "\(phase + 1) / 4" : "UI PREVIEW")
            }
            .font(.om.mono(9, weight: .semibold))
            .kerning(0.8)
            .foregroundStyle(Color.omFgSubtle)
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 12)
        .background(Color.omPaper2)
        .overlay(alignment: .top) {
            Rectangle().fill(Color.omBorder).frame(height: 1)
        }
    }

    private func demoChip(_ text: String) -> some View {
        Text(text)
            .font(.om.mono(9, weight: .semibold))
            .kerning(0.7)
            .foregroundStyle(Color.omInk2)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color.omPaper2, in: Capsule())
            .overlay { Capsule().stroke(Color.omBorder, lineWidth: 1) }
    }

    private func entityRow(icon: String, label: String, value: String, tone: Color) -> some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(tone)
                .frame(width: 30, height: 30)
                .background(tone.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.om.mono(8, weight: .semibold))
                    .kerning(0.8)
                    .foregroundStyle(Color.omFgSubtle)
                Text(value)
                    .font(.om.body(15, weight: .semibold))
                    .foregroundStyle(Color.omInk)
            }
            Spacer()
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Color.omTealAccent)
        }
    }

    private func relationNode(_ text: String) -> some View {
        Text(text)
            .font(.om.body(11, weight: .semibold))
            .foregroundStyle(Color.omInk)
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color.omStone100, in: Capsule())
    }

    private func beginPlayback() {
        phase = 0
        downloadProgress = 0
        isPlaying = true
        playbackID += 1
    }

    @MainActor
    private func runPlayback() async {
        try? await Task.sleep(nanoseconds: 160_000_000)
        withAnimation(.linear(duration: 2.8)) {
            downloadProgress = 1
        }

        try? await Task.sleep(nanoseconds: 3_300_000_000)
        guard !Task.isCancelled else { return }
        withAnimation { phase = 1 }

        try? await Task.sleep(nanoseconds: 3_500_000_000)
        guard !Task.isCancelled else { return }
        withAnimation { phase = 2 }

        try? await Task.sleep(nanoseconds: 3_500_000_000)
        guard !Task.isCancelled else { return }
        withAnimation { phase = 3 }

        try? await Task.sleep(nanoseconds: 4_700_000_000)
        guard !Task.isCancelled else { return }
        withAnimation { isPlaying = false }
    }
}
