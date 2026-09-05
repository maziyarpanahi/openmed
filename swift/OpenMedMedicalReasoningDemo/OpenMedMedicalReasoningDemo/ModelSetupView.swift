import OpenMedKit
import SwiftUI

struct ModelSetupView: View {
    @ObservedObject var store: MedicalReasoningStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 26) {
                MedicalReasoningBrand()

                VStack(alignment: .leading, spacing: 12) {
                    Text("MODEL SETUP")
                        .font(.system(size: 11, weight: .semibold, design: .monospaced))
                        .tracking(2.2)
                        .foregroundStyle(MedicalReasoningTheme.teal)
                    Text("Clinical evidence chat, entirely on device.")
                        .font(.system(size: 39, weight: .bold, design: .rounded))
                        .foregroundStyle(MedicalReasoningTheme.ink)
                        .fixedSize(horizontal: false, vertical: true)
                    Text("Download the pinned LFM2.5 model once, then ask multi-turn questions against a de-identified clinical context without a cloud fallback.")
                        .font(.system(size: 17, weight: .regular, design: .rounded))
                        .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                        .lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)
                }

                modelCard
                boundaryCard
            }
            .padding(.horizontal, MedicalReasoningTheme.pagePadding)
            .padding(.top, 18)
            .padding(.bottom, 130)
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            primaryAction
        }
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top, spacing: 14) {
                ZStack {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(MedicalReasoningTheme.teal)
                        .frame(width: 52, height: 52)
                    Image(systemName: "brain.head.profile.fill")
                        .font(.system(size: 21, weight: .semibold))
                        .foregroundStyle(.white)
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("LiquidAI LFM2.5")
                        .font(.system(size: 21, weight: .bold, design: .rounded))
                        .foregroundStyle(MedicalReasoningTheme.ink)
                    Text("2.6B · MLX · 4-bit")
                        .font(.system(size: 12, weight: .medium, design: .monospaced))
                        .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                }

                Spacer(minLength: 0)
                statePill
            }

            Divider().overlay(MedicalReasoningTheme.border)

            VStack(alignment: .leading, spacing: 10) {
                detailRow(label: "Artifact", value: OpenMedLFM.repositoryID)
                detailRow(
                    label: "Revision",
                    value: String(OpenMedLFM.pinnedRevision.prefix(12))
                )
                detailRow(label: "Download", value: "1.60 GB · seven verified files")
                detailRow(label: "Execution", value: "Local MLX · no remote fallback")
            }

            downloadProgress
        }
        .medicalCard(padding: 20)
    }

    private var statePill: some View {
        Group {
            switch store.downloadState {
            case .ready:
                MedicalStatusPill(title: "Ready", systemImage: "checkmark.circle.fill")
            case .downloading:
                MedicalStatusPill(title: "Downloading", systemImage: "arrow.down.circle")
            case .partial:
                MedicalStatusPill(title: "Resume", systemImage: "pause.circle")
            case .failed:
                MedicalStatusPill(
                    title: "Retry",
                    systemImage: "exclamationmark.circle",
                    accent: MedicalReasoningTheme.coral
                )
            case .cancelled:
                MedicalStatusPill(title: "Paused", systemImage: "pause.circle")
            case .missing:
                MedicalStatusPill(title: "Required", systemImage: "arrow.down.circle")
            }
        }
    }

    @ViewBuilder
    private var downloadProgress: some View {
        switch store.downloadState {
        case .downloading(let downloaded, let expected):
            VStack(alignment: .leading, spacing: 9) {
                ProgressView(value: Double(downloaded), total: Double(expected))
                    .tint(MedicalReasoningTheme.teal)
                HStack {
                    Text(store.activeDownloadFile)
                        .lineLimit(1)
                    Spacer()
                    Text("\(bytes(downloaded)) / \(bytes(expected))")
                }
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(MedicalReasoningTheme.secondaryInk)
            }
        case .partial(let bytesOnDisk):
            Label(
                "\(bytes(bytesOnDisk)) is cached. Resume to verify the complete pinned artifact.",
                systemImage: "internaldrive"
            )
            .font(.system(size: 12, design: .rounded))
            .foregroundStyle(MedicalReasoningTheme.secondaryInk)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .font(.system(size: 12, design: .rounded))
                .foregroundStyle(MedicalReasoningTheme.coral)
        case .ready:
            Label(
                "Architecture, required files, and exact 4-bit weights verified.",
                systemImage: "checkmark.shield.fill"
            )
            .font(.system(size: 12, weight: .medium, design: .rounded))
            .foregroundStyle(MedicalReasoningTheme.teal)
        case .missing, .cancelled:
            EmptyView()
        }
    }

    private var boundaryCard: some View {
        HStack(alignment: .top, spacing: 13) {
            Image(systemName: "lock.shield.fill")
                .font(.system(size: 18))
                .foregroundStyle(MedicalReasoningTheme.teal)
            VStack(alignment: .leading, spacing: 5) {
                Text("Local by construction")
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.ink)
                Text("Prompts and clinical context are not uploaded. Downloading model files is the only network step.")
                    .font(.system(size: 13, design: .rounded))
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .medicalCard()
    }

    private var primaryAction: some View {
        VStack(spacing: 10) {
            if store.downloadState.isDownloading {
                Button("Pause download", action: store.cancelDownload)
                    .buttonStyle(.bordered)
                    .tint(MedicalReasoningTheme.ink)
            } else {
                Button(action: primaryTapped) {
                    HStack {
                        Text(store.modelIsReady ? "Continue to clinical context" : "Download LFM2.5")
                        Spacer()
                        Image(systemName: store.modelIsReady ? "arrow.right" : "arrow.down")
                    }
                    .font(.system(size: 16, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 20)
                    .frame(height: 54)
                    .background(MedicalReasoningTheme.ink, in: Capsule())
                }
                .buttonStyle(.plain)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, MedicalReasoningTheme.pagePadding)
        .padding(.top, 14)
        .padding(.bottom, 10)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) { Divider() }
    }

    private func primaryTapped() {
        if store.modelIsReady {
            store.showClinicalContext()
        } else {
            store.startDownload()
        }
    }

    private func detailRow(label: String, value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 14) {
            Text(label.uppercased())
                .font(.system(size: 9, weight: .semibold, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(MedicalReasoningTheme.secondaryInk)
                .frame(width: 70, alignment: .leading)
            Text(value)
                .font(.system(size: 12, weight: .medium, design: .monospaced))
                .foregroundStyle(MedicalReasoningTheme.ink)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func bytes(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }
}

#Preview("Missing model") {
    ModelSetupView(store: .preview(route: .modelSetup, downloadState: .missing))
        .background(MedicalReasoningTheme.canvas)
}

#Preview("Model ready") {
    ModelSetupView(store: .preview(route: .modelSetup, downloadState: .ready))
        .background(MedicalReasoningTheme.canvas)
}
