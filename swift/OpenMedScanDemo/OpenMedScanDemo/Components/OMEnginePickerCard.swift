import SwiftUI

/// Two-column picker: each column represents a PII engine (OpenMed or OpenAI Nemotron)
/// and shows its download state. Tapping an engine selects it; tapping its
/// download button starts the download. Exists to make the choice explicit
/// up-front instead of burying it on the De-identification screen.
public struct OMEnginePickerCard: View {
    public let selectedEngine: ScanFlowViewModel.PIIEngine
    public let entries: [ScanModelID: ModelDownloadManager.Entry]
    public let onSelect: (ScanFlowViewModel.PIIEngine) -> Void
    public let onDownload: (ScanFlowViewModel.PIIEngine) -> Void
    public let onCancel: (ScanFlowViewModel.PIIEngine) -> Void

    public init(
        selectedEngine: ScanFlowViewModel.PIIEngine,
        entries: [ScanModelID: ModelDownloadManager.Entry],
        onSelect: @escaping (ScanFlowViewModel.PIIEngine) -> Void,
        onDownload: @escaping (ScanFlowViewModel.PIIEngine) -> Void,
        onCancel: @escaping (ScanFlowViewModel.PIIEngine) -> Void
    ) {
        self.selectedEngine = selectedEngine
        self.entries = entries
        self.onSelect = onSelect
        self.onDownload = onDownload
        self.onCancel = onCancel
    }

    public var body: some View {
        OMCard(padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                header
                HStack(alignment: .top, spacing: OM.Space.s3) {
                    column(for: .openMed)
                    column(for: .privacyFilter)
                }
                footnote
            }
        }
    }

    // MARK: Subviews

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("PII ENGINE").omEyebrow()
            Text("Choose one to download first")
                .font(.om.heading(19, weight: .semibold))
                .foregroundStyle(Color.omInk)
            Text("Two independent redactors. You only need one — download the other later to compare.")
                .font(.om.body(13))
                .foregroundStyle(Color.omFgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func column(for engine: ScanFlowViewModel.PIIEngine) -> some View {
        let modelID = engine.modelID
        let entry = entries[modelID]
        let isSelected = engine == selectedEngine
        let state = entry?.state ?? .missing

        return VStack(alignment: .leading, spacing: OM.Space.s3) {
            Button {
                onSelect(engine)
            } label: {
                VStack(alignment: .leading, spacing: OM.Space.s2) {
                    HStack(spacing: 4) {
                        if isSelected {
                            Image(systemName: "checkmark.circle.fill")
                                .foregroundStyle(Color.omTealAccent)
                                .font(.system(size: 12))
                        } else {
                            Circle()
                                .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                                .frame(width: 12, height: 12)
                        }
                        Text(engine == .openMed ? "OPENMED" : "OPENAI NEMOTRON")
                            .omMonoTag(size: 10)
                            .foregroundStyle(isSelected ? Color.omTealAccent : Color.omFgMuted)
                    }
                    Text(engine.shortTitle)
                        .font(.om.display(19, weight: .medium))
                        .foregroundStyle(Color.omInk)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                    Text(engine.blurb)
                        .font(.om.body(12))
                        .foregroundStyle(Color.omFgMuted)
                        .fixedSize(horizontal: false, vertical: true)
                        .multilineTextAlignment(.leading)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .buttonStyle(.plain)

            stateAndAction(engine: engine, state: state, entry: entry)
        }
        .padding(OM.Space.s3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            isSelected ? Color.omTealSoft : Color.omPaper,
            in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                .strokeBorder(isSelected ? Color.omTealAccent : Color.omBorderStrong,
                              lineWidth: isSelected ? 1.5 : OM.Stroke.hairline)
        )
    }

    @ViewBuilder
    private func stateAndAction(engine: ScanFlowViewModel.PIIEngine, state: ModelDownloadState, entry: ModelDownloadManager.Entry?) -> some View {
        switch state {
        case .ready:
            HStack(spacing: 4) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.omTealAccent)
                Text("Cached")
                    .font(.om.mono(11, weight: .medium))
                    .foregroundStyle(Color.omTealAccent)
            }
        case .downloading(let bytes, let total, _):
            VStack(alignment: .leading, spacing: 4) {
                OMProgressBar(
                    mode: (entry?.fraction ?? nil).map { .determinate(progress: $0) } ?? .indeterminate,
                    height: 4
                )
                HStack {
                    Text(ByteFormatter.percent(entry?.fraction))
                        .font(.om.mono(10, weight: .semibold))
                    Spacer()
                    Text(ByteFormatter.progressString(bytes: bytes, total: total ?? entry?.bytesEstimatedTotal))
                        .font(.om.mono(9))
                        .foregroundStyle(Color.omFgSubtle)
                }
                Button("Cancel") { onCancel(engine) }
                    .buttonStyle(.omGhost(.ink))
            }
        case .queued:
            Text("QUEUED")
                .font(.om.mono(10, weight: .medium))
                .foregroundStyle(Color.omFgMuted)
        case .installing:
            Text("INSTALLING")
                .font(.om.mono(10, weight: .medium))
                .foregroundStyle(Color.omTealAccent)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 4) {
                Text(message)
                    .font(.om.body(11))
                    .foregroundStyle(Color.omSignal)
                    .lineLimit(3)
                Button("Retry") { onDownload(engine) }
                    .buttonStyle(.omSecondary(.sm))
            }
        case .partial, .cancelled, .missing:
            Button {
                onDownload(engine)
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 11))
                    Text("Download · \(engine.sizeLabel)")
                }
            }
            .buttonStyle(.omSecondary(.sm))
        }
    }

    private var footnote: some View {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
            Text("You can switch engines or cache both at any time.")
        }
        .foregroundStyle(Color.omFgSubtle)
        .font(.om.body(11))
    }
}

// MARK: - Engine copy helpers

extension ScanFlowViewModel.PIIEngine {
    var shortTitle: String {
        switch self {
        case .openMed:       return "OpenMed PII LiteClinical"
        case .privacyFilter: return "OpenAI Nemotron Privacy Filter"
        }
    }

    var blurb: String {
        switch self {
        case .openMed:
            return "66M-param DistilBERT trained for clinical PII. Fast, balanced precision/recall."
        case .privacyFilter:
            return "OpenAI Nemotron 8-bit Privacy Filter with BIOES decoding. Broad coverage, slightly heavier."
        }
    }

    var sizeLabel: String {
        modelID.estimatedSizeLabel.replacingOccurrences(of: "~", with: "")
    }
}
