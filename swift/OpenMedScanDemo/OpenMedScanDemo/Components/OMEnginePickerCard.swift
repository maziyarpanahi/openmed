import SwiftUI

/// Compact PII model picker. Each row is an on-device redactor and includes
/// its cache/download state so three choices stay readable on iPhone.
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
                engineList
                footnote
            }
        }
    }

    // MARK: Subviews

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("PII ENGINE").omEyebrow()
            Text("Choose a redaction model")
                .font(.om.heading(19, weight: .semibold))
                .foregroundStyle(Color.omInk)
            Text("Three local PII models: clinical, Nemotron, and multilingual 8-bit.")
                .font(.om.body(13))
                .foregroundStyle(Color.omFgMuted)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var engineList: some View {
        let engines = ScanFlowViewModel.PIIEngine.allCases
        return VStack(spacing: 0) {
            ForEach(engines.indices, id: \.self) { index in
                let engine = engines[index]
                row(for: engine)
                if index < engines.index(before: engines.endIndex) {
                    OMRule()
                        .padding(.vertical, OM.Space.s2)
                }
            }
        }
    }

    private func row(for engine: ScanFlowViewModel.PIIEngine) -> some View {
        let modelID = engine.modelID
        let entry = entries[modelID]
        let isSelected = engine == selectedEngine
        let state = entry?.state ?? .missing

        return VStack(alignment: .leading, spacing: OM.Space.s2) {
            HStack(alignment: .top, spacing: OM.Space.s2) {
                Button {
                    onSelect(engine)
                } label: {
                    HStack(alignment: .top, spacing: OM.Space.s3) {
                        selectionIcon(isSelected)
                            .padding(.top, 2)

                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 6) {
                                Text(engine.eyebrowTitle)
                                    .omMonoTag(size: 9)
                                    .foregroundStyle(isSelected ? Color.omTealAccent : Color.omFgMuted)
                                OMBadge(engine.modelBadge, tone: isSelected ? .accent : .neutral)
                                Text(engine.sizeLabel)
                                    .font(.om.mono(10, weight: .semibold))
                                    .foregroundStyle(Color.omFgSubtle)
                                    .lineLimit(1)
                            }
                            Text(engine.shortTitle)
                                .font(.om.heading(15, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                                .lineLimit(1)
                                .minimumScaleFactor(0.82)
                            Text(engine.compactBlurb)
                                .font(.om.body(11))
                                .foregroundStyle(Color.omFgMuted)
                                .lineLimit(2)
                                .fixedSize(horizontal: false, vertical: true)
                                .multilineTextAlignment(.leading)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, alignment: .leading)

                compactStateAndAction(engine: engine, state: state, entry: entry)
            }
            .padding(10)
            .background(
                isSelected ? Color.omTealSoft : Color.clear,
                in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
            )
        }
    }

    @ViewBuilder
    private func selectionIcon(_ isSelected: Bool) -> some View {
        if isSelected {
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(Color.omTealAccent)
                .font(.system(size: 16))
        } else {
            Circle()
                .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                .frame(width: 16, height: 16)
        }
    }

    @ViewBuilder
    private func compactStateAndAction(engine: ScanFlowViewModel.PIIEngine, state: ModelDownloadState, entry: ModelDownloadManager.Entry?) -> some View {
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
            .padding(.top, 4)
        case .downloading:
            VStack(alignment: .leading, spacing: 4) {
                OMProgressBar(
                    mode: (entry?.fraction ?? nil).map { .determinate(progress: $0) } ?? .indeterminate,
                    height: 4
                )
                HStack {
                    Text(ByteFormatter.percent(entry?.fraction))
                        .font(.om.mono(10, weight: .semibold))
                }
                Button("Cancel") { onCancel(engine) }
                    .buttonStyle(.omGhost(.ink))
            }
            .frame(width: 86, alignment: .leading)
        case .queued:
            Text("QUEUED")
                .font(.om.mono(10, weight: .medium))
                .foregroundStyle(Color.omFgMuted)
                .padding(.top, 4)
        case .installing:
            Text("INSTALLING")
                .font(.om.mono(10, weight: .medium))
                .foregroundStyle(Color.omTealAccent)
                .padding(.top, 4)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 4) {
                Text(message)
                    .font(.om.body(11))
                    .foregroundStyle(Color.omSignal)
                    .lineLimit(1)
                Button("Retry") { onDownload(engine) }
                    .buttonStyle(.omSecondary(.sm))
            }
            .frame(width: 86, alignment: .leading)
        case .partial, .cancelled, .missing:
            Button {
                onDownload(engine)
            } label: {
                Image(systemName: "arrow.down.circle.fill")
                    .font(.system(size: 18, weight: .semibold))
                    .foregroundStyle(Color.omInk)
                    .frame(width: 44, height: 44)
                    .background(Color.omBgElevated, in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                            .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                    )
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Download \(engine.shortTitle)")
            .frame(width: 52, alignment: .trailing)
        }
    }

    private var footnote: some View {
        HStack(spacing: 6) {
            Image(systemName: "info.circle")
                .font(.system(size: 11))
            Text("Cache any model now; the selected one runs first.")
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
        case .multilingual:  return "OpenMed Multilingual Privacy Filter"
        }
    }

    var blurb: String {
        switch self {
        case .openMed:
            return "66M-param DistilBERT trained for clinical PII. Fast, balanced precision/recall."
        case .privacyFilter:
            return "OpenAI Nemotron 8-bit Privacy Filter with BIOES decoding. Broad coverage, slightly heavier."
        case .multilingual:
            return "OpenMed 8-bit privacy filter with official support for 16 languages."
        }
    }

    var compactBlurb: String {
        switch self {
        case .openMed:
            return "Clinical PII, fastest default."
        case .privacyFilter:
            return "Nemotron BIOES redactor."
        case .multilingual:
            return "16-language 8-bit redactor."
        }
    }

    var eyebrowTitle: String {
        switch self {
        case .openMed:       return "OPENMED"
        case .privacyFilter: return "NEMOTRON"
        case .multilingual:  return "MULTI"
        }
    }

    var modelBadge: String {
        switch self {
        case .openMed:       return "Clinical"
        case .privacyFilter: return "8-bit"
        case .multilingual:  return "8-bit"
        }
    }

    var toggleTitle: String {
        switch self {
        case .openMed:       return "OpenMed"
        case .privacyFilter: return "Nemotron"
        case .multilingual:  return "Multilingual"
        }
    }

    var sizeLabel: String {
        modelID.estimatedSizeLabel.replacingOccurrences(of: "~", with: "")
    }
}
