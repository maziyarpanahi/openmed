import SwiftUI

/// Inline download card: model name, size badge, progress bar, primary CTA.
/// Appears above the primary button when a stage's required model isn't ready.
public struct OMDownloadRow: View {
    public let modelID: ScanModelID
    public let entry: ModelDownloadManager.Entry
    public let onStart: () -> Void
    public let onCancel: () -> Void

    public init(
        modelID: ScanModelID,
        entry: ModelDownloadManager.Entry,
        onStart: @escaping () -> Void,
        onCancel: @escaping () -> Void
    ) {
        self.modelID = modelID
        self.entry = entry
        self.onStart = onStart
        self.onCancel = onCancel
    }

    public var body: some View {
        OMCard(padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                header
                progressSection
                actionRow
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: OM.Space.s3) {
            Image(systemName: icon)
                .font(.system(size: 17, weight: .regular))
                .foregroundStyle(tone)
                .frame(width: 28, height: 28, alignment: .center)
                .background(Color.omTealSoft, in: RoundedRectangle(cornerRadius: OM.Radius.sm, style: .continuous))
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(modelID.displayName)
                        .font(.om.heading(15, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                    OMBadge(stateBadgeText, tone: stateBadgeTone)
                }
                Text(modelID.artifactRepoID)
                    .font(.om.mono(11))
                    .foregroundStyle(Color.omFgSubtle)
                    .lineLimit(1)
            }
            Spacer(minLength: 0)
        }
    }

    @ViewBuilder
    private var progressSection: some View {
        switch entry.state {
        case .missing, .cancelled, .failed:
            HStack(spacing: OM.Space.s2) {
                Text("SIZE")
                    .font(.om.mono(10, weight: .medium))
                    .foregroundStyle(Color.omFgSubtle)
                    .kerning(1.1)
                Text(modelID.estimatedSizeLabel)
                    .font(.om.body(14, weight: .semibold))
                    .foregroundStyle(Color.omInk)
            }
        case .partial(let bytesOnDisk, let bytesExpected):
            progressStack(
                bytes: bytesOnDisk,
                total: bytesExpected ?? entry.bytesEstimatedTotal,
                fraction: entry.fraction,
                rate: nil,
                caption: "PAUSED"
            )
        case .queued:
            OMPhaseIndicator(phase: "QUEUED", detail: "Waiting for a slot")
        case .downloading(let bytesDone, let total, let rate):
            progressStack(
                bytes: bytesDone,
                total: total ?? entry.bytesEstimatedTotal,
                fraction: entry.fraction,
                rate: rate,
                caption: "DOWNLOADING"
            )
        case .installing:
            OMPhaseIndicator(phase: "INSTALLING", detail: "Unpacking artifact")
        case .ready:
            HStack(spacing: OM.Space.s2) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 13, weight: .regular))
                    .foregroundStyle(Color.omTealAccent)
                Text("Cached — ready for offline runs")
                    .font(.om.body(13, weight: .medium))
                    .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    private func progressStack(bytes: Int64, total: Int64?, fraction: Double?, rate: Double?, caption: String) -> some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            HStack {
                Text(caption)
                    .font(.om.mono(10, weight: .medium))
                    .kerning(1.1)
                    .foregroundStyle(Color.omTealAccent)
                Spacer()
                Text(ByteFormatter.percent(fraction))
                    .font(.om.mono(12, weight: .semibold))
                    .foregroundStyle(Color.omInk)
            }
            OMProgressBar(mode: fraction == nil ? .indeterminate : .determinate(progress: fraction ?? 0))
            HStack {
                Text(ByteFormatter.progressString(bytes: bytes, total: total))
                    .font(.om.mono(11))
                    .foregroundStyle(Color.omFgMuted)
                Spacer()
                if let rateStr = ByteFormatter.rate(rate) {
                    Text(rateStr)
                        .font(.om.mono(11))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
        }
    }

    private var actionRow: some View {
        HStack(spacing: OM.Space.s2) {
            switch entry.state {
            case .missing, .cancelled, .partial:
                Button("Download", action: onStart)
                    .buttonStyle(.omPrimary(.sm))
            case .queued, .downloading, .installing:
                Button("Cancel", action: onCancel)
                    .buttonStyle(.omSecondary(.sm))
            case .failed:
                Button("Retry", action: onStart)
                    .buttonStyle(.omPrimary(.sm))
                Button("Dismiss", action: onCancel)
                    .buttonStyle(.omGhost)
            case .ready:
                EmptyView()
            }
        }
    }

    // MARK: Style helpers

    private var icon: String {
        switch modelID {
        case .piiLiteClinical:     return "shield.lefthalf.filled"
        case .openaiPrivacyFilter: return "lock.shield.fill"
        case .glinerRelex:         return "sparkles"
        }
    }

    private var tone: Color {
        switch entry.state {
        case .failed: return .omSignal
        default:      return .omTealAccent
        }
    }

    private var stateBadgeText: String {
        switch entry.state {
        case .missing:     return "NEEDED"
        case .partial:     return "PAUSED"
        case .queued:      return "QUEUED"
        case .downloading: return "LIVE"
        case .installing:  return "INSTALLING"
        case .ready:       return "READY"
        case .failed:      return "FAILED"
        case .cancelled:   return "CANCELLED"
        }
    }

    private var stateBadgeTone: OMBadge.Tone {
        switch entry.state {
        case .ready:       return .positive
        case .failed:      return .signal
        case .downloading, .queued, .installing: return .accent
        default:           return .neutral
        }
    }
}
