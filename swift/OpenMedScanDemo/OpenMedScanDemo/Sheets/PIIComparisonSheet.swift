import SwiftUI

public struct PIIComparisonSheet: View {
    public let sourceText: String
    public let leftOutput: PIIOutput?
    public let rightOutput: PIIOutput?
    public let leftEngine: ScanFlowViewModel.PIIEngine
    public let rightEngine: ScanFlowViewModel.PIIEngine

    @Environment(\.dismiss) private var dismiss

    public init(
        sourceText: String,
        leftOutput: PIIOutput?,
        rightOutput: PIIOutput?,
        leftEngine: ScanFlowViewModel.PIIEngine = .openMed,
        rightEngine: ScanFlowViewModel.PIIEngine = .privacyFilter
    ) {
        self.sourceText = sourceText
        self.leftOutput = leftOutput
        self.rightOutput = rightOutput
        self.leftEngine = leftEngine
        self.rightEngine = rightEngine
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: OM.Space.s4) {
                    header
                    statsStrip
                    diffList
                }
                .padding(OM.Space.s5)
            }
            .background(Color.omPaper)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Close") { dismiss() }
                        .buttonStyle(.omGhost(.ink))
                }
            }
            .toolbarBackground(Color.omPaper, for: .navigationBar)
            .navigationTitle("Engine comparison")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            Text("SIDE-BY-SIDE").omEyebrow()
            OMDisplayHeadline([
                .plain("Where do they "),
                .accent("disagree"),
                .plain("?"),
            ], scale: .sm)
        }
    }

    private var diff: (summary: EngineDiff.Summary, rows: [EngineDiff.Row]) {
        EngineDiff.build(
            left: leftOutput?.entities ?? [],
            right: rightOutput?.entities ?? [],
            leftLabel: leftEngine.displayName,
            rightLabel: rightEngine.displayName
        )
    }

    private var statsStrip: some View {
        OMCard(padding: OM.Space.s4) {
            HStack {
                statCell("AGREE", value: "\(diff.summary.agreements)", tone: .omTealAccent)
                Divider().frame(height: 32).background(Color.omBorder)
                statCell("ONLY OPENMED", value: "\(diff.summary.onlyLeft)", tone: .omInk)
                Divider().frame(height: 32).background(Color.omBorder)
                statCell("ONLY OPENAI", value: "\(diff.summary.onlyRight)", tone: .omInk)
            }
        }
    }

    private func statCell(_ label: String, value: String, tone: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).omEyebrow()
                .foregroundStyle(Color.omFgSubtle)
            Text(value)
                .font(.om.display(28, weight: .medium))
                .foregroundStyle(tone)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var diffList: some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            Text("SPAN DIFF").omEyebrow()
            ForEach(diff.rows) { row in
                diffRow(row)
            }
        }
    }

    private func diffRow(_ row: EngineDiff.Row) -> some View {
        OMCard(padding: OM.Space.s3) {
            HStack(alignment: .top, spacing: OM.Space.s3) {
                Text("\(row.start)–\(row.end)")
                    .font(.om.mono(10))
                    .foregroundStyle(Color.omFgSubtle)
                    .frame(width: 52, alignment: .leading)

                VStack(alignment: .leading, spacing: 4) {
                    Text(row.text)
                        .font(.om.body(15, weight: .medium))
                        .foregroundStyle(Color.omInk)
                    HStack(spacing: 6) {
                        OMBadge(originBadge(for: row.origin), tone: originTone(row.origin))
                        if let l = row.leftLabel { Text("OM: \(l)").font(.om.mono(10)).foregroundStyle(Color.omFgMuted) }
                        if let r = row.rightLabel { Text("AI: \(r)").font(.om.mono(10)).foregroundStyle(Color.omFgMuted) }
                    }
                }

                Spacer()
            }
        }
    }

    private func originBadge(for origin: EngineDiff.Row.Origin) -> String {
        switch origin {
        case .both:  return "BOTH"
        case .left:  return "OM ONLY"
        case .right: return "AI ONLY"
        }
    }

    private func originTone(_ origin: EngineDiff.Row.Origin) -> OMBadge.Tone {
        switch origin {
        case .both:  return .positive
        case .left:  return .accent
        case .right: return .highlight
        }
    }
}
