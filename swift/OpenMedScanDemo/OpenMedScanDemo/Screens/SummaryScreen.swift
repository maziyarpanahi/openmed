import CryptoKit
import SwiftUI

public struct SummaryScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    public let onShowComparison: () -> Void
    public let onStartOver: () -> Void

    public init(
        flow: ScanFlowViewModel,
        onShowComparison: @escaping () -> Void,
        onStartOver: @escaping () -> Void
    ) {
        self.flow = flow
        self.onShowComparison = onShowComparison
        self.onStartOver = onStartOver
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.summary.eyebrow,
                spans: headlineSpans,
                subhead: "Review detected entities, export the JSON, or compare engines side by side.",
                scale: .lg
            )

            filterBar
            entitySections
            relationSection
            mapleBriefSection

            secondaryActions
        }
    }

    private var allEntities: [DetectedEntity] {
        (flow.currentPIIOutput?.entities ?? []) + (flow.clinicalOutput?.entities ?? [])
    }

    private var headlineSpans: [OMDisplayHeadline.Span] {
        let count = allEntities.count
        let cats = Set(allEntities.map(\.category)).count
        if count == 0 {
            return [.plain("No entities "), .accent("yet"), .plain(".")]
        }
        return [
            .plain("Found "),
            .accent("\(count) entities"),
            .plain(" across \(cats) categor\(cats == 1 ? "y" : "ies")."),
        ]
    }

    private var filterBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: OM.Space.s2) {
                OMChip(
                    "ALL · \(allEntities.count)",
                    tone: flow.summaryCategoryFilter == nil ? .ink : .neutral,
                    action: {
                        flow.summaryCategoryFilter = nil
                        HapticsCenter.selection()
                    }
                )
                ForEach(EntityGrouping.categoryCounts(allEntities), id: \.0) { cat, n in
                    OMChip(
                        "\(cat.displayName.uppercased()) · \(n)",
                        tone: flow.summaryCategoryFilter == cat ? .ink : .neutral,
                        action: {
                            flow.summaryCategoryFilter = cat
                            HapticsCenter.selection()
                        }
                    )
                }
            }
            .padding(.vertical, 2)
        }
    }

    @ViewBuilder
    private var entitySections: some View {
        let sections = EntityGrouping.group(allEntities, filter: flow.summaryCategoryFilter)
        if sections.isEmpty {
            OMCard {
                VStack(alignment: .leading, spacing: OM.Space.s2) {
                    Text("NO RESULTS").omEyebrow()
                    Text("Nothing in this category. Try another filter or rerun the pipeline.")
                        .font(.om.body(15))
                        .foregroundStyle(Color.omFgMuted)
                }
            }
        } else {
            ForEach(sections) { section in
                OMCard {
                    VStack(alignment: .leading, spacing: OM.Space.s3) {
                        HStack {
                            Text(section.category.displayName.uppercased()).omEyebrow()
                                .foregroundStyle(section.category.tone.accent)
                            Spacer()
                            Text("\(section.count)")
                                .font(.om.mono(12, weight: .semibold))
                                .foregroundStyle(Color.omFgSubtle)
                        }
                        ForEach(section.entities) { entity in
                            entityRow(entity)
                            if entity != section.entities.last {
                                OMRule()
                            }
                        }
                    }
                }
            }
        }
    }

    private func entityRow(_ entity: DetectedEntity) -> some View {
        HStack(alignment: .top, spacing: OM.Space.s3) {
            Text("\(entity.start)–\(entity.end)")
                .font(.om.mono(10))
                .foregroundStyle(Color.omFgSubtle)
                .frame(width: 56, alignment: .leading)
            VStack(alignment: .leading, spacing: 2) {
                Text(entity.text)
                    .font(.om.body(16, weight: .medium))
                    .foregroundStyle(Color.omInk)
                HStack(spacing: 6) {
                    Text(entity.label.uppercased())
                        .font(.om.mono(10, weight: .medium))
                        .foregroundStyle(entity.category.tone.accent)
                    Text("·")
                        .font(.om.mono(10))
                        .foregroundStyle(Color.omFgSubtle)
                    if let confidence = entity.confidence {
                        Text(String(format: "%.0f%%", confidence * 100))
                            .font(.om.mono(10))
                            .foregroundStyle(Color.omFgSubtle)
                    } else {
                        Text("GENERATIVE")
                            .font(.om.mono(10))
                            .foregroundStyle(Color.omFgSubtle)
                    }
                }
            }
            Spacer()
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var relationSection: some View {
        let relations = flow.clinicalOutput?.relations ?? []
        if !relations.isEmpty {
            OMCard {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        Text("RELATIONSHIPS").omEyebrow()
                        Spacer()
                        OMBadge("\(relations.count)", tone: .accent)
                    }
                    ForEach(relations) { relation in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 7) {
                                Text(relation.head)
                                    .font(.om.body(14, weight: .semibold))
                                Image(systemName: "arrow.right")
                                    .font(.system(size: 10, weight: .bold))
                                    .foregroundStyle(Color.omTealAccent)
                                Text(relation.tail)
                                    .font(.om.body(14, weight: .semibold))
                            }
                            Text(relation.label.uppercased())
                                .font(.om.mono(10, weight: .medium))
                                .foregroundStyle(Color.omFgMuted)
                        }
                        if relation != relations.last { OMRule() }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var mapleBriefSection: some View {
        if let brief = flow.mapleBrief {
            OMCard(elevation: .raised) {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        Text("MAPLE BRIEF").omEyebrow()
                        Spacer()
                        OMBadge("On-device", tone: .positive, systemImage: "lock.fill")
                    }
                    Text(brief)
                        .font(.om.body(15))
                        .foregroundStyle(Color.omInk)
                        .textSelection(.enabled)
                }
            }
        }
    }

    private var secondaryActions: some View {
        VStack(spacing: OM.Space.s2) {
            if completedPIIOutputs >= 2 {
                Button {
                    onShowComparison()
                } label: {
                    HStack {
                        Image(systemName: "rectangle.split.2x1")
                        Text("View engine comparison")
                        Spacer()
                        Image(systemName: "chevron.right")
                            .font(.system(size: 11, weight: .semibold))
                    }
                }
                .buttonStyle(.omSecondary(.md))
            }

            ShareLink(item: jsonExport(), preview: SharePreview("OpenMed Scan Export")) {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text("Export JSON")
                    Spacer()
                    Image(systemName: "chevron.right")
                        .font(.system(size: 11, weight: .semibold))
                }
            }
            .buttonStyle(.omSecondary(.md))

            Button {
                onStartOver()
            } label: {
                HStack {
                    Image(systemName: "arrow.counterclockwise")
                    Text("Start new scan")
                    Spacer()
                }
            }
            .buttonStyle(.omGhost)
        }
    }

    private func jsonExport() -> String {
        let payload: [String: Any] = [
            "stage": "summary",
            "sourceLength": flow.trimmedText.count,
            "piiEntities": (flow.currentPIIOutput?.entities ?? []).map {
                [
                    "label": $0.label, "textSHA256": sha256($0.text), "start": $0.start,
                    "end": $0.end,
                    "category": $0.category.rawValue,
                    "confidence": $0.confidence.map { $0 as Any } ?? NSNull(),
                ]
            },
            "clinicalEntities": (flow.clinicalOutput?.entities ?? []).map {
                [
                    "label": $0.label, "text": $0.text, "start": $0.start, "end": $0.end,
                    "category": $0.category.rawValue,
                    "confidence": $0.confidence.map { $0 as Any } ?? NSNull(),
                ]
            },
            "relations": (flow.clinicalOutput?.relations ?? []).map {
                [
                    "label": $0.label,
                    "head": $0.head,
                    "tail": $0.tail,
                    "confidence": $0.confidence.map { $0 as Any } ?? NSNull(),
                ]
            },
            "mapleBrief": flow.mapleBrief.map { $0 as Any } ?? NSNull(),
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) else {
            return "{}"
        }
        return String(data: data, encoding: .utf8) ?? "{}"
    }

    private func sha256(_ value: String) -> String {
        SHA256.hash(data: Data(value.utf8))
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private var completedPIIOutputs: Int {
        ScanFlowViewModel.PIIEngine.allCases
            .filter { flow.output(for: $0) != nil }
            .count
    }
}
