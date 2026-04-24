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
                    Text(String(format: "%.0f%%", entity.confidence * 100))
                        .font(.om.mono(10))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
            Spacer()
        }
        .padding(.vertical, 4)
    }

    private var secondaryActions: some View {
        VStack(spacing: OM.Space.s2) {
            if flow.openMedPIIOutput != nil && flow.privacyFilterPIIOutput != nil {
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
            "piiEntities": (flow.currentPIIOutput?.entities ?? []).map { [
                "label": $0.label, "text": $0.text, "start": $0.start, "end": $0.end,
                "category": $0.category.rawValue, "confidence": $0.confidence,
            ] },
            "clinicalEntities": (flow.clinicalOutput?.entities ?? []).map { [
                "label": $0.label, "text": $0.text, "start": $0.start, "end": $0.end,
                "category": $0.category.rawValue, "confidence": $0.confidence,
            ] },
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted]) else {
            return "{}"
        }
        return String(data: data, encoding: .utf8) ?? "{}"
    }
}
