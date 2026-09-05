import SwiftUI

public struct DeIDScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager
    public let onShowComparison: () -> Void

    public init(
        flow: ScanFlowViewModel,
        downloads: ModelDownloadManager,
        onShowComparison: @escaping () -> Void
    ) {
        self.flow = flow
        self.downloads = downloads
        self.onShowComparison = onShowComparison
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.deidentify.eyebrow,
                spans: [.plain("Redact, then "), .accent("compare"), .plain(".")],
                subhead: "Three local PII engines can mask the same note. Yellow = detected PHI.",
                scale: .lg
            )

            engineToggle

            modelGate

            ForEach(ScanFlowViewModel.PIIEngine.allCases) { engine in
                outputPanel(
                    engine: engine,
                    output: flow.output(for: engine),
                    eyebrow: engine.eyebrow
                )
            }

            if flow.comparisonPIIEngines != nil {
                comparisonRow
            }
        }
    }

    private var engineToggle: some View {
        OMCard(padding: OM.Space.s3) {
            VStack(alignment: .leading, spacing: OM.Space.s2) {
                Text("ENGINE").omEyebrow()
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: OM.Space.s2) {
                        ForEach(ScanFlowViewModel.PIIEngine.allCases) { engine in
                            OMChip(
                                engine.toggleTitle,
                                tone: flow.piiEngine == engine ? .ink : .neutral,
                                action: {
                                    flow.piiEngine = engine
                                    HapticsCenter.selection()
                                }
                            )
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
        }
    }

    @ViewBuilder
    private var modelGate: some View {
        let modelID = flow.piiEngine.modelID
        if let entry = downloads.entries[modelID], entry.state != .ready {
            OMDownloadRow(
                modelID: modelID,
                entry: entry,
                onStart: { downloads.prepare(modelID) },
                onCancel: { downloads.cancel(modelID) }
            )
        }
    }

    private func outputPanel(engine: ScanFlowViewModel.PIIEngine, output: PIIOutput?, eyebrow: String) -> some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text(eyebrow).omEyebrow()
                    Spacer()
                    if engine == flow.piiEngine, flow.selectedPIIRequiresRun {
                        OMBadge(output == nil ? "READY TO RUN" : "RUN SELECTED", tone: .neutral)
                    } else if let output {
                        OMBadge("\(output.entities.count) spans", tone: .accent)
                    }
                }

                if let output {
                    OMEntityHighlight(
                        text: flow.trimmedText,
                        entities: output.entities,
                        bodyFont: .om.display(17),
                        showsLabels: true
                    )
                } else {
                    Text("Run the pipeline to see masked output.")
                        .font(.om.body(14, italic: true))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
        }
    }

    @ViewBuilder
    private var comparisonRow: some View {
        if let engines = flow.comparisonPIIEngines {
            let diff = EngineDiff.build(
                left: flow.output(for: engines.left)?.entities ?? [],
                right: flow.output(for: engines.right)?.entities ?? [],
                leftLabel: engines.left.displayName,
                rightLabel: engines.right.displayName
            ).summary

            OMCard(padding: OM.Space.s4) {
                HStack(alignment: .center, spacing: OM.Space.s3) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("\(diff.agreements) AGREE · \(diff.onlyLeft) ONLY LEFT · \(diff.onlyRight) ONLY RIGHT")
                            .omMonoTag(size: 10)
                            .foregroundStyle(Color.omFgMuted)
                        Text("Compare side by side")
                            .font(.om.heading(17, weight: .semibold))
                            .foregroundStyle(Color.omInk)
                    }
                    Spacer()
                    Button("Open diff") { onShowComparison() }
                        .buttonStyle(.omGhost)
                }
            }
        }
    }
}
