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
                subhead: "Two independent engines mask the same note. Yellow = detected PHI.",
                scale: .lg
            )

            engineToggle

            modelGate

            outputPanel(
                engine: .openMed,
                output: flow.openMedPIIOutput,
                eyebrow: ScanFlowViewModel.PIIEngine.openMed.eyebrow
            )

            outputPanel(
                engine: .privacyFilter,
                output: flow.privacyFilterPIIOutput,
                eyebrow: ScanFlowViewModel.PIIEngine.privacyFilter.eyebrow
            )

            if flow.openMedPIIOutput != nil && flow.privacyFilterPIIOutput != nil {
                comparisonRow
            }
        }
    }

    private var engineToggle: some View {
        OMCard(padding: OM.Space.s3) {
            HStack(spacing: OM.Space.s3) {
                Text("ENGINE").omEyebrow()
                Spacer()
                OMChip(
                    ScanFlowViewModel.PIIEngine.openMed.displayName,
                    tone: flow.piiEngine == .openMed ? .ink : .neutral,
                    action: { flow.piiEngine = .openMed; HapticsCenter.selection() }
                )
                OMChip(
                    ScanFlowViewModel.PIIEngine.privacyFilter.displayName,
                    tone: flow.piiEngine == .privacyFilter ? .ink : .neutral,
                    action: { flow.piiEngine = .privacyFilter; HapticsCenter.selection() }
                )
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
                    if let output {
                        OMBadge("\(output.entities.count) spans", tone: .accent)
                    } else if engine == flow.piiEngine {
                        OMBadge("READY TO RUN", tone: .neutral)
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

    private var comparisonRow: some View {
        let diff = EngineDiff.build(
            left: flow.openMedPIIOutput?.entities ?? [],
            right: flow.privacyFilterPIIOutput?.entities ?? [],
            leftLabel: ScanFlowViewModel.PIIEngine.openMed.displayName,
            rightLabel: ScanFlowViewModel.PIIEngine.privacyFilter.displayName
        ).summary

        return OMCard(padding: OM.Space.s4) {
            HStack(alignment: .center, spacing: OM.Space.s3) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(diff.agreements) AGREE · \(diff.onlyLeft) ONLY OPENMED · \(diff.onlyRight) ONLY OPENAI")
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
