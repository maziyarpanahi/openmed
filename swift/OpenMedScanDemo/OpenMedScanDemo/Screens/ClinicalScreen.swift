import SwiftUI

/// Explicit clinical NER runner. Model selection is inert; the shared action
/// button performs one load/run/unload cycle for the selected cached model.
public struct ClinicalScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager

    public init(
        flow: ScanFlowViewModel,
        downloads: ModelDownloadManager
    ) {
        self.flow = flow
        self.downloads = downloads
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.clinical.eyebrow,
                spans: [.plain("Run one "), .accent("NER model"), .plain(" at a time.")],
                subhead: "Choose a downloaded token-classification model. Selection alone does not run it.",
                scale: .lg
            )

            lifecycleCard
            modelPicker
            modelGate
            results
        }
    }

    private var lifecycleCard: some View {
        OMCard(elevation: .raised, padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("MODEL LIFECYCLE").omEyebrow()
                    Spacer()
                    OMBadge("One at a time", tone: .positive, systemImage: "memorychip")
                }
                Text("Download → load → run → unload")
                    .font(.om.heading(18, weight: .semibold))
                    .foregroundStyle(Color.omInk)
                Text("Every explicit run releases the NER runtime and clears the MLX buffer cache before control returns. Switch models and press Run to repeat on the same de-identified note.")
                    .font(.om.body(13))
                    .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    private var modelPicker: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("CLINICAL NER MODEL").omEyebrow()
                    Spacer()
                    Text("3 available")
                        .font(.om.mono(11))
                        .foregroundStyle(Color.omFgSubtle)
                }

                ForEach(Array(ScanFlowViewModel.NERModel.allCases.enumerated()), id: \.element.id) {
                    index,
                    model in
                    modelRow(model)
                    if index < ScanFlowViewModel.NERModel.allCases.count - 1 {
                        OMRule()
                    }
                }
            }
        }
    }

    private func modelRow(_ model: ScanFlowViewModel.NERModel) -> some View {
        let isSelected = flow.nerModel == model
        let state = downloads.state(for: model.modelID)
        let output = flow.nerOutputs[model]

        return Button {
            flow.nerModel = model
            HapticsCenter.selection()
        } label: {
            HStack(alignment: .center, spacing: OM.Space.s3) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(isSelected ? Color.omTealAccent : Color.omBorderStrong)

                VStack(alignment: .leading, spacing: 3) {
                    Text(model.displayName)
                        .font(.om.body(15, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                    Text(model.detail)
                        .font(.om.body(12))
                        .foregroundStyle(Color.omFgMuted)
                }

                Spacer(minLength: 0)

                if isSelected, state == .ready, flow.selectedNERRequiresRun {
                    OMBadge(output == nil ? "READY TO RUN" : "RUN SELECTED", tone: .neutral)
                } else if let output {
                    OMBadge("\(output.entities.count) spans", tone: .accent)
                } else {
                    stateBadge(state)
                }
            }
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private func stateBadge(_ state: ModelDownloadState) -> some View {
        switch state {
        case .ready:
            OMBadge("Cached", tone: .positive)
        case .downloading, .queued, .installing:
            OMBadge("Preparing", tone: .neutral)
        case .failed:
            OMBadge("Retry", tone: .signal)
        case .missing, .partial, .cancelled:
            OMBadge("Download", tone: .neutral)
        }
    }

    @ViewBuilder
    private var modelGate: some View {
        let modelID = flow.nerModel.modelID
        if let entry = downloads.entries[modelID], entry.state != .ready {
            OMDownloadRow(
                modelID: modelID,
                entry: entry,
                onStart: { downloads.prepare(modelID) },
                onCancel: { downloads.cancel(modelID) }
            )
        }
    }

    @ViewBuilder
    private var results: some View {
        if flow.completedNERModels.isEmpty {
            OMCard {
                VStack(alignment: .leading, spacing: OM.Space.s2) {
                    Text("NO NER RUN YET").omEyebrow()
                    Text("Download a model, select it, then press the action button.")
                        .font(.om.body(14, italic: true))
                        .foregroundStyle(Color.omFgSubtle)
                }
            }
        } else {
            ForEach(flow.completedNERModels) { model in
                if let output = flow.nerOutputs[model] {
                    resultCard(model: model, output: output)
                }
            }
        }
    }

    private func resultCard(
        model: ScanFlowViewModel.NERModel,
        output: NEROutput
    ) -> some View {
        OMCard(elevation: .raised) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(model.displayName.uppercased()).omEyebrow()
                        Text("NER result · runtime unloaded")
                            .font(.om.heading(18, weight: .semibold))
                            .foregroundStyle(Color.omInk)
                    }
                    Spacer()
                    OMBadge("\(output.entities.count) spans", tone: .accent)
                }

                if output.entities.isEmpty {
                    Text("This model found no matching entity above the confidence threshold.")
                        .font(.om.body(13, italic: true))
                        .foregroundStyle(Color.omFgSubtle)
                } else {
                    ForEach(Array(output.entities.prefix(8).enumerated()), id: \.element.id) {
                        index,
                        entity in
                        HStack(alignment: .firstTextBaseline, spacing: OM.Space.s3) {
                            Text(entity.label.uppercased())
                                .font(.om.mono(9, weight: .semibold))
                                .foregroundStyle(entity.category.tone.accent)
                                .frame(width: 88, alignment: .leading)
                            Text(entity.text)
                                .font(.om.body(14, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                            Spacer(minLength: 0)
                            if let confidence = entity.confidence {
                                Text(String(format: "%.0f%%", confidence * 100))
                                    .font(.om.mono(10))
                                    .foregroundStyle(Color.omFgSubtle)
                            }
                        }
                        if index < min(output.entities.count, 8) - 1 { OMRule() }
                    }
                }
            }
        }
    }
}
