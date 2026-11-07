import SwiftUI

public struct ClinicalScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager
    @ObservedObject public var presets: ClinicalPresetsStore
    public let onSaveAsNewPreset: () -> Void

    public init(
        flow: ScanFlowViewModel,
        downloads: ModelDownloadManager,
        presets: ClinicalPresetsStore,
        onSaveAsNewPreset: @escaping () -> Void
    ) {
        self.flow = flow
        self.downloads = downloads
        self.presets = presets
        self.onSaveAsNewPreset = onSaveAsNewPreset
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.clinical.eyebrow,
                spans: [.plain("Pick what to "), .accent("extract"), .plain(".")],
                subhead: "Choose a task preset or edit the label set. Maple extracts entities and their relationships on-device.",
                scale: .lg
            )

            promptPreviewCard
            presetPicker
            labelEditor
            extractionResultsCard
            extractionContractCard
            modelGate
        }
    }

    private var promptPreviewCard: some View {
        OMCard(elevation: .raised, padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    HStack(spacing: 6) {
                        Image(systemName: "text.quote")
                        Text("PROMPTED EXTRACTION")
                    }
                    .font(.om.eyebrow())
                    .textCase(.uppercase)
                    .kerning(1.4)
                    .foregroundStyle(Color.omTealAccent)
                    Spacer()
                    OMBadge("Local", tone: .positive, systemImage: "lock.fill")
                }

                Text(promptPreview)
                    .font(.om.mono(12))
                    .foregroundStyle(Color.omInk)
                    .lineSpacing(4)
                    .padding(OM.Space.s3)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(
                        Color.omPaper2,
                        in: RoundedRectangle(
                            cornerRadius: OM.Radius.md,
                            style: .continuous
                        )
                    )

                Label(
                    "The model returns JSON; results appear only after span and vocabulary validation.",
                    systemImage: "checkmark.shield.fill"
                )
                .font(.om.body(11))
                .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    private var promptPreview: String {
        let labels = presets.selectedPreset.labels
        let visibleLabels = labels.prefix(5).joined(separator: ", ")
        let remainder = labels.count > 5 ? ", +\(labels.count - 5) more" : ""
        return "“Extract only [\(visibleLabels)\(remainder)] from the de-identified note. Preserve exact source spans and return only supported relations as JSON.”"
    }

    private var presetPicker: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("PRESET").omEyebrow()
                    Spacer()
                    Text("\(presets.allPresets.count) available")
                        .font(.om.mono(11))
                        .foregroundStyle(Color.omFgSubtle)
                }
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: OM.Space.s2) {
                        ForEach(presets.allPresets) { preset in
                            presetChip(preset)
                        }
                        Button {
                            onSaveAsNewPreset()
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "plus")
                                    .font(.system(size: 10, weight: .semibold))
                                Text("NEW")
                                    .font(.om.mono(12, weight: .medium))
                                    .kerning(1.2)
                            }
                            .foregroundStyle(Color.omTealAccent)
                            .padding(.vertical, 6)
                            .padding(.horizontal, 12)
                            .overlay(
                                Capsule().strokeBorder(Color.omTealAccent, style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                            )
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.vertical, 2)
                }
                if !presets.selectedPreset.summary.isEmpty {
                    Text(presets.selectedPreset.summary)
                        .font(.om.body(13))
                        .foregroundStyle(Color.omFgMuted)
                }
            }
        }
    }

    private func presetChip(_ preset: ClinicalPreset) -> some View {
        let selected = preset.id == presets.selectedID
        return OMChip(
            preset.name,
            tone: selected ? .ink : .neutral,
            leadingSystemImage: preset.isBuiltIn ? "lock.fill" : nil,
            action: {
                presets.select(preset)
                HapticsCenter.selection()
            }
        )
    }

    private var labelEditor: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("LABELS").omEyebrow()
                    Spacer()
                    Text("\(presets.selectedPreset.labels.count) active")
                        .font(.om.mono(11))
                        .foregroundStyle(Color.omFgSubtle)
                }
                OMLabelChipGrid(
                    labels: labelsBinding,
                    onChanged: nil
                )
                HStack {
                    Button("Save as new preset…") {
                        onSaveAsNewPreset()
                    }
                    .buttonStyle(.omGhost)
                    Spacer()
                    if !presets.selectedPreset.isBuiltIn {
                        Button("Delete") {
                            presets.delete(presets.selectedPreset)
                            HapticsCenter.impact(.rigid)
                        }
                        .buttonStyle(.omGhost(.signal))
                    }
                }
            }
        }
    }

    /// A binding that reads from the selected preset's labels and writes back
    /// via the store (auto-forking built-ins when needed).
    private var labelsBinding: Binding<[String]> {
        Binding(
            get: { presets.selectedPreset.labels },
            set: { presets.updateLabelsOnSelected(to: $0) }
        )
    }

    private var extractionContractCard: some View {
        OMCard(padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("STRUCTURED OUTPUT").omEyebrow()
                    Spacer()
                    OMBadge("Validated", tone: .positive, systemImage: "checkmark.shield.fill")
                }
                Text("Maple is generative, so it does not expose calibrated confidence scores. OpenMedKit validates source spans, enforces the selected label set, and drops relations whose endpoints were not extracted.")
                    .font(.om.body(13))
                    .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    @ViewBuilder
    private var extractionResultsCard: some View {
        if let output = flow.clinicalOutput {
            OMCard(elevation: .raised) {
                VStack(alignment: .leading, spacing: OM.Space.s3) {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("PROMPT RESPONSE").omEyebrow()
                            Text("Validated clinical structure")
                                .font(.om.heading(18, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                        }
                        Spacer()
                        OMBadge("\(output.entities.count) entities", tone: .accent)
                    }

                    ForEach(Array(output.entities.prefix(6).enumerated()), id: \.element.id) {
                        index,
                        entity in
                        HStack(alignment: .firstTextBaseline, spacing: OM.Space.s3) {
                            Text(entity.label)
                                .font(.om.mono(9, weight: .semibold))
                                .kerning(0.8)
                                .foregroundStyle(entity.category.tone.accent)
                                .frame(width: 84, alignment: .leading)
                            Text(entity.text)
                                .font(.om.body(14, weight: .semibold))
                                .foregroundStyle(Color.omInk)
                            Spacer(minLength: 0)
                            Image(systemName: "checkmark.circle.fill")
                                .font(.system(size: 12))
                                .foregroundStyle(Color.omTealAccent)
                        }
                        if index < min(output.entities.count, 6) - 1 { OMRule() }
                    }

                    if !output.relations.isEmpty {
                        HStack(spacing: OM.Space.s2) {
                            Image(systemName: "point.3.connected.trianglepath.dotted")
                                .foregroundStyle(Color.omTealAccent)
                            Text("\(output.relations.count) supported relation\(output.relations.count == 1 ? "" : "s") ready for the Maple workspace")
                                .font(.om.body(12, weight: .medium))
                                .foregroundStyle(Color.omFgMuted)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var modelGate: some View {
        if let entry = downloads.entries[flow.clinicalModelID], entry.state != .ready {
            OMDownloadRow(
                modelID: flow.clinicalModelID,
                entry: entry,
                onStart: { downloads.prepare(flow.clinicalModelID) },
                onCancel: { downloads.cancel(flow.clinicalModelID) }
            )
        }
    }
}
