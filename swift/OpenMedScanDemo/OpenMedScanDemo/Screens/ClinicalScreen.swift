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
                subhead: "Choose a task preset or edit the label set. GLiNER Relex does the rest.",
                scale: .lg
            )

            presetPicker
            labelEditor
            thresholdCard
            modelGate
        }
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

    private var thresholdCard: some View {
        OMCard(padding: OM.Space.s4) {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("THRESHOLD").omEyebrow()
                    Spacer()
                    Text(String(format: "%.2f", flow.clinicalThreshold))
                        .font(.om.mono(13, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                }
                Slider(value: $flow.clinicalThreshold, in: 0.1...0.9, step: 0.05)
                    .tint(Color.omTealAccent)
                Text("Lower values surface more tentative signals; higher values keep only strong matches.")
                    .font(.om.body(13))
                    .foregroundStyle(Color.omFgMuted)
            }
        }
    }

    @ViewBuilder
    private var modelGate: some View {
        if let entry = downloads.entries[.glinerRelex], entry.state != .ready {
            OMDownloadRow(
                modelID: .glinerRelex,
                entry: entry,
                onStart: { downloads.prepare(.glinerRelex) },
                onCancel: { downloads.cancel(.glinerRelex) }
            )
        }
    }
}
