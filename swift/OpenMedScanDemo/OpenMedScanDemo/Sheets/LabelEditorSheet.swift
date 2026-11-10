import SwiftUI

public struct LabelEditorSheet: View {
    @ObservedObject public var presets: ClinicalPresetsStore
    @Environment(\.dismiss) private var dismiss

    @State private var draftName: String
    @State private var draftLabels: [String]
    @State private var error: String?

    public init(presets: ClinicalPresetsStore) {
        self.presets = presets
        let current = presets.selectedPreset
        _draftName = State(initialValue: current.isBuiltIn ? "" : current.name)
        _draftLabels = State(initialValue: current.labels)
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: OM.Space.s4) {
                    header
                    nameCard
                    labelsCard
                    if let error {
                        Text(error)
                            .font(.om.body(13))
                            .foregroundStyle(Color.omSignal)
                    }
                }
                .padding(OM.Space.s5)
            }
            .background(Color.omPaper)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                        .buttonStyle(.omGhost(.ink))
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save", action: save)
                        .font(.om.body(14, weight: .semibold))
                        .foregroundStyle(Color.omTealAccent)
                        .disabled(draftName.trimmingCharacters(in: .whitespaces).isEmpty || draftLabels.isEmpty)
                }
            }
            .toolbarBackground(Color.omPaper, for: .navigationBar)
            .navigationTitle("Save as preset")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            Text("NEW PRESET").omEyebrow()
            Text("Name this label set so you can reuse it later.")
                .font(.om.body(14))
                .foregroundStyle(Color.omFgMuted)
        }
    }

    private var nameCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s2) {
                Text("NAME").omEyebrow()
                TextField("e.g. Discharge note", text: $draftName)
                    .textInputAutocapitalization(.sentences)
                    .font(.om.body(16, weight: .medium))
                    .padding(OM.Space.s3)
                    .background(Color.omPaper, in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                            .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                    )
            }
        }
    }

    private var labelsCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                HStack {
                    Text("LABELS").omEyebrow()
                    Spacer()
                    Text("\(draftLabels.count) active").font(.om.mono(11)).foregroundStyle(Color.omFgSubtle)
                }
                OMLabelChipGrid(labels: $draftLabels)
                Text("Labels are normalised (trimmed, lowercase, deduped).")
                    .font(.om.body(12))
                    .foregroundStyle(Color.omFgSubtle)
            }
        }
    }

    private func save() {
        let name = draftName.trimmingCharacters(in: .whitespaces)
        guard !name.isEmpty else { error = "Name is required."; return }
        guard !draftLabels.isEmpty else { error = "Add at least one label."; return }
        guard presets.userPresets.count < ClinicalPresetsStore.maxUserPresets else {
            error = "Preset cap reached (\(ClinicalPresetsStore.maxUserPresets)). Remove one first."
            return
        }
        if presets.saveAsNew(name: name, labels: draftLabels) != nil {
            HapticsCenter.notify(.success)
            dismiss()
        } else {
            error = "Could not save preset."
        }
    }
}
