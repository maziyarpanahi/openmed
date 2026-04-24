import SwiftUI

public struct ModelDownloadSheet: View {
    @ObservedObject public var downloads: ModelDownloadManager
    @Environment(\.dismiss) private var dismiss

    public init(downloads: ModelDownloadManager) {
        self.downloads = downloads
    }

    public var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: OM.Space.s4) {
                    header

                    ForEach(ScanModelID.allCases) { modelID in
                        if let entry = downloads.entries[modelID] {
                            OMDownloadRow(
                                modelID: modelID,
                                entry: entry,
                                onStart: { downloads.prepare(modelID) },
                                onCancel: { downloads.cancel(modelID) }
                            )
                        }
                    }

                    footer
                }
                .padding(OM.Space.s5)
            }
            .background(Color.omPaper)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Close") { dismiss() }
                        .buttonStyle(.omGhost(.ink))
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        prepareAll()
                    } label: {
                        Text("Prepare all")
                            .font(.om.body(14, weight: .semibold))
                    }
                    .foregroundStyle(Color.omTealAccent)
                    .disabled(!anyMissing)
                }
            }
            .toolbarBackground(Color.omPaper, for: .navigationBar)
            .toolbarColorScheme(.light, for: .navigationBar)
            .navigationTitle("On-device models")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: OM.Space.s2) {
            Text("LOCAL RUNTIME").omEyebrow()
            OMDisplayHeadline([
                .plain("Cache "),
                .accent("models"),
                .plain(" for offline runs."),
            ], scale: .sm)
            Text("Each artifact is downloaded once from Hugging Face and cached on this device. Cancel, resume, or delete anytime.")
                .font(.om.body(14))
                .foregroundStyle(Color.omFgMuted)
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("SECURITY").omEyebrow()
                .foregroundStyle(Color.omFgSubtle)
            Text("Inference never leaves your device. Model files are encrypted at rest by iOS and bundled into the app's private cache.")
                .font(.om.body(12))
                .foregroundStyle(Color.omFgSubtle)
        }
        .padding(.top, OM.Space.s4)
    }

    private var anyMissing: Bool {
        downloads.entries.values.contains(where: {
            switch $0.state {
            case .ready: return false
            default:     return true
            }
        })
    }

    private func prepareAll() {
        for modelID in ScanModelID.allCases {
            guard let entry = downloads.entries[modelID] else { continue }
            switch entry.state {
            case .ready, .downloading, .installing, .queued: continue
            default: downloads.prepare(modelID)
            }
        }
    }
}
