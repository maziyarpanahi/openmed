import SwiftUI

public struct ReviewScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager

    public init(flow: ScanFlowViewModel, downloads: ModelDownloadManager) {
        self.flow = flow
        self.downloads = downloads
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.review.eyebrow,
                spans: [.plain("Your "), .accent("document"), .plain(".")],
                subhead: "Edit OCR mistakes before masking. Nothing leaves this device.",
                scale: .lg
            )

            metaStrip

            if flow.needsOCR, flow.currentSource == .scan {
                ocrCard
            } else {
                editorCard
            }

            modelGate
        }
    }

    private var metaStrip: some View {
        HStack(spacing: OM.Space.s3) {
            metaItem("SOURCE", value: flow.currentSource.label)
            Divider().frame(height: 26).background(Color.omBorder)
            metaItem("CHARS", value: "\(flow.pastedOrScannedText.count)")
            Divider().frame(height: 26).background(Color.omBorder)
            metaItem("WORDS", value: "\(wordCount)")
            Divider().frame(height: 26).background(Color.omBorder)
            metaItem("PAGES", value: "\(max(flow.pageCount, 1))")
        }
    }

    private func metaItem(_ label: String, value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).omEyebrow()
            Text(value)
                .font(.om.display(20, weight: .medium))
                .foregroundStyle(Color.omInk)
        }
    }

    private var wordCount: Int {
        flow.trimmedText.isEmpty ? 0 : flow.trimmedText
            .split(whereSeparator: { $0.isWhitespace || $0.isNewline })
            .count
    }

    private var ocrCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                Text("VISION OCR").omEyebrow()
                Text("Extract text from the scan")
                    .font(.om.heading(19))
                    .foregroundStyle(Color.omInk)
                Text("Apple Vision runs locally. Results populate this card when ready.")
                    .font(.om.body(14))
                    .foregroundStyle(Color.omFgMuted)

                Button {
                    Task { await flow.runOCRIfNeeded() }
                } label: {
                    HStack {
                        Image(systemName: "text.viewfinder")
                        Text(flow.isWorking ? "Running OCR…" : "Run OCR")
                    }
                }
                .buttonStyle(.omPrimary(.sm))
                .disabled(flow.isWorking)
            }
        }
    }

    private var editorCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                Text("TRANSCRIPT").omEyebrow()
                TextEditor(text: $flow.pastedOrScannedText)
                    .scrollContentBackground(.hidden)
                    .font(.om.display(17))
                    .foregroundStyle(Color.omInk)
                    .frame(minHeight: 260)
                    .padding(14)
                    .background(Color.omPaper, in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                            .strokeBorder(Color.omBorder, lineWidth: OM.Stroke.hairline)
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
}
