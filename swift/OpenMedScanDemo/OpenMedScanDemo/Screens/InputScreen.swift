import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

public struct InputScreen: View {
    @ObservedObject public var flow: ScanFlowViewModel
    @ObservedObject public var downloads: ModelDownloadManager
    public let onShowScanner: () -> Void
    public let onShowModelSheet: () -> Void

    @State private var pasteBuffer: String = ""
    @State private var selectedSampleLanguage: SampleClinicalText.Language = .en
    @FocusState private var pasteFocused: Bool

    public init(
        flow: ScanFlowViewModel,
        downloads: ModelDownloadManager,
        onShowScanner: @escaping () -> Void,
        onShowModelSheet: @escaping () -> Void
    ) {
        self.flow = flow
        self.downloads = downloads
        self.onShowScanner = onShowScanner
        self.onShowModelSheet = onShowModelSheet
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: OM.Space.s5) {
            ScanStageHeader(
                eyebrow: ScanStage.input.eyebrow,
                spans: [.plain("Bring your clinical "), .accent("text"), .plain(".")],
                subhead: "Paste, scan, or load the sample. Everything from here stays on this device.",
                scale: .lg
            )

            enginePickerCard
            clinicalModelCard

            inputChoiceHeader

            pasteCard
            scanCard
            sampleCard

            privacyFooter
        }
    }

    // MARK: Engine picker + clinical model

    private var enginePickerCard: some View {
        OMEnginePickerCard(
            selectedEngine: flow.piiEngine,
            entries: downloads.entries,
            onSelect: { engine in
                flow.piiEngine = engine
                HapticsCenter.selection()
            },
            onDownload: { engine in
                downloads.prepare(engine.modelID)
            },
            onCancel: { engine in
                downloads.cancel(engine.modelID)
            }
        )
    }

    private var clinicalModelCard: some View {
        OMCard(padding: OM.Space.s4) {
            HStack(alignment: .top, spacing: OM.Space.s3) {
                VStack(alignment: .leading, spacing: OM.Space.s2) {
                    Text("CLINICAL EXTRACTOR").omEyebrow()
                    Text("GLiNER Relex Base")
                        .font(.om.heading(17, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                    Text("Zero-shot NER for condition, medication, and care-plan labels you define.")
                        .font(.om.body(12))
                        .foregroundStyle(Color.omFgMuted)
                        .fixedSize(horizontal: false, vertical: true)

                    clinicalStateRow
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Button("Manage") { onShowModelSheet() }
                    .buttonStyle(.omGhost)
            }
        }
    }

    @ViewBuilder
    private var clinicalStateRow: some View {
        let entry = downloads.entries[.glinerRelex]
        let state = entry?.state ?? .missing
        switch state {
        case .ready:
            HStack(spacing: 4) {
                Image(systemName: "checkmark.seal.fill")
                    .font(.system(size: 11))
                    .foregroundStyle(Color.omTealAccent)
                Text("Cached")
                    .font(.om.mono(11, weight: .medium))
                    .foregroundStyle(Color.omTealAccent)
            }
            .padding(.top, 4)
        case .downloading(let bytes, let total, _):
            VStack(alignment: .leading, spacing: 4) {
                OMProgressBar(
                    mode: (entry?.fraction).map { .determinate(progress: $0) } ?? .indeterminate,
                    height: 4
                )
                HStack {
                    Text(ByteFormatter.percent(entry?.fraction))
                        .font(.om.mono(10, weight: .semibold))
                    Spacer()
                    Text(ByteFormatter.progressString(bytes: bytes, total: total ?? entry?.bytesEstimatedTotal))
                        .font(.om.mono(9))
                        .foregroundStyle(Color.omFgSubtle)
                }
                Button("Cancel") { downloads.cancel(.glinerRelex) }
                    .buttonStyle(.omGhost(.ink))
            }
            .padding(.top, 4)
        case .failed(let message):
            VStack(alignment: .leading, spacing: 4) {
                Text(message)
                    .font(.om.body(11))
                    .foregroundStyle(Color.omSignal)
                    .lineLimit(2)
                Button("Retry") { downloads.prepare(.glinerRelex) }
                    .buttonStyle(.omSecondary(.sm))
            }
            .padding(.top, 4)
        default:
            Button {
                downloads.prepare(.glinerRelex)
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.down.circle.fill")
                        .font(.system(size: 11))
                    Text("Download · \(ScanModelID.glinerRelex.estimatedSizeLabel.replacingOccurrences(of: "~", with: ""))")
                }
            }
            .buttonStyle(.omSecondary(.sm))
            .padding(.top, 4)
        }
    }

    // MARK: Input choice cards

    private var inputChoiceHeader: some View {
        VStack(alignment: .leading, spacing: 6) {
            OMRule()
            Text("DOCUMENT").omEyebrow()
                .padding(.top, 4)
        }
    }

    private var pasteCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                Text("PASTE").omEyebrow()
                Text("Paste clinical text")
                    .font(.om.heading(19))
                    .foregroundStyle(Color.omInk)

                TextEditor(text: $pasteBuffer)
                    .focused($pasteFocused)
                    .scrollContentBackground(.hidden)
                    .font(.om.mono(13))
                    .frame(minHeight: 150)
                    .padding(10)
                    .background(Color.omPaper2, in: RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                            .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                    )

                HStack {
                    Button("Paste from clipboard") {
                        #if canImport(UIKit)
                        if let string = UIPasteboard.general.string {
                            pasteBuffer = string
                            HapticsCenter.selection()
                        }
                        #endif
                    }
                    .buttonStyle(.omGhost)

                    Spacer()

                    Button("Use text") {
                        let text = pasteBuffer.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !text.isEmpty else { return }
                        flow.useText(text)
                        HapticsCenter.notify(.success)
                        pasteFocused = false
                    }
                    .buttonStyle(.omSecondary(.sm))
                    .disabled(pasteBuffer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .frame(maxWidth: 180)
                }
            }
        }
    }

    private var scanCard: some View {
        OMCard {
            VStack(alignment: .leading, spacing: OM.Space.s3) {
                Text("SCAN").omEyebrow()
                Text("Scan a paper document")
                    .font(.om.heading(19))
                    .foregroundStyle(Color.omInk)
                Text("Capture pages with the iOS document scanner. OCR runs on-device with Vision before anything leaves this screen.")
                    .font(.om.body(15))
                    .foregroundStyle(Color.omFgMuted)

                HStack(spacing: OM.Space.s2) {
                    Button {
                        onShowScanner()
                    } label: {
                        Label("Open camera", systemImage: "camera")
                    }
                    .buttonStyle(.omPrimary(.sm))
                }
            }
        }
    }

    private var sampleCard: some View {
        OMCard {
            HStack(alignment: .top, spacing: OM.Space.s4) {
                VStack(alignment: .leading, spacing: OM.Space.s2) {
                    Text("SAMPLE").omEyebrow()
                    Text("Try a multilingual note")
                        .font(.om.heading(19))
                        .foregroundStyle(Color.omInk)
                    Text("Synthetic EN, FR, and AR documents with names, IDs, addresses, phones, email, insurance, employer, and emergency contacts.")
                        .font(.om.body(15))
                        .foregroundStyle(Color.omFgMuted)

                    HStack(spacing: OM.Space.s2) {
                        ForEach(SampleClinicalText.Language.allCases) { language in
                            OMChip(
                                language.buttonTitle,
                                tone: selectedSampleLanguage == language ? .ink : .neutral,
                                leadingSystemImage: "doc.text",
                                action: {
                                    selectedSampleLanguage = language
                                    flow.useSample(text: language.note)
                                    HapticsCenter.notify(.success)
                                }
                            )
                            .frame(minWidth: 62)
                            .accessibilityLabel("Use \(language.displayName) sample")
                        }
                    }
                    .fixedSize(horizontal: false, vertical: true)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                sampleThumbnail
            }
        }
    }

    @ViewBuilder
    private var sampleThumbnail: some View {
        #if canImport(UIKit)
        if let image = UIImage(named: selectedSampleLanguage.assetName) {
            Image(uiImage: image)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fill)
                .frame(width: 86, height: 118)
                .clipped()
                .clipShape(RoundedRectangle(cornerRadius: OM.Radius.sm, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: OM.Radius.sm, style: .continuous)
                        .strokeBorder(Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
                )
        }
        #endif
    }

    private var privacyFooter: some View {
        HStack(spacing: 6) {
            Image(systemName: "lock.shield")
                .font(.system(size: 11))
            Text("All redaction and extraction runs on-device. Nothing leaves your iPhone.")
                .font(.om.body(12))
        }
        .foregroundStyle(Color.omFgSubtle)
        .padding(.top, OM.Space.s2)
    }
}
