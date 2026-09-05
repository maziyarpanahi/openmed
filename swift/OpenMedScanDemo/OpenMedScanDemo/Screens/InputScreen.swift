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
                    Text("CLINICAL NER").omEyebrow()
                    Text("Three focused NER models · ~134 MB each")
                        .font(.om.heading(17, weight: .semibold))
                        .foregroundStyle(Color.omInk)
                    Text("Disease, medication, and anatomy token classifiers. Each model loads for one explicit run, then unloads.")
                        .font(.om.body(12))
                        .foregroundStyle(Color.omFgMuted)
                        .fixedSize(horizontal: false, vertical: true)

                    Text("\(cachedNERModelCount) of 3 cached")
                        .font(.om.mono(11, weight: .medium))
                        .foregroundStyle(cachedNERModelCount > 0 ? Color.omTealAccent : Color.omFgSubtle)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Button("Manage") { onShowModelSheet() }
                    .buttonStyle(.omGhost)
            }
        }
    }

    private var cachedNERModelCount: Int {
        ScanFlowViewModel.NERModel.allCases.filter {
            downloads.state(for: $0.modelID) == .ready
        }.count
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
            Text("All redaction and clinical NER runs on-device. Nothing leaves your iPhone.")
                .font(.om.body(12))
        }
        .foregroundStyle(Color.omFgSubtle)
        .padding(.top, OM.Space.s2)
    }
}
