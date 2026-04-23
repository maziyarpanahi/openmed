import Foundation
import OpenMedKit
import SwiftUI
import Vision
#if canImport(UIKit) && canImport(VisionKit)
import UIKit
import VisionKit
#endif

struct ContentView: View {
    private let pipeline = ScanDemoPipelineDescriptor.defaultPipeline

    @State private var huggingFaceToken = ScanDemoSecrets.huggingFaceToken
    @State private var piiEngine: ScanDemoPIIEngine = .openMed
    @State private var documentImages: [UIImage] = []
    @State private var extractedText = ""
    @State private var entities: [DetectedEntity] = []
    @State private var focusEntities: [DetectedEntity] = []
    @State private var focusSourceText = ""
    @State private var status: PipelineStatus?
    @State private var errorMessage: String?
    @State private var isShowingScanner = false
    @State private var isShowingModelSetup = false
    @State private var isRecognizingText = false
    @State private var isAnalyzing = false
    @State private var isPreparingModels = false
    @State private var modelSetupMessage: String?
    @State private var modelSetupError: String?
    @State private var inferenceTime: Double?
    @State private var scannedPageCount = 0
    @State private var lastSource: ScanSource?
    @State private var hasRunAnalysis = false
    @State private var needsDocumentOCR = false

    private var trimmedText: String {
        extractedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var trimmedToken: String {
        huggingFaceToken.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var isBusy: Bool {
        isRecognizingText || isAnalyzing
    }

    private var hasDocumentSource: Bool {
        !documentImages.isEmpty
    }

    private var glinerCacheState: OpenMedMLXModelCacheState {
        (try? OpenMedModelStore.mlxModelCacheState(
            repoID: glinerModel.artifactRepoID,
            revision: "main"
        )) ?? .missing
    }

    private var piiCacheState: OpenMedMLXModelCacheState {
        (try? OpenMedModelStore.mlxModelCacheState(
            repoID: piiModel.artifactRepoID,
            revision: "main"
        )) ?? .missing
    }

    private var hasRunnableInput: Bool {
        if needsDocumentOCR {
            return hasDocumentSource
        }
        return !trimmedText.isEmpty
    }

    private var canAnalyze: Bool {
        hasRunnableInput && !isBusy && !isPreparingModels && DemoPlatform.supportsOnDeviceMLX && canRunClinicalExtractor
    }

    private var hasResults: Bool {
        !entities.isEmpty || !focusEntities.isEmpty
    }

    private var modelsAreOfflineReady: Bool {
        piiCacheState == .ready && glinerCacheState == .ready
    }

    private var canRunClinicalExtractor: Bool {
        glinerCacheState == .ready || !trimmedToken.isEmpty
    }

    private var piiModel: ScanDemoModelDescriptor {
        pipeline.piiModel
    }

    private var glinerModel: ScanDemoModelDescriptor {
        pipeline.glinerModel
    }

    private var scanButtonTitle: String {
        DocumentScannerSupport.isSupported ? "Scan Document" : "Scanner Unavailable"
    }

    private var primaryActionTitle: String {
        if isRecognizingText {
            return "Running OCR..."
        }
        if isAnalyzing {
            return "Running \(piiEngine.title) + GLiNER..."
        }
        if needsDocumentOCR && hasDocumentSource {
            return "Run OCR + De-identify + Extract"
        }
        return "De-identify + Extract On Device"
    }

    private var actionHint: String {
        if isRecognizingText {
            return "Reading the document image with Vision OCR."
        }
        if isAnalyzing {
            return "Running native OpenMedKit + MLX off the main thread."
        }
        if isPreparingModels {
            return "Preparing the local model cache for offline demo runs."
        }
        if hasResults {
            return "Results are ready. Edit the transcript or load another note to rerun."
        }
        if needsDocumentOCR && hasDocumentSource {
            return "Document staged. Start the full local pipeline when you are ready."
        }
        if !hasRunnableInput {
            return "Start with the sample clinical note or scan a real document."
        }
        if !canRunClinicalExtractor {
            return "Model setup is needed before the local run. Use the small Setup control in the label pack."
        }
        return "Ready to run \(piiEngine.title) for de-identification, then GLiNER for clinical extraction."
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    heroCard
                    pipelineStoryCard

                    if let status {
                        PipelineStatusCard(status: status, pipeline: pipeline)
                    }

                    if let errorMessage {
                        errorCard(errorMessage)
                    }

                    if hasDocumentSource {
                        documentPreviewCard
                    }

                    if !trimmedText.isEmpty {
                        extractedTextCard
                    }

                    piiEngineCard
                    glinerConfigurationCard

                    if !entities.isEmpty {
                        maskedOutputCard
                        entityListCard
                    } else if hasRunAnalysis && !isBusy && errorMessage == nil && !trimmedText.isEmpty {
                        noEntityCard
                    }

                    if !focusEntities.isEmpty {
                        clinicalHighlightsCard
                        clinicalEntityListCard
                    } else if hasRunAnalysis && !isBusy && errorMessage == nil && !trimmedText.isEmpty {
                        noClinicalEntityCard
                    }
                }
                .padding()
                .padding(.bottom, 104)
            }
            .background(demoBackground)
            .safeAreaInset(edge: .bottom) {
                actionBar
            }
            .toolbar(.hidden, for: .navigationBar)
            .scrollDismissesKeyboard(.interactively)
        }
        .sheet(isPresented: $isShowingScanner) {
            DocumentScannerSheet(
                onComplete: handleScannedPages,
                onCancel: { isShowingScanner = false },
                onError: handleScannerError
            )
        }
        .sheet(isPresented: $isShowingModelSetup) {
            ModelSetupSheet(
                token: $huggingFaceToken,
                piiCacheState: piiCacheState,
                glinerCacheState: glinerCacheState,
                isPreparing: isPreparingModels,
                message: modelSetupMessage,
                errorMessage: modelSetupError,
                onPrepare: prepareOfflineModels
            )
            .presentationDetents([.medium, .large])
        }
        .onChange(of: extractedText) { _, newValue in
            guard !isRecognizingText else { return }
            if hasDocumentSource {
                needsDocumentOCR = newValue.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            clearAnalysisState(keepError: false)
        }
        .onChange(of: piiEngine) { _, _ in
            clearAnalysisState(keepError: false)
        }
    }

    private var demoBackground: some View {
        ZStack {
            LinearGradient(
                colors: [
                    Color(red: 0.96, green: 0.99, blue: 0.98),
                    Color(red: 0.91, green: 0.96, blue: 0.98),
                    Color(red: 0.98, green: 0.96, blue: 0.91),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            )

            Circle()
                .fill(Color.teal.opacity(0.16))
                .frame(width: 280, height: 280)
                .blur(radius: 40)
                .offset(x: -150, y: -260)

            Circle()
                .fill(Color.orange.opacity(0.12))
                .frame(width: 260, height: 260)
                .blur(radius: 50)
                .offset(x: 180, y: 260)
        }
        .ignoresSafeArea()
    }

    private var heroCard: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .center) {
                Label("OpenMed", systemImage: "cross.case.fill")
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(.white)

                Spacer()

                Text(DemoPlatform.supportsOnDeviceMLX ? "ON-DEVICE" : "UI PREVIEW")
                    .font(.caption.weight(.black))
                    .tracking(1.3)
                    .foregroundStyle(Color(red: 0.05, green: 0.25, blue: 0.24))
                    .padding(.horizontal, 11)
                    .padding(.vertical, 7)
                    .background(.white.opacity(0.86), in: Capsule())
            }

            VStack(alignment: .leading, spacing: 10) {
                Text("Clinical notes, de-identified in seconds.")
                    .font(.system(size: 31, weight: .black, design: .rounded))
                    .foregroundStyle(.white)
                    .fixedSize(horizontal: false, vertical: true)

                Text("Scan a document, mask patient identifiers locally, then extract clinical signals with native GLiNER over the safe note.")
                    .font(.system(.body, design: .rounded).weight(.medium))
                    .foregroundStyle(.white.opacity(0.82))
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 8) {
                heroBadge(title: "Privacy", value: "PII masked", tint: .white)
                heroBadge(title: "Clinical AI", value: "GLiNER labels", tint: .white)
                heroBadge(title: "Runtime", value: "Offline MLX", tint: .white)
            }
        }
        .padding(22)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.02, green: 0.25, blue: 0.24),
                        Color(red: 0.05, green: 0.42, blue: 0.39),
                        Color(red: 0.10, green: 0.50, blue: 0.46),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )

                Circle()
                    .fill(Color.white.opacity(0.16))
                    .frame(width: 190, height: 190)
                    .blur(radius: 6)
                    .offset(x: 155, y: -92)

                Circle()
                    .stroke(Color.white.opacity(0.18), lineWidth: 18)
                    .frame(width: 180, height: 180)
                    .offset(x: 150, y: 118)
            }
        )
        .clipShape(RoundedRectangle(cornerRadius: 32, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 32, style: .continuous)
                .stroke(Color.white.opacity(0.25), lineWidth: 1)
        )
        .shadow(color: Color(red: 0.02, green: 0.22, blue: 0.22).opacity(0.24), radius: 24, x: 0, y: 16)
    }

    private func heroBadge(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.caption2.weight(.bold))
                .foregroundStyle(.white.opacity(0.68))
                .lineLimit(1)
                .minimumScaleFactor(0.72)
            Text(value)
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.72)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.13), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .stroke(tint.opacity(0.20), lineWidth: 1)
        )
    }

    private var pipelineStoryCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Video flow")
                        .font(.caption.weight(.black))
                        .tracking(1.2)
                        .foregroundStyle(.secondary)
                    Text("One clean on-device pipeline")
                        .font(.headline.weight(.bold))
                }

                Spacer()

                Label("Cached after first run", systemImage: "icloud.and.arrow.down.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.teal)
                    .labelStyle(.titleAndIcon)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    pipelineStep(index: "1", title: "Scan", subtitle: "Vision OCR", icon: "doc.viewfinder", tint: .orange)
                    pipelineStep(index: "2", title: "De-ID", subtitle: piiModel.displayName, icon: "shield.lefthalf.filled", tint: .teal)
                    pipelineStep(index: "3", title: "Extract", subtitle: "GLiNER clinical labels", icon: "sparkles", tint: .blue)
                }

                VStack(spacing: 10) {
                    pipelineStep(index: "1", title: "Scan", subtitle: "Vision OCR", icon: "doc.viewfinder", tint: .orange)
                    pipelineStep(index: "2", title: "De-ID", subtitle: piiModel.displayName, icon: "shield.lefthalf.filled", tint: .teal)
                    pipelineStep(index: "3", title: "Extract", subtitle: "GLiNER clinical labels", icon: "sparkles", tint: .blue)
                }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.78), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private func pipelineStep(
        index: String,
        title: String,
        subtitle: String,
        icon: String,
        tint: Color
    ) -> some View {
        HStack(spacing: 10) {
            ZStack(alignment: .bottomTrailing) {
                Image(systemName: icon)
                    .font(.headline)
                    .foregroundStyle(tint)
                    .frame(width: 34, height: 34)
                    .background(tint.opacity(0.13), in: RoundedRectangle(cornerRadius: 12, style: .continuous))

                Text(index)
                    .font(.caption2.weight(.black))
                    .foregroundStyle(.white)
                    .frame(width: 16, height: 16)
                    .background(tint, in: Circle())
                    .offset(x: 3, y: 3)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.bold))
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .minimumScaleFactor(0.82)
            }

            Spacer(minLength: 0)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.07), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private var documentPreviewCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label(lastSource == .sample ? "Sample clinical note" : "Scanned clinical note", systemImage: "doc.text.image")
                    .font(.headline.weight(.bold))

                Spacer()

                Text(needsDocumentOCR ? "Ready for OCR" : "OCR complete")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background((needsDocumentOCR ? Color.orange : Color.teal).opacity(0.12))
                    .foregroundStyle(needsDocumentOCR ? .orange : .teal)
                    .clipShape(Capsule())
            }

            Text(
                needsDocumentOCR
                    ? "The document image is staged. Tap the bottom button when you are ready to run OCR, de-identification, and clinical extraction."
                    : "This is the source image behind the transcript below. Edit the OCR text if needed, then rerun the local pipeline."
            )
            .font(.subheadline)
            .foregroundStyle(.secondary)

            if let previewImage = documentImages.first {
                Image(uiImage: previewImage)
                    .resizable()
                    .scaledToFit()
                    .padding(8)
                    .background(Color.white, in: RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .stroke(Color.black.opacity(0.07), lineWidth: 1)
                    )
                    .shadow(color: Color.black.opacity(0.06), radius: 16, x: 0, y: 10)
            }

            if let sourceSummary {
                Text(sourceSummary)
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var extractedTextCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("OCR transcript", systemImage: "text.viewfinder")
                    .font(.headline.weight(.bold))

                Spacer()

                if let sourceSummary {
                    Text(sourceSummary)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(Color.gray.opacity(0.12))
                        .clipShape(Capsule())
                }
            }

            Text("A quick review checkpoint before masking. If OCR misreads a clinical term, correct it here and rerun.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            TextEditor(text: $extractedText)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 180)
                .padding(10)
                .background(Color(red: 0.98, green: 0.99, blue: 0.98), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(Color.black.opacity(0.06), lineWidth: 1)
                )
                .scrollContentBackground(.hidden)

            if trimmedText.isEmpty {
                Text("Start with a native scan or load the sample document below.")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var maskedOutputCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Safe note", systemImage: "theatermasks.fill")
                    .font(.headline.weight(.bold))

                Spacer()

                if let inferenceTime {
                    Text(String(format: "%.0f ms", inferenceTime * 1000))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }

            Text("Identifiers are replaced inline using \(piiEngine.title), keeping the clinical story readable without exposing patient details.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            MaskedDocumentView(text: extractedText, entities: entities)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(red: 0.98, green: 0.99, blue: 0.98), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.92), Color.teal.opacity(0.08)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 26, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.78), lineWidth: 1)
        )
    }

    private var piiEngineCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("PII engine", systemImage: "shield.lefthalf.filled")
                    .font(.headline.weight(.bold))

                Spacer()

                Text("Stage 1")
                    .font(.caption.weight(.black))
                    .tracking(0.8)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.teal.opacity(0.12), in: Capsule())
                    .foregroundStyle(.teal)
            }

            Text("Compare the dedicated OpenMed PII model against the GLiNER PII span model on the same scanned note.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Picker("PII engine", selection: $piiEngine) {
                ForEach(ScanDemoPIIEngine.allCases) { engine in
                    Text(engine.title).tag(engine)
                }
            }
            .pickerStyle(.segmented)

            HStack(alignment: .top, spacing: 10) {
                Image(systemName: piiEngine.systemImage)
                    .foregroundStyle(piiEngine.tint)
                    .frame(width: 28, height: 28)
                    .background(piiEngine.tint.opacity(0.13), in: RoundedRectangle(cornerRadius: 9, style: .continuous))

                VStack(alignment: .leading, spacing: 4) {
                    Text(piiEngine.detailTitle)
                        .font(.subheadline.weight(.bold))

                    Text(piiEngine.detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(12)
            .background(Color.black.opacity(0.035), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var glinerConfigurationCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("Clinical label pack", systemImage: "sparkles.rectangle.stack.fill")
                    .font(.headline.weight(.bold))

                Spacer()

                Button {
                    isShowingModelSetup = true
                } label: {
                    Label(modelsAreOfflineReady ? "Ready" : "Setup", systemImage: "slider.horizontal.3")
                        .labelStyle(.titleAndIcon)
                }
                .font(.caption.weight(.semibold))
                .buttonStyle(.bordered)
                .buttonBorderShape(.capsule)
                .tint(modelsAreOfflineReady ? .teal : .secondary)

                Text("60%+ confidence")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.11))
                    .foregroundStyle(.blue)
                    .clipShape(Capsule())
            }

            Text("After masking, GLiNER searches the safe note for clinical concepts that make the demo useful: symptoms, conditions, medications, plans, and follow-up context.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 108), spacing: 8)], spacing: 8) {
                ForEach(pipeline.focusLabels, id: \.self) { label in
                    Text(label)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            colorForEntityKey(entityCategory(for: label)).opacity(0.14),
                            in: RoundedRectangle(cornerRadius: 14, style: .continuous)
                        )
                        .foregroundStyle(colorForEntityKey(entityCategory(for: label)))
                }
            }

            if glinerCacheState == .ready {
                Label("GLiNER is cached. The second stage can run offline.", systemImage: "checkmark.shield.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.teal)
                    .padding(.vertical, 6)
            }

            Text("Runtime: \(piiEngine.title) + \(glinerModel.displayName)")
                .font(.caption.monospaced())
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var entityListCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Identifiers removed", systemImage: "person.text.rectangle.fill")
                    .font(.headline.weight(.bold))

                Spacer()

                Text("\(entities.count)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.white)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.teal, in: Capsule())
            }

            Text("Detected by \(piiEngine.title). Switch the PII engine above and rerun to compare coverage on the same note.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            ForEach(entities) { entity in
                EntityRow(entity: entity)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var noEntityCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("No PII Detected", systemImage: "checkmark.shield")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text("\(piiEngine.title) completed the local run but did not return any spans above the confidence threshold for the current OCR text.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var clinicalHighlightsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Clinical signals", systemImage: "waveform.path.ecg.text.page.fill")
                    .font(.headline.weight(.bold))

                Spacer()

                Text("\(focusEntities.count) found")
                    .font(.caption.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.blue.opacity(0.11))
                    .foregroundStyle(.blue)
                    .clipShape(Capsule())
            }

            Text("GLiNER runs on the de-identified note, proving the clinical extraction stage still works after direct identifiers are removed.")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            HighlightedTextView(text: focusSourceText, entities: focusEntities)
                .padding(14)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(red: 0.98, green: 0.99, blue: 0.98), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [Color.white.opacity(0.92), Color.blue.opacity(0.07)],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 26, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.78), lineWidth: 1)
        )
    }

    private var clinicalEntityListCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Label("Extracted clinical entities", systemImage: "sparkles")
                .font(.headline.weight(.bold))

            ForEach(focusEntities) { entity in
                EntityRow(entity: entity)
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private var noClinicalEntityCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("No GLiNER Matches", systemImage: "sparkles.slash")
                .font(.headline)
                .foregroundStyle(.secondary)

            Text("The native GLiNER pass completed on the masked note, but none of the predefined labels cleared the confidence threshold.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(.white.opacity(0.72), lineWidth: 1)
        )
    }

    private func errorCard(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.orange)

            Text(message)
                .font(.subheadline)
                .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.12), in: RoundedRectangle(cornerRadius: 20, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(Color.orange.opacity(0.22), lineWidth: 1)
        )
    }

    private var actionBar: some View {
        VStack(spacing: 10) {
            Text(actionHint)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: .infinity)

            Button {
                analyzeExtractedText()
            } label: {
                HStack(spacing: 10) {
                    if isBusy {
                        ProgressView()
                            .controlSize(.small)
                            .tint(.white)
                    } else {
                        Image(systemName: "sparkles.rectangle.stack.fill")
                    }

                    Text(primaryActionTitle)
                        .fontWeight(.bold)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 4)
            }
            .buttonStyle(.borderedProminent)
            .buttonBorderShape(.capsule)
            .tint(Color(red: 0.02, green: 0.38, blue: 0.35))
            .disabled(!canAnalyze)

            HStack(spacing: 10) {
                Button {
                    isShowingScanner = true
                } label: {
                    Label(scanButtonTitle, systemImage: "doc.viewfinder")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .buttonBorderShape(.capsule)
                .disabled(isBusy || !DocumentScannerSupport.isSupported)

                Button {
                    loadSampleDocument()
                } label: {
                    Label("Try Sample", systemImage: "doc.richtext")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .buttonBorderShape(.capsule)
                .disabled(isBusy)
            }
        }
        .padding(.horizontal)
        .padding(.top, 12)
        .padding(.bottom, 8)
        .background(.regularMaterial)
        .overlay(alignment: .top) {
            Divider()
        }
    }

    private var sourceSummary: String? {
        guard let lastSource else { return nil }

        switch lastSource {
        case .camera:
            let pageLabel = scannedPageCount == 1 ? "1 page" : "\(scannedPageCount) pages"
            return "Scanned \(pageLabel)"
        case .sample:
            return "Sample document"
        }
    }

    private func loadSampleDocument() {
        guard let image = UIImage(named: "SampleClinicalDocument") else {
            errorMessage = ScanDemoError.missingSampleDocument.errorDescription
            return
        }

        documentImages = [image]
        extractedText = ""
        scannedPageCount = 1
        lastSource = .sample
        needsDocumentOCR = true
        clearAnalysisState(keepError: false)
    }

    private func handleScannedPages(_ images: [UIImage]) {
        isShowingScanner = false

        guard !images.isEmpty else {
            return
        }

        documentImages = images
        extractedText = ""
        scannedPageCount = images.count
        lastSource = .camera
        needsDocumentOCR = true
        errorMessage = nil
        status = nil
        clearAnalysisState(keepError: false)
    }

    private func handleScannerError(_ error: Error) {
        isShowingScanner = false
        errorMessage = error.localizedDescription
    }

    private func prepareOfflineModels() {
        guard DemoPlatform.supportsOnDeviceMLX else {
            modelSetupError = ScanDemoError.unsupportedDevice.errorDescription
            return
        }

        modelSetupMessage = nil
        modelSetupError = nil
        isPreparingModels = true
        let setupToken = trimmedToken

        Task(priority: .utility) {
            do {
                try await ScanDemoRuntime.shared.prepareForOfflineUse(
                    pipeline: pipeline,
                    authToken: setupToken,
                    progress: { newStatus in
                        await MainActor.run {
                            modelSetupMessage = newStatus.detail
                        }
                    }
                )

                await MainActor.run {
                    isPreparingModels = false
                    huggingFaceToken = ""
                    modelSetupMessage = "Models are cached and ready for offline demo runs."
                }
            } catch {
                await MainActor.run {
                    isPreparingModels = false
                    modelSetupError = error.localizedDescription
                }
            }
        }
    }

    private func analyzeExtractedText() {
        guard DemoPlatform.supportsOnDeviceMLX else {
            errorMessage = ScanDemoError.unsupportedDevice.errorDescription
            return
        }

        let shouldRunOCR = needsDocumentOCR && hasDocumentSource
        let sourceImages = documentImages
        let originalText = extractedText

        guard shouldRunOCR || !originalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        errorMessage = nil
        inferenceTime = nil
        entities = []
        focusEntities = []
        focusSourceText = ""
        hasRunAnalysis = false
        isRecognizingText = shouldRunOCR
        isAnalyzing = !shouldRunOCR

        if shouldRunOCR {
            status = PipelineStatus(
                phase: .recognizing,
                detail: "Running Vision OCR on \(sourceImages.count) \(sourceImages.count == 1 ? "document" : "document pages")."
            )
        }

        Task(priority: .utility) {
            let start = CFAbsoluteTimeGetCurrent()
            var analysisText = originalText

            do {
                if shouldRunOCR {
                    let recognitionResult = try await DocumentTextRecognizer.extractText(from: sourceImages)
                    let recognizedText = recognitionResult.text.trimmingCharacters(in: .whitespacesAndNewlines)

                    guard !recognizedText.isEmpty else {
                        throw ScanDemoError.emptyRecognizedText
                    }

                    analysisText = recognizedText

                    await MainActor.run {
                        extractedText = recognizedText
                        scannedPageCount = recognitionResult.pageCount
                        needsDocumentOCR = false
                        isRecognizingText = false
                        isAnalyzing = true
                    }
                }

                await MainActor.run {
                    hasRunAnalysis = true
                }

                let results = try await ScanDemoRuntime.shared.analyze(
                    text: analysisText,
                    pipeline: pipeline,
                    piiEngine: piiEngine,
                    authToken: trimmedToken,
                    progress: { newStatus in
                        await MainActor.run {
                            status = newStatus
                        }
                    }
                )

                await MainActor.run {
                    isRecognizingText = false
                    isAnalyzing = false
                    status = nil

                    guard extractedText == analysisText else {
                        return
                    }

                    inferenceTime = CFAbsoluteTimeGetCurrent() - start
                    entities = results.piiEntities
                        .filter { entity in
                            entity.start >= 0 &&
                            entity.end > entity.start &&
                            entity.end <= analysisText.count
                        }
                        .sorted { lhs, rhs in
                            if lhs.start == rhs.start {
                                return lhs.end > rhs.end
                            }
                            return lhs.start < rhs.start
                        }
                    focusSourceText = results.maskedText
                    focusEntities = results.focusEntities
                        .filter { entity in
                            entity.start >= 0 &&
                            entity.end > entity.start &&
                            entity.end <= results.maskedText.count
                        }
                        .sorted { lhs, rhs in
                            if lhs.start == rhs.start {
                                return lhs.end > rhs.end
                            }
                            return lhs.start < rhs.start
                        }
                }
            } catch {
                await MainActor.run {
                    isRecognizingText = false
                    isAnalyzing = false
                    status = nil

                    guard analysisText.isEmpty || extractedText == analysisText else {
                        return
                    }

                    errorMessage = error.localizedDescription
                }
            }
        }
    }

    private func clearAnalysisState(keepError: Bool) {
        guard !isAnalyzing else { return }

        entities = []
        focusEntities = []
        focusSourceText = ""
        inferenceTime = nil
        status = nil
        hasRunAnalysis = false

        if !keepError {
            errorMessage = nil
        }
    }
}

private enum ScanDemoPIIEngine: String, CaseIterable, Identifiable, Sendable {
    case openMed
    case gliner

    var id: String { rawValue }

    var title: String {
        switch self {
        case .openMed:
            return "OpenMed PII"
        case .gliner:
            return "GLiNER PII"
        }
    }

    var detailTitle: String {
        switch self {
        case .openMed:
            return "Dedicated PII extractor"
        case .gliner:
            return "Zero-shot PII span model"
        }
    }

    var detail: String {
        switch self {
        case .openMed:
            return "Runs the OpenMed token-classification model plus the semantic merge rules used for production-style masking."
        case .gliner:
            return "Runs the GLiNER PII model with predefined identifier labels at 60%+ confidence before masking."
        }
    }

    var systemImage: String {
        switch self {
        case .openMed:
            return "checkmark.shield.fill"
        case .gliner:
            return "sparkles.rectangle.stack.fill"
        }
    }

    var tint: Color {
        switch self {
        case .openMed:
            return .teal
        case .gliner:
            return .blue
        }
    }
}

private struct ScanDemoModelDescriptor: Sendable {
    let displayName: String
    let sourceModelID: String
    let artifactRepoID: String
    let note: String
    let requiresAuthToken: Bool

    static let liteClinical = ScanDemoModelDescriptor(
        displayName: "LiteClinical Small",
        sourceModelID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1",
        artifactRepoID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx",
        note: "DistilBERT-family OpenMed PII model used here for a small, native on-device demo.",
        requiresAuthToken: false
    )

    static let glinerMultiPII = ScanDemoModelDescriptor(
        displayName: "GLiNER Multi PII v1",
        sourceModelID: "urchade/gliner_multi_pii-v1",
        artifactRepoID: "OpenMed/gliner-multi-pii-v1-mlx",
        note: "Zero-shot GLiNER span model used here for optional PII masking and second-stage clinical extraction.",
        requiresAuthToken: true
    )
}

private struct ScanDemoPipelineDescriptor: Sendable {
    let piiModel: ScanDemoModelDescriptor
    let glinerModel: ScanDemoModelDescriptor
    let piiLabels: [String]
    let piiGlinerThreshold: Float
    let focusLabels: [String]
    let glinerThreshold: Float

    static let defaultPipeline = ScanDemoPipelineDescriptor(
        piiModel: .liteClinical,
        glinerModel: .glinerMultiPII,
        piiLabels: [
            "person",
            "patient name",
            "doctor name",
            "organization",
            "hospital",
            "location",
            "address",
            "phone number",
            "email",
            "date",
            "date of birth",
            "age",
            "medical record number",
            "patient id",
            "insurance id",
            "account number",
            "social security number",
        ],
        piiGlinerThreshold: 0.6,
        focusLabels: [
            "symptom",
            "condition",
            "medical history",
            "medication",
            "dosage",
            "allergy",
            "treatment",
            "procedure",
            "follow-up plan",
            "care plan",
            "care setting",
            "work status",
        ],
        glinerThreshold: 0.6
    )
}

private struct ScanDemoAnalysisResult: Sendable {
    let piiEntities: [DetectedEntity]
    let maskedText: String
    let focusEntities: [DetectedEntity]
}

private enum PipelinePhase: Int, CaseIterable, Identifiable, Sendable {
    case recognizing
    case downloading
    case loading
    case inferencing

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .recognizing:
            return "OCR"
        case .downloading:
            return "Download"
        case .loading:
            return "Load"
        case .inferencing:
            return "Analyze"
        }
    }

    var headline: String {
        switch self {
        case .recognizing:
            return "Extracting text from the scan"
        case .downloading:
            return "Downloading the OpenMed artifacts"
        case .loading:
            return "Loading the local MLX runtimes"
        case .inferencing:
            return "Running de-identification and GLiNER"
        }
    }

    var systemImage: String {
        switch self {
        case .recognizing:
            return "text.viewfinder"
        case .downloading:
            return "arrow.down.circle.fill"
        case .loading:
            return "shippingbox.fill"
        case .inferencing:
            return "shield.lefthalf.filled"
        }
    }

    var tint: Color {
        switch self {
        case .recognizing:
            return .orange
        case .downloading:
            return .blue
        case .loading:
            return .teal
        case .inferencing:
            return .green
        }
    }
}

private struct PipelineStatus: Sendable {
    let phase: PipelinePhase
    let detail: String
}

private struct PipelineStatusCard: View {
    let status: PipelineStatus
    let pipeline: ScanDemoPipelineDescriptor

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: status.phase.systemImage)
                    .font(.title3.weight(.bold))
                    .foregroundStyle(status.phase.tint)
                    .frame(width: 44, height: 44)
                    .background(status.phase.tint.opacity(0.14), in: RoundedRectangle(cornerRadius: 16, style: .continuous))

                VStack(alignment: .leading, spacing: 6) {
                    Text("Pipeline running")
                        .font(.caption.weight(.black))
                        .tracking(1.1)
                        .foregroundStyle(.secondary)

                    Text(status.phase.headline)
                        .font(.headline.weight(.bold))

                    Text(status.detail)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                Spacer()

                ProgressView()
                    .tint(status.phase.tint)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 10) {
                    ForEach(PipelinePhase.allCases) { phase in
                        phaseTile(phase)
                    }
                }

                VStack(spacing: 10) {
                    ForEach(PipelinePhase.allCases) { phase in
                        phaseTile(phase)
                    }
                }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.white.opacity(0.86), in: RoundedRectangle(cornerRadius: 26, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 26, style: .continuous)
                .stroke(status.phase.tint.opacity(0.18), lineWidth: 1)
        )
    }

    private func phaseTile(_ phase: PipelinePhase) -> some View {
        let isCompleted = phase.rawValue < status.phase.rawValue
        let isActive = phase == status.phase

        return HStack(spacing: 10) {
            Image(systemName: isCompleted ? "checkmark.circle.fill" : phase.systemImage)
                .foregroundStyle(isCompleted ? Color.green : phase.tint)

            VStack(alignment: .leading, spacing: 2) {
                Text(phase.title)
                    .font(.subheadline.weight(.semibold))
                Text(isActive ? "Working" : (isCompleted ? "Done" : "Queued"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(isActive ? phase.tint.opacity(0.11) : Color.black.opacity(0.035))
        )
    }
}

private struct ModelSetupSheet: View {
    @Environment(\.dismiss) private var dismiss

    @Binding var token: String

    let piiCacheState: OpenMedMLXModelCacheState
    let glinerCacheState: OpenMedMLXModelCacheState
    let isPreparing: Bool
    let message: String?
    let errorMessage: String?
    let onPrepare: () -> Void

    private var trimmedToken: String {
        token.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var needsAccessToken: Bool {
        glinerCacheState != .ready
    }

    private var canPrepare: Bool {
        !isPreparing && (!needsAccessToken || !trimmedToken.isEmpty)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Offline model setup", systemImage: "shippingbox.and.arrow.backward.fill")
                            .font(.title3.weight(.bold))

                        Text("Use this once before filming. It prepares the local model cache so the visible demo can stay clean and run without network after setup.")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    VStack(spacing: 10) {
                        cacheRow(title: "OpenMed PII model", state: piiCacheState, tint: .teal)
                        cacheRow(title: "GLiNER model", state: glinerCacheState, tint: .blue)
                    }

                    if needsAccessToken {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Access token")
                                .font(.caption.weight(.bold))
                                .foregroundStyle(.secondary)

                            SecureField("hf_...", text: $token)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                                .privacySensitive()
                                .font(.caption.monospaced())
                                .padding(.horizontal, 12)
                                .padding(.vertical, 12)
                                .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                                        .stroke(Color.black.opacity(0.06), lineWidth: 1)
                                )

                            Text("Kept only for this app session. Once the artifacts are cached, you can run the demo offline without entering it again.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }

                    if let message {
                        Label(message, systemImage: "checkmark.circle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.teal)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.orange)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Button {
                        onPrepare()
                    } label: {
                        HStack {
                            if isPreparing {
                                ProgressView()
                                    .controlSize(.small)
                                    .tint(.white)
                            } else {
                                Image(systemName: "arrow.down.to.line.compact")
                            }

                            Text(isPreparing ? "Preparing Models..." : "Prepare Offline Models")
                                .fontWeight(.bold)
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .buttonBorderShape(.capsule)
                    .tint(Color(red: 0.02, green: 0.38, blue: 0.35))
                    .disabled(!canPrepare)
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("Setup")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func cacheRow(
        title: String,
        state: OpenMedMLXModelCacheState,
        tint: Color
    ) -> some View {
        HStack(spacing: 12) {
            Image(systemName: state == .ready ? "checkmark.shield.fill" : "shippingbox.fill")
                .foregroundStyle(tint)
                .frame(width: 32, height: 32)
                .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.subheadline.weight(.semibold))
                Text(cacheStateDescription(state))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(cacheStateBadge(state))
                .font(.caption2.weight(.black))
                .padding(.horizontal, 9)
                .padding(.vertical, 5)
                .background(cacheStateTint(state).opacity(0.12), in: Capsule())
                .foregroundStyle(cacheStateTint(state))
        }
        .padding(12)
        .background(Color(.secondarySystemGroupedBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }

    private func cacheStateBadge(_ state: OpenMedMLXModelCacheState) -> String {
        switch state {
        case .ready:
            return "READY"
        case .partial:
            return "PARTIAL"
        case .missing:
            return "NEEDED"
        }
    }

    private func cacheStateDescription(_ state: OpenMedMLXModelCacheState) -> String {
        switch state {
        case .ready:
            return "Available from local cache"
        case .partial:
            return "Cache exists but needs completion"
        case .missing:
            return "Not cached on this device yet"
        }
    }

    private func cacheStateTint(_ state: OpenMedMLXModelCacheState) -> Color {
        switch state {
        case .ready:
            return .teal
        case .partial:
            return .orange
        case .missing:
            return .secondary
        }
    }
}

private actor ScanDemoRuntime {
    static let shared = ScanDemoRuntime()

    private let blockingWorkQueue = DispatchQueue(
        label: "openmed.scan-demo.runtime",
        qos: .utility
    )
    private var piiRuntimes: [String: OpenMed] = [:]
    private var glinerRuntimes: [String: OpenMedZeroShotNER] = [:]

    func prepareForOfflineUse(
        pipeline: ScanDemoPipelineDescriptor,
        authToken: String?,
        progress: @escaping @Sendable (PipelineStatus) async -> Void
    ) async throws {
        guard DemoPlatform.supportsOnDeviceMLX else {
            throw ScanDemoError.unsupportedDevice
        }

        _ = try await loadPIIRuntime(model: pipeline.piiModel, progress: progress)
        _ = try await loadGLiNERRuntime(
            model: pipeline.glinerModel,
            authToken: authToken,
            progress: progress
        )
    }

    func analyze(
        text: String,
        pipeline: ScanDemoPipelineDescriptor,
        piiEngine: ScanDemoPIIEngine,
        authToken: String?,
        progress: @escaping @Sendable (PipelineStatus) async -> Void
    ) async throws -> ScanDemoAnalysisResult {
        guard DemoPlatform.supportsOnDeviceMLX else {
            throw ScanDemoError.unsupportedDevice
        }

        var glinerForClinicalStage: OpenMedZeroShotNER?
        let piiEntities: [DetectedEntity]

        switch piiEngine {
        case .openMed:
            let openmed = try await loadPIIRuntime(model: pipeline.piiModel, progress: progress)

            await progress(
                PipelineStatus(
                    phase: .inferencing,
                    detail: "Running OpenMed token classification and semantic PII merging on-device."
                )
            )

            piiEntities = try await runBlockingWork {
                try openmed.extractPII(text).map { prediction in
                    DetectedEntity(
                        label: prediction.label,
                        text: prediction.text,
                        confidence: prediction.confidence,
                        start: prediction.start,
                        end: prediction.end,
                        category: entityCategory(for: prediction.label)
                    )
                }
            }

        case .gliner:
            let gliner = try await loadGLiNERRuntime(
                model: pipeline.glinerModel,
                authToken: authToken,
                progress: progress
            )
            glinerForClinicalStage = gliner

            await progress(
                PipelineStatus(
                    phase: .inferencing,
                    detail: "Running GLiNER PII zero-shot extraction on the original note at 60%+ confidence."
                )
            )

            let glinerPIIEntities = try await extractChunkedGLiNEREntities(
                from: text,
                labels: pipeline.piiLabels,
                threshold: pipeline.piiGlinerThreshold,
                gliner: gliner
            ).map { entity in
                DetectedEntity(
                    label: entity.label,
                    text: entity.text,
                    confidence: entity.score,
                    start: entity.start,
                    end: entity.end,
                    category: entityCategory(for: entity.label)
                )
            }
            piiEntities = suppressOverlappingPIIEntities(glinerPIIEntities)
        }

        let maskedText = maskedTextByReplacingEntities(in: text, entities: piiEntities)
        let gliner: OpenMedZeroShotNER
        if let glinerForClinicalStage {
            gliner = glinerForClinicalStage
        } else {
            gliner = try await loadGLiNERRuntime(
                model: pipeline.glinerModel,
                authToken: authToken,
                progress: progress
            )
        }

        await progress(
            PipelineStatus(
                phase: .inferencing,
                detail: "Running native GLiNER zero-shot NER on the masked note with section-aware chunks to recover more clinical signals."
            )
        )

        let focusEntities = try await extractChunkedGLiNEREntities(
            from: maskedText,
            labels: pipeline.focusLabels,
            threshold: pipeline.glinerThreshold,
            gliner: gliner
        ).map { entity in
            DetectedEntity(
                label: entity.label,
                text: entity.text,
                confidence: entity.score,
                start: entity.start,
                end: entity.end,
                category: entityCategory(for: entity.label)
            )
        }

        return ScanDemoAnalysisResult(
            piiEntities: piiEntities,
            maskedText: maskedText,
            focusEntities: focusEntities
        )
    }

    private func loadPIIRuntime(
        model: ScanDemoModelDescriptor,
        progress: @escaping @Sendable (PipelineStatus) async -> Void
    ) async throws -> OpenMed {
        if let piiRuntime = piiRuntimes[model.artifactRepoID] {
            return piiRuntime
        }

        let preparedArtifact = try await prepareModelDirectory(
            model: model,
            authToken: nil,
            progress: progress
        )
        await progress(
            PipelineStatus(
                phase: .loading,
                detail: preparedArtifact.downloadedNow
                    ? "Preparing tokenizer assets and initializing the OpenMed PII runtime inside OpenMedKit."
                    : "Using the cached \(model.displayName) artifact and initializing the OpenMed PII runtime."
            )
        )
        let runtime = try await runBlockingWork {
            try OpenMed(backend: .mlx(modelDirectoryURL: preparedArtifact.directory))
        }
        piiRuntimes[model.artifactRepoID] = runtime
        return runtime
    }

    private func loadGLiNERRuntime(
        model: ScanDemoModelDescriptor,
        authToken: String?,
        progress: @escaping @Sendable (PipelineStatus) async -> Void
    ) async throws -> OpenMedZeroShotNER {
        if let glinerRuntime = glinerRuntimes[model.artifactRepoID] {
            return glinerRuntime
        }

        let normalizedToken = authToken?.trimmingCharacters(in: .whitespacesAndNewlines)
        let cacheState = try OpenMedModelStore.mlxModelCacheState(
            repoID: model.artifactRepoID,
            revision: "main"
        )
        if model.requiresAuthToken && cacheState != .ready && (normalizedToken?.isEmpty ?? true) {
            throw ScanDemoError.missingAuthToken(model.artifactRepoID)
        }

        let preparedArtifact = try await prepareModelDirectory(
            model: model,
            authToken: normalizedToken,
            progress: progress
        )
        await progress(
            PipelineStatus(
                phase: .loading,
                detail: preparedArtifact.downloadedNow
                    ? "Preparing tokenizer assets and initializing the native GLiNER runtime in OpenMedKit."
                    : "Using the cached \(model.displayName) artifact and initializing the native GLiNER runtime."
            )
        )
        let runtime = try await runBlockingWork {
            try OpenMedZeroShotNER(modelDirectoryURL: preparedArtifact.directory)
        }
        glinerRuntimes[model.artifactRepoID] = runtime
        return runtime
    }

    private func extractChunkedGLiNEREntities(
        from text: String,
        labels: [String],
        threshold: Float,
        gliner: OpenMedZeroShotNER
    ) async throws -> [OpenMedZeroShotEntity] {
        let chunks = makeGLiNERChunks(from: text)
        var mergedEntities: [ChunkedGLiNEREntityKey: OpenMedZeroShotEntity] = [:]

        for chunk in chunks {
            let predictions = try await runBlockingWork {
                try gliner.extract(
                    chunk.text,
                    labels: labels,
                    threshold: threshold,
                    flatNER: true
                )
            }

            for prediction in predictions {
                let rebasedEntity = OpenMedZeroShotEntity(
                    text: prediction.text,
                    label: prediction.label,
                    score: prediction.score,
                    start: prediction.start + chunk.offset,
                    end: prediction.end + chunk.offset
                )
                let key = ChunkedGLiNEREntityKey(
                    label: rebasedEntity.label,
                    start: rebasedEntity.start,
                    end: rebasedEntity.end,
                    normalizedText: rebasedEntity.text.lowercased()
                )
                if let existing = mergedEntities[key] {
                    if rebasedEntity.score > existing.score {
                        mergedEntities[key] = rebasedEntity
                    }
                } else {
                    mergedEntities[key] = rebasedEntity
                }
            }
        }

        return suppressOverlappingChunkedEntities(Array(mergedEntities.values))
    }

    private func suppressOverlappingPIIEntities(
        _ entities: [DetectedEntity]
    ) -> [DetectedEntity] {
        let sorted = entities.sorted { lhs, rhs in
            if lhs.confidence != rhs.confidence {
                return lhs.confidence > rhs.confidence
            }
            let lhsLength = lhs.end - lhs.start
            let rhsLength = rhs.end - rhs.start
            if lhsLength != rhsLength {
                return lhsLength > rhsLength
            }
            if lhs.start != rhs.start {
                return lhs.start < rhs.start
            }
            return lhs.end < rhs.end
        }

        var kept = [DetectedEntity]()

        for entity in sorted {
            let overlapsExisting = kept.contains { existing in
                max(existing.start, entity.start) < min(existing.end, entity.end)
            }
            if !overlapsExisting {
                kept.append(entity)
            }
        }

        return kept.sorted { lhs, rhs in
            if lhs.start == rhs.start {
                return lhs.end > rhs.end
            }
            return lhs.start < rhs.start
        }
    }

    private func runBlockingWork<T>(
        _ work: @escaping () throws -> T
    ) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            blockingWorkQueue.async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func suppressOverlappingChunkedEntities(
        _ entities: [OpenMedZeroShotEntity]
    ) -> [OpenMedZeroShotEntity] {
        let sorted = entities.sorted { lhs, rhs in
            if lhs.label != rhs.label {
                return lhs.label < rhs.label
            }
            if lhs.score != rhs.score {
                return lhs.score > rhs.score
            }
            let lhsLength = lhs.end - lhs.start
            let rhsLength = rhs.end - rhs.start
            if lhsLength != rhsLength {
                return lhsLength > rhsLength
            }
            if lhs.start != rhs.start {
                return lhs.start < rhs.start
            }
            return lhs.end < rhs.end
        }

        var kept = [OpenMedZeroShotEntity]()

        for entity in sorted {
            let overlapsExisting = kept.contains { existing in
                existing.label == entity.label &&
                max(existing.start, entity.start) < min(existing.end, entity.end)
            }
            if !overlapsExisting {
                kept.append(entity)
            }
        }

        return kept.sorted { lhs, rhs in
            if lhs.start == rhs.start {
                if lhs.end == rhs.end {
                    if lhs.score == rhs.score {
                        return lhs.label < rhs.label
                    }
                    return lhs.score > rhs.score
                }
                return lhs.end > rhs.end
            }
            return lhs.start < rhs.start
        }
    }

    private func prepareModelDirectory(
        model: ScanDemoModelDescriptor,
        authToken: String?,
        progress: @escaping @Sendable (PipelineStatus) async -> Void
    ) async throws -> PreparedArtifact {
        let cacheState = try OpenMedModelStore.mlxModelCacheState(
            repoID: model.artifactRepoID,
            revision: "main"
        )

        if cacheState == .ready {
            return PreparedArtifact(
                directory: try OpenMedModelStore.cachedMLXModelDirectory(
                    repoID: model.artifactRepoID,
                    revision: "main"
                ),
                downloadedNow: false
            )
        }

        let progressDetail: String
        switch cacheState {
        case .missing:
            progressDetail = "Preparing \(model.displayName) and saving it in the local OpenMed cache."
        case .partial:
            progressDetail = "Finishing the cached \(model.displayName) download by fetching only the missing artifact files."
        case .ready:
            progressDetail = "Using cached \(model.displayName)."
        }

        await progress(
            PipelineStatus(
                phase: .downloading,
                detail: progressDetail
            )
        )
        return PreparedArtifact(
            directory: try await OpenMedModelStore.downloadMLXModel(
                repoID: model.artifactRepoID,
                authToken: authToken,
                revision: "main"
            ),
            downloadedNow: true
        )
    }
}

private struct GLiNERTextChunk: Sendable {
    let offset: Int
    let text: String
}

private struct ChunkedGLiNEREntityKey: Hashable, Sendable {
    let label: String
    let start: Int
    let end: Int
    let normalizedText: String
}

private struct PreparedArtifact: Sendable {
    let directory: URL
    let downloadedNow: Bool
}

private struct DocumentRecognitionResult: Sendable {
    let text: String
    let pageCount: Int
}

private enum DocumentTextRecognizer {
    static func extractText(from images: [UIImage]) async throws -> DocumentRecognitionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    let pageTexts = try images.enumerated().map { _, image in
                        try recognizeText(in: image)
                    }

                    let combinedText = pageTexts
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }
                        .joined(separator: "\n\n")
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                    continuation.resume(
                        returning: DocumentRecognitionResult(
                            text: combinedText,
                            pageCount: images.count
                        )
                    )
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private static func recognizeText(in image: UIImage) throws -> String {
        guard let cgImage = image.cgImage else {
            throw ScanDemoError.invalidScanImage
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.automaticallyDetectsLanguage = true

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let observations = (request.results ?? []).sorted(by: recognitionSort)
        let lines = observations.compactMap { observation -> String? in
            observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return lines
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func recognitionSort(
        lhs: VNRecognizedTextObservation,
        rhs: VNRecognizedTextObservation
    ) -> Bool {
        let verticalDelta = lhs.boundingBox.maxY - rhs.boundingBox.maxY

        if abs(verticalDelta) > 0.02 {
            return verticalDelta > 0
        }

        return lhs.boundingBox.minX < rhs.boundingBox.minX
    }
}

private enum ScanSource {
    case camera
    case sample
}

private enum ScanDemoError: LocalizedError {
    case unsupportedDevice
    case invalidScanImage
    case emptyRecognizedText
    case missingSampleDocument
    case missingAuthToken(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedDevice:
            return "This demo needs a real iPhone or iPad for the local MLX path. iOS Simulator is not supported for end-to-end inference."
        case .invalidScanImage:
            return "One of the scanned pages could not be converted into an OCR image."
        case .emptyRecognizedText:
            return "The document loaded correctly, but Vision did not extract readable text from it."
        case .missingSampleDocument:
            return "The bundled sample document asset could not be loaded."
        case .missingAuthToken(_):
            return "Model setup is not complete on this device yet. Cache the GLiNER artifact once before recording, then the demo can run offline."
        }
    }
}

private enum ScanDemoSecrets {
    static var huggingFaceToken: String {
        let candidates = [
            "OPENMED_HF_TOKEN",
            "OPENMED_HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACE_HUB_TOKEN",
        ]

        for name in candidates {
            let token = ProcessInfo.processInfo.environment[name]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !token.isEmpty {
                return token
            }
        }

        return ""
    }
}

private enum DemoPlatform {
    static var supportsOnDeviceMLX: Bool {
        #if targetEnvironment(simulator)
        false
        #elseif arch(arm64) || arch(arm64e)
        true
        #else
        false
        #endif
    }
}

private enum DocumentScannerSupport {
    static var isSupported: Bool {
        #if canImport(UIKit) && canImport(VisionKit)
        VNDocumentCameraViewController.isSupported
        #else
        false
        #endif
    }
}

struct DetectedEntity: Identifiable, Sendable {
    let id = UUID()
    let label: String
    let text: String
    let confidence: Float
    let start: Int
    let end: Int
    let category: String

    var color: Color {
        colorForEntityKey(category)
    }

    var maskedToken: String {
        "[\(category.replacingOccurrences(of: " ", with: "_"))]"
    }
}

struct HighlightedTextView: View {
    let text: String
    let entities: [DetectedEntity]

    var body: some View {
        Text(highlightedString)
            .font(.body.monospaced())
            .textSelection(.enabled)
    }

    private var highlightedString: AttributedString {
        var attributed = AttributedString(text)
        let textLength = text.count
        let sortedEntities = entities.sorted { $0.start > $1.start }

        for entity in sortedEntities {
            let safeStart = min(max(entity.start, 0), textLength)
            let safeEnd = min(max(entity.end, safeStart), textLength)

            guard safeStart < safeEnd else {
                continue
            }

            let startIndex = attributed.index(attributed.startIndex, offsetByCharacters: safeStart)
            let endIndex = attributed.index(attributed.startIndex, offsetByCharacters: safeEnd)
            let range = startIndex..<endIndex

            attributed[range].backgroundColor = entity.color.opacity(0.22)
            attributed[range].foregroundColor = entity.color
            attributed[range].font = .body.monospaced().bold()
        }

        return attributed
    }
}

struct MaskedDocumentView: View {
    let text: String
    let entities: [DetectedEntity]

    var body: some View {
        Text(maskedString)
            .font(.body.monospaced())
            .textSelection(.enabled)
    }

    private var maskedString: AttributedString {
        var output = AttributedString()
        let sortedEntities = entities.sorted { lhs, rhs in
            if lhs.start == rhs.start {
                return lhs.end > rhs.end
            }
            return lhs.start < rhs.start
        }

        var cursor = 0

        for entity in sortedEntities {
            let safeStart = min(max(entity.start, 0), text.count)
            let safeEnd = min(max(entity.end, safeStart), text.count)

            guard safeStart >= cursor, safeStart < safeEnd else {
                continue
            }

            output += AttributedString(text.substring(from: cursor, to: safeStart))

            var token = AttributedString(" \(entity.maskedToken) ")
            token.foregroundColor = .white
            token.backgroundColor = entity.color
            token.font = .body.monospaced().bold()
            output += token

            cursor = safeEnd
        }

        output += AttributedString(text.substring(from: cursor, to: text.count))
        return output
    }
}

struct EntityRow: View {
    let entity: DetectedEntity

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 4)
                .fill(entity.color)
                .frame(width: 5, height: 48)

            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(entity.maskedToken)
                        .font(.caption.weight(.bold))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 5)
                        .background(entity.color.opacity(0.16))
                        .foregroundStyle(entity.color)
                        .clipShape(Capsule())

                    Text(entity.label)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                }

                Text(entity.text)
                    .font(.body.weight(.semibold))

                Text("\(String(format: "%.0f%%", entity.confidence * 100)) confidence")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Spacer()
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.82), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(Color.black.opacity(0.04), lineWidth: 1)
        )
    }
}

private func entityCategory(for label: String) -> String {
    let normalized = label.lowercased()

    if normalized.contains("record")
        || normalized.contains("insurance")
        || normalized.contains("license")
        || normalized.contains("passport")
        || normalized.contains("employee")
        || normalized.contains("account")
        || normalized.contains("routing")
        || normalized.contains("policy")
        || normalized.contains("id")
    {
        return "ID"
    }
    if normalized.contains("name")
        || normalized == "person"
        || normalized == "patient"
        || normalized.contains("doctor")
        || normalized.contains("provider")
        || normalized.contains("clinician")
    {
        return "NAME"
    }
    if normalized.contains("age") {
        return "AGE"
    }
    if normalized.contains("date") || normalized.contains("dob") {
        return "DATE"
    }
    if normalized.contains("phone") || normalized.contains("fax") {
        return "PHONE"
    }
    if normalized.contains("ssn") {
        return "SSN"
    }
    if normalized.contains("email") {
        return "EMAIL"
    }
    if normalized.contains("address")
        || normalized.contains("location")
        || normalized.contains("city")
        || normalized.contains("state")
        || normalized.contains("postcode")
        || normalized.contains("postal")
    {
        return "ADDRESS"
    }
    if normalized.contains("organization")
        || normalized.contains("employer")
        || normalized.contains("company")
    {
        return "ORG"
    }
    return normalized.uppercased()
}

private func colorForEntityKey(_ key: String) -> Color {
    switch key {
    case "NAME": return .blue
    case "DATE": return .purple
    case "PHONE": return .green
    case "SSN": return .red
    case "ADDRESS": return .orange
    case "EMAIL": return .mint
    case "AGE": return .cyan
    case "ID": return .pink
    case "ORG": return .indigo
    case "SYMPTOM": return .red
    case "CONDITION": return .orange
    case "MEDICATION": return .blue
    case "DOSAGE": return .teal
    case "TREATMENT": return .green
    case "CARE PLAN": return .cyan
    case "CARE SETTING": return .brown
    case "WORK STATUS": return .purple
    default:
        let palette: [Color] = [.blue, .teal, .mint, .indigo, .orange, .pink, .cyan, .brown]
        let seed = key.unicodeScalars.reduce(0) { partial, scalar in
            (partial * 31 + Int(scalar.value)) % 997
        }
        return palette[abs(seed) % palette.count]
    }
}

private func maskedTextByReplacingEntities(in text: String, entities: [DetectedEntity]) -> String {
    let sortedEntities = entities.sorted { lhs, rhs in
        if lhs.start == rhs.start {
            return lhs.end > rhs.end
        }
        return lhs.start < rhs.start
    }

    var output = ""
    var cursor = 0

    for entity in sortedEntities {
        let safeStart = min(max(entity.start, 0), text.count)
        let safeEnd = min(max(entity.end, safeStart), text.count)

        guard safeStart >= cursor, safeStart < safeEnd else {
            continue
        }

        output += text.substring(from: cursor, to: safeStart)
        output += " \(entity.maskedToken) "
        cursor = safeEnd
    }

    output += text.substring(from: cursor, to: text.count)
    return output
}

private func makeGLiNERChunks(from text: String) -> [GLiNERTextChunk] {
    let trimmedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmedText.isEmpty else {
        return []
    }

    var chunks = [GLiNERTextChunk(offset: 0, text: text)]
    let sectionRanges = clinicalSectionRanges(in: text)

    for range in sectionRanges {
        let sectionText = text.substring(from: range.lowerBound, to: range.upperBound)
        let sectionChunks = splitChunkText(
            sectionText,
            baseOffset: range.lowerBound,
            preferredWindowLength: 700,
            overlapLength: 120
        )
        chunks.append(contentsOf: sectionChunks)
    }

    if sectionRanges.isEmpty {
        chunks.append(
            contentsOf: splitChunkText(
                text,
                baseOffset: 0,
                preferredWindowLength: 700,
                overlapLength: 120
            )
        )
    }

    var dedupedChunks = [GLiNERTextChunk]()
    var seen = Set<String>()
    for chunk in chunks {
        let normalizedText = chunk.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedText.isEmpty else {
            continue
        }
        let key = "\(chunk.offset)|\(normalizedText)"
        if seen.insert(key).inserted {
            dedupedChunks.append(GLiNERTextChunk(offset: chunk.offset, text: normalizedText))
        }
    }
    return dedupedChunks
}

private func clinicalSectionRanges(in text: String) -> [Range<Int>] {
    let headings = findClinicalSectionHeadings(in: text)
    guard !headings.isEmpty else {
        return []
    }

    var ranges = [Range<Int>]()
    for (index, heading) in headings.enumerated() {
        let start = heading.offset
        let end = index + 1 < headings.count ? headings[index + 1].offset : text.count
        if start < end {
            ranges.append(start..<end)
        }
    }
    return ranges
}

private func findClinicalSectionHeadings(in text: String) -> [(title: String, offset: Int)] {
    let headingKeywords = [
        "CHIEF",
        "HISTORY",
        "MEDICATION",
        "ALLERGY",
        "EXAM",
        "VITAL",
        "LAB",
        "PROCEDURE",
        "ASSESSMENT",
        "PLAN",
        "FOLLOW-UP",
        "WORK STATUS",
    ]

    var headings = [(title: String, offset: Int)]()
    var cursor = 0

    for line in text.components(separatedBy: .newlines) {
        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
        defer {
            cursor += line.count + 1
        }

        guard !trimmed.isEmpty else {
            continue
        }

        let normalized = trimmed.uppercased()
        let containsKeyword = headingKeywords.contains { normalized.contains($0) }
        let wordCount = trimmed.split(whereSeparator: \.isWhitespace).count
        let letters = trimmed.filter(\.isLetter)
        let uppercaseLetters = trimmed.filter { $0.isLetter && $0.isUppercase }
        let isUppercaseLine = !letters.isEmpty && uppercaseLetters.count * 100 >= letters.count * 85

        if containsKeyword && isUppercaseLine && wordCount <= 6 {
            headings.append((title: trimmed, offset: cursor))
        }
    }

    return headings
}

private func splitChunkText(
    _ text: String,
    baseOffset: Int,
    preferredWindowLength: Int,
    overlapLength: Int
) -> [GLiNERTextChunk] {
    let normalizedText = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard normalizedText.count > preferredWindowLength else {
        return [GLiNERTextChunk(offset: baseOffset, text: text)]
    }

    var chunks = [GLiNERTextChunk]()
    var windowStart = 0
    let safeOverlap = max(0, min(overlapLength, preferredWindowLength / 2))

    while windowStart < text.count {
        let remaining = text.count - windowStart
        if remaining <= preferredWindowLength {
            let chunkText = text.substring(from: windowStart, to: text.count)
            chunks.append(GLiNERTextChunk(offset: baseOffset + windowStart, text: chunkText))
            break
        }

        var windowEnd = min(text.count, windowStart + preferredWindowLength)
        let searchStart = max(windowStart, windowEnd - 120)
        if let breakIndex = text.lastIndex(
            where: { $0 == "\n" || $0 == "." },
            from: searchStart,
            to: windowEnd
        ) {
            windowEnd = breakIndex + 1
        }

        let chunkText = text.substring(from: windowStart, to: windowEnd)
        chunks.append(GLiNERTextChunk(offset: baseOffset + windowStart, text: chunkText))

        let nextStart = max(windowStart + 1, windowEnd - safeOverlap)
        if nextStart <= windowStart {
            break
        }
        windowStart = nextStart
    }

    return chunks
}

private extension String {
    func substring(from start: Int, to end: Int) -> String {
        guard start < end else { return "" }

        let safeStart = index(startIndex, offsetBy: max(0, min(start, count)))
        let safeEnd = index(startIndex, offsetBy: max(0, min(end, count)))
        return String(self[safeStart..<safeEnd])
    }

    func lastIndex(
        where predicate: (Character) -> Bool,
        from start: Int,
        to end: Int
    ) -> Int? {
        guard start < end else {
            return nil
        }

        let safeStart = max(0, min(start, count))
        let safeEnd = max(safeStart, min(end, count))
        guard safeStart < safeEnd else {
            return nil
        }

        var currentIndex = index(startIndex, offsetBy: safeEnd)
        let lowerBound = index(startIndex, offsetBy: safeStart)

        while currentIndex > lowerBound {
            let previousIndex = index(before: currentIndex)
            if predicate(self[previousIndex]) {
                return distance(from: startIndex, to: previousIndex)
            }
            currentIndex = previousIndex
        }

        return nil
    }
}

#if canImport(UIKit) && canImport(VisionKit)
private struct DocumentScannerSheet: UIViewControllerRepresentable {
    let onComplete: ([UIImage]) -> Void
    let onCancel: () -> Void
    let onError: (Error) -> Void

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let controller = VNDocumentCameraViewController()
        controller.delegate = context.coordinator
        return controller
    }

    func updateUIViewController(_ uiViewController: VNDocumentCameraViewController, context: Context) {}

    final class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        private let parent: DocumentScannerSheet

        init(_ parent: DocumentScannerSheet) {
            self.parent = parent
        }

        func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) {
            parent.onCancel()
        }

        func documentCameraViewController(
            _ controller: VNDocumentCameraViewController,
            didFailWithError error: Error
        ) {
            parent.onError(error)
        }

        func documentCameraViewController(
            _ controller: VNDocumentCameraViewController,
            didFinishWith scan: VNDocumentCameraScan
        ) {
            let images = (0..<scan.pageCount).map { scan.imageOfPage(at: $0) }
            parent.onComplete(images)
        }
    }
}
#else
private struct DocumentScannerSheet: View {
    let onComplete: ([UIImage]) -> Void
    let onCancel: () -> Void
    let onError: (Error) -> Void

    var body: some View {
        Text("Document scanning is unavailable on this platform.")
    }
}
#endif

#Preview {
    ContentView()
}
