import OpenMedKit
import SwiftUI

struct ContentView: View {
    private static let showcaseSampleText = """
    Patient: John Doe, DOB: 01/15/1970, SSN: 000-00-0000, MRN: MRN-TEST-88421, Address: 123 Example Street, Apt 4B, Springfield, CA 90000, Phone: (555) 010-2244, Email: john.doe@example.test, Insurance ID: TEST-POLICY-778291, Driver License: DLT-TEST-442190, Passport: P-TEST-998877, Emergency Contact: Jane Doe, (555) 010-7788, Employer: Example Manufacturing LLC, Employee ID: EMP-20481, Bank Account: ACCT-TEST-55667788, Routing: 000000000.
    """

    @AppStorage("openmed.demo.hfToken") private var huggingFaceToken = ""
    @State private var inputText = Self.showcaseSampleText
    @State private var availableModels: [DemoModelDescriptor] = [.missingBundledModel]
    @State private var selectedModelID = DemoModelDescriptor.missingBundledModel.id
    @State private var entities: [DetectedEntity] = []
    @State private var isAnalyzing = false
    @State private var analysisStatus: DemoAnalysisStatus?
    @State private var isShowingModelPicker = false
    @State private var errorMessage: String?
    @State private var inferenceTime: Double?
    @State private var didAutostartShowcase = false

    private enum AnalysisInvalidationReason {
        case textChanged
        case modelChanged
    }

    private var selectedModel: DemoModelDescriptor {
        availableModels.first(where: { $0.id == selectedModelID }) ?? .missingBundledModel
    }

    private var canAnalyzeSelectedModel: Bool {
        selectedModel.isRunnableInDemoApp
    }

    private var analyzeButtonTitle: String {
        if isAnalyzing {
            return analysisStatus?.phase.buttonTitle ?? "Preparing Run..."
        }
        switch selectedModel.runtimeKind {
        case .bundled:
            return canAnalyzeSelectedModel ? "Detect PII Entities" : "Add Bundled CoreML Model"
        case .mlx:
            return canAnalyzeSelectedModel ? "Download & Detect PII" : "Run on Apple Silicon Device"
        case .unavailable:
            return "Add a Supported Model"
        }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Label("Model", systemImage: "shippingbox")
                            .font(.headline)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Button("Choose Model") {
                            refreshAvailableModels()
                            isShowingModelPicker = true
                        }
                        .buttonStyle(.bordered)
                        .disabled(isAnalyzing)
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .firstTextBaseline, spacing: 10) {
                            Text(selectedModel.displayName)
                                .font(.title3.weight(.semibold))

                            Text(selectedModel.badgeText)
                                .font(.caption.weight(.semibold))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(selectedModel.badgeColor.opacity(0.14))
                                .foregroundStyle(selectedModel.badgeColor)
                                .clipShape(Capsule())
                        }

                        Text(selectedModel.sourceModelID)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)

                        if let artifactRepoID = selectedModel.artifactRepoID {
                            Text(artifactRepoID)
                                .font(.caption2)
                                .foregroundStyle(.tertiary)
                                .textSelection(.enabled)
                        }

                        Text(selectedModel.detailText)
                            .font(.caption)
                            .foregroundStyle(.tertiary)

                        if selectedModel.requiresAuthToken {
                            HStack(spacing: 8) {
                                Image(systemName: huggingFaceToken.isEmpty ? "key.horizontal" : "checkmark.circle.fill")
                                    .foregroundStyle(huggingFaceToken.isEmpty ? .orange : .green)

                                Text(
                                    huggingFaceToken.isEmpty
                                        ? "Manage the private Hugging Face access token in Choose Model."
                                        : "Private Hugging Face token is saved on this device."
                                )
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                            }
                            .padding(.top, 2)
                        } else if selectedModel.runtimeKind == .mlx {
                            HStack(spacing: 8) {
                                Image(systemName: "globe.badge.chevron.backward")
                                    .foregroundStyle(.mint)

                                Text("Public Hugging Face artifact. No token required.")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.top, 2)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color.gray.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                    if availableModels.filter(\.isBundled).isEmpty {
                        Text("This app build does not currently include a bundled CoreML model. You can still test the published OpenMed MLX artifacts on Apple Silicon macOS or a physical iPhone/iPad by selecting an MLX model below.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                VStack(alignment: .leading, spacing: 8) {
                    Label("Clinical Note", systemImage: "doc.text")
                        .font(.headline)
                        .foregroundStyle(.secondary)

                    TextEditor(text: $inputText)
                        .font(.body.monospaced())
                        .frame(minHeight: 120, maxHeight: 200)
                        .padding(8)
                        .background(Color.gray.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if let analysisStatus {
                    AnalysisStatusCard(status: analysisStatus)
                        .transition(.move(edge: .top).combined(with: .opacity))
                }

                if let error = errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(error)
                            .font(.caption)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if !entities.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        HStack {
                            Label("Detected Entities", systemImage: "shield.lefthalf.filled")
                                .font(.headline)
                                .foregroundStyle(.secondary)
                            Spacer()
                            if let time = inferenceTime {
                                Text(String(format: "%.0fms", time * 1000))
                                    .font(.caption)
                                    .foregroundStyle(.tertiary)
                            }
                        }

                        HighlightedTextView(text: inputText, entities: entities)
                            .padding(12)
                            .background(Color.gray.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 10))

                        ForEach(entities) { entity in
                            EntityRow(entity: entity)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                .padding(.bottom, 96)
            }
            .safeAreaInset(edge: .bottom) {
                VStack(spacing: 0) {
                    Divider()
                    Button(action: analyzeText) {
                        HStack {
                            if isAnalyzing {
                                ProgressView()
                                    .controlSize(.small)
                            } else {
                                Image(systemName: "wand.and.stars")
                            }
                            Text(analyzeButtonTitle)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isAnalyzing || inputText.isEmpty || !canAnalyzeSelectedModel)
                    .padding(.horizontal)
                    .padding(.top, 12)
                    .padding(.bottom, 8)
                }
                .background(.ultraThinMaterial)
            }
            .navigationTitle("OpenMed PII Demo")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            .scrollDismissesKeyboard(.interactively)
            #endif
        }
        .onAppear {
            refreshAvailableModels()
        }
        .task {
            refreshAvailableModels()
            autostartShowcaseIfNeeded()
        }
        .onChange(of: inputText) { _, _ in
            invalidateAnalysisResults(reason: .textChanged)
        }
        .onChange(of: selectedModelID) { _, _ in
            invalidateAnalysisResults(reason: .modelChanged)
            autostartShowcaseIfNeeded()
        }
        .sheet(isPresented: $isShowingModelPicker) {
            ModelPickerSheet(
                models: availableModels.isEmpty ? [.missingBundledModel] : availableModels,
                selectedModelID: $selectedModelID,
                huggingFaceToken: $huggingFaceToken
            )
            #if os(iOS)
            .presentationDetents([.large])
            .presentationDragIndicator(.visible)
            #endif
        }
        .animation(.spring(response: 0.38, dampingFraction: 0.88), value: analysisStatus?.phase)
    }

    private func analyzeText() {
        let analysisText = inputText
        let analysisModel = selectedModel

        isAnalyzing = true
        analysisStatus = nil
        errorMessage = nil
        inferenceTime = nil
        entities = []

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            defer {
                isAnalyzing = false
                analysisStatus = nil
            }

            do {
                let result = try await runInference(
                    text: analysisText,
                    model: analysisModel,
                    authToken: huggingFaceToken,
                    progress: { status in
                        await MainActor.run {
                            analysisStatus = status
                        }
                    }
                )

                guard inputText == analysisText, selectedModelID == analysisModel.id else {
                    return
                }

                inferenceTime = CFAbsoluteTimeGetCurrent() - start
                entities = result
                    .filter { entity in
                        entity.start >= 0 &&
                        entity.end >= 0 &&
                        entity.start < entity.end &&
                        entity.end <= analysisText.count
                    }
                    .sorted { lhs, rhs in
                        if lhs.start == rhs.start {
                            return lhs.end < rhs.end
                        }
                        return lhs.start < rhs.start
                    }
                errorMessage = nil
            } catch {
                guard inputText == analysisText, selectedModelID == analysisModel.id else {
                    return
                }
                errorMessage = error.localizedDescription
            }
        }
    }

    private func invalidateAnalysisResults(reason: AnalysisInvalidationReason) {
        if isAnalyzing {
            return
        }

        analysisStatus = nil
        inferenceTime = nil
        entities = []

        if case .modelChanged = reason {
            errorMessage = nil
        }
    }

    private func refreshAvailableModels() {
        let discovered = DemoModelCatalog.discover(from: .main)
        availableModels = discovered

        let preferredID =
            discovered.first {
                $0.sourceModelID == DemoModelCatalog.preferredDefaultSourceModelID && $0.isRunnableInDemoApp
            }?.id ??
            discovered.first(where: \.isRunnableInDemoApp)?.id ??
            discovered.first?.id ??
            DemoModelDescriptor.missingBundledModel.id

        if selectedModelID == DemoModelDescriptor.missingBundledModel.id {
            selectedModelID = preferredID
        } else if !discovered.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = preferredID
        }
    }

    private func autostartShowcaseIfNeeded() {
        guard !didAutostartShowcase,
              inputText == Self.showcaseSampleText,
              entities.isEmpty,
              !isAnalyzing,
              canAnalyzeSelectedModel
        else {
            return
        }

        didAutostartShowcase = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) {
            analyzeText()
        }
    }
}

private func runInference(
    text: String,
    model: DemoModelDescriptor,
    authToken: String,
    progress: @escaping @Sendable (DemoAnalysisStatus) async -> Void
) async throws -> [DetectedEntity] {
    switch model.runtimeKind {
    case .bundled:
        return try await OpenMedRuntimeCache.shared.analyze(
            text: text,
            using: model,
            authToken: nil,
            progress: progress
        )
    case .mlx:
        return try await OpenMedRuntimeCache.shared.analyze(
            text: text,
            using: model,
            authToken: authToken,
            progress: progress
        )
    case .unavailable:
        throw DemoError.unavailableInSwift(
            "Select a bundled CoreML model or a supported MLX model before running inference."
        )
    }
}

private enum DemoAnalysisPhase: Int, CaseIterable, Identifiable, Sendable {
    case downloadingModel
    case loadingModel
    case inferencing

    var id: Int { rawValue }

    var title: String {
        switch self {
        case .downloadingModel:
            return "Download"
        case .loadingModel:
            return "Load"
        case .inferencing:
            return "Inference"
        }
    }

    var buttonTitle: String {
        switch self {
        case .downloadingModel:
            return "Downloading Model..."
        case .loadingModel:
            return "Loading Model..."
        case .inferencing:
            return "Running Inference..."
        }
    }

    var systemImage: String {
        switch self {
        case .downloadingModel:
            return "arrow.down.circle.fill"
        case .loadingModel:
            return "shippingbox.fill"
        case .inferencing:
            return "waveform.path.ecg.rectangle.fill"
        }
    }

    var tint: Color {
        switch self {
        case .downloadingModel:
            return Color(red: 0.13, green: 0.42, blue: 0.85)
        case .loadingModel:
            return Color(red: 0.04, green: 0.60, blue: 0.66)
        case .inferencing:
            return Color(red: 0.12, green: 0.64, blue: 0.44)
        }
    }
}

private enum DemoAnalysisStageAvailability: Sendable {
    case required
    case cached
    case notNeeded
    case warm
}

private enum DemoAnalysisStageVisualState {
    case active
    case completed
    case pending
    case skipped

    var tint: Color {
        switch self {
        case .active:
            return Color.primary
        case .completed:
            return Color(red: 0.10, green: 0.52, blue: 0.50)
        case .pending:
            return Color.secondary
        case .skipped:
            return Color.secondary.opacity(0.85)
        }
    }

    var background: Color {
        switch self {
        case .active:
            return Color.white.opacity(0.88)
        case .completed:
            return Color.white.opacity(0.72)
        case .pending:
            return Color.white.opacity(0.50)
        case .skipped:
            return Color.white.opacity(0.42)
        }
    }

    var border: Color {
        switch self {
        case .active:
            return Color.white.opacity(0.95)
        case .completed:
            return Color.white.opacity(0.75)
        case .pending:
            return Color.white.opacity(0.55)
        case .skipped:
            return Color.white.opacity(0.45)
        }
    }
}

private struct DemoAnalysisStagePresentation {
    let title: String
    let systemImage: String
    let badgeText: String
    let caption: String
    let visualState: DemoAnalysisStageVisualState
    let accent: Color
}

private struct DemoAnalysisPlan: Sendable {
    let downloadAvailability: DemoAnalysisStageAvailability
    let loadAvailability: DemoAnalysisStageAvailability

    func status(
        for phase: DemoAnalysisPhase,
        model: DemoModelDescriptor
    ) -> DemoAnalysisStatus {
        DemoAnalysisStatus(
            phase: phase,
            modelDisplayName: model.displayName,
            runtimeKind: model.runtimeKind,
            downloadAvailability: downloadAvailability,
            loadAvailability: loadAvailability
        )
    }
}

private struct DemoAnalysisStatus: Sendable {
    let phase: DemoAnalysisPhase
    let modelDisplayName: String
    let runtimeKind: DemoModelRuntimeKind
    let downloadAvailability: DemoAnalysisStageAvailability
    let loadAvailability: DemoAnalysisStageAvailability

    var pipelineLabel: String {
        switch runtimeKind {
        case .bundled:
            return "Bundled CoreML Pipeline"
        case .mlx:
            return "On-Device MLX Pipeline"
        case .unavailable:
            return "OpenMed Pipeline"
        }
    }

    var headline: String {
        switch phase {
        case .downloadingModel:
            return "Downloading \(modelDisplayName)"
        case .loadingModel:
            return "Loading \(modelDisplayName)"
        case .inferencing:
            return "Running PII Inference"
        }
    }

    var detailText: String {
        switch phase {
        case .downloadingModel:
            return "Fetching the MLX artifact from Hugging Face and storing it in the local OpenMed cache so future runs start faster."
        case .loadingModel:
            return "Preparing tokenizer assets and initializing the model runtime before the note can be analyzed."
        case .inferencing:
            return "Applying token classification and OpenMed's smart PII merging on-device so the entities are production-grade."
        }
    }

    func presentation(for stage: DemoAnalysisPhase) -> DemoAnalysisStagePresentation {
        switch stage {
        case .downloadingModel:
            return makeDownloadPresentation()
        case .loadingModel:
            return makeLoadPresentation()
        case .inferencing:
            return makeInferencePresentation()
        }
    }

    private func makeDownloadPresentation() -> DemoAnalysisStagePresentation {
        switch downloadAvailability {
        case .required:
            let isActive = phase == .downloadingModel
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.downloadingModel.title,
                systemImage: DemoAnalysisPhase.downloadingModel.systemImage,
                badgeText: isActive ? "Downloading" : "Ready",
                caption: isActive
                    ? "Fetching the MLX artifact and snapshot files."
                    : "Artifact stored locally and ready for reuse.",
                visualState: isActive ? .active : .completed,
                accent: DemoAnalysisPhase.downloadingModel.tint
            )
        case .cached:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.downloadingModel.title,
                systemImage: DemoAnalysisPhase.downloadingModel.systemImage,
                badgeText: "Cached",
                caption: "Artifact already exists on this device.",
                visualState: .completed,
                accent: DemoAnalysisPhase.downloadingModel.tint
            )
        case .notNeeded:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.downloadingModel.title,
                systemImage: "tray.and.arrow.down.fill",
                badgeText: "Bundled",
                caption: "No download needed for bundled CoreML assets.",
                visualState: .skipped,
                accent: DemoAnalysisPhase.downloadingModel.tint
            )
        case .warm:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.downloadingModel.title,
                systemImage: DemoAnalysisPhase.downloadingModel.systemImage,
                badgeText: "Warm",
                caption: "Model files are already ready to go.",
                visualState: .completed,
                accent: DemoAnalysisPhase.downloadingModel.tint
            )
        }
    }

    private func makeLoadPresentation() -> DemoAnalysisStagePresentation {
        switch loadAvailability {
        case .warm:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.loadingModel.title,
                systemImage: "checkmark.circle.fill",
                badgeText: "Warm",
                caption: "Runtime is already initialized in memory.",
                visualState: .completed,
                accent: DemoAnalysisPhase.loadingModel.tint
            )
        case .required:
            let state: DemoAnalysisStageVisualState
            let badgeText: String
            let caption: String

            if phase == .loadingModel {
                state = .active
                badgeText = "Loading"
                caption = "Creating tokenizer and model runtime."
            } else if phase.rawValue > DemoAnalysisPhase.loadingModel.rawValue {
                state = .completed
                badgeText = "Ready"
                caption = "Runtime is loaded and standing by."
            } else {
                state = .pending
                badgeText = "Next"
                caption = "Starts once the model files are ready."
            }

            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.loadingModel.title,
                systemImage: DemoAnalysisPhase.loadingModel.systemImage,
                badgeText: badgeText,
                caption: caption,
                visualState: state,
                accent: DemoAnalysisPhase.loadingModel.tint
            )
        case .cached:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.loadingModel.title,
                systemImage: DemoAnalysisPhase.loadingModel.systemImage,
                badgeText: "Cached",
                caption: "Model resources are ready for initialization.",
                visualState: .completed,
                accent: DemoAnalysisPhase.loadingModel.tint
            )
        case .notNeeded:
            return DemoAnalysisStagePresentation(
                title: DemoAnalysisPhase.loadingModel.title,
                systemImage: DemoAnalysisPhase.loadingModel.systemImage,
                badgeText: "Skipped",
                caption: "No runtime loading step was needed.",
                visualState: .skipped,
                accent: DemoAnalysisPhase.loadingModel.tint
            )
        }
    }

    private func makeInferencePresentation() -> DemoAnalysisStagePresentation {
        let state: DemoAnalysisStageVisualState = phase == .inferencing ? .active : .pending
        let badgeText = phase == .inferencing ? "Live" : "Next"
        let caption =
            phase == .inferencing
            ? "Running token classification and smart PII merging."
            : "Begins as soon as the runtime is ready."

        return DemoAnalysisStagePresentation(
            title: DemoAnalysisPhase.inferencing.title,
            systemImage: DemoAnalysisPhase.inferencing.systemImage,
            badgeText: badgeText,
            caption: caption,
            visualState: state,
            accent: DemoAnalysisPhase.inferencing.tint
        )
    }
}

private struct AnalysisStatusCard: View {
    let status: DemoAnalysisStatus

    var body: some View {
        ZStack(alignment: .topTrailing) {
            RoundedRectangle(cornerRadius: 22, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [
                            Color(red: 0.95, green: 0.98, blue: 0.99),
                            Color(red: 0.94, green: 0.98, blue: 0.97),
                            Color(red: 0.98, green: 0.99, blue: 1.00),
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 22, style: .continuous)
                        .stroke(Color.white.opacity(0.92), lineWidth: 1)
                )

            Circle()
                .fill(status.phase.tint.opacity(0.16))
                .frame(width: 180, height: 180)
                .blur(radius: 24)
                .offset(x: 56, y: -56)

            VStack(alignment: .leading, spacing: 18) {
                HStack(alignment: .top, spacing: 14) {
                    VStack(alignment: .leading, spacing: 6) {
                        Label(status.pipelineLabel, systemImage: "bolt.shield")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)

                        Text(status.headline)
                            .font(.title3.weight(.semibold))
                            .foregroundStyle(.primary)

                        Text(status.detailText)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    Spacer(minLength: 16)

                    HStack(spacing: 8) {
                        ProgressView()
                            .controlSize(.small)
                            .tint(status.phase.tint)

                        Text(status.phase.title.uppercased())
                            .font(.caption.weight(.bold))
                            .tracking(0.6)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(status.phase.tint.opacity(0.12))
                            .foregroundStyle(status.phase.tint)
                            .clipShape(Capsule())
                    }
                }

                ViewThatFits(in: .horizontal) {
                    HStack(spacing: 12) {
                        ForEach(DemoAnalysisPhase.allCases) { phase in
                            AnalysisPhaseTile(presentation: status.presentation(for: phase))
                        }
                    }

                    VStack(spacing: 12) {
                        ForEach(DemoAnalysisPhase.allCases) { phase in
                            AnalysisPhaseTile(presentation: status.presentation(for: phase))
                        }
                    }
                }
            }
            .padding(18)
        }
        .shadow(color: status.phase.tint.opacity(0.10), radius: 18, x: 0, y: 10)
    }
}

private struct AnalysisPhaseTile: View {
    let presentation: DemoAnalysisStagePresentation

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 10) {
                ZStack {
                    Circle()
                        .fill(presentation.accent.opacity(iconBackgroundOpacity))
                        .frame(width: 34, height: 34)

                    Image(systemName: presentation.systemImage)
                        .font(.system(size: 14, weight: .semibold))
                        .foregroundStyle(presentation.accent)
                }

                VStack(alignment: .leading, spacing: 2) {
                    Text(presentation.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)

                    Text(presentation.badgeText)
                        .font(.caption2.weight(.bold))
                        .tracking(0.5)
                        .foregroundStyle(presentation.visualState.tint)
                }

                Spacer(minLength: 8)
            }

            Text(presentation.caption)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(presentation.visualState.background)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(presentation.visualState.border, lineWidth: 1)
        )
    }

    private var iconBackgroundOpacity: Double {
        switch presentation.visualState {
        case .active:
            return 0.18
        case .completed:
            return 0.14
        case .pending:
            return 0.10
        case .skipped:
            return 0.08
        }
    }
}

enum DemoError: LocalizedError {
    case invalidBundle(String)
    case missingAuthToken(String)
    case unavailableInSwift(String)

    var errorDescription: String? {
        switch self {
        case .invalidBundle(let detail):
            return "Bundled model is incomplete: \(detail)"
        case .missingAuthToken(let detail):
            return detail
        case .unavailableInSwift(let detail):
            return detail
        }
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
        switch category {
        case "NAME": return .blue
        case "DATE": return .purple
        case "PHONE": return .green
        case "SSN": return .red
        case "ADDRESS": return .orange
        case "EMAIL": return .mint
        case "ID": return .pink
        case "ORG": return .indigo
        default: return .gray
        }
    }
}

private enum DemoModelRuntimeKind: String, Hashable, Sendable {
    case unavailable
    case bundled
    case mlx
}

private struct DemoModelDescriptor: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let sourceModelID: String
    let artifactRepoID: String?
    let tokenizerName: String
    let modelURL: URL?
    let id2labelURL: URL?
    let tokenizerFolderURL: URL?
    let runtimeKind: DemoModelRuntimeKind
    let note: String

    static let missingBundledModel = DemoModelDescriptor(
        id: "missing-bundled-model",
        displayName: "No Supported Model",
        sourceModelID: "Add a bundled CoreML model or select a supported MLX artifact",
        artifactRepoID: nil,
        tokenizerName: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1",
        modelURL: nil,
        id2labelURL: nil,
        tokenizerFolderURL: nil,
        runtimeKind: .unavailable,
        note: "OpenMedKit now supports BERT-family MLX artifacts on Apple Silicon macOS and physical iPhone/iPad devices, or bundled CoreML assets across Apple platforms."
    )

    var isBundled: Bool { runtimeKind == .bundled }
    var requiresAuthToken: Bool { false }
    var isRunnableInDemoApp: Bool {
        switch runtimeKind {
        case .bundled:
            return true
        case .mlx:
            return DemoPlatform.supportsOnDeviceMLX
        case .unavailable:
            return false
        }
    }

    var badgeText: String {
        switch runtimeKind {
        case .unavailable: return "Unavailable"
        case .bundled: return "Bundled CoreML"
        case .mlx: return "MLX Download"
        }
    }

    var badgeColor: Color {
        switch runtimeKind {
        case .unavailable: return .orange
        case .bundled: return .green
        case .mlx: return .blue
        }
    }

    var detailText: String {
        switch runtimeKind {
        case .unavailable:
            return note
        case .bundled:
            if let tokenizerFolderURL {
                return "Runs through OpenMedKit with bundled tokenizer assets at \(tokenizerFolderURL.lastPathComponent)."
            }
            return "Runs through OpenMedKit and uses tokenizerName \(tokenizerName)."
        case .mlx:
            if DemoPlatform.supportsOnDeviceMLX {
                return "Downloads the public MLX artifact from Hugging Face, caches it locally, and runs it on-device with OpenMedKit + MLX."
            }
            return "This MLX model requires Apple Silicon macOS or a physical iPhone/iPad. iOS Simulator is not supported."
        }
    }

    var searchText: String {
        [displayName, sourceModelID, artifactRepoID ?? "", tokenizerName, note]
            .joined(separator: " ")
            .lowercased()
    }

}

private struct BundledModelManifest: Decodable {
    let displayName: String
    let sourceModelID: String
    let tokenizerName: String?
    let tokenizerFolderName: String?
    let compiledModelName: String?
    let compiledModelExtension: String?
    let id2labelFileName: String?
    let note: String?

    enum CodingKeys: String, CodingKey {
        case displayName
        case sourceModelID = "sourceModelId"
        case tokenizerName
        case tokenizerFolderName
        case compiledModelName
        case compiledModelExtension
        case id2labelFileName
        case note
    }
}

private enum DemoModelCatalog {
    static let preferredDefaultSourceModelID = "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1"

    static func discover(from bundle: Bundle) -> [DemoModelDescriptor] {
        var discovered: [DemoModelDescriptor] = swiftMLXCatalog()

        if let legacy = discoverLegacyBundle(from: bundle) {
            discovered.append(legacy)
        }

        discovered.append(contentsOf: discoverManifestBundles(from: bundle))

        var seen = Set<String>()
        let unique = discovered
            .filter { seen.insert($0.id).inserted }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }
        return unique.isEmpty ? [.missingBundledModel] : unique
    }

    private static func swiftMLXCatalog() -> [DemoModelDescriptor] {
        [
            DemoModelDescriptor(
                id: "mlx-clinicale5-small",
                displayName: "ClinicalE5 Small",
                sourceModelID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                artifactRepoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx",
                tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                modelURL: nil,
                id2labelURL: nil,
                tokenizerFolderURL: nil,
                runtimeKind: .mlx,
                note: "BERT family, 33M parameters. Smallest published OpenMed PII model for Swift MLX validation."
            ),
            DemoModelDescriptor(
                id: "mlx-liteclinical-small",
                displayName: "LiteClinical Small",
                sourceModelID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1",
                artifactRepoID: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx",
                tokenizerName: "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1",
                modelURL: nil,
                id2labelURL: nil,
                tokenizerFolderURL: nil,
                runtimeKind: .mlx,
                note: "DistilBERT family, 66M parameters."
            ),
            DemoModelDescriptor(
                id: "mlx-fastclinical-small",
                displayName: "FastClinical Small",
                sourceModelID: "OpenMed/OpenMed-PII-FastClinical-Small-82M-v1",
                artifactRepoID: "OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx",
                tokenizerName: "OpenMed/OpenMed-PII-FastClinical-Small-82M-v1",
                modelURL: nil,
                id2labelURL: nil,
                tokenizerFolderURL: nil,
                runtimeKind: .mlx,
                note: "RoBERTa family, 82M parameters."
            ),
            DemoModelDescriptor(
                id: "mlx-biomedelectra-base",
                displayName: "BiomedELECTRA Base",
                sourceModelID: "OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1",
                artifactRepoID: "OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx",
                tokenizerName: "OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1",
                modelURL: nil,
                id2labelURL: nil,
                tokenizerFolderURL: nil,
                runtimeKind: .mlx,
                note: "ELECTRA family, 110M parameters."
            ),
            DemoModelDescriptor(
                id: "mlx-bigmed-large",
                displayName: "BigMed Large",
                sourceModelID: "OpenMed/OpenMed-PII-BigMed-Large-278M-v1",
                artifactRepoID: "OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx",
                tokenizerName: "OpenMed/OpenMed-PII-BigMed-Large-278M-v1",
                modelURL: nil,
                id2labelURL: nil,
                tokenizerFolderURL: nil,
                runtimeKind: .mlx,
                note: "XLM-RoBERTa family, 278M parameters."
            ),
        ]
    }

    private static func discoverLegacyBundle(from bundle: Bundle) -> DemoModelDescriptor? {
        guard let resourceURL = bundle.resourceURL else {
            return nil
        }

        let modelURL =
            bundle.url(forResource: "OpenMedPII", withExtension: "mlmodelc") ??
            bundle.url(forResource: "OpenMedPII", withExtension: "mlpackage")

        guard let modelURL,
              let labelsURL = bundle.url(forResource: "id2label", withExtension: "json") else {
            return nil
        }

        let tokenizerFolder = resourceURL.appendingPathComponent("TokenizerAssets", isDirectory: true)

        return DemoModelDescriptor(
            id: "legacy-openmedpii",
            displayName: "Bundled OpenMedPII",
            sourceModelID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
            artifactRepoID: nil,
            tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
            modelURL: modelURL,
            id2labelURL: labelsURL,
            tokenizerFolderURL: FileManager.default.fileExists(atPath: tokenizerFolder.path) ? tokenizerFolder : nil,
            runtimeKind: .bundled,
            note: "Legacy single-model bundle."
        )
    }

    private static func discoverManifestBundles(from bundle: Bundle) -> [DemoModelDescriptor] {
        guard let resourceURL = bundle.resourceURL else {
            return []
        }

        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var models: [DemoModelDescriptor] = []

        for case let fileURL as URL in enumerator {
            guard fileURL.lastPathComponent == "openmed-model.json" else {
                continue
            }

            do {
                let data = try Data(contentsOf: fileURL)
                let manifest = try JSONDecoder().decode(BundledModelManifest.self, from: data)
                let directoryURL = fileURL.deletingLastPathComponent()
                let compiledName = manifest.compiledModelName ?? "OpenMedPII"
                let compiledExt = manifest.compiledModelExtension ?? "mlmodelc"
                let id2labelName = manifest.id2labelFileName ?? "id2label.json"

                let modelURL = directoryURL.appendingPathComponent(compiledName).appendingPathExtension(compiledExt)
                let labelsURL = directoryURL.appendingPathComponent(id2labelName)

                guard FileManager.default.fileExists(atPath: modelURL.path) else {
                    continue
                }

                guard FileManager.default.fileExists(atPath: labelsURL.path) else {
                    continue
                }

                let tokenizerFolderURL: URL?
                if let tokenizerFolderName = manifest.tokenizerFolderName {
                    let candidate = directoryURL.appendingPathComponent(tokenizerFolderName, isDirectory: true)
                    tokenizerFolderURL = FileManager.default.fileExists(atPath: candidate.path) ? candidate : nil
                } else {
                    tokenizerFolderURL = nil
                }

                models.append(
                    DemoModelDescriptor(
                        id: manifest.sourceModelID,
                        displayName: manifest.displayName,
                        sourceModelID: manifest.sourceModelID,
                        artifactRepoID: nil,
                        tokenizerName: manifest.tokenizerName ?? manifest.sourceModelID,
                        modelURL: modelURL,
                        id2labelURL: labelsURL,
                        tokenizerFolderURL: tokenizerFolderURL,
                        runtimeKind: .bundled,
                        note: manifest.note ?? "Bundle-discovered model."
                    )
                )
            } catch {
                continue
            }
        }

        return models
    }
}

private actor OpenMedRuntimeCache {
    static let shared = OpenMedRuntimeCache()

    private var runtimes: [String: OpenMed] = [:]

    func analyze(
        text: String,
        using model: DemoModelDescriptor,
        authToken: String?,
        progress: @escaping @Sendable (DemoAnalysisStatus) async -> Void
    ) async throws -> [DetectedEntity] {
        let plan = try analysisPlan(for: model)
        let runtime: OpenMed

        if let cached = runtimes[model.id] {
            runtime = cached
        } else {
            let openmed: OpenMed
            switch model.runtimeKind {
            case .bundled:
                guard let modelURL = model.modelURL,
                      let id2labelURL = model.id2labelURL else {
                    throw DemoError.invalidBundle("Model selection \(model.displayName) is not loadable.")
                }

                await progress(plan.status(for: .loadingModel, model: model))
                openmed = try OpenMed(
                    modelURL: modelURL,
                    id2labelURL: id2labelURL,
                    tokenizerName: model.tokenizerName,
                    tokenizerFolderURL: model.tokenizerFolderURL
                )

            case .mlx:
                guard let artifactRepoID = model.artifactRepoID else {
                    throw DemoError.unavailableInSwift(
                        "The selected MLX model does not define a Hugging Face artifact repo."
                    )
                }
                let normalizedToken = authToken?.trimmingCharacters(in: .whitespacesAndNewlines)
                if model.requiresAuthToken && plan.downloadAvailability == .required && (normalizedToken?.isEmpty ?? true) {
                    throw DemoError.missingAuthToken(
                        "Enter a Hugging Face token before downloading \(artifactRepoID). The repo is private for now."
                    )
                }

                let modelDirectory: URL
                switch plan.downloadAvailability {
                case .required:
                    await progress(plan.status(for: .downloadingModel, model: model))
                    modelDirectory = try await OpenMedModelStore.downloadMLXModel(
                        repoID: artifactRepoID,
                        authToken: normalizedToken,
                        revision: "main"
                    )
                case .cached, .warm:
                    modelDirectory = try OpenMedModelStore.cachedMLXModelDirectory(
                        repoID: artifactRepoID,
                        revision: "main"
                    )
                case .notNeeded:
                    throw DemoError.unavailableInSwift(
                        "The selected MLX model cannot skip its download stage."
                    )
                }

                await progress(plan.status(for: .loadingModel, model: model))
                openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))

            case .unavailable:
                throw DemoError.unavailableInSwift(
                    "The selected model cannot run in this app."
                )
            }

            runtimes[model.id] = openmed
            runtime = openmed
        }

        await progress(plan.status(for: .inferencing, model: model))
        let predictions = try runtime.extractPII(text)
        return predictions.map { prediction in
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

    private func analysisPlan(for model: DemoModelDescriptor) throws -> DemoAnalysisPlan {
        if runtimes[model.id] != nil {
            return DemoAnalysisPlan(
                downloadAvailability: model.runtimeKind == .mlx ? .cached : .notNeeded,
                loadAvailability: .warm
            )
        }

        switch model.runtimeKind {
        case .bundled:
            return DemoAnalysisPlan(downloadAvailability: .notNeeded, loadAvailability: .required)
        case .mlx:
            guard let artifactRepoID = model.artifactRepoID else {
                throw DemoError.unavailableInSwift(
                    "The selected MLX model does not define a Hugging Face artifact repo."
                )
            }

            let isCached = try OpenMedModelStore.isMLXModelCached(
                repoID: artifactRepoID,
                revision: "main"
            )
            return DemoAnalysisPlan(
                downloadAvailability: isCached ? .cached : .required,
                loadAvailability: .required
            )
        case .unavailable:
            throw DemoError.unavailableInSwift(
                "The selected model cannot run in this app."
            )
        }
    }
}

struct HighlightedTextView: View {
    let text: String
    let entities: [DetectedEntity]

    var body: some View {
        let attributed = buildAttributedString()
        Text(attributed)
            .font(.body.monospaced())
    }

    private func buildAttributedString() -> AttributedString {
        var attrStr = AttributedString(text)
        let textLength = text.count
        let sorted = entities.sorted { $0.start > $1.start }

        for entity in sorted {
            let safeStart = min(max(entity.start, 0), textLength)
            let safeEnd = min(max(entity.end, safeStart), textLength)

            guard safeStart < safeEnd else {
                continue
            }

            let startIdx = attrStr.index(attrStr.startIndex, offsetByCharacters: safeStart)
            let endIdx = attrStr.index(attrStr.startIndex, offsetByCharacters: safeEnd)
            let range = startIdx..<endIdx

            attrStr[range].backgroundColor = entity.color.opacity(0.25)
            attrStr[range].foregroundColor = entity.color
            attrStr[range].font = .body.monospaced().bold()
        }

        return attrStr
    }
}

struct EntityRow: View {
    let entity: DetectedEntity

    var body: some View {
        HStack {
            RoundedRectangle(cornerRadius: 3)
                .fill(entity.color)
                .frame(width: 4)

            VStack(alignment: .leading, spacing: 2) {
                Text(entity.text)
                    .font(.body.bold())
                Text(entity.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Text(String(format: "%.0f%%", entity.confidence * 100))
                .font(.caption.monospacedDigit())
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(entity.color.opacity(0.15))
                .clipShape(Capsule())
        }
        .padding(.horizontal)
        .padding(.vertical, 4)
    }
}

private struct ModelPickerSheet: View {
    let models: [DemoModelDescriptor]
    @Binding var selectedModelID: String
    @Binding var huggingFaceToken: String

    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""

    private var filteredModels: [DemoModelDescriptor] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return models }
        return models.filter { $0.searchText.contains(query) }
    }

    private var selectedModel: DemoModelDescriptor? {
        models.first(where: { $0.id == selectedModelID })
    }

    private var showsTokenCard: Bool {
        models.contains(where: \.requiresAuthToken)
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 16) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Choose the runtime path for this demo")
                            .font(.title3.weight(.semibold))
                        Text("Browse bundled CoreML models and the OpenMed MLX catalog without losing space on the main note view.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    }

                    HStack(spacing: 10) {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(.secondary)
                        TextField("Search OpenMed models", text: $searchText)
                            #if os(iOS)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                            #endif
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 12)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(Color.primary.opacity(0.06), lineWidth: 1)
                    }
                }
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 14)

                Divider()

                ScrollView {
                    if filteredModels.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "magnifyingglass")
                                .font(.system(size: 28, weight: .semibold))
                                .foregroundStyle(.secondary)
                            Text("No Matching Models")
                                .font(.headline)
                            Text("Try a model family, language, or runtime keyword like ClinicalE5, Dutch, or MLX.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                                .multilineTextAlignment(.center)
                                .frame(maxWidth: 420)
                        }
                        .frame(maxWidth: .infinity, minHeight: 280)
                        .padding(24)
                    } else {
                        LazyVStack(spacing: 14) {
                            ForEach(filteredModels) { model in
                                modelCard(for: model)
                            }
                        }
                        .padding(.horizontal, 20)
                        .padding(.vertical, 18)
                    }
                }
            }
            .background(Color.gray.opacity(0.06))
            .safeAreaInset(edge: .bottom) {
                if showsTokenCard {
                    tokenManagementCard
                }
            }
            .navigationTitle("Choose Model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            #if os(macOS)
            .frame(minWidth: 620, minHeight: 560)
            #endif
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
    }

    private func modelCard(for model: DemoModelDescriptor) -> some View {
        let isSelected = selectedModelID == model.id

        return Button {
            selectedModelID = model.id
        } label: {
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(model.displayName)
                            .font(.headline)
                            .foregroundStyle(.primary)
                            .multilineTextAlignment(.leading)
                            .lineLimit(3)
                            .layoutPriority(1)

                        if isSelected {
                            Text("Selected")
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 7)
                                .padding(.vertical, 4)
                                .background(model.badgeColor.opacity(0.14))
                                .foregroundStyle(model.badgeColor)
                                .clipShape(Capsule())
                        }
                    }

                    Text(model.sourceModelID)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                        .lineLimit(2)

                    Text(model.detailText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                        .lineLimit(3)

                    if model.requiresAuthToken {
                        Label("Private Hugging Face artifact", systemImage: "lock.fill")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.orange)
                    }
                }

                Spacer(minLength: 12)

                VStack(alignment: .trailing, spacing: 10) {
                    Text(model.badgeText)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 9)
                        .padding(.vertical, 5)
                        .background(model.badgeColor.opacity(0.14))
                        .foregroundStyle(model.badgeColor)
                        .clipShape(Capsule())

                    Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                        .font(.title3.weight(.semibold))
                        .foregroundStyle(isSelected ? model.badgeColor : Color.secondary.opacity(0.55))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(16)
            .background(
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .fill(isSelected ? model.badgeColor.opacity(0.11) : Color.gray.opacity(0.09))
            )
            .overlay {
                RoundedRectangle(cornerRadius: 20, style: .continuous)
                    .stroke(
                        isSelected ? model.badgeColor.opacity(0.42) : Color.primary.opacity(0.06),
                        lineWidth: isSelected ? 1.5 : 1
                    )
            }
            .shadow(
                color: isSelected ? model.badgeColor.opacity(0.12) : .clear,
                radius: 14,
                x: 0,
                y: 8
            )
            .contentShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private var tokenManagementCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Label("Private MLX Access", systemImage: "key.horizontal")
                    .font(.headline)
                    .foregroundStyle(.secondary)

                Spacer()

                if !huggingFaceToken.isEmpty {
                    Label("Saved", systemImage: "checkmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.green)
                } else if selectedModel?.requiresAuthToken == true {
                    Label("Needed for current pick", systemImage: "exclamationmark.circle.fill")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                }
            }

            tokenField

            HStack(alignment: .top) {
                Text("Only needed while the MLX repos stay private. Once the Hugging Face artifacts are public, you can leave this empty.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer(minLength: 12)

                if !huggingFaceToken.isEmpty {
                    Button("Clear") {
                        huggingFaceToken = ""
                    }
                    .font(.caption.weight(.semibold))
                    .buttonStyle(.borderless)
                }
            }

            Text("Your model choice is saved immediately. Close this sheet when everything looks right.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 20)
        .padding(.top, 14)
        .padding(.bottom, 14)
        .background(.ultraThinMaterial)
        .overlay(alignment: .top) {
            Divider()
        }
    }

    @ViewBuilder
    private var tokenField: some View {
        #if os(iOS)
        SecureField("hf_...", text: $huggingFaceToken)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.body.monospaced())
            .padding(12)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        #else
        SecureField("hf_...", text: $huggingFaceToken)
            .font(.body.monospaced())
            .padding(12)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        #endif
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

private func entityCategory(for label: String) -> String {
    let normalized = label.lowercased()

    if normalized.contains("name") {
        return "NAME"
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

    return normalized.uppercased()
}

#Preview {
    ContentView()
}
