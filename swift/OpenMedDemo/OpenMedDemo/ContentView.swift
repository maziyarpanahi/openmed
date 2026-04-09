import OpenMedKit
import SwiftUI

struct ContentView: View {
    @AppStorage("openmed.demo.hfToken") private var huggingFaceToken = ""
    @State private var inputText = "Patient John Doe, DOB 1990-05-15, phone 555-123-4567, SSN 123-45-6789. Diagnosed with Type 2 diabetes."
    @State private var availableModels: [DemoModelDescriptor] = [.missingBundledModel]
    @State private var selectedModelID = DemoModelDescriptor.missingBundledModel.id
    @State private var entities: [DetectedEntity] = []
    @State private var isAnalyzing = false
    @State private var isShowingModelPicker = false
    @State private var errorMessage: String?
    @State private var inferenceTime: Double?

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
            return "Analyzing..."
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
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color.gray.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                    if availableModels.filter(\.isBundled).isEmpty {
                        Text("This app build does not currently include a bundled CoreML model. You can still test the published OpenMed MLX artifacts on Apple Silicon macOS or a physical iPhone/iPad by selecting an MLX model below and providing a Hugging Face token while the repos remain private.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if selectedModel.runtimeKind == .mlx {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("Hugging Face Token", systemImage: "key.horizontal")
                            .font(.headline)
                            .foregroundStyle(.secondary)

                        huggingFaceTokenField

                        Text("Required while these MLX repos are private. Once the repos are public, you can leave this blank.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

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
        }
        .onChange(of: inputText) { _, _ in
            invalidateAnalysisResults(reason: .textChanged)
        }
        .onChange(of: selectedModelID) { _, _ in
            invalidateAnalysisResults(reason: .modelChanged)
        }
        .sheet(isPresented: $isShowingModelPicker) {
            ModelPickerSheet(
                models: availableModels.isEmpty ? [.missingBundledModel] : availableModels,
                selectedModelID: $selectedModelID
            )
        }
    }

    private func analyzeText() {
        let analysisText = inputText
        let analysisModel = selectedModel

        isAnalyzing = true
        errorMessage = nil
        entities = []

        Task {
            let start = CFAbsoluteTimeGetCurrent()
            defer { isAnalyzing = false }

            do {
                let result = try await runInference(
                    text: analysisText,
                    model: analysisModel,
                    authToken: huggingFaceToken
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
            discovered.first(where: \.isRunnableInDemoApp)?.id ??
            discovered.first?.id ??
            DemoModelDescriptor.missingBundledModel.id

        if selectedModelID == DemoModelDescriptor.missingBundledModel.id {
            selectedModelID = preferredID
        } else if !discovered.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = preferredID
        }
    }

    @ViewBuilder
    private var huggingFaceTokenField: some View {
        #if os(iOS)
        SecureField("hf_...", text: $huggingFaceToken)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.body.monospaced())
            .padding(12)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        #else
        SecureField("hf_...", text: $huggingFaceToken)
            .font(.body.monospaced())
            .padding(12)
            .background(Color.gray.opacity(0.12))
            .clipShape(RoundedRectangle(cornerRadius: 10))
        #endif
    }
}

private func runInference(
    text: String,
    model: DemoModelDescriptor,
    authToken: String
) async throws -> [DetectedEntity] {
    switch model.runtimeKind {
    case .bundled:
        return try await OpenMedRuntimeCache.shared.analyze(
            text: text,
            using: model,
            authToken: nil
        )
    case .mlx:
        return try await OpenMedRuntimeCache.shared.analyze(
            text: text,
            using: model,
            authToken: authToken
        )
    case .unavailable:
        throw DemoError.unavailableInSwift(
            "Select a bundled CoreML model or a supported MLX model before running inference."
        )
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
        tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        modelURL: nil,
        id2labelURL: nil,
        tokenizerFolderURL: nil,
        runtimeKind: .unavailable,
        note: "OpenMedKit now supports BERT-family MLX artifacts on Apple Silicon macOS and physical iPhone/iPad devices, or bundled CoreML assets across Apple platforms."
    )

    var isBundled: Bool { runtimeKind == .bundled }
    var requiresAuthToken: Bool { runtimeKind == .mlx && artifactRepoID != nil }
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
                return "Downloads the MLX artifact from Hugging Face, caches it locally, and runs it on-device with OpenMedKit + MLX."
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
        authToken: String?
    ) async throws -> [DetectedEntity] {
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
                if model.requiresAuthToken && (normalizedToken?.isEmpty ?? true) {
                    throw DemoError.missingAuthToken(
                        "Enter a Hugging Face token before downloading \(artifactRepoID). The repo is private for now."
                    )
                }

                let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
                    repoID: artifactRepoID,
                    authToken: normalizedToken,
                    revision: "main"
                )
                openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))

            case .unavailable:
                throw DemoError.unavailableInSwift(
                    "The selected model cannot run in this app."
                )
            }

            runtimes[model.id] = openmed
            runtime = openmed
        }

        let predictions = try runtime.analyzeText(text)
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

    @Environment(\.dismiss) private var dismiss
    @State private var searchText = ""

    private var filteredModels: [DemoModelDescriptor] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return models }
        return models.filter { $0.searchText.contains(query) }
    }

    var body: some View {
        NavigationStack {
            Group {
                if filteredModels.isEmpty {
                    VStack(spacing: 12) {
                        Image(systemName: "magnifyingglass")
                            .font(.system(size: 28, weight: .semibold))
                            .foregroundStyle(.secondary)
                        Text("No Matching Models")
                            .font(.headline)
                        Text("Search the bundled CoreML models and the Swift-MLX-compatible OpenMed model catalog.")
                            .font(.callout)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 420)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .padding()
                } else {
                    List {
                        ForEach(filteredModels) { model in
                            Button {
                                selectedModelID = model.id
                                dismiss()
                            } label: {
                                HStack(alignment: .top, spacing: 12) {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(model.displayName)
                                            .font(.body.weight(.semibold))
                                            .foregroundStyle(.primary)
                                        Text(model.sourceModelID)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                            .multilineTextAlignment(.leading)
                                        Text(model.detailText)
                                            .font(.caption)
                                            .foregroundStyle(.tertiary)
                                            .multilineTextAlignment(.leading)
                                    }

                                    Spacer()

                                    VStack(alignment: .trailing, spacing: 6) {
                                        Text(model.badgeText)
                                            .font(.caption.weight(.semibold))
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 4)
                                            .background(model.badgeColor.opacity(0.14))
                                            .foregroundStyle(model.badgeColor)
                                            .clipShape(Capsule())

                                        if selectedModelID == model.id {
                                            Image(systemName: "checkmark.circle.fill")
                                                .foregroundStyle(.green)
                                        }
                                    }
                                }
                                .contentShape(Rectangle())
                                .padding(.vertical, 4)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
            }
            .searchable(text: $searchText, prompt: "Search OpenMed models")
            .navigationTitle("Choose Model")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .frame(minWidth: 620, minHeight: 420)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
            }
        }
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
    if normalized.contains("address") || normalized.contains("location") {
        return "ADDRESS"
    }

    return normalized.uppercased()
}

#Preview {
    ContentView()
}
