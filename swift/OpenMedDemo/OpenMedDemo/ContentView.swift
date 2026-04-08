import CoreML
import OpenMedKit
import SwiftUI

struct ContentView: View {
    @State private var inputText = "Patient John Doe, DOB 1990-05-15, phone 555-123-4567, SSN 123-45-6789. Diagnosed with Type 2 diabetes."
    @State private var availableModels: [DemoModelDescriptor] = [.demo]
    @State private var selectedModelID = DemoModelDescriptor.demo.id
    @State private var entities: [DetectedEntity] = []
    @State private var isAnalyzing = false
    @State private var isShowingModelPicker = false
    @State private var errorMessage: String?
    @State private var inferenceTime: Double?

    private var selectedModel: DemoModelDescriptor {
        availableModels.first(where: { $0.id == selectedModelID }) ?? .demo
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
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

                        Text(selectedModel.detailText)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
                    .background(Color.gray.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 10))

                    if availableModels.filter(\.isBundled).isEmpty {
                        Text("No bundled CoreML model was found in this app build, so the demo is running in mock mode. Uploaded MLX repos are for Python on macOS; Swift apps still need bundled CoreML models.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding()

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
                .padding()

                Button(action: analyzeText) {
                    HStack {
                        if isAnalyzing {
                            ProgressView()
                                .controlSize(.small)
                        } else {
                            Image(systemName: "wand.and.stars")
                        }
                        Text(isAnalyzing ? "Analyzing..." : "Detect PII Entities")
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isAnalyzing || inputText.isEmpty)
                .padding(.horizontal)

                if let error = errorMessage {
                    HStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(error)
                            .font(.caption)
                    }
                    .padding(.horizontal)
                    .padding(.top, 8)
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
                    .padding()
                }

                Spacer()
            }
            .navigationTitle("OpenMed PII Demo")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
        .onAppear {
            refreshAvailableModels()
        }
        .task {
            refreshAvailableModels()
        }
        .sheet(isPresented: $isShowingModelPicker) {
            ModelPickerSheet(
                models: availableModels.isEmpty ? [.demo] : availableModels,
                selectedModelID: $selectedModelID
            )
        }
    }

    private func analyzeText() {
        isAnalyzing = true
        errorMessage = nil
        entities = []

        Task {
            let start = CFAbsoluteTimeGetCurrent()

            do {
                let result = try await runInference(text: inputText, model: selectedModel)
                inferenceTime = CFAbsoluteTimeGetCurrent() - start
                entities = result.sorted { lhs, rhs in
                    if lhs.start == rhs.start {
                        return lhs.end < rhs.end
                    }
                    return lhs.start < rhs.start
                }
            } catch {
                errorMessage = error.localizedDescription
            }

            isAnalyzing = false
        }
    }

    private func refreshAvailableModels() {
        let discovered = DemoModelCatalog.discover(from: .main)
        availableModels = discovered

        let firstBundledID = discovered.first(where: \.isBundled)?.id

        if selectedModelID == DemoModelDescriptor.demo.id, let firstBundledID {
            selectedModelID = firstBundledID
        } else if !discovered.contains(where: { $0.id == selectedModelID }) {
            selectedModelID = firstBundledID ?? DemoModelDescriptor.demo.id
        }
    }
}

private func runInference(
    text: String,
    model: DemoModelDescriptor
) async throws -> [DetectedEntity] {
    switch model.runtimeKind {
    case .demo:
        return mockEntities(for: text)
    case .bundled:
        return try await OpenMedRuntimeCache.shared.analyze(text: text, using: model)
    }
}

private func mockEntities(for text: String) -> [DetectedEntity] {
    var found: [DetectedEntity] = []

    let patterns: [(String, String, String)] = [
        ("John Doe", "first_name", "NAME"),
        ("1990-05-15", "date_of_birth", "DATE"),
        ("555-123-4567", "phone_number", "PHONE"),
        ("123-45-6789", "ssn", "SSN"),
    ]

    for (needle, label, category) in patterns {
        if let range = text.range(of: needle) {
            let start = text.distance(from: text.startIndex, to: range.lowerBound)
            let end = text.distance(from: text.startIndex, to: range.upperBound)
            found.append(
                DetectedEntity(
                    label: label,
                    text: needle,
                    confidence: Float.random(in: 0.88...0.98),
                    start: start,
                    end: end,
                    category: category
                )
            )
        }
    }

    return found
}

enum DemoError: LocalizedError {
    case invalidBundle(String)

    var errorDescription: String? {
        switch self {
        case .invalidBundle(let detail):
            return "Bundled model is incomplete: \(detail)"
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
    case demo
    case bundled
}

private struct DemoModelDescriptor: Identifiable, Hashable, Sendable {
    let id: String
    let displayName: String
    let sourceModelID: String
    let tokenizerName: String
    let modelURL: URL?
    let id2labelURL: URL?
    let tokenizerFolderURL: URL?
    let runtimeKind: DemoModelRuntimeKind
    let note: String

    static let demo = DemoModelDescriptor(
        id: "demo-mode",
        displayName: "Demo Mode",
        sourceModelID: "UI-only mock entities",
        tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        modelURL: nil,
        id2labelURL: nil,
        tokenizerFolderURL: nil,
        runtimeKind: .demo,
        note: "Works with no bundled model. Useful for validating macOS/iOS UI flow."
    )

    var isBundled: Bool { runtimeKind == .bundled }

    var badgeText: String {
        switch runtimeKind {
        case .demo: return "Demo"
        case .bundled: return "Bundled CoreML"
        }
    }

    var badgeColor: Color {
        switch runtimeKind {
        case .demo: return .orange
        case .bundled: return .green
        }
    }

    var detailText: String {
        switch runtimeKind {
        case .demo:
            return note
        case .bundled:
            if let tokenizerFolderURL {
                return "Runs through OpenMedKit with bundled tokenizer assets at \(tokenizerFolderURL.lastPathComponent)."
            }
            return "Runs through OpenMedKit and uses tokenizerName \(tokenizerName)."
        }
    }

    var searchText: String {
        [displayName, sourceModelID, tokenizerName, note].joined(separator: " ").lowercased()
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
        var discovered: [DemoModelDescriptor] = [.demo]

        if let legacy = discoverLegacyBundle(from: bundle) {
            discovered.append(legacy)
        }

        discovered.append(contentsOf: discoverManifestBundles(from: bundle))

        var seen = Set<String>()
        let unique = discovered.filter { seen.insert($0.id).inserted }
        let demo = unique.filter { $0.runtimeKind == .demo }
        let bundled = unique
            .filter { $0.runtimeKind == .bundled }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }

        return demo + bundled
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

    func analyze(text: String, using model: DemoModelDescriptor) throws -> [DetectedEntity] {
        guard model.runtimeKind == .bundled,
              let modelURL = model.modelURL,
              let id2labelURL = model.id2labelURL else {
            throw DemoError.invalidBundle("Model selection \(model.displayName) is not loadable.")
        }

        let runtime: OpenMed
        if let cached = runtimes[model.id] {
            runtime = cached
        } else {
            _ = try Data(contentsOf: id2labelURL)

            let openmed = try OpenMed(
                modelURL: modelURL,
                id2labelURL: id2labelURL,
                tokenizerName: model.tokenizerName,
                tokenizerFolderURL: model.tokenizerFolderURL
            )
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
        let sorted = entities.sorted { $0.start > $1.start }

        for entity in sorted {
            let startIdx = attrStr.index(attrStr.startIndex, offsetByCharacters: entity.start)
            let endIdx = attrStr.index(attrStr.startIndex, offsetByCharacters: min(entity.end, text.count))
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
                        Text("This demo only lists bundled CoreML models plus Demo Mode. Try a different search, or bundle a model into the app before launching it.")
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
            .searchable(text: $searchText, prompt: "Search bundled models or demo mode")
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
