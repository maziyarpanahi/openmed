import SwiftUI
import CoreML

struct ContentView: View {
    @State private var inputText = "Patient John Doe, DOB 1990-05-15, phone 555-123-4567, SSN 123-45-6789. Diagnosed with Type 2 diabetes."
    @State private var entities: [DetectedEntity] = []
    @State private var isAnalyzing = false
    @State private var errorMessage: String?
    @State private var inferenceTime: Double?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Input section
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

                // Analyze button
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

                // Error message
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

                // Results
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

                        // Highlighted text
                        HighlightedTextView(text: inputText, entities: entities)
                            .padding(12)
                            .background(Color.gray.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 10))

                        // Entity list
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
    }

    private func analyzeText() {
        isAnalyzing = true
        errorMessage = nil
        entities = []

        Task {
            let start = CFAbsoluteTimeGetCurrent()

            do {
                let result = try await runInference(text: inputText)
                inferenceTime = CFAbsoluteTimeGetCurrent() - start
                entities = result
            } catch {
                errorMessage = error.localizedDescription
            }

            isAnalyzing = false
        }
    }
}

// MARK: - Inference

private func runInference(text: String) async throws -> [DetectedEntity] {
    // Try loading the bundled CoreML model
    // The model must be added to the Xcode project as OpenMedPII.mlmodelc
    guard let modelURL = Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc") else {
        // Fallback: demo mode with mock data when no model is bundled
        return mockEntities(for: text)
    }

    guard let labelsURL = Bundle.main.url(forResource: "id2label", withExtension: "json") else {
        throw DemoError.missingFile("id2label.json")
    }

    // Verify the model loads (validates the .mlmodelc is valid)
    _ = try MLModel(contentsOf: modelURL)
    _ = try Data(contentsOf: labelsURL)

    // TODO: Replace with real inference once OpenMedKit is wired up:
    //   let openmed = try OpenMed(modelURL: modelURL, id2labelURL: labelsURL)
    //   return try openmed.analyzeText(text).map { ... }
    //
    // For now, use mock entities to demonstrate the UI flow.
    return mockEntities(for: text)
}

/// Demo entities for when no CoreML model is bundled.
/// Replace with real inference once you've converted a model.
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
            found.append(DetectedEntity(
                label: label,
                text: needle,
                confidence: Float.random(in: 0.88...0.98),
                start: start,
                end: end,
                category: category
            ))
        }
    }

    return found
}

enum DemoError: LocalizedError {
    case missingFile(String)

    var errorDescription: String? {
        switch self {
        case .missingFile(let name):
            return "Missing required file: \(name). See README for setup instructions."
        }
    }
}

// MARK: - Data Model

struct DetectedEntity: Identifiable {
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

// MARK: - Highlighted Text View

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

        // Sort entities by start position (reverse to avoid offset issues)
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

// MARK: - Entity Row

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

#Preview {
    ContentView()
}
