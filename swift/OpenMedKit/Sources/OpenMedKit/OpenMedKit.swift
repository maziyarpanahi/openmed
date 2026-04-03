import Foundation

/// OpenMedKit — On-device clinical NLP for iOS and macOS.
///
/// Provides NER and PII detection using CoreML models converted from
/// the OpenMed Python library.
///
/// ## Quick Start
///
/// ```swift
/// let openmed = try OpenMed(
///     modelURL: Bundle.main.url(forResource: "OpenMedPII", withExtension: "mlmodelc")!,
///     id2labelURL: Bundle.main.url(forResource: "id2label", withExtension: "json")!,
///     tokenizerName: "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
/// )
/// let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
/// for entity in entities {
///     print(entity)  // [first_name] "John Doe" (8:16) conf=0.95
/// }
/// ```
public class OpenMed {

    private let pipeline: NERPipeline
    private let tokenizerName: String

    /// Initialize OpenMed with a CoreML model and tokenizer.
    ///
    /// - Parameters:
    ///   - modelURL: URL to the compiled CoreML model (`.mlmodelc` or `.mlpackage`).
    ///   - id2labelURL: URL to the `id2label.json` label mapping file.
    ///   - tokenizerName: HuggingFace tokenizer name for text tokenization.
    ///   - maxSeqLength: Maximum token sequence length (default: 512).
    public init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        maxSeqLength: Int = 512
    ) throws {
        self.pipeline = try NERPipeline(
            modelURL: modelURL,
            id2labelURL: id2labelURL,
            maxSeqLength: maxSeqLength
        )
        self.tokenizerName = tokenizerName
    }

    /// Run NER on the given text and return detected entities.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - confidenceThreshold: Minimum confidence to include an entity (default: 0.5).
    /// - Returns: Array of detected entities above the confidence threshold.
    public func analyzeText(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) throws -> [EntityPrediction] {
        let (inputIds, attentionMask, offsets) = try tokenize(text)

        let entities = try pipeline.predict(
            inputIds: inputIds,
            attentionMask: attentionMask,
            offsets: offsets,
            text: text
        )

        return entities.filter { $0.confidence >= confidenceThreshold }
    }

    /// Run PII detection on the given text.
    ///
    /// Alias for ``analyzeText(_:confidenceThreshold:)`` — the model must be
    /// a PII-trained model for meaningful results.
    public func extractPII(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) throws -> [EntityPrediction] {
        return try analyzeText(text, confidenceThreshold: confidenceThreshold)
    }

    // MARK: - Private

    private func tokenize(_ text: String) throws -> ([Int], [Int], [(Int, Int)]) {
        // Use swift-transformers for tokenization
        // This ensures token IDs match the Python HuggingFace tokenizer
        let tokenizer = try AutoTokenizer.from(pretrained: tokenizerName)
        let encoding = tokenizer(text)

        let inputIds = encoding.inputIds
        let attentionMask = Array(repeating: 1, count: inputIds.count)

        // Build offset mapping from token spans
        // swift-transformers provides offsets via the encoding
        let offsets: [(Int, Int)]
        if let tokenOffsets = encoding.offsets {
            offsets = tokenOffsets.map { ($0.start, $0.end) }
        } else {
            // Fallback: approximate offsets (not ideal)
            offsets = inputIds.enumerated().map { i, _ in
                i == 0 || i == inputIds.count - 1 ? (0, 0) : (i, i + 1)
            }
        }

        return (inputIds, attentionMask, offsets)
    }
}

// Re-export swift-transformers tokenizer
import Transformers
typealias AutoTokenizer = Transformers.AutoTokenizer
