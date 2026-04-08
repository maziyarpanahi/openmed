import Dispatch
import Foundation
import Tokenizers

/// OpenMedKit — On-device clinical NLP for iOS and macOS.
///
/// Provides NER and PII detection using either CoreML or MLX models
/// produced by the OpenMed Python library.
///
/// ## Quick Start
///
/// ```swift
/// let modelDirectory = try await OpenMedModelStore.downloadMLXModel(
///     repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx",
///     authToken: ProcessInfo.processInfo.environment["HF_TOKEN"]
/// )
/// let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))
/// let entities = try openmed.analyzeText("Patient John Doe, SSN 123-45-6789")
/// for entity in entities {
///     print(entity)  // [first_name] "John Doe" (8:16) conf=0.95
/// }
/// ```
public final class OpenMed {
    private enum Runtime {
        case coreML(NERPipeline)
        case mlx(MLXTokenClassificationPipeline)
    }

    private let runtime: Runtime
    private let tokenizer: any Tokenizer
    private let maxSeqLength: Int

    /// Initialize OpenMed with an explicit backend.
    public init(
        backend: OpenMedBackend,
        maxSeqLength: Int = 512
    ) throws {
        switch backend {
        case .coreML(let modelURL, let id2labelURL, let tokenizerName, let tokenizerFolderURL):
            let pipeline = try NERPipeline(
                modelURL: modelURL,
                id2labelURL: id2labelURL,
                maxSeqLength: maxSeqLength
            )
            self.runtime = .coreML(pipeline)
            self.tokenizer = try Self.loadTokenizer(
                tokenizerName: tokenizerName,
                tokenizerFolderURL: tokenizerFolderURL
            )
            self.maxSeqLength = maxSeqLength

        case .mlx(let modelDirectoryURL):
            let pipeline = try MLXTokenClassificationPipeline(
                modelDirectoryURL: modelDirectoryURL,
                maxSeqLength: maxSeqLength
            )
            self.runtime = .mlx(pipeline)
            self.tokenizer = try Self.loadTokenizer(
                tokenizerName: pipeline.tokenizerName ?? modelDirectoryURL.path,
                tokenizerFolderURL: pipeline.tokenizerDirectoryURL
            )
            self.maxSeqLength = pipeline.resolvedMaxSequenceLength
        }
    }

    /// Initialize OpenMed with a CoreML model and tokenizer.
    ///
    /// - Parameters:
    ///   - modelURL: URL to the compiled CoreML model (`.mlmodelc` or `.mlpackage`).
    ///   - id2labelURL: URL to the `id2label.json` label mapping file.
    ///   - tokenizerName: HuggingFace tokenizer name for text tokenization.
    ///   - tokenizerFolderURL: Optional local tokenizer asset directory for offline use.
    ///   - maxSeqLength: Maximum token sequence length (default: 512).
    public convenience init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        tokenizerFolderURL: URL? = nil,
        maxSeqLength: Int = 512
    ) throws {
        try self.init(
            backend: .coreML(
                modelURL: modelURL,
                id2labelURL: id2labelURL,
                tokenizerName: tokenizerName,
                tokenizerFolderURL: tokenizerFolderURL
            ),
            maxSeqLength: maxSeqLength
        )
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
        let (inputIDs, attentionMask, tokenTypeIDs, offsets) = try tokenize(text)

        let entities: [EntityPrediction]
        switch runtime {
        case .coreML(let pipeline):
            entities = try pipeline.predict(
                inputIds: inputIDs,
                attentionMask: attentionMask,
                offsets: offsets,
                text: text
            )
        case .mlx(let pipeline):
            entities = try pipeline.predict(
                inputIDs: inputIDs,
                attentionMask: attentionMask,
                tokenTypeIDs: tokenTypeIDs,
                offsets: offsets,
                text: text
            )
        }

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

    private func tokenize(_ text: String) throws -> ([Int], [Int], [Int], [(Int, Int)]) {
        // Use swift-transformers for tokenization
        // This ensures token IDs match the Python HuggingFace tokenizer
        let inputIds = Array(tokenizer(text, addSpecialTokens: true).prefix(maxSeqLength))
        let tokens = tokenizer.convertIdsToTokens(inputIds).map { $0 ?? "" }
        let attentionMask = Array(repeating: 1, count: inputIds.count)
        let tokenTypeIDs = Array(repeating: 0, count: inputIds.count)
        let offsets = Self.buildOffsets(tokens: tokens, in: text)

        return (inputIds, attentionMask, tokenTypeIDs, offsets)
    }

    private static func loadTokenizer(
        tokenizerName: String,
        tokenizerFolderURL: URL?
    ) throws -> any Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<any Tokenizer, Error>?

        Task.detached {
            do {
                let tokenizer: any Tokenizer
                if let tokenizerFolderURL {
                    tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerFolderURL)
                } else {
                    tokenizer = try await AutoTokenizer.from(pretrained: tokenizerName)
                }
                result = .success(tokenizer)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        return try result!.get()
    }

    private static func buildOffsets(
        tokens: [String],
        in text: String
    ) -> [(Int, Int)] {
        var offsets: [(Int, Int)] = []
        var cursor = text.startIndex

        for token in tokens {
            if isSpecialToken(token) {
                offsets.append((0, 0))
                continue
            }

            let normalized = normalize(token: token)
            let piece = normalized.piece

            if piece.isEmpty {
                offsets.append((0, 0))
                continue
            }

            var searchStart = cursor
            if normalized.skipLeadingWhitespace {
                while searchStart < text.endIndex && text[searchStart].isWhitespace {
                    searchStart = text.index(after: searchStart)
                }
            }

            if let range = text[searchStart...].range(of: piece) ??
                text[searchStart...].range(
                    of: piece,
                    options: [.caseInsensitive, .diacriticInsensitive]
                ) {
                let start = text.distance(from: text.startIndex, to: range.lowerBound)
                let end = text.distance(from: text.startIndex, to: range.upperBound)
                offsets.append((start, end))
                cursor = range.upperBound
                continue
            }

            let start = text.distance(from: text.startIndex, to: searchStart)
            let endIndex = text.index(
                searchStart,
                offsetBy: piece.count,
                limitedBy: text.endIndex
            ) ?? text.endIndex
            let end = text.distance(from: text.startIndex, to: endIndex)
            offsets.append((start, end))
            cursor = endIndex
        }

        return offsets
    }

    private static func normalize(token: String) -> (piece: String, skipLeadingWhitespace: Bool) {
        if token == "Ċ" {
            return ("\n", false)
        }
        if token.hasPrefix("##") {
            return (String(token.dropFirst(2)), false)
        }
        if token.hasPrefix("▁") || token.hasPrefix("Ġ") {
            return (String(token.dropFirst()), true)
        }
        return (token, false)
    }

    private static func isSpecialToken(_ token: String) -> Bool {
        switch token {
        case "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>", "<pad>", "<mask>":
            return true
        default:
            return false
        }
    }
}
