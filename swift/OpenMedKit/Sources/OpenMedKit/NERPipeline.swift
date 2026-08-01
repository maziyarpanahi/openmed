import CoreML
import Foundation

/// Runs NER / token-classification inference using a CoreML model.
///
/// The model is expected to accept `input_ids` and `attention_mask` as
/// `MLMultiArray` inputs and produce `logits` of shape `(1, seq_len, num_labels)`.
public class NERPipeline {

    private let model: MLModel
    private let id2label: [Int: String]
    private let maxSeqLength: Int

    /// Initialize the pipeline with a compiled CoreML model.
    ///
    /// - Parameters:
    ///   - modelURL: Path to the `.mlmodelc` or `.mlpackage` file.
    ///   - id2labelURL: Path to the `id2label.json` file mapping label IDs to names.
    ///   - maxSeqLength: Maximum input sequence length the model supports.
    @available(watchOS, unavailable, message: "Use PlatformModel for Nano budget enforcement.")
    @available(visionOS, unavailable, message: "Use PlatformModel for Nano budget enforcement.")
    public convenience init(
        modelURL: URL,
        id2labelURL: URL,
        maxSeqLength: Int = 512
    ) throws {
        try self.init(
            resolvedModelURL: Self.resolveModelURL(modelURL),
            id2labelURL: id2labelURL,
            maxSeqLength: maxSeqLength
        )
    }

    convenience init(
        validatedDescriptor descriptor: PlatformModelDescriptor,
        configuration: PlatformModelConfiguration
    ) throws {
        guard configuration.allows(descriptor) else {
            throw PlatformModelError.noCompatibleModel(configuration.platform)
        }
        try self.init(
            resolvedModelURL: Self.resolveModelURL(descriptor.modelURL),
            id2labelURL: descriptor.id2labelURL,
            maxSeqLength: configuration.maximumSequenceLength
        )
    }

    private init(
        resolvedModelURL: URL,
        id2labelURL: URL,
        maxSeqLength: Int
    ) throws {
        self.model = try MLModel(contentsOf: resolvedModelURL)
        self.maxSeqLength = maxSeqLength

        let data = try Data(contentsOf: id2labelURL)
        let raw = try JSONDecoder().decode([String: String].self, from: data)
        self.id2label = Dictionary(
            uniqueKeysWithValues: raw.compactMap { k, v in
                Int(k).map { ($0, v) }
            })
    }

    private static func resolveModelURL(_ modelURL: URL) throws -> URL {
        switch modelURL.pathExtension.lowercased() {
        case "mlpackage", "mlmodel":
            #if os(watchOS) || os(visionOS)
                throw NERPipelineError.uncompiledModelUnsupported(modelURL)
            #else
                return try MLModel.compileModel(at: modelURL)
            #endif
        default:
            return modelURL
        }
    }

    /// Run token classification on the given token IDs and offsets.
    ///
    /// - Parameters:
    ///   - inputIds: Token IDs from the tokenizer.
    ///   - attentionMask: Attention mask (1 for real tokens, 0 for padding).
    ///   - offsets: Character-level `(start, end)` offsets for each token.
    ///   - text: The original input text (used for span extraction).
    ///   - strategy: Aggregation strategy for multi-token entities.
    /// - Returns: An array of `EntityPrediction` instances.
    public func predict(
        inputIds: [Int],
        attentionMask: [Int],
        offsets: [(Int, Int)],
        text: String,
        strategy: PostProcessing.AggregationStrategy = .average
    ) throws -> [EntityPrediction] {
        let seqLen = inputIds.count

        // Create MLMultiArray inputs
        let inputIdsArray = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)
        let maskArray = try MLMultiArray(shape: [1, seqLen as NSNumber], dataType: .int32)

        for i in 0..<seqLen {
            inputIdsArray[[0, i] as [NSNumber]] = NSNumber(value: inputIds[i])
            maskArray[[0, i] as [NSNumber]] = NSNumber(value: attentionMask[i])
        }

        // Create input feature provider
        let inputFeatures = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: inputIdsArray),
            "attention_mask": MLFeatureValue(multiArray: maskArray),
        ])

        // Run inference
        let output = try model.prediction(from: inputFeatures)

        guard let logitsValue = output.featureValue(for: "logits"),
            let logits = logitsValue.multiArrayValue
        else {
            throw NERPipelineError.missingOutput("logits")
        }

        // Decode logits → per-token predictions
        let numLabels = id2label.count
        var tokenPredictions: [PostProcessing.TokenPrediction] = []

        for i in 0..<seqLen {
            let offset = offsets[i]
            // Skip special tokens ([CLS], [SEP], padding)
            if offset.0 == 0 && offset.1 == 0 { continue }

            // Softmax over labels
            var maxScore: Float = -.infinity
            var maxLabelId = 0
            var expSum: Float = 0.0

            for j in 0..<numLabels {
                let val = logits[[0, i, j] as [NSNumber]].floatValue
                if val > maxScore {
                    maxScore = val
                    maxLabelId = j
                }
                expSum += exp(val)
            }

            let score = exp(maxScore) / expSum
            let label = id2label[maxLabelId] ?? "O"

            if label != "O" {
                tokenPredictions.append(
                    PostProcessing.TokenPrediction(
                        labelId: maxLabelId,
                        label: label,
                        score: score,
                        startOffset: offset.0,
                        endOffset: offset.1
                    ))
            } else {
                // Still pass "O" tokens for boundary detection
                tokenPredictions.append(
                    PostProcessing.TokenPrediction(
                        labelId: maxLabelId,
                        label: "O",
                        score: score,
                        startOffset: offset.0,
                        endOffset: offset.1
                    ))
            }
        }

        return PostProcessing.decodeEntities(
            tokens: tokenPredictions,
            text: text,
            strategy: strategy
        )
    }
}

/// Errors thrown by the NER pipeline.
public enum NERPipelineError: Error, LocalizedError {
    case missingOutput(String)
    case uncompiledModelUnsupported(URL)

    public var errorDescription: String? {
        switch self {
        case .missingOutput(let name):
            return "CoreML model output '\(name)' not found"
        case .uncompiledModelUnsupported(let url):
            return "\(url.lastPathComponent) must be compiled to .mlmodelc before bundling on watchOS or visionOS"
        }
    }
}
