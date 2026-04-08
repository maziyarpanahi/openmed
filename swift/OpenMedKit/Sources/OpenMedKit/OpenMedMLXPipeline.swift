import Foundation
import MLX

enum OpenMedMLXRuntimeError: LocalizedError {
    case unsupportedPlatform

    var errorDescription: String? {
        switch self {
        case .unsupportedPlatform:
            return "Swift MLX inference requires Apple Silicon macOS or a real iPhone/iPad device."
        }
    }
}

final class MLXTokenClassificationPipeline {
    private let artifact: OpenMedMLXArtifact
    private let model: OpenMedBertForTokenClassification
    private let maxSeqLength: Int

    init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
        guard Self.isRuntimeSupported else {
            throw OpenMedMLXRuntimeError.unsupportedPlatform
        }

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
        self.artifact = artifact
        self.model = try OpenMedMLXModelLoader.loadTokenClassifier(from: artifact)
        self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
    }

    var tokenizerDirectoryURL: URL? {
        artifact.tokenizerDirectoryURL
    }

    var tokenizerName: String? {
        artifact.tokenizerName
    }

    var resolvedMaxSequenceLength: Int {
        maxSeqLength
    }

    func predict(
        inputIDs: [Int],
        attentionMask: [Int],
        tokenTypeIDs: [Int],
        offsets: [(Int, Int)],
        text: String,
        strategy: PostProcessing.AggregationStrategy = .average
    ) throws -> [EntityPrediction] {
        let sequenceLength = inputIDs.count

        let inputArray = MLXArray(inputIDs, [1, sequenceLength])
        let attentionArray = MLXArray(attentionMask, [1, sequenceLength]).asType(.float32)
        let tokenTypeArray: MLXArray? =
            artifact.configuration.typeVocabularySize > 0
            ? MLXArray(tokenTypeIDs, [1, sequenceLength])
            : nil

        let logits = model(
            inputArray,
            tokenTypeIDs: tokenTypeArray,
            attentionMask: attentionArray
        )
        eval(logits)

        let probabilities = softmax(logits[0], axis: -1)
        let predictions = probabilities.argMax(axis: -1)
        eval(probabilities, predictions)

        let flatProbabilities = probabilities.asArray(Float.self)
        let predictedLabelIDs = predictions.asArray(Int32.self).map(Int.init)
        let numLabels = artifact.configuration.numLabels

        var tokenPredictions = [PostProcessing.TokenPrediction]()
        tokenPredictions.reserveCapacity(sequenceLength)

        for tokenIndex in 0..<sequenceLength {
            let offset = offsets[tokenIndex]
            if offset == (0, 0) {
                continue
            }

            let labelID = predictedLabelIDs[tokenIndex]
            let label = artifact.id2label[labelID] ?? "O"
            let score = flatProbabilities[tokenIndex * numLabels + labelID]

            tokenPredictions.append(
                PostProcessing.TokenPrediction(
                    labelId: labelID,
                    label: label,
                    score: score,
                    startOffset: offset.0,
                    endOffset: offset.1
                )
            )
        }

        return PostProcessing.decodeEntities(
            tokens: tokenPredictions,
            text: text,
            strategy: strategy
        )
    }

    private static var isRuntimeSupported: Bool {
        #if targetEnvironment(simulator)
        false
        #elseif arch(arm64) || arch(arm64e)
        true
        #else
        false
        #endif
    }
}
