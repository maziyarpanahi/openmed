import Foundation
import MLX
import MLXNN

private final class OpenMedBertEmbeddings: Module {
    private let typeVocabularySize: Int
    private let positionOffset: Int

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(_ configuration: OpenMedMLXBertConfiguration) {
        typeVocabularySize = configuration.typeVocabularySize
        positionOffset = configuration.positionOffset

        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        _positionEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.maxPositionEmbeddings,
            dimensions: configuration.hiddenSize
        )
        if configuration.typeVocabularySize > 0 {
            _tokenTypeEmbeddings.wrappedValue = Embedding(
                embeddingCount: configuration.typeVocabularySize,
                dimensions: configuration.hiddenSize
            )
        }
        _norm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        tokenTypeIDs: MLXArray?
    ) -> MLXArray {
        let positionIDs = broadcast(
            MLXArray.arange(inputIDs.dim(1)) + positionOffset,
            to: inputIDs.shape
        )

        var embeddings = wordEmbeddings(inputIDs) + positionEmbeddings(positionIDs)
        if typeVocabularySize > 0, let tokenTypeEmbeddings {
            let tokenTypeIDs = tokenTypeIDs ?? MLXArray.zeros(like: inputIDs)
            embeddings += tokenTypeEmbeddings(tokenTypeIDs)
        }

        return norm(embeddings)
    }
}

private final class OpenMedBertEncoderLayer: Module {
    let attention: MultiHeadAttention

    @ModuleInfo(key: "ln1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "ln2") var outputNorm: LayerNorm
    @ModuleInfo(key: "linear1") var upProjection: Linear
    @ModuleInfo(key: "linear2") var downProjection: Linear

    init(_ configuration: OpenMedMLXBertConfiguration) {
        attention = MultiHeadAttention(
            dimensions: configuration.hiddenSize,
            numHeads: configuration.numAttentionHeads,
            bias: true
        )
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
        _outputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.layerNormEps
        )
        _upProjection.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize
        )
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize
        )
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?) -> MLXArray {
        let attentionOutput = attention(inputs, keys: inputs, values: inputs, mask: mask)
        let attentionResidual = attentionNorm(inputs + attentionOutput)
        let feedForwardOutput = downProjection(gelu(upProjection(attentionResidual)))
        return outputNorm(attentionResidual + feedForwardOutput)
    }
}

private final class OpenMedBertEncoder: Module {
    let layers: [OpenMedBertEncoderLayer]

    init(_ configuration: OpenMedMLXBertConfiguration) {
        layers = (0..<configuration.numHiddenLayers).map { _ in
            OpenMedBertEncoderLayer(configuration)
        }
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?) -> MLXArray {
        var hiddenStates = inputs
        for layer in layers {
            hiddenStates = layer(hiddenStates, mask: mask)
        }
        return hiddenStates
    }
}

final class OpenMedBertForTokenClassification: Module {
    @ModuleInfo(key: "embeddings") fileprivate var embeddings: OpenMedBertEmbeddings
    @ModuleInfo(key: "classifier") var classifier: Linear

    let configuration: OpenMedMLXBertConfiguration
    fileprivate let encoder: OpenMedBertEncoder

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.configuration = configuration
        self.encoder = OpenMedBertEncoder(configuration)
        _embeddings.wrappedValue = OpenMedBertEmbeddings(configuration)
        _classifier.wrappedValue = Linear(configuration.hiddenSize, configuration.numLabels)
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        tokenTypeIDs: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        var inputs = inputIDs
        if inputs.ndim == 1 {
            inputs = inputs.reshaped(1, -1)
        }

        let embedded = embeddings(inputs, tokenTypeIDs: tokenTypeIDs)
        let mask: MLXArray?
        if let attentionMask {
            mask = attentionMask
                .asType(embedded.dtype)
                .expandedDimensions(axes: [1, 2])
                .log()
        } else {
            mask = nil
        }

        let hiddenStates = encoder(embedded, mask: mask)
        return classifier(hiddenStates)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            key != "embeddings.position_ids" && !key.hasPrefix("_")
        }
    }
}

enum OpenMedMLXModelLoader {
    private static func loadedWeights(for artifact: OpenMedMLXArtifact) throws -> [String: MLXArray] {
        try OpenMedMLXWeightArchive.loadWeights(from: artifact.weightCandidateURLs)
    }

    static func loadTokenClassifier(
        from artifact: OpenMedMLXArtifact
    ) throws -> OpenMedBertForTokenClassification {
        var weights = try loadedWeights(for: artifact)
        let model = OpenMedBertForTokenClassification(artifact.configuration)
        weights = model.sanitize(weights: weights)

        if let bits = artifact.configuration.quantizationBits {
            quantize(model: model) { path, _ in
                if weights["\(path).scales"] != nil {
                    return (64, bits, .affine)
                } else {
                    return nil
                }
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func loadPrivacyFilter(
        from artifact: OpenMedMLXArtifact
    ) throws -> OpenMedPrivacyFilterForTokenClassification {
        var weights = try loadedWeights(for: artifact)
        let model = OpenMedPrivacyFilterForTokenClassification(artifact.configuration)
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func loadGLiNERSpanModel(
        from artifact: OpenMedMLXArtifact
    ) throws -> OpenMedGLiNERSpanModel {
        var weights = try loadedWeights(for: artifact)
        let model = OpenMedGLiNERSpanModel(artifact.configuration)
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func loadGLiClassUniEncoderModel(
        from artifact: OpenMedMLXArtifact
    ) throws -> OpenMedGLiClassUniEncoderModel {
        var weights = try loadedWeights(for: artifact)
        let model = OpenMedGLiClassUniEncoderModel(artifact.configuration)
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func loadGLiNERRelexModel(
        from artifact: OpenMedMLXArtifact
    ) throws -> OpenMedGLiNERRelexModel {
        var weights = try loadedWeights(for: artifact)
        let model = OpenMedGLiNERRelexModel(artifact.configuration)
        weights = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
}
