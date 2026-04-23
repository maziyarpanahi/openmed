import Foundation
import MLX
import MLXNN

private func openMedFloatScalar(_ value: Float, like array: MLXArray) -> MLXArray {
    MLXArray(value).asType(array.dtype)
}

private func openMedLogBucketPosition(
    _ relativePosition: MLXArray,
    bucketSize: Int,
    maxPosition: Int
) -> MLXArray {
    let signValue = sign(relativePosition)
    let mid = bucketSize / 2
    let absPosition = `where`(
        (relativePosition .< mid) .&& (relativePosition .> -mid),
        MLXArray(mid - 1).asType(relativePosition.dtype),
        abs(relativePosition)
    )

    let absPositionFloat = absPosition.asType(.float32)
    let midFloat = MLXArray(Float(mid))
    let logBase = log(MLXArray(Float(maxPosition - 1) / Float(mid)))
    let logPosition = ceil(log(absPositionFloat / midFloat) / logBase * Float(mid - 1)) + Float(mid)
    let bucketPosition = `where`(
        absPosition .<= mid,
        relativePosition.asType(logPosition.dtype),
        logPosition * signValue.asType(logPosition.dtype)
    )
    return bucketPosition.asType(relativePosition.dtype)
}

private func openMedBuildRelativePosition(
    queryLayer: MLXArray,
    keyLayer: MLXArray,
    bucketSize: Int,
    maxPosition: Int
) -> MLXArray {
    let querySize = queryLayer.dim(-2)
    let keySize = keyLayer.dim(-2)
    let queryIDs = MLXArray.arange(querySize, dtype: .int32)
    let keyIDs = MLXArray.arange(keySize, dtype: .int32)
    var relativePosition = queryIDs.expandedDimensions(axis: 1) - keyIDs.expandedDimensions(axis: 0)

    if bucketSize > 0 && maxPosition > 0 {
        relativePosition = openMedLogBucketPosition(
            relativePosition,
            bucketSize: bucketSize,
            maxPosition: maxPosition
        )
    }

    return relativePosition.asType(.int32).expandedDimensions(axis: 0)
}

private func openMedBuildRPosition(
    queryLayer: MLXArray,
    keyLayer: MLXArray,
    relativePosition: MLXArray,
    positionBuckets: Int,
    maxRelativePositions: Int
) -> MLXArray {
    if keyLayer.dim(-2) != queryLayer.dim(-2) {
        return openMedBuildRelativePosition(
            queryLayer: keyLayer,
            keyLayer: keyLayer,
            bucketSize: positionBuckets,
            maxPosition: maxRelativePositions
        )
    }
    return relativePosition
}

private func openMedRepeatBatches(_ array: MLXArray, batchSize: Int) -> MLXArray {
    let targetShape = [batchSize, array.dim(0), array.dim(1), array.dim(2)]
    return broadcast(array.expandedDimensions(axis: 0), to: targetShape)
        .reshaped(batchSize * array.dim(0), array.dim(1), array.dim(2))
}

final class OpenMedDebertaV2Embeddings: Module {
    private let embeddingSize: Int
    private let hiddenSize: Int
    private let positionBiasedInput: Bool

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding?
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding?
    @ModuleInfo(key: "embed_proj") var embedProjection: Linear?
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

    init(_ configuration: OpenMedMLXBertConfiguration) {
        embeddingSize = configuration.embeddingSize
        hiddenSize = configuration.encoderHiddenSize
        positionBiasedInput = configuration.positionBiasedInput

        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: embeddingSize
        )
        if positionBiasedInput {
            _positionEmbeddings.wrappedValue = Embedding(
                embeddingCount: configuration.maxPositionEmbeddings,
                dimensions: embeddingSize
            )
        }
        if configuration.typeVocabularySize > 0 {
            _tokenTypeEmbeddings.wrappedValue = Embedding(
                embeddingCount: configuration.typeVocabularySize,
                dimensions: embeddingSize
            )
        }
        if embeddingSize != hiddenSize {
            _embedProjection.wrappedValue = Linear(embeddingSize, hiddenSize, bias: false)
        }
        _layerNorm.wrappedValue = LayerNorm(
            dimensions: hiddenSize,
            eps: configuration.layerNormEps
        )
    }

    func callAsFunction(
        inputIDs: MLXArray,
        tokenTypeIDs: MLXArray? = nil,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        let seqLen = inputIDs.dim(1)
        var embeddings = wordEmbeddings(inputIDs)

        if let positionEmbeddings {
            let positionIDs = MLXArray.arange(seqLen, dtype: inputIDs.dtype)
                .expandedDimensions(axis: 0)
            embeddings = embeddings + positionEmbeddings(positionIDs)
        }

        if let tokenTypeEmbeddings {
            let tokenTypeIDs = tokenTypeIDs ?? MLXArray.zeros(like: inputIDs)
            embeddings = embeddings + tokenTypeEmbeddings(tokenTypeIDs)
        }

        if let embedProjection {
            embeddings = embedProjection(embeddings)
        }

        embeddings = layerNorm(embeddings)

        if let attentionMask {
            var mask = attentionMask
            if mask.ndim == 4 {
                mask = mask[0..., 0, 0, 0...]
            }
            if mask.ndim == 2 {
                mask = mask.expandedDimensions(axis: -1)
            }
            embeddings = embeddings * mask.asType(embeddings.dtype)
        }

        return embeddings
    }
}

final class OpenMedDisentangledSelfAttention: Module {
    private let numAttentionHeads: Int
    private let attentionHeadSize: Int
    private let allHeadSize: Int
    private let shareAttentionKey: Bool
    private let positionAttentionTypes: Set<String>
    private let relativeAttention: Bool
    private let positionBuckets: Int
    private let maxRelativePositions: Int
    private let positionEmbeddingSize: Int

    @ModuleInfo(key: "query_proj") var queryProjection: Linear
    @ModuleInfo(key: "key_proj") var keyProjection: Linear
    @ModuleInfo(key: "value_proj") var valueProjection: Linear
    @ModuleInfo(key: "pos_key_proj") var positionKeyProjection: Linear?
    @ModuleInfo(key: "pos_query_proj") var positionQueryProjection: Linear?

    init(_ configuration: OpenMedMLXBertConfiguration) {
        numAttentionHeads = configuration.numAttentionHeads
        attentionHeadSize = configuration.encoderHiddenSize / configuration.numAttentionHeads
        allHeadSize = numAttentionHeads * attentionHeadSize
        shareAttentionKey = configuration.shareAttentionKey
        positionAttentionTypes = Set(configuration.positionAttentionTypes)
        relativeAttention = configuration.relativeAttention
        positionBuckets = configuration.positionBuckets
        var resolvedMaxRelativePositions = configuration.maxRelativePositions
        if resolvedMaxRelativePositions < 1 {
            resolvedMaxRelativePositions = configuration.maxPositionEmbeddings
        }
        maxRelativePositions = resolvedMaxRelativePositions
        positionEmbeddingSize =
            positionBuckets > 0 ? positionBuckets : resolvedMaxRelativePositions

        _queryProjection.wrappedValue = Linear(configuration.encoderHiddenSize, allHeadSize)
        _keyProjection.wrappedValue = Linear(configuration.encoderHiddenSize, allHeadSize)
        _valueProjection.wrappedValue = Linear(configuration.encoderHiddenSize, allHeadSize)

        if relativeAttention && !shareAttentionKey {
            if positionAttentionTypes.contains("c2p") {
                _positionKeyProjection.wrappedValue = Linear(
                    configuration.encoderHiddenSize,
                    allHeadSize
                )
            }
            if positionAttentionTypes.contains("p2c") {
                _positionQueryProjection.wrappedValue = Linear(
                    configuration.encoderHiddenSize,
                    allHeadSize
                )
            }
        }
    }

    private func transposeForScores(_ array: MLXArray) -> MLXArray {
        let batchSize = array.dim(0)
        let seqLen = array.dim(1)
        return array
            .reshaped(batchSize, seqLen, numAttentionHeads, attentionHeadSize)
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize * numAttentionHeads, seqLen, attentionHeadSize)
    }

    private func disentangledAttentionBias(
        queryLayer: MLXArray,
        keyLayer: MLXArray,
        relativePosition: MLXArray?,
        relativeEmbeddings: MLXArray,
        scaleFactor: Int
    ) -> MLXArray {
        var relativePosition = relativePosition ?? openMedBuildRelativePosition(
            queryLayer: queryLayer,
            keyLayer: keyLayer,
            bucketSize: positionBuckets,
            maxPosition: maxRelativePositions
        )

        if relativePosition.ndim == 2 {
            relativePosition = relativePosition.expandedDimensions(axes: [0, 1])
        } else if relativePosition.ndim == 3 {
            relativePosition = relativePosition.expandedDimensions(axis: 1)
        }

        relativePosition = relativePosition.asType(.int32)
        let attentionSpan = positionEmbeddingSize
        let relEmbeddings = relativeEmbeddings[0..<(attentionSpan * 2), 0...]
            .expandedDimensions(axis: 0)

        let batchSize = queryLayer.dim(0) / numAttentionHeads
        let positionQueryLayer: MLXArray?
        let positionKeyLayer: MLXArray?
        if shareAttentionKey {
            positionQueryLayer = openMedRepeatBatches(
                transposeForScores(queryProjection(relEmbeddings)),
                batchSize: batchSize
            )
            positionKeyLayer = openMedRepeatBatches(
                transposeForScores(keyProjection(relEmbeddings)),
                batchSize: batchSize
            )
        } else {
            if positionAttentionTypes.contains("p2c"), let positionQueryProjection {
                positionQueryLayer = openMedRepeatBatches(
                    transposeForScores(positionQueryProjection(relEmbeddings)),
                    batchSize: batchSize
                )
            } else {
                positionQueryLayer = nil
            }
            if positionAttentionTypes.contains("c2p"), let positionKeyProjection {
                positionKeyLayer = openMedRepeatBatches(
                    transposeForScores(positionKeyProjection(relEmbeddings)),
                    batchSize: batchSize
                )
            } else {
                positionKeyLayer = nil
            }
        }

        var score: MLXArray?

        if positionAttentionTypes.contains("c2p"), let positionKeyLayer {
            let c2pAttention = queryLayer.matmul(positionKeyLayer.transposed(0, 2, 1))
            let c2pPosition = clip(
                relativePosition + attentionSpan,
                min: 0,
                max: attentionSpan * 2 - 1
            )
            let c2pIndex = broadcast(
                c2pPosition.squeezed(axis: 0),
                to: [queryLayer.dim(0), queryLayer.dim(1), relativePosition.dim(-1)]
            ).asType(.int32)
            let c2pScale = openMedFloatScalar(
                sqrt(Float(positionKeyLayer.dim(-1) * scaleFactor)),
                like: c2pAttention
            )
            let c2pScore = takeAlong(c2pAttention, c2pIndex, axis: -1) / c2pScale
            score = c2pScore
        }

        if positionAttentionTypes.contains("p2c"), let positionQueryLayer {
            let rPosition = openMedBuildRPosition(
                queryLayer: queryLayer,
                keyLayer: keyLayer,
                relativePosition: relativePosition,
                positionBuckets: positionBuckets,
                maxRelativePositions: maxRelativePositions
            )
            let p2cPosition = clip(
                -rPosition + attentionSpan,
                min: 0,
                max: attentionSpan * 2 - 1
            )
            let p2cAttention = keyLayer.matmul(positionQueryLayer.transposed(0, 2, 1))
            let p2cIndex = broadcast(
                p2cPosition.squeezed(axis: 0),
                to: [queryLayer.dim(0), keyLayer.dim(-2), keyLayer.dim(-2)]
            ).asType(.int32)
            let p2cScale = openMedFloatScalar(
                sqrt(Float(positionQueryLayer.dim(-1) * scaleFactor)),
                like: p2cAttention
            )
            let p2cScore = takeAlong(p2cAttention, p2cIndex, axis: -1)
                .transposed(0, 2, 1)
                / p2cScale
            score = score.map { $0 + p2cScore } ?? p2cScore
        }

        return score ?? MLXArray.zeros(
            [queryLayer.dim(0), queryLayer.dim(1), keyLayer.dim(1)],
            type: Float.self
        )
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        queryStates: MLXArray? = nil,
        relativePosition: MLXArray? = nil,
        relativeEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let queryStates = queryStates ?? hiddenStates
        let queryLayer = transposeForScores(queryProjection(queryStates))
        let keyLayer = transposeForScores(keyProjection(hiddenStates))
        let valueLayer = transposeForScores(valueProjection(hiddenStates))

        var scaleFactor = 1
        if positionAttentionTypes.contains("c2p") {
            scaleFactor += 1
        }
        if positionAttentionTypes.contains("p2c") {
            scaleFactor += 1
        }

        let scale = openMedFloatScalar(
            sqrt(Float(queryLayer.dim(-1) * scaleFactor)),
            like: keyLayer
        )
        var attentionScores = queryLayer.matmul(keyLayer.transposed(0, 2, 1) / scale)

        if relativeAttention, let relativeEmbeddings {
            attentionScores = attentionScores + disentangledAttentionBias(
                queryLayer: queryLayer,
                keyLayer: keyLayer,
                relativePosition: relativePosition,
                relativeEmbeddings: relativeEmbeddings,
                scaleFactor: scaleFactor
            )
        }

        let batchSize = hiddenStates.dim(0)
        let queryLen = queryLayer.dim(1)
        let keyLen = keyLayer.dim(1)
        attentionScores = attentionScores.reshaped(
            batchSize,
            numAttentionHeads,
            queryLen,
            keyLen
        )

        attentionScores = `where`(
            attentionMask .> 0,
            attentionScores,
            openMedFloatScalar(-3.4028235e38, like: attentionScores)
        )
        let attentionProbs = softmax(attentionScores, axis: -1)
        let contextLayer = attentionProbs
            .reshaped(batchSize * numAttentionHeads, queryLen, keyLen)
            .matmul(valueLayer)
            .reshaped(batchSize, numAttentionHeads, queryLen, attentionHeadSize)
            .transposed(0, 2, 1, 3)
        return contextLayer.reshaped(batchSize, queryLen, allHeadSize)
    }
}

final class OpenMedDebertaV2Attention: Module {
    @ModuleInfo(key: "self") var selfAttention: OpenMedDisentangledSelfAttention
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    init(_ configuration: OpenMedMLXBertConfiguration) {
        _selfAttention.wrappedValue = OpenMedDisentangledSelfAttention(configuration)
        _outputProjection.wrappedValue = Linear(
            configuration.encoderHiddenSize,
            configuration.encoderHiddenSize
        )
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        queryStates: MLXArray? = nil,
        relativePosition: MLXArray? = nil,
        relativeEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let context = selfAttention(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            queryStates: queryStates,
            relativePosition: relativePosition,
            relativeEmbeddings: relativeEmbeddings
        )
        return outputProjection(context)
    }
}

final class OpenMedDebertaV2Layer: Module {
    private let hiddenActivation: String

    @ModuleInfo(key: "attention") var attention: OpenMedDebertaV2Attention
    @ModuleInfo(key: "ln1") var attentionNorm: LayerNorm
    @ModuleInfo(key: "linear1") var upProjection: Linear
    @ModuleInfo(key: "linear2") var downProjection: Linear
    @ModuleInfo(key: "ln2") var outputNorm: LayerNorm

    init(_ configuration: OpenMedMLXBertConfiguration) {
        hiddenActivation = configuration.hiddenAct
        _attention.wrappedValue = OpenMedDebertaV2Attention(configuration)
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: configuration.encoderHiddenSize,
            eps: configuration.layerNormEps
        )
        _upProjection.wrappedValue = Linear(
            configuration.encoderHiddenSize,
            configuration.intermediateSize
        )
        _downProjection.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.encoderHiddenSize
        )
        _outputNorm.wrappedValue = LayerNorm(
            dimensions: configuration.encoderHiddenSize,
            eps: configuration.layerNormEps
        )
    }

    private func activate(_ array: MLXArray) -> MLXArray {
        switch hiddenActivation {
        case "gelu", "gelu_fast", "gelu_new":
            return gelu(array)
        case "relu":
            return relu(array)
        default:
            return gelu(array)
        }
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        attentionMask: MLXArray,
        queryStates: MLXArray? = nil,
        relativePosition: MLXArray? = nil,
        relativeEmbeddings: MLXArray? = nil
    ) -> MLXArray {
        let residual = queryStates ?? hiddenStates
        let attentionOutput = attention(
            hiddenStates: hiddenStates,
            attentionMask: attentionMask,
            queryStates: queryStates,
            relativePosition: relativePosition,
            relativeEmbeddings: relativeEmbeddings
        )
        let attentionResidual = attentionNorm(residual + attentionOutput)
        let feedForward = downProjection(activate(upProjection(attentionResidual)))
        return outputNorm(attentionResidual + feedForward)
    }
}

final class OpenMedDebertaV2Encoder: Module {
    private let relativeAttention: Bool
    private let positionBuckets: Int
    private let maxRelativePositions: Int
    private let normalizedRelativeEmbeddings: Set<String>

    @ModuleInfo(key: "layer") var layer: [OpenMedDebertaV2Layer]
    @ModuleInfo(key: "rel_embeddings") var relativeEmbeddings: Embedding?
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm?

    init(_ configuration: OpenMedMLXBertConfiguration) {
        relativeAttention = configuration.relativeAttention
        positionBuckets = configuration.positionBuckets
        var resolvedMaxRelativePositions = configuration.maxRelativePositions
        if resolvedMaxRelativePositions < 1 {
            resolvedMaxRelativePositions = configuration.maxPositionEmbeddings
        }
        maxRelativePositions = resolvedMaxRelativePositions
        normalizedRelativeEmbeddings = Set(
            configuration.normRelativeEmbedding
                .lowercased()
                .split(separator: "|")
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        )

        _layer.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
            OpenMedDebertaV2Layer(configuration)
        }
        if relativeAttention {
            let embeddingCount =
                positionBuckets > 0 ? positionBuckets * 2 : resolvedMaxRelativePositions * 2
            _relativeEmbeddings.wrappedValue = Embedding(
                embeddingCount: embeddingCount,
                dimensions: configuration.encoderHiddenSize
            )
        }
        if normalizedRelativeEmbeddings.contains("layer_norm") {
            _layerNorm.wrappedValue = LayerNorm(
                dimensions: configuration.encoderHiddenSize,
                eps: configuration.layerNormEps
            )
        }
    }

    private func getRelativeEmbeddings() -> MLXArray? {
        guard relativeAttention, let relativeEmbeddings else {
            return nil
        }
        if normalizedRelativeEmbeddings.contains("layer_norm"), let layerNorm {
            return layerNorm(relativeEmbeddings.weight)
        }
        return relativeEmbeddings.weight
    }

    private func getAttentionMask(_ attentionMask: MLXArray) -> MLXArray {
        if attentionMask.ndim <= 2 {
            return attentionMask
                .expandedDimensions(axis: 1)
                .expandedDimensions(axis: 3)
                * attentionMask
                .expandedDimensions(axis: 1)
                .expandedDimensions(axis: 2)
        }
        if attentionMask.ndim == 3 {
            return attentionMask.expandedDimensions(axis: 1)
        }
        return attentionMask
    }

    private func getRelativePosition(
        hiddenStates: MLXArray,
        queryStates: MLXArray? = nil,
        relativePosition: MLXArray? = nil
    ) -> MLXArray? {
        guard relativeAttention, relativePosition == nil else {
            return relativePosition
        }
        if let queryStates {
            return openMedBuildRelativePosition(
                queryLayer: queryStates,
                keyLayer: hiddenStates,
                bucketSize: positionBuckets,
                maxPosition: maxRelativePositions
            )
        }
        return openMedBuildRelativePosition(
            queryLayer: hiddenStates,
            keyLayer: hiddenStates,
            bucketSize: positionBuckets,
            maxPosition: maxRelativePositions
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray, attentionMask: MLXArray) -> MLXArray {
        let attentionMask = getAttentionMask(attentionMask)
        let relativePosition = getRelativePosition(hiddenStates: hiddenStates)
        let relativeEmbeddings = getRelativeEmbeddings()

        var states = hiddenStates
        for layer in layer {
            states = layer(
                hiddenStates: states,
                attentionMask: attentionMask,
                relativePosition: relativePosition,
                relativeEmbeddings: relativeEmbeddings
            )
        }
        return states
    }
}

final class OpenMedDebertaV2Model: Module {
    @ModuleInfo(key: "embeddings") var embeddings: OpenMedDebertaV2Embeddings
    @ModuleInfo(key: "encoder") var encoder: OpenMedDebertaV2Encoder

    init(_ configuration: OpenMedMLXBertConfiguration) {
        _embeddings.wrappedValue = OpenMedDebertaV2Embeddings(configuration)
        _encoder.wrappedValue = OpenMedDebertaV2Encoder(configuration)
    }

    func callAsFunction(
        inputIDs: MLXArray,
        attentionMask: MLXArray? = nil,
        tokenTypeIDs: MLXArray? = nil
    ) -> MLXArray {
        let resolvedAttentionMask = attentionMask ?? MLXArray.ones(inputIDs.shape, type: Float.self)
        let embeddingOutput = embeddings(
            inputIDs: inputIDs,
            tokenTypeIDs: tokenTypeIDs,
            attentionMask: resolvedAttentionMask
        )
        return encoder(embeddingOutput, attentionMask: resolvedAttentionMask)
    }
}
