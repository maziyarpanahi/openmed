import Foundation
import MLX
import MLXNN

private func privacyFilterParameterDType(_ configuration: OpenMedMLXBertConfiguration) -> DType {
    switch configuration.parameterDType.lowercased() {
    case "bf16", "bfloat16":
        return .bfloat16
    default:
        return .float32
    }
}

private func privacyFilterScalar(_ value: Float, like array: MLXArray) -> MLXArray {
    MLXArray(value).asType(array.dtype)
}

private func privacyFilterLinearInput(_ input: MLXArray, for linear: Linear) -> MLXArray {
    if linear is Quantized {
        return input
    }
    return input.asType(linear.weight.dtype)
}

private func privacyFilterExpertInput(
    _ input: MLXArray,
    for expert: OpenMedPrivacyFilterExpertLinear
) -> MLXArray {
    if expert is Quantized {
        return input
    }
    return input.asType(expert.weight.dtype)
}

private final class OpenMedPrivacyFilterRMSNorm: Module {
    private let eps: Float

    @ParameterInfo(key: "scale") var scale: MLXArray

    init(hiddenSize: Int, eps: Float) {
        self.eps = eps
        _scale.wrappedValue = MLXArray.ones([hiddenSize], type: Float.self)
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let dtype = input.dtype
        let normalized = input.asType(.float32)
        let variance = mean(normalized * normalized, axis: -1, keepDims: true)
        return (normalized * rsqrt(variance + MLXArray(eps)) * scale).asType(dtype)
    }
}

private func applyPrivacyFilterRotaryEmbedding(
    _ input: MLXArray,
    cos: MLXArray,
    sin: MLXArray
) -> MLXArray {
    let dtype = input.dtype
    let shape = input.shape
    let headDim = shape[3]
    let paired = input.reshaped(shape[0], shape[1], shape[2], headDim / 2, 2)
    let x1 = paired[0..., 0..., 0..., 0..., 0]
    let x2 = paired[0..., 0..., 0..., 0..., 1]
    let cos = cos.expandedDimensions(axes: [0, 2]).asType(dtype)
    let sin = sin.expandedDimensions(axes: [0, 2]).asType(dtype)
    let rotated = stacked([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis: -1)
    return rotated.reshaped(shape).asType(dtype)
}

private final class OpenMedPrivacyFilterRotaryEmbedding {
    private let headDim: Int
    private let base: Float
    private let initialContextLength: Int
    private let scalingFactor: Float
    private let ntkAlpha: Float
    private let ntkBeta: Float

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.headDim = configuration.headDim
        self.base = configuration.ropeTheta
        self.initialContextLength = configuration.initialContextLength
        self.scalingFactor = configuration.ropeScalingFactor
        self.ntkAlpha = configuration.ropeNTKAlpha
        self.ntkBeta = configuration.ropeNTKBeta
    }

    private func cosSin(numTokens: Int) -> (MLXArray, MLXArray) {
        let halfDim = headDim / 2
        let exponents = MLXArray.arange(0, headDim, step: 2, dtype: .float32) / Float(headDim)
        let frequency = pow(base, exponents)

        let inverseFrequency: MLXArray
        let concentration: Float
        if scalingFactor > 1.0 {
            concentration = 0.1 * Foundation.log(scalingFactor) + 1.0
            let halfDimFloat = Float(headDim) / 2.0
            let denominator = Foundation.log(base)
            let low =
                halfDimFloat
                * Foundation.log(Float(initialContextLength) / (ntkBeta * 2.0 * Float.pi))
                / denominator
            let high =
                halfDimFloat
                * Foundation.log(Float(initialContextLength) / (ntkAlpha * 2.0 * Float.pi))
                / denominator
            let interpolation = 1.0 / (scalingFactor * frequency)
            let extrapolation = 1.0 / frequency
            let ramp = (MLXArray.arange(halfDim, dtype: .float32) - low) / (high - low)
            let mask = 1.0 - clip(ramp, min: 0.0, max: 1.0)
            inverseFrequency = interpolation * (1.0 - mask) + extrapolation * mask
        } else {
            concentration = 1.0
            inverseFrequency = 1.0 / frequency
        }

        let positions = MLXArray.arange(numTokens, dtype: .float32)
        let frequencies = positions.expandedDimensions(axis: 1) * inverseFrequency.expandedDimensions(axis: 0)
        return (cos(frequencies) * concentration, sin(frequencies) * concentration)
    }

    func callAsFunction(query: MLXArray, key: MLXArray) -> (MLXArray, MLXArray) {
        let (cos, sin) = cosSin(numTokens: query.dim(1))
        return (
            applyPrivacyFilterRotaryEmbedding(query, cos: cos, sin: sin),
            applyPrivacyFilterRotaryEmbedding(key, cos: cos, sin: sin)
        )
    }
}

private func privacyFilterTopK(_ values: MLXArray, k: Int) -> (MLXArray, MLXArray) {
    let indices = argPartition(-values, kth: k - 1, axis: -1)[0..., 0..<k]
    var topValues = takeAlong(values, indices, axis: -1)
    let order = argSort(-topValues, axis: -1)
    let sortedIndices = takeAlong(indices, order, axis: -1).asType(.int32)
    topValues = takeAlong(topValues, order, axis: -1)
    return (topValues, sortedIndices)
}

private func privacyFilterSwiGLU(
    _ input: MLXArray,
    alpha: Float = 1.702,
    limit: Float
) -> MLXArray {
    let half = input.dim(-1) / 2
    let glu = minimum(input[0..., 0..., 0..<half], MLXArray(limit).asType(input.dtype))
    let linear = clip(
        input[0..., 0..., half..<input.dim(-1)],
        min: -limit,
        max: limit
    )
    return (glu / (1.0 + exp(-alpha * glu))) * (linear + 1.0)
}

fileprivate class OpenMedPrivacyFilterExpertLinear: Module, Quantizable {
    fileprivate let numExperts: Int
    fileprivate let inputSize: Int
    fileprivate let outputSize: Int

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(numExperts: Int, inputSize: Int, outputSize: Int, dtype: DType) {
        self.numExperts = numExperts
        self.inputSize = inputSize
        self.outputSize = outputSize
        _weight.wrappedValue =
            MLXArray.zeros([numExperts, inputSize, outputSize], dtype: dtype)
        _bias.wrappedValue =
            MLXArray.zeros([numExperts, outputSize], dtype: dtype)
        super.init()
    }

    fileprivate init(
        numExperts: Int,
        inputSize: Int,
        outputSize: Int,
        weight: MLXArray,
        bias: MLXArray
    ) {
        self.numExperts = numExperts
        self.inputSize = inputSize
        self.outputSize = outputSize
        _weight.wrappedValue = weight
        _bias.wrappedValue = bias
        super.init()
    }

    func callAsFunction(_ input: MLXArray, expertIndices: MLXArray) -> MLXArray {
        let inputShape = input.shape
        let flatInput = input.reshaped(-1, 1, input.dim(-1)).asType(weight.dtype)
        let flatIndices = expertIndices.reshaped(-1).asType(.int32)
        var output = gatherMM(flatInput, weight, rhsIndices: flatIndices).squeezed(axis: -2)
        output = output + bias.take(flatIndices, axis: 0)
        return output.reshaped(Array(inputShape.dropLast()) + [outputSize])
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        OpenMedPrivacyFilterQuantizedExpertLinear(
            self,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

fileprivate final class OpenMedPrivacyFilterQuantizedExpertLinear:
    OpenMedPrivacyFilterExpertLinear, Quantized
{
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode
    let scales: MLXArray
    let biases: MLXArray?

    init(
        _ other: OpenMedPrivacyFilterExpertLinear,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let transposedWeight = other.weight.swappedAxes(-1, -2)
        let quantizedWeights = MLX.quantized(
            transposedWeight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self.scales = quantizedWeights.scales
        self.biases = quantizedWeights.biases
        super.init(
            numExperts: other.numExperts,
            inputSize: other.inputSize,
            outputSize: other.outputSize,
            weight: quantizedWeights.wq,
            bias: other.bias
        )
        freeze()
    }

    /// Shape-only placeholder init used by the quantized loader path. Allocates
    /// empty tensors with the right dtypes/shapes so `update(parameters:)` can
    /// fill them in — avoiding the wasteful quantization of dummy zero weights
    /// that the `init(_ other:)` path performs.
    init(
        numExperts: Int,
        inputSize: Int,
        outputSize: Int,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode,
        dtype: DType
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        let packedFeatures = (inputSize * bits) / 32
        let scaleGroups = inputSize / groupSize
        self.scales = MLXArray.zeros(
            [numExperts, outputSize, scaleGroups], dtype: dtype)
        self.biases = MLXArray.zeros(
            [numExperts, outputSize, scaleGroups], dtype: dtype)
        super.init(
            numExperts: numExperts,
            inputSize: inputSize,
            outputSize: outputSize,
            weight: MLXArray.zeros(
                [numExperts, outputSize, packedFeatures], dtype: .uint32),
            bias: MLXArray.zeros([numExperts, outputSize], dtype: dtype)
        )
        freeze()
    }

    override func callAsFunction(_ input: MLXArray, expertIndices: MLXArray) -> MLXArray {
        let inputShape = input.shape
        let flatInput = input.reshaped(-1, 1, input.dim(-1))
        let flatIndices = expertIndices.reshaped(-1).asType(.int32)
        var output = gatherQuantizedMM(
            flatInput,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: flatIndices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        ).squeezed(axis: -2)
        output = output + bias.take(flatIndices, axis: 0)
        return output.reshaped(Array(inputShape.dropLast()) + [outputSize])
    }
}

private func privacyFilterLocalAttention(
    query: MLXArray,
    key: MLXArray,
    value: MLXArray,
    sinks: MLXArray,
    leftContext: Int,
    rightContext: Int,
    attentionMask: MLXArray?
) -> MLXArray {
    let batchSize = query.dim(0)
    let numTokens = query.dim(1)
    let numKVHeads = query.dim(2)
    let queryMultiplier = query.dim(3)
    let headDim = query.dim(4)
    let window = leftContext + rightContext + 1
    let paddedTokens = numTokens + leftContext + rightContext

    let keyPadded = padded(key, widths: [0, [leftContext, rightContext], 0, 0])
    let valuePadded = padded(value, widths: [0, [leftContext, rightContext], 0, 0])
    let strides = [
        paddedTokens * numKVHeads * headDim,
        numKVHeads * headDim,
        numKVHeads * headDim,
        headDim,
        1,
    ]
    let keyWindows = asStrided(
        keyPadded,
        [batchSize, numTokens, window, numKVHeads, headDim],
        strides: strides
    )
    let valueWindows = asStrided(
        valuePadded,
        [batchSize, numTokens, window, numKVHeads, headDim],
        strides: strides
    )

    var scores = einsum("bthqd,btwhd->bthqw", query, keyWindows).asType(.float32)
    let offsets = MLXArray.arange(window, dtype: .int32) - Int32(leftContext)
    let positions = MLXArray.arange(numTokens, dtype: .int32).expandedDimensions(axis: 1)
        + offsets.expandedDimensions(axis: 0)
    var valid = ((positions .>= 0) & (positions .< Int32(numTokens)))
        .expandedDimensions(axes: [0, 2, 3])

    if let attentionMask {
        let maskPadded = padded(
            attentionMask.asType(.bool),
            widths: [0, [leftContext, rightContext]],
            value: MLXArray(false)
        )
        let maskWindows = asStrided(
            maskPadded,
            [batchSize, numTokens, window],
            strides: [paddedTokens, 1, 1]
        )
        valid = valid & maskWindows.expandedDimensions(axes: [2, 3])
    }

    scores = `where`(
        valid,
        scores,
        privacyFilterScalar(-1.0e9, like: scores)
    )

    let sinkScores = (sinks * Foundation.log(2.0)).reshaped(numKVHeads, queryMultiplier)
    let broadcastSinkScores = broadcast(
        sinkScores.expandedDimensions(axes: [0, 1, 4]),
        to: [batchSize, numTokens, numKVHeads, queryMultiplier, 1]
    )
    let allWeights = softmax(concatenated([scores, broadcastSinkScores], axis: -1), axis: -1)
    let weights = allWeights[0..., 0..., 0..., 0..., 0..<window]
    let attention = einsum("bthqw,btwhd->bthqd", weights.asType(value.dtype), valueWindows)
    return attention.reshaped(batchSize, numTokens, numKVHeads * queryMultiplier * headDim)
}

private final class OpenMedPrivacyFilterAttentionBlock: Module {
    private let headDim: Int
    private let numAttentionHeads: Int
    private let numKeyValueHeads: Int
    private let queryMultiplier: Int
    private let leftContext: Int
    private let rightContext: Int
    private let qkScale: Float
    private let rope: OpenMedPrivacyFilterRotaryEmbedding

    @ModuleInfo(key: "norm") var norm: OpenMedPrivacyFilterRMSNorm
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ParameterInfo(key: "sinks") var sinks: MLXArray

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.headDim = configuration.headDim
        self.numAttentionHeads = configuration.numAttentionHeads
        self.numKeyValueHeads = configuration.numKeyValueHeads
        self.queryMultiplier = configuration.numAttentionHeads / configuration.numKeyValueHeads
        self.leftContext = configuration.bidirectionalLeftContext
        self.rightContext = configuration.bidirectionalRightContext
        self.qkScale = 1.0 / sqrt(sqrt(Float(configuration.headDim)))
        self.rope = OpenMedPrivacyFilterRotaryEmbedding(configuration)
        let qkvSize = configuration.headDim
            * (configuration.numAttentionHeads + 2 * configuration.numKeyValueHeads)

        _norm.wrappedValue = OpenMedPrivacyFilterRMSNorm(
            hiddenSize: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        _qkv.wrappedValue = Linear(configuration.hiddenSize, qkvSize)
        _out.wrappedValue = Linear(configuration.headDim * configuration.numAttentionHeads, configuration.hiddenSize)
        _sinks.wrappedValue = MLXArray.zeros([configuration.numAttentionHeads], type: Float.self)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let batchSize = input.dim(0)
        let numTokens = input.dim(1)
        let qkvStates = qkv(norm(input))
        let queryEnd = numAttentionHeads * headDim
        let keyEnd = queryEnd + numKeyValueHeads * headDim
        var query = qkvStates[0..., 0..., 0..<queryEnd]
            .reshaped(batchSize, numTokens, numAttentionHeads, headDim)
        var key = qkvStates[0..., 0..., queryEnd..<keyEnd]
            .reshaped(batchSize, numTokens, numKeyValueHeads, headDim)
        let value = qkvStates[0..., 0..., keyEnd..<qkvStates.dim(-1)]
            .reshaped(batchSize, numTokens, numKeyValueHeads, headDim)

        (query, key) = rope(query: query, key: key)
        query = (query * qkScale).reshaped(
            batchSize,
            numTokens,
            numKeyValueHeads,
            queryMultiplier,
            headDim
        )
        key = key * qkScale
        let attentionOutput = privacyFilterLocalAttention(
            query: query,
            key: key,
            value: value,
            sinks: sinks,
            leftContext: leftContext,
            rightContext: rightContext,
            attentionMask: attentionMask
        )
        return input + out(privacyFilterLinearInput(attentionOutput, for: out)).asType(input.dtype)
    }
}

private final class OpenMedPrivacyFilterMLPBlock: Module {
    private let expertsPerToken: Int
    private let swigluLimit: Float

    @ModuleInfo(key: "norm") var norm: OpenMedPrivacyFilterRMSNorm
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "swiglu") var swiglu: OpenMedPrivacyFilterExpertLinear
    @ModuleInfo(key: "out") var out: OpenMedPrivacyFilterExpertLinear

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.expertsPerToken = configuration.expertsPerToken
        self.swigluLimit = configuration.swigluLimit
        let dtype = privacyFilterParameterDType(configuration)
        _norm.wrappedValue = OpenMedPrivacyFilterRMSNorm(
            hiddenSize: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        _gate.wrappedValue = Linear(configuration.hiddenSize, configuration.numExperts)
        _swiglu.wrappedValue = OpenMedPrivacyFilterExpertLinear(
            numExperts: configuration.numExperts,
            inputSize: configuration.hiddenSize,
            outputSize: configuration.intermediateSize * 2,
            dtype: dtype
        )
        _out.wrappedValue = OpenMedPrivacyFilterExpertLinear(
            numExperts: configuration.numExperts,
            inputSize: configuration.intermediateSize,
            outputSize: configuration.hiddenSize,
            dtype: dtype
        )
        super.init()
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let batchShape = Array(input.shape.dropLast())
        let hiddenSize = input.dim(-1)
        let normalized = norm(input).reshaped(-1, hiddenSize)
        let gateLogits = gate(privacyFilterLinearInput(normalized, for: gate)).asType(.float32)
        let (expertValues, expertIndices) = privacyFilterTopK(gateLogits, k: expertsPerToken)
        let expertWeights = softmax(expertValues, axis: -1) / Float(expertsPerToken)
        let expandedInput = broadcast(
            privacyFilterExpertInput(normalized, for: swiglu).expandedDimensions(axis: 1),
            to: [normalized.dim(0), expertsPerToken, hiddenSize]
        )
        var hidden = swiglu(expandedInput, expertIndices: expertIndices).asType(.float32)
        hidden = privacyFilterSwiGLU(hidden, limit: swigluLimit)
        let output = out(
            privacyFilterExpertInput(hidden, for: out),
            expertIndices: expertIndices
        ).asType(.float32)
        let mixed = sum(output * expertWeights.expandedDimensions(axis: -1), axis: 1)
            * Float(expertsPerToken)
        return input + mixed.reshaped(batchShape + [hiddenSize]).asType(input.dtype)
    }
}

private final class OpenMedPrivacyFilterTransformerBlock: Module {
    @ModuleInfo(key: "attn") var attention: OpenMedPrivacyFilterAttentionBlock
    @ModuleInfo(key: "mlp") var mlp: OpenMedPrivacyFilterMLPBlock

    init(_ configuration: OpenMedMLXBertConfiguration) {
        _attention.wrappedValue = OpenMedPrivacyFilterAttentionBlock(configuration)
        _mlp.wrappedValue = OpenMedPrivacyFilterMLPBlock(configuration)
        super.init()
    }

    func callAsFunction(_ input: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        mlp(attention(input, attentionMask: attentionMask))
    }
}

final class OpenMedPrivacyFilterForTokenClassification: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "block") fileprivate var block: [OpenMedPrivacyFilterTransformerBlock]
    @ModuleInfo(key: "norm") fileprivate var norm: OpenMedPrivacyFilterRMSNorm
    @ModuleInfo(key: "unembedding") var unembedding: Linear

    let configuration: OpenMedMLXBertConfiguration

    init(_ configuration: OpenMedMLXBertConfiguration) {
        self.configuration = configuration
        _embedding.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        _block.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
            OpenMedPrivacyFilterTransformerBlock(configuration)
        }
        _norm.wrappedValue = OpenMedPrivacyFilterRMSNorm(
            hiddenSize: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        // The original openai/privacy-filter has a bias-less classifier head;
        // the Nemotron-PII fine-tunes (`classifier_bias: true`) ship with a
        // learned bias. Honor whichever the config requests.
        _unembedding.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.numLabels,
            bias: configuration.classifierBias
        )
        super.init()
    }

    func callAsFunction(
        _ inputIDs: MLXArray,
        attentionMask: MLXArray? = nil
    ) -> MLXArray {
        precondition(inputIDs.ndim == 2, "Privacy Filter expects input IDs with shape [batch, tokens].")
        let resolvedAttentionMask = attentionMask?.asType(.bool)
        var hiddenStates = embedding(inputIDs)
        for layer in block {
            hiddenStates = layer(hiddenStates, attentionMask: resolvedAttentionMask)
        }
        hiddenStates = norm(hiddenStates)
        return unembedding(privacyFilterLinearInput(hiddenStates, for: unembedding))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in !key.hasPrefix("_") }
    }

    /// Install quantized-module placeholders for every path flagged by
    /// ``hasScales``. Custom MoE expert modules use a shape-only placeholder
    /// (no `MLX.quantized` on dummy zero weights); standard `Linear`/`Embedding`
    /// layers fall through to MLX's default `quantizeSingle`. Real tensors are
    /// loaded afterwards via `update(parameters:)`.
    func installQuantizedPlaceholders(
        where hasScales: (String) -> Bool,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) {
        quantize(
            model: self,
            filter: { path, _ in
                hasScales(path) ? (groupSize, bits, mode) : nil
            },
            apply: { layer, groupSize, bits, mode in
                if let expert = layer as? OpenMedPrivacyFilterExpertLinear,
                   !(expert is OpenMedPrivacyFilterQuantizedExpertLinear)
                {
                    return OpenMedPrivacyFilterQuantizedExpertLinear(
                        numExperts: expert.numExperts,
                        inputSize: expert.inputSize,
                        outputSize: expert.outputSize,
                        groupSize: groupSize,
                        bits: bits,
                        mode: mode,
                        dtype: expert.weight.dtype
                    )
                }
                return quantizeSingle(
                    layer: layer,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
            }
        )
    }
}
