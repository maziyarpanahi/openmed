#if canImport(MLXLMCommon) && !os(watchOS) && !os(visionOS)
    import Foundation
    import MLX
    import MLXLMCommon
    import MLXNN

    private let compassPositionIDsKey = LMOutput.Key<MLXArray>(
        "openmed.compass.positionIDs"
    )
    private let compassRopeDeltaKey = LMOutput.Key<MLXArray>(
        "openmed.compass.ropeDelta"
    )

    struct OpenMedCompassConfiguration: Codable, Sendable {
        struct RopeConfiguration: Codable, Sendable {
            let theta: Float
            let sections: [Int]

            enum CodingKeys: String, CodingKey {
                case theta = "rope_theta"
                case sections = "mrope_section"
            }
        }

        struct RopeParameters: Codable, Sendable {
            let slidingAttention: RopeConfiguration
            let fullAttention: RopeConfiguration?

            enum CodingKeys: String, CodingKey {
                case slidingAttention = "sliding_attention"
                case fullAttention = "full_attention"
            }
        }

        struct TextConfiguration: Codable, Sendable {
            let modelType: String
            let vocabularySize: Int
            let hiddenSize: Int
            let intermediateSize: Int
            let hiddenLayers: Int
            let attentionHeads: Int
            let keyValueHeads: Int
            let headDimension: Int
            let layerNormEpsilon: Float
            let tieWordEmbeddings: Bool
            let attentionBias: Bool
            let slidingWindow: Int
            let layerTypes: [String]
            let logitScale: Float
            let ropeParameters: RopeParameters

            enum CodingKeys: String, CodingKey {
                case modelType = "model_type"
                case vocabularySize = "vocab_size"
                case hiddenSize = "hidden_size"
                case intermediateSize = "intermediate_size"
                case hiddenLayers = "num_hidden_layers"
                case attentionHeads = "num_attention_heads"
                case keyValueHeads = "num_key_value_heads"
                case headDimension = "head_dim"
                case layerNormEpsilon = "layer_norm_eps"
                case tieWordEmbeddings = "tie_word_embeddings"
                case attentionBias = "attention_bias"
                case slidingWindow = "sliding_window"
                case layerTypes = "layer_types"
                case logitScale = "logit_scale"
                case ropeParameters = "rope_parameters"
            }
        }

        struct VisionConfiguration: Codable, Sendable {
            let modelType: String
            let depth: Int
            let hiddenSize: Int
            let intermediateSize: Int
            let heads: Int
            let inputChannels: Int
            let patchSize: Int
            let mergeSize: Int
            let temporalPatchSize: Int
            let outputHiddenSize: Int
            let positionEmbeddings: Int
            let deepstackIndexes: [Int]

            enum CodingKeys: String, CodingKey {
                case modelType = "model_type"
                case depth
                case hiddenSize = "hidden_size"
                case intermediateSize = "intermediate_size"
                case heads = "num_heads"
                case inputChannels = "in_channels"
                case patchSize = "patch_size"
                case mergeSize = "spatial_merge_size"
                case temporalPatchSize = "temporal_patch_size"
                case outputHiddenSize = "out_hidden_size"
                case positionEmbeddings = "num_position_embeddings"
                case deepstackIndexes = "deepstack_visual_indexes"
            }
        }

        let modelType: String
        let text: TextConfiguration
        let vision: VisionConfiguration
        let imageTokenID: Int
        let visionStartTokenID: Int
        let visionEndTokenID: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case text = "text_config"
            case vision = "vision_config"
            case imageTokenID = "image_token_id"
            case visionStartTokenID = "vision_start_token_id"
            case visionEndTokenID = "vision_end_token_id"
        }

        func validate() throws {
            guard modelType == "cohere_compass" else {
                throw ModelFactoryError.invalidConfiguration(
                    "expected cohere_compass, got \(modelType)"
                )
            }
            guard text.modelType == "cohere_compass_text" else {
                throw ModelFactoryError.invalidConfiguration(
                    "unsupported Compass text model \(text.modelType)"
                )
            }
            guard vision.modelType == "cohere_compass_vision" else {
                throw ModelFactoryError.invalidConfiguration(
                    "unsupported Compass vision model \(vision.modelType)"
                )
            }
            guard text.layerTypes.count == text.hiddenLayers else {
                throw ModelFactoryError.invalidConfiguration(
                    "layer_types must contain one entry per text layer"
                )
            }
            let sections = text.ropeParameters.slidingAttention.sections
            guard sections.count == 3, sections.reduce(0, +) == text.headDimension / 2 else {
                throw ModelFactoryError.invalidConfiguration(
                    "mrope_section must partition half the attention head"
                )
            }
            guard vision.positionEmbeddings > 0 else {
                throw ModelFactoryError.invalidConfiguration(
                    "vision position embedding count must be positive"
                )
            }
        }
    }

    private final class CompassLayerNorm: Module, UnaryLayer {
        let weight: MLXArray
        let epsilon: Float

        init(dimensions: Int, epsilon: Float) {
            weight = MLXArray.ones([dimensions])
            self.epsilon = epsilon
            super.init()
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let sourceType = input.dtype
            let values = input.asType(.float32)
            let average = values.mean(axis: -1, keepDims: true)
            let variance = square(values - average).mean(axis: -1, keepDims: true)
            let normalized = (values - average) * rsqrt(variance + epsilon)
            return (normalized * weight.asType(.float32)).asType(sourceType)
        }
    }

    private final class CompassRotaryEmbedding: Module {
        let enabled: Bool
        private let _selector: MLXArray?
        private let _inverseFrequencies: MLXArray

        init(
            configuration: OpenMedCompassConfiguration.TextConfiguration,
            layerType: String
        ) {
            let rope =
                layerType == "sliding_attention"
                ? configuration.ropeParameters.slidingAttention
                : configuration.ropeParameters.fullAttention
            enabled = rope != nil
            let theta = rope?.theta ?? 10_000
            let exponents = MLXArray(
                stride(from: 0, to: configuration.headDimension, by: 2)
            ).asType(.float32)
            _inverseFrequencies =
                1
                / pow(
                    MLXArray(theta),
                    exponents / Float(configuration.headDimension)
                )
            if let rope {
                var values = Array(repeating: Int32(0), count: configuration.headDimension / 2)
                for axis in 1...2 {
                    var index = axis
                    let upperBound = min(rope.sections[axis] * 3, values.count)
                    while index < upperBound {
                        values[index] = Int32(axis)
                        index += 3
                    }
                }
                _selector = MLXArray(values)
            } else {
                _selector = nil
            }
            super.init()
        }

        func values(
            reference: MLXArray,
            positionIDs: MLXArray
        ) -> (MLXArray, MLXArray)? {
            guard enabled else { return nil }
            let frequencies: MLXArray
            if positionIDs.ndim == 3, let selector = _selector {
                let positions = take(positionIDs, selector, axis: 0)
                    .transposed(1, 2, 0)
                    .asType(.float32)
                frequencies = positions * _inverseFrequencies
            } else {
                frequencies =
                    positionIDs.asType(.float32)[.ellipsis, .newAxis]
                    * _inverseFrequencies
            }
            let angles = concatenated([frequencies, frequencies], axis: -1)
            return (
                cos(angles).asType(reference.dtype),
                sin(angles).asType(reference.dtype)
            )
        }
    }

    private func compassRotateHalf(_ input: MLXArray) -> MLXArray {
        let half = input.dim(-1) / 2
        return concatenated(
            [-input[.ellipsis, half...], input[.ellipsis, ..<half]],
            axis: -1
        )
    }

    private final class CompassTextAttention: Module {
        let layerType: String
        let queryHeads: Int
        let keyValueHeads: Int
        let headDimension: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var queryProjection: Linear
        @ModuleInfo(key: "k_proj") var keyProjection: Linear
        @ModuleInfo(key: "v_proj") var valueProjection: Linear
        @ModuleInfo(key: "o_proj") var outputProjection: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: CompassRotaryEmbedding

        init(
            _ configuration: OpenMedCompassConfiguration.TextConfiguration,
            layerIndex: Int
        ) {
            layerType = configuration.layerTypes[layerIndex]
            queryHeads = configuration.attentionHeads
            keyValueHeads = configuration.keyValueHeads
            headDimension = configuration.headDimension
            scale = pow(Float(headDimension), -0.5)
            _queryProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                queryHeads * headDimension,
                bias: configuration.attentionBias
            )
            _keyProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                keyValueHeads * headDimension,
                bias: configuration.attentionBias
            )
            _valueProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                keyValueHeads * headDimension,
                bias: configuration.attentionBias
            )
            _outputProjection.wrappedValue = Linear(
                queryHeads * headDimension,
                configuration.hiddenSize,
                bias: configuration.attentionBias
            )
            _rotaryEmbedding.wrappedValue = CompassRotaryEmbedding(
                configuration: configuration,
                layerType: layerType
            )
            super.init()
        }

        func callAsFunction(
            _ input: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIDs: MLXArray
        ) -> MLXArray {
            let batch = input.dim(0)
            let length = input.dim(1)
            var queries = queryProjection(input)
                .reshaped(batch, length, queryHeads, headDimension)
                .transposed(0, 2, 1, 3)
            var keys = keyProjection(input)
                .reshaped(batch, length, keyValueHeads, headDimension)
                .transposed(0, 2, 1, 3)
            let values = valueProjection(input)
                .reshaped(batch, length, keyValueHeads, headDimension)
                .transposed(0, 2, 1, 3)

            if let (cosine, sine) = rotaryEmbedding.values(
                reference: input,
                positionIDs: positionIDs
            ) {
                let cosine = cosine.expandedDimensions(axis: 1)
                let sine = sine.expandedDimensions(axis: 1)
                queries = (queries * cosine + compassRotateHalf(queries) * sine)
                    .asType(queries.dtype)
                keys = (keys * cosine + compassRotateHalf(keys) * sine)
                    .asType(keys.dtype)
            }
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            return output.transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)
                .pipe(outputProjection)
        }
    }

    private final class CompassTextMLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProjection: Linear
        @ModuleInfo(key: "up_proj") var upProjection: Linear
        @ModuleInfo(key: "down_proj") var downProjection: Linear

        init(_ configuration: OpenMedCompassConfiguration.TextConfiguration) {
            _gateProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.intermediateSize,
                bias: false
            )
            _upProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.intermediateSize,
                bias: false
            )
            _downProjection.wrappedValue = Linear(
                configuration.intermediateSize,
                configuration.hiddenSize,
                bias: false
            )
            super.init()
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            downProjection(silu(gateProjection(input)) * upProjection(input))
        }
    }

    private final class CompassDecoderLayer: Module {
        let attentionType: String
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: CompassLayerNorm
        @ModuleInfo(key: "self_attn") var attention: CompassTextAttention
        @ModuleInfo var mlp: CompassTextMLP

        init(
            _ configuration: OpenMedCompassConfiguration.TextConfiguration,
            layerIndex: Int
        ) {
            attentionType = configuration.layerTypes[layerIndex]
            _inputLayerNorm.wrappedValue = CompassLayerNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.layerNormEpsilon
            )
            _attention.wrappedValue = CompassTextAttention(
                configuration,
                layerIndex: layerIndex
            )
            _mlp.wrappedValue = CompassTextMLP(configuration)
            super.init()
        }

        func callAsFunction(
            _ input: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIDs: MLXArray
        ) -> MLXArray {
            let normalized = inputLayerNorm(input)
            return input
                + attention(
                    normalized,
                    mask: mask,
                    cache: cache,
                    positionIDs: positionIDs
                )
                + mlp(normalized)
        }
    }

    private func compassVisualIndices(_ mask: MLXArray) -> MLXArray {
        let values = mask.flattened().asArray(Bool.self)
        let indices = values.enumerated().compactMap { index, value in
            value ? Int32(index) : nil
        }
        return MLXArray(indices)
    }

    private func compassInject(
        hidden: MLXArray,
        mask: MLXArray?,
        visual: MLXArray,
        add: Bool
    ) -> MLXArray {
        guard let mask else { return hidden }
        let indices = compassVisualIndices(mask)
        guard indices.size > 0 else { return hidden }
        let width = hidden.dim(-1)
        let flattened = hidden.reshaped(-1, width)
        let source = visual[..<indices.dim(0), 0...]
        flattened[indices] = add ? flattened[indices] + source : source
        return flattened.reshaped(hidden.shape)
    }

    private final class CompassTextModel: Module {
        let configuration: OpenMedCompassConfiguration.TextConfiguration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [CompassDecoderLayer]
        @ModuleInfo var norm: CompassLayerNorm

        init(_ configuration: OpenMedCompassConfiguration.TextConfiguration) {
            self.configuration = configuration
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: configuration.vocabularySize,
                dimensions: configuration.hiddenSize
            )
            _layers.wrappedValue = (0..<configuration.hiddenLayers).map {
                CompassDecoderLayer(configuration, layerIndex: $0)
            }
            _norm.wrappedValue = CompassLayerNorm(
                dimensions: configuration.hiddenSize,
                epsilon: configuration.layerNormEpsilon
            )
            super.init()
        }

        func callAsFunction(
            inputIDs: MLXArray,
            inputEmbeddings: MLXArray? = nil,
            cache: [KVCache]? = nil,
            positionIDs: MLXArray,
            visualMask: MLXArray? = nil,
            deepstack: [MLXArray]? = nil
        ) -> MLXArray {
            var hidden = inputEmbeddings ?? embedTokens(inputIDs)
            let globalIndex = configuration.layerTypes.firstIndex(of: "full_attention")
            let slidingIndex = configuration.layerTypes.firstIndex(of: "sliding_attention")
            let globalCache = globalIndex.flatMap { cache?[$0] }
            let slidingCache = slidingIndex.flatMap { cache?[$0] }
            let globalMask = createAttentionMask(
                h: hidden,
                cache: globalCache
            )
            let slidingMask = createAttentionMask(
                h: hidden,
                cache: slidingCache,
                windowSize: configuration.slidingWindow
            )
            for (index, layer) in layers.enumerated() {
                hidden = layer(
                    hidden,
                    mask: layer.attentionType == "full_attention"
                        ? globalMask : slidingMask,
                    cache: cache?[index],
                    positionIDs: positionIDs
                )
                if let deepstack, index < deepstack.count {
                    hidden = compassInject(
                        hidden: hidden,
                        mask: visualMask,
                        visual: deepstack[index],
                        add: true
                    )
                }
            }
            return norm(hidden)
        }
    }

    private final class CompassLanguageModel: Module {
        let configuration: OpenMedCompassConfiguration.TextConfiguration
        @ModuleInfo var model: CompassTextModel
        @ModuleInfo(key: "lm_head") var languageHead: Linear?

        init(_ configuration: OpenMedCompassConfiguration.TextConfiguration) {
            self.configuration = configuration
            _model.wrappedValue = CompassTextModel(configuration)
            if !configuration.tieWordEmbeddings {
                _languageHead.wrappedValue = Linear(
                    configuration.hiddenSize,
                    configuration.vocabularySize,
                    bias: false
                )
            }
            super.init()
        }

        var layers: [CompassDecoderLayer] { model.layers }

        private func project(_ hidden: MLXArray) -> MLXArray {
            let logits = languageHead?(hidden) ?? model.embedTokens.asLinear(hidden)
            return logits * configuration.logitScale
        }

        func prefill(
            inputIDs: MLXArray,
            inputEmbeddings: MLXArray?,
            cache: [KVCache],
            positionIDs: MLXArray,
            visualMask: MLXArray?,
            deepstack: [MLXArray]?
        ) -> MLXArray {
            project(
                model(
                    inputIDs: inputIDs,
                    inputEmbeddings: inputEmbeddings,
                    cache: cache,
                    positionIDs: positionIDs,
                    visualMask: visualMask,
                    deepstack: deepstack
                )
            )
        }

        func decode(
            _ input: LMInput.Text,
            cache: [KVCache]?,
            state: LMOutput.State?
        ) -> LMOutput {
            let batch = input.tokens.ndim == 2 ? input.tokens.dim(0) : 1
            let length = input.tokens.ndim == 2 ? input.tokens.dim(1) : input.tokens.dim(0)
            let offset = cache?.first?.offset ?? 0
            var positions = MLXArray(0..<length).asType(.int32)
                .reshaped(1, length)
            positions = broadcast(positions, to: [batch, length])
            if let delta = state?[compassRopeDeltaKey] {
                positions = positions + offset + delta.asType(.int32)
            } else {
                positions = positions + offset
            }
            let positionIDs = broadcast(
                positions[.newAxis, 0..., 0...],
                to: [3, batch, length]
            )
            let hidden = model(
                inputIDs: input.tokens,
                cache: cache,
                positionIDs: positionIDs
            )
            return LMOutput(logits: project(hidden), state: state)
        }
    }

    private final class CompassPatchEmbedding: Module, UnaryLayer {
        let inputChannels: Int
        let temporalPatchSize: Int
        let patchSize: Int
        let hiddenSize: Int
        @ModuleInfo var proj: Conv3d

        init(_ configuration: OpenMedCompassConfiguration.VisionConfiguration) {
            inputChannels = configuration.inputChannels
            temporalPatchSize = configuration.temporalPatchSize
            patchSize = configuration.patchSize
            hiddenSize = configuration.hiddenSize
            let kernel = IntOrTriple([
                temporalPatchSize,
                patchSize,
                patchSize,
            ])
            _proj.wrappedValue = Conv3d(
                inputChannels: inputChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
            super.init()
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            var values = input.reshaped(
                -1,
                inputChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            )
            values = values.movedAxis(source: 1, destination: 4)
            return proj(values).reshaped(-1, hiddenSize)
        }
    }

    private final class CompassVisionRotaryEmbedding: Module {
        let dimensions: Int
        private let _inverseFrequencies: MLXArray

        init(dimensions: Int) {
            self.dimensions = dimensions
            let exponents = MLXArray(
                stride(from: 0, to: dimensions, by: 2)
            ).asType(.float32)
            _inverseFrequencies =
                1
                / pow(
                    MLXArray(Float(10_000)),
                    exponents / Float(dimensions)
                )
            super.init()
        }

        func callAsFunction(_ positions: MLXArray) -> MLXArray {
            (positions[0..., .newAxis].asType(.float32) * _inverseFrequencies)
                .reshaped(positions.dim(0), -1)
        }
    }

    private func compassVisionRotateHalf(_ input: MLXArray) -> MLXArray {
        let half = input.dim(-1) / 2
        return concatenated(
            [-input[.ellipsis, half...], input[.ellipsis, ..<half]],
            axis: -1
        )
    }

    private final class CompassVisionAttention: Module {
        let heads: Int
        let headDimension: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ configuration: OpenMedCompassConfiguration.VisionConfiguration) {
            heads = configuration.heads
            headDimension = configuration.hiddenSize / configuration.heads
            scale = pow(Float(headDimension), -0.5)
            _qkv.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.hiddenSize * 3,
                bias: true
            )
            _proj.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.hiddenSize,
                bias: true
            )
            super.init()
        }

        func callAsFunction(
            _ input: MLXArray,
            sequenceEnds: [Int],
            rotary: MLXArray?
        ) -> MLXArray {
            let length = input.dim(0)
            let packed = qkv(input).reshaped(length, 3, heads, headDimension)
            let parts = split(packed, parts: 3, axis: 1)
            var query = parts[0].squeezed(axis: 1)
            var key = parts[1].squeezed(axis: 1)
            let value = parts[2].squeezed(axis: 1)
            if let rotary {
                let cosine = concatenated([cos(rotary), cos(rotary)], axis: -1)
                    .expandedDimensions(axis: 1)
                    .asType(.float32)
                let sine = concatenated([sin(rotary), sin(rotary)], axis: -1)
                    .expandedDimensions(axis: 1)
                    .asType(.float32)
                let queryType = query.dtype
                let keyType = key.dtype
                query =
                    (query.asType(.float32) * cosine
                    + compassVisionRotateHalf(query.asType(.float32)) * sine).asType(queryType)
                key =
                    (key.asType(.float32) * cosine
                    + compassVisionRotateHalf(key.asType(.float32)) * sine).asType(keyType)
            }
            query = query.transposed(1, 0, 2).expandedDimensions(axis: 0)
            key = key.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let values = value.transposed(1, 0, 2).expandedDimensions(axis: 0)
            var outputs = [MLXArray]()
            for index in 1..<sequenceEnds.count {
                let start = sequenceEnds[index - 1]
                let end = sequenceEnds[index]
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: query[0..., 0..., start..<end, 0...],
                        keys: key[0..., 0..., start..<end, 0...],
                        values: values[0..., 0..., start..<end, 0...],
                        scale: scale,
                        mask: .none
                    )
                )
            }
            let output = concatenated(outputs, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(length, -1)
            return proj(output)
        }
    }

    private final class CompassVisionMLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var first: Linear
        @ModuleInfo(key: "linear_fc2") var second: Linear
        @ModuleInfo(key: "act_fn") var activation: GELU

        init(_ configuration: OpenMedCompassConfiguration.VisionConfiguration) {
            _first.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.intermediateSize
            )
            _second.wrappedValue = Linear(
                configuration.intermediateSize,
                configuration.hiddenSize
            )
            _activation.wrappedValue = GELU(approximation: .tanh)
            super.init()
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            second(activation(first(input)))
        }
    }

    private final class CompassVisionBlock: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var attn: CompassVisionAttention
        @ModuleInfo var mlp: CompassVisionMLP

        init(_ configuration: OpenMedCompassConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(
                dimensions: configuration.hiddenSize,
                eps: 1e-6
            )
            _norm2.wrappedValue = LayerNorm(
                dimensions: configuration.hiddenSize,
                eps: 1e-6
            )
            _attn.wrappedValue = CompassVisionAttention(configuration)
            _mlp.wrappedValue = CompassVisionMLP(configuration)
            super.init()
        }

        func callAsFunction(
            _ input: MLXArray,
            sequenceEnds: [Int],
            rotary: MLXArray?
        ) -> MLXArray {
            var hidden =
                input
                + attn(
                    norm1(input),
                    sequenceEnds: sequenceEnds,
                    rotary: rotary
                )
            hidden = hidden + mlp(norm2(hidden))
            return hidden
        }
    }

    private final class CompassPatchMerger: Module, UnaryLayer {
        let mergedSize: Int
        let postshuffleNorm: Bool
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var first: Linear
        @ModuleInfo(key: "linear_fc2") var second: Linear
        @ModuleInfo(key: "act_fn") var activation: GELU

        init(
            _ configuration: OpenMedCompassConfiguration.VisionConfiguration,
            postshuffleNorm: Bool = false
        ) {
            mergedSize =
                configuration.hiddenSize
                * configuration.mergeSize
                * configuration.mergeSize
            self.postshuffleNorm = postshuffleNorm
            _norm.wrappedValue = LayerNorm(
                dimensions: postshuffleNorm ? mergedSize : configuration.hiddenSize,
                eps: 1e-6
            )
            _first.wrappedValue = Linear(mergedSize, mergedSize)
            _second.wrappedValue = Linear(
                mergedSize,
                configuration.outputHiddenSize
            )
            _activation.wrappedValue = GELU()
            super.init()
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let values =
                postshuffleNorm
                ? norm(input.reshaped(-1, mergedSize))
                : norm(input).reshaped(-1, mergedSize)
            return second(activation(first(values)))
        }
    }

    private final class CompassVisionTower: Module {
        let configuration: OpenMedCompassConfiguration.VisionConfiguration
        let gridSide: Int
        @ModuleInfo(key: "patch_embed") var patchEmbed: CompassPatchEmbedding
        @ModuleInfo(key: "pos_embed") var positionEmbedding: Embedding
        @ModuleInfo var blocks: [CompassVisionBlock]
        @ModuleInfo var merger: CompassPatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [CompassPatchMerger]
        private let _rotaryEmbedding: CompassVisionRotaryEmbedding

        init(_ configuration: OpenMedCompassConfiguration.VisionConfiguration) {
            self.configuration = configuration
            gridSide = Int(Double(configuration.positionEmbeddings).squareRoot())
            _patchEmbed.wrappedValue = CompassPatchEmbedding(configuration)
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: configuration.positionEmbeddings,
                dimensions: configuration.hiddenSize
            )
            _blocks.wrappedValue = (0..<configuration.depth).map { _ in
                CompassVisionBlock(configuration)
            }
            _merger.wrappedValue = CompassPatchMerger(configuration)
            _deepstackMergers.wrappedValue = configuration.deepstackIndexes.map { _ in
                CompassPatchMerger(configuration, postshuffleNorm: true)
            }
            let headDimension = configuration.hiddenSize / configuration.heads
            _rotaryEmbedding = CompassVisionRotaryEmbedding(
                dimensions: headDimension / 2
            )
            super.init()
        }

        private func learnedPositions(_ frame: THW) -> MLXArray {
            let merge = configuration.mergeSize
            let rowCoordinates = linspace(
                Float(0),
                Float(gridSide - 1),
                count: frame.h
            ).asArray(Float.self)
            let columnCoordinates = linspace(
                Float(0),
                Float(gridSide - 1),
                count: frame.w
            ).asArray(Float.self)
            var cornerIndices = Array(repeating: [Int32](), count: 4)
            var cornerWeights = Array(repeating: [Float](), count: 4)
            for mergedRow in 0..<frame.h / merge {
                for mergedColumn in 0..<frame.w / merge {
                    for innerRow in 0..<merge {
                        for innerColumn in 0..<merge {
                            let row = mergedRow * merge + innerRow
                            let column = mergedColumn * merge + innerColumn
                            let rowValue = rowCoordinates[row]
                            let columnValue = columnCoordinates[column]
                            let rowFloor = Int(floor(rowValue))
                            let columnFloor = Int(floor(columnValue))
                            let rowCeil = min(rowFloor + 1, gridSide - 1)
                            let columnCeil = min(columnFloor + 1, gridSide - 1)
                            let rowFraction = rowValue - Float(rowFloor)
                            let columnFraction = columnValue - Float(columnFloor)
                            cornerIndices[0].append(
                                Int32(rowFloor * gridSide + columnFloor)
                            )
                            cornerIndices[1].append(
                                Int32(rowFloor * gridSide + columnCeil)
                            )
                            cornerIndices[2].append(
                                Int32(rowCeil * gridSide + columnFloor)
                            )
                            cornerIndices[3].append(
                                Int32(rowCeil * gridSide + columnCeil)
                            )
                            cornerWeights[0].append(
                                (1 - rowFraction) * (1 - columnFraction)
                            )
                            cornerWeights[1].append(
                                (1 - rowFraction) * columnFraction
                            )
                            cornerWeights[2].append(
                                rowFraction * (1 - columnFraction)
                            )
                            cornerWeights[3].append(
                                rowFraction * columnFraction
                            )
                        }
                    }
                }
            }
            let count = frame.h * frame.w
            let indices = MLXArray(cornerIndices.flatMap { $0 }, [4, count])
            let weights = MLXArray(cornerWeights.flatMap { $0 }, [4, count])
            let spatial = (positionEmbedding(indices) * weights[0..., 0..., .newAxis])
                .sum(axis: 0)
            return frame.t > 1
                ? tiled(spatial, repetitions: [frame.t, 1])
                : spatial
        }

        private func rotaryPositions(_ frame: THW) -> MLXArray {
            let merge = configuration.mergeSize
            var rows = [Int32]()
            var columns = [Int32]()
            for _ in 0..<frame.t {
                for mergedRow in 0..<frame.h / merge {
                    for mergedColumn in 0..<frame.w / merge {
                        for innerRow in 0..<merge {
                            for innerColumn in 0..<merge {
                                rows.append(Int32(mergedRow * merge + innerRow))
                                columns.append(Int32(mergedColumn * merge + innerColumn))
                            }
                        }
                    }
                }
            }
            let table = _rotaryEmbedding(
                MLXArray(0..<max(frame.h, frame.w)).asType(.int32)
            )
            return concatenated(
                [table[MLXArray(rows)], table[MLXArray(columns)]],
                axis: -1
            )
        }

        private func encodeOne(
            _ patches: MLXArray,
            frame: THW
        ) -> (MLXArray, [MLXArray]) {
            var hidden = patchEmbed(patches)
            let positions = learnedPositions(frame).asType(hidden.dtype)
            hidden = hidden + positions
            let rotary = rotaryPositions(frame)
            var sequenceEnds = [0]
            for _ in 0..<frame.t {
                sequenceEnds.append(sequenceEnds.last! + frame.h * frame.w)
            }
            var deepstack = [MLXArray]()
            for (index, block) in blocks.enumerated() {
                hidden = block(
                    hidden,
                    sequenceEnds: sequenceEnds,
                    rotary: rotary
                )
                if let mergerIndex = configuration.deepstackIndexes.firstIndex(of: index) {
                    deepstack.append(deepstackMergers[mergerIndex](hidden))
                }
            }
            let merged = merger(hidden)
            return (merged, deepstack)
        }

        func callAsFunction(
            _ patches: MLXArray,
            frames: [THW]
        ) -> (MLXArray, [MLXArray]) {
            var cumulative = 0
            var splitPoints = [Int]()
            for frame in frames.dropLast() {
                cumulative += frame.product
                splitPoints.append(cumulative)
            }
            let groups =
                splitPoints.isEmpty
                ? [patches]
                : split(patches, indices: splitPoints, axis: 0)
            var features = [MLXArray]()
            var deepFeatures = configuration.deepstackIndexes.map { _ in [MLXArray]() }
            for (patchGroup, frame) in zip(groups, frames) {
                let (output, deepstack) = encodeOne(patchGroup, frame: frame)
                features.append(output)
                for index in deepstack.indices {
                    deepFeatures[index].append(deepstack[index])
                }
            }
            return (
                concatenated(features, axis: 0),
                deepFeatures.map { concatenated($0, axis: 0) }
            )
        }
    }

    final class OpenMedCompassModel: Module, LanguageModel, LoRAModel {
        let configuration: OpenMedCompassConfiguration
        @ModuleInfo(key: "language_model") private var languageModel: CompassLanguageModel
        @ModuleInfo(key: "vision_tower") private var visionTower: CompassVisionTower

        var vocabularySize: Int { configuration.text.vocabularySize }
        var kvHeads: [Int] {
            Array(
                repeating: configuration.text.keyValueHeads,
                count: configuration.text.hiddenLayers
            )
        }
        var loraLayers: [Module] { languageModel.layers }

        init(_ configuration: OpenMedCompassConfiguration) {
            self.configuration = configuration
            _languageModel.wrappedValue = CompassLanguageModel(configuration.text)
            _visionTower.wrappedValue = CompassVisionTower(configuration.vision)
            super.init()
        }

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            configuration.text.layerTypes.map { layerType in
                if layerType == "sliding_attention" {
                    return RotatingKVCache(
                        maxSize: configuration.text.slidingWindow,
                        keep: 0
                    )
                }
                return KVCacheSimple()
            }
        }

        private func ropePositions(
            inputIDs: MLXArray,
            frames: [THW]
        ) -> (MLXArray, MLXArray) {
            let tokens = inputIDs[0].asArray(Int32.self).map(Int.init)
            guard !frames.isEmpty else {
                let positions = MLXArray(0..<tokens.count).asType(.int32)
                return (
                    broadcast(
                        positions[.newAxis, .newAxis, 0...],
                        to: [3, 1, tokens.count]
                    ),
                    MLXArray([Int32(0)], [1, 1])
                )
            }
            var axes = Array(
                repeating: Array(repeating: Int32(1), count: tokens.count),
                count: 3
            )
            var start = 0
            var nextPosition = 0
            for frame in frames {
                guard
                    let imageStart = tokens[start...].firstIndex(
                        of: configuration.imageTokenID
                    )
                else { continue }
                for tokenIndex in start..<imageStart {
                    for axis in 0..<3 {
                        axes[axis][tokenIndex] = Int32(
                            nextPosition + tokenIndex - start
                        )
                    }
                }
                nextPosition += imageStart - start
                let gridHeight = frame.h / configuration.vision.mergeSize
                let gridWidth = frame.w / configuration.vision.mergeSize
                var visualOffset = 0
                for time in 0..<frame.t {
                    for row in 0..<gridHeight {
                        for column in 0..<gridWidth {
                            let tokenIndex = imageStart + visualOffset
                            axes[0][tokenIndex] = Int32(nextPosition + time)
                            axes[1][tokenIndex] = Int32(nextPosition + row)
                            axes[2][tokenIndex] = Int32(nextPosition + column)
                            visualOffset += 1
                        }
                    }
                }
                nextPosition += max(frame.t, max(gridHeight, gridWidth))
                start = imageStart + visualOffset
            }
            if start < tokens.count {
                for tokenIndex in start..<tokens.count {
                    for axis in 0..<3 {
                        axes[axis][tokenIndex] = Int32(
                            nextPosition + tokenIndex - start
                        )
                    }
                }
            }
            let maximum = axes.flatMap { $0 }.max().map(Int.init) ?? 0
            let delta = maximum + 1 - tokens.count
            return (
                MLXArray(axes.flatMap { $0 }, [3, 1, tokens.count]),
                MLXArray([Int32(delta)], [1, 1])
            )
        }

        func prepare(
            _ input: LMInput,
            cache: [KVCache],
            windowSize: Int?
        ) throws -> PrepareResult {
            _ = windowSize
            let frames = input.image?.frames ?? []
            let (positionIDs, ropeDelta) = ropePositions(
                inputIDs: input.text.tokens,
                frames: frames
            )
            var state = LMOutput.State()
            state[compassPositionIDsKey] = positionIDs
            state[compassRopeDeltaKey] = ropeDelta

            var inputEmbeddings: MLXArray?
            var visualMask: MLXArray?
            var deepstack: [MLXArray]?
            if let pixels = input.image?.pixels, !frames.isEmpty {
                let embeddings = languageModel.model.embedTokens(input.text.tokens)
                let (visualFeatures, deepFeatures) = visionTower(
                    pixels.asType(visionTower.patchEmbed.proj.weight.dtype),
                    frames: frames
                )
                let mask = input.text.tokens .== configuration.imageTokenID
                guard mask.sum().item(Int.self) == visualFeatures.dim(0) else {
                    throw OpenMedCompassError.imageTokenCountMismatch
                }
                inputEmbeddings = compassInject(
                    hidden: embeddings,
                    mask: mask,
                    visual: visualFeatures,
                    add: false
                )
                visualMask = mask
                deepstack = deepFeatures
            }
            let logits = languageModel.prefill(
                inputIDs: input.text.tokens,
                inputEmbeddings: inputEmbeddings,
                cache: cache,
                positionIDs: positionIDs,
                visualMask: visualMask,
                deepstack: deepstack
            )
            return .logits(LMOutput(logits: logits, state: state))
        }

        func callAsFunction(
            _ input: LMInput.Text,
            cache: [KVCache]?,
            state: LMOutput.State?
        ) -> LMOutput {
            languageModel.decode(input, cache: cache, state: state)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            weights.filter { key, _ in
                !(configuration.text.tieWordEmbeddings && key == "lm_head.weight")
            }
        }
    }

    extension MLXArray {
        fileprivate func pipe(_ layer: Linear) -> MLXArray {
            layer(self)
        }
    }
#endif
