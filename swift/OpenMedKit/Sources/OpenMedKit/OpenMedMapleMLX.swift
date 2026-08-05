#if canImport(MLX) && canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import MLX
    import MLXLMCommon
    import MLXLLM
    import MLXNN
    import Tokenizers

    /// Configuration for DeepGrove's Maple architecture.
    public struct OpenMedMapleConfiguration: Codable, Sendable {
        let modelType: String
        let hiddenSize: Int
        let intermediateSize: Int
        let moeIntermediateSize: Int
        let hiddenLayers: Int
        let attentionHeads: Int
        let kvHeads: Int
        let headDim: Int
        let numExperts: Int
        let numExpertsPerToken: Int
        let firstDenseLayers: Int
        let rmsNormEpsilon: Float
        let ropeTheta: Float
        let ropeScaling: [String: StringOrNumber]?
        let partialRotaryFactor: Float
        let maxPositionEmbeddings: Int
        let vocabularySize: Int
        let slidingWindow: Int
        let layerTypes: [String]
        let useQKNorm: Bool
        let useBias: Bool
        let tieWordEmbeddings: Bool
        let quantization: Quantization?

        struct Quantization: Codable, Sendable {
            let bits: Int
            let groupSize: Int

            enum CodingKeys: String, CodingKey {
                case bits
                case groupSize = "group_size"
            }
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case numExperts = "num_experts"
            case numExpertsPerToken = "num_experts_per_tok"
            case firstDenseLayers = "first_k_dense_replace"
            case rmsNormEpsilon = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabularySize = "vocab_size"
            case slidingWindow = "sliding_window"
            case layerTypes = "layer_types"
            case useQKNorm = "use_qk_norm"
            case useBias = "use_bias"
            case tieWordEmbeddings = "tie_word_embeddings"
            case quantization
        }

        public init(from decoder: any Swift.Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try values.decodeIfPresent(String.self, forKey: .modelType) ?? "maple"
            hiddenSize = try values.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048
            intermediateSize =
                try values.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5_120
            moeIntermediateSize =
                try values.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512
            hiddenLayers = try values.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            attentionHeads =
                try values.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try values.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
            headDim = try values.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
            numExperts = try values.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256
            numExpertsPerToken =
                try values.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
            firstDenseLayers =
                try values.decodeIfPresent(Int.self, forKey: .firstDenseLayers) ?? 0
            rmsNormEpsilon =
                try values.decodeIfPresent(Float.self, forKey: .rmsNormEpsilon) ?? 1e-6
            ropeTheta = try values.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            ropeScaling = try values.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            )
            partialRotaryFactor =
                try values.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
            maxPositionEmbeddings =
                try values.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 128_000
            vocabularySize =
                try values.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            slidingWindow =
                try values.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
            layerTypes =
                try values.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? Array(repeating: "full_attention", count: hiddenLayers)
            useQKNorm = try values.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
            useBias = try values.decodeIfPresent(Bool.self, forKey: .useBias) ?? false
            tieWordEmbeddings =
                try values.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            quantization = try values.decodeIfPresent(Quantization.self, forKey: .quantization)
        }

        func validate() throws {
            guard modelType == "maple" else {
                throw OpenMedMapleRuntimeError.unsupportedArchitecture(modelType)
            }
            guard hiddenLayers == layerTypes.count else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "num_hidden_layers must equal layer_types.count"
                )
            }
            guard
                layerTypes.allSatisfy({
                    $0 == "sliding_attention" || $0 == "full_attention"
                })
            else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "layer_types contains an unsupported attention type"
                )
            }
            guard firstDenseLayers == 0 else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "this Maple runtime requires first_k_dense_replace = 0"
                )
            }
            guard useQKNorm else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "this Maple runtime requires use_qk_norm = true"
                )
            }
            guard attentionHeads > 0, kvHeads > 0, headDim > 0,
                hiddenSize == attentionHeads * headDim,
                attentionHeads.isMultiple(of: kvHeads)
            else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "invalid attention head dimensions"
                )
            }
            guard numExperts > 0, numExpertsPerToken > 0,
                numExpertsPerToken <= numExperts
            else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "invalid expert routing dimensions"
                )
            }
            guard partialRotaryFactor > 0, partialRotaryFactor <= 1 else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "partial_rotary_factor must be in (0, 1]"
                )
            }
        }
    }

    private final class MapleRMSNorm: RMSNorm {
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(
                x.asType(.float32),
                weight: weight.asType(.float32),
                eps: eps
            ).asType(x.dtype)
        }
    }

    private final class MapleAttention: Module {
        let attentionHeads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let usesRotaryPosition: Bool

        @ModuleInfo(key: "q_proj") var queryProjection: Linear
        @ModuleInfo(key: "k_proj") var keyProjection: Linear
        @ModuleInfo(key: "v_proj") var valueProjection: Linear
        @ModuleInfo(key: "o_proj") var outputProjection: Linear
        @ModuleInfo(key: "q_norm") var queryNorm: MapleRMSNorm
        @ModuleInfo(key: "k_norm") var keyNorm: MapleRMSNorm

        let rope: RoPELayer?

        init(_ configuration: OpenMedMapleConfiguration, layerIndex: Int) {
            attentionHeads = configuration.attentionHeads
            kvHeads = configuration.kvHeads
            headDim = configuration.headDim
            scale = pow(Float(configuration.headDim), -0.5)
            usesRotaryPosition = configuration.layerTypes[layerIndex] == "sliding_attention"

            _queryProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.attentionHeads * configuration.headDim,
                bias: configuration.useBias
            )
            _keyProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.kvHeads * configuration.headDim,
                bias: configuration.useBias
            )
            _valueProjection.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.kvHeads * configuration.headDim,
                bias: configuration.useBias
            )
            _outputProjection.wrappedValue = Linear(
                configuration.attentionHeads * configuration.headDim,
                configuration.hiddenSize,
                bias: configuration.useBias
            )
            _queryNorm.wrappedValue = MapleRMSNorm(
                dimensions: configuration.headDim,
                eps: configuration.rmsNormEpsilon
            )
            _keyNorm.wrappedValue = MapleRMSNorm(
                dimensions: configuration.headDim,
                eps: configuration.rmsNormEpsilon
            )

            if usesRotaryPosition {
                rope = initializeRope(
                    dims: Int(Float(configuration.headDim) * configuration.partialRotaryFactor),
                    base: configuration.ropeTheta,
                    traditional: false,
                    scalingConfig: configuration.ropeScaling,
                    maxPositionEmbeddings: configuration.maxPositionEmbeddings
                )
            } else {
                rope = nil
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)

            var queries = queryProjection(x)
                .reshaped(batch, length, attentionHeads, headDim)
            var keys = keyProjection(x)
                .reshaped(batch, length, kvHeads, headDim)
            var values = valueProjection(x)
                .reshaped(batch, length, kvHeads, headDim)

            queries = queryNorm(queries).transposed(0, 2, 1, 3)
            keys = keyNorm(keys).transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            if usesRotaryPosition, let rope {
                let offset = cache?.ropeOffset
                queries = applyRotaryPosition(rope, to: queries, offset: offset)
                keys = applyRotaryPosition(rope, to: keys, offset: offset)
            }

            var output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask
            )
            output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
            return outputProjection(output)
        }
    }

    private final class MapleGate: Module {
        let expertCount: Int
        let expertsPerToken: Int

        @ParameterInfo var weight: MLXArray

        init(_ configuration: OpenMedMapleConfiguration) {
            expertCount = configuration.numExperts
            expertsPerToken = configuration.numExpertsPerToken
            _weight.wrappedValue = MLXArray.zeros([
                configuration.numExperts,
                configuration.hiddenSize,
            ])
        }

        func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
            let logits = matmul(
                x.asType(.float32),
                weight.asType(.float32).transposed()
            )
            let allScores = softmax(logits, axis: -1)
            let k = expertsPerToken
            let indices = argPartition(-allScores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            var selectedScores = takeAlong(allScores, indices, axis: -1)
            selectedScores =
                selectedScores
                / (selectedScores.sum(axis: -1, keepDims: true) + 1e-20)
            return (indices, selectedScores)
        }
    }

    private final class MapleSwitchGLU: Module {
        @ModuleInfo(key: "up_proj") var upProjection: SwitchLinear
        @ModuleInfo(key: "gate_proj") var gateProjection: SwitchLinear
        @ModuleInfo(key: "down_proj") var downProjection: SwitchLinear

        init(_ configuration: OpenMedMapleConfiguration) {
            _upProjection.wrappedValue = SwitchLinear(
                inputDims: configuration.hiddenSize,
                outputDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts,
                bias: configuration.useBias
            )
            _gateProjection.wrappedValue = SwitchLinear(
                inputDims: configuration.hiddenSize,
                outputDims: configuration.moeIntermediateSize,
                numExperts: configuration.numExperts,
                bias: configuration.useBias
            )
            _downProjection.wrappedValue = SwitchLinear(
                inputDims: configuration.moeIntermediateSize,
                outputDims: configuration.hiddenSize,
                numExperts: configuration.numExperts,
                bias: configuration.useBias
            )
        }

        func callAsFunction(_ input: MLXArray, indices: MLXArray) -> MLXArray {
            var expanded = MLX.expandedDimensions(input, axes: [-2, -3])
            let shouldSort = indices.size >= 64
            var selectedIndices = indices
            var inverseOrder = MLXArray()

            if shouldSort {
                (expanded, selectedIndices, inverseOrder) = gatherSort(
                    x: expanded,
                    indices: indices
                )
            }

            let up = upProjection(expanded, selectedIndices, sortedIndices: shouldSort)
            let gate = gateProjection(expanded, selectedIndices, sortedIndices: shouldSort)
            let activated =
                silu(MLX.minimum(gate, 7.0))
                * MLX.clip(up, min: -7.0, max: 7.0)
            var output = downProjection(
                activated,
                selectedIndices,
                sortedIndices: shouldSort
            )

            if shouldSort {
                output = scatterUnsort(
                    x: output,
                    invOrder: inverseOrder,
                    shape: indices.shape
                )
            }
            return MLX.squeezed(output, axis: -2)
        }
    }

    private final class MapleSparseMoEBlock: Module {
        @ModuleInfo var gate: MapleGate
        @ModuleInfo(key: "switch_mlp") var switchMLP: MapleSwitchGLU

        init(_ configuration: OpenMedMapleConfiguration) {
            _gate.wrappedValue = MapleGate(configuration)
            _switchMLP.wrappedValue = MapleSwitchGLU(configuration)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let selection = gate(x)
            let expertOutputs = switchMLP(x, indices: selection.indices)
            return
                (expertOutputs.asType(.float32)
                * MLX.expandedDimensions(selection.scores, axis: -1))
                .sum(axis: -2)
                .asType(expertOutputs.dtype)
        }
    }

    private final class MapleDecoderLayer: Module {
        let usesSlidingAttention: Bool

        @ModuleInfo(key: "self_attn") var selfAttention: MapleAttention
        @ModuleInfo var mlp: MapleSparseMoEBlock
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: MapleRMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: MapleRMSNorm

        init(_ configuration: OpenMedMapleConfiguration, layerIndex: Int) {
            usesSlidingAttention =
                configuration.layerTypes[layerIndex] == "sliding_attention"
            _selfAttention.wrappedValue = MapleAttention(
                configuration,
                layerIndex: layerIndex
            )
            _mlp.wrappedValue = MapleSparseMoEBlock(configuration)
            _inputLayerNorm.wrappedValue = MapleRMSNorm(
                dimensions: configuration.hiddenSize,
                eps: configuration.rmsNormEpsilon
            )
            _postAttentionLayerNorm.wrappedValue = MapleRMSNorm(
                dimensions: configuration.hiddenSize,
                eps: configuration.rmsNormEpsilon
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?
        ) -> MLXArray {
            let hidden = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            return hidden + mlp(postAttentionLayerNorm(hidden))
        }
    }

    private final class MapleModelInner: Module {
        let slidingWindow: Int
        let fullAttentionIndex: Int?
        let slidingAttentionIndex: Int?

        @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
        @ModuleInfo var layers: [MapleDecoderLayer]
        @ModuleInfo var norm: MapleRMSNorm

        init(_ configuration: OpenMedMapleConfiguration) {
            slidingWindow = configuration.slidingWindow
            fullAttentionIndex = configuration.layerTypes.firstIndex(of: "full_attention")
            slidingAttentionIndex = configuration.layerTypes.firstIndex(
                of: "sliding_attention"
            )
            _wordEmbeddings.wrappedValue = Embedding(
                embeddingCount: configuration.vocabularySize,
                dimensions: configuration.hiddenSize
            )
            _layers.wrappedValue = configuration.layerTypes.indices.map {
                MapleDecoderLayer(configuration, layerIndex: $0)
            }
            _norm.wrappedValue = MapleRMSNorm(
                dimensions: configuration.hiddenSize,
                eps: configuration.rmsNormEpsilon
            )
        }

        func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
            var hidden = wordEmbeddings(tokens)
            let fullMask =
                fullAttentionIndex.map {
                    createAttentionMask(h: hidden, cache: cache?[$0])
                } ?? .none
            let slidingMask =
                slidingAttentionIndex.map {
                    createAttentionMask(
                        h: hidden,
                        cache: cache?[$0],
                        windowSize: slidingWindow
                    )
                } ?? .none

            for (index, layer) in layers.enumerated() {
                hidden = layer(
                    hidden,
                    mask: layer.usesSlidingAttention ? slidingMask : fullMask,
                    cache: cache?[index]
                )
            }
            return norm(hidden)
        }
    }

    enum OpenMedMapleWeightSanitizer {
        static func sanitize(
            _ weights: [String: MLXArray],
            configuration: OpenMedMapleConfiguration
        ) -> [String: MLXArray] {
            var result = weights.filter { key, _ in
                !key.hasPrefix("lm_head_flash.")
                    && !key.contains("rotary_emb.inv_freq")
            }

            if configuration.tieWordEmbeddings {
                result = result.filter { !$0.key.hasPrefix("lm_head.") }
            }

            let groupSize = configuration.quantization?.groupSize ?? 128
            let rowAlphaKeys = result.keys.filter { $0.hasSuffix(".row_alpha") }
            for key in rowAlphaKeys {
                guard let alpha = result.removeValue(forKey: key) else { continue }
                let prefix = String(key.dropLast(".row_alpha".count))
                guard let packed = result["\(prefix).weight"] else { continue }
                let groupCount = (packed.dim(-1) * 16) / groupSize
                let scales = MLX.contiguous(
                    MLX.broadcast(
                        MLX.expandedDimensions(alpha, axis: -1),
                        to: alpha.shape + [groupCount]
                    )
                )
                result["\(prefix).scales"] = scales
                result["\(prefix).biases"] = -scales
            }
            return result
        }
    }

    private final class MapleMLXModel: Module, LLMModel, KVCacheDimensionProvider {
        let configuration: OpenMedMapleConfiguration
        let vocabularySize: Int
        let kvHeads: [Int]
        let layerUsesSlidingAttention: [Bool]

        @ModuleInfo var model: MapleModelInner
        @ModuleInfo(key: "lm_head") var languageHead: Linear?

        init(_ configuration: OpenMedMapleConfiguration) {
            self.configuration = configuration
            vocabularySize = configuration.vocabularySize
            kvHeads = Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
            layerUsesSlidingAttention = configuration.layerTypes.map {
                $0 == "sliding_attention"
            }
            _model.wrappedValue = MapleModelInner(configuration)
            if !configuration.tieWordEmbeddings {
                _languageHead.wrappedValue = Linear(
                    configuration.hiddenSize,
                    configuration.vocabularySize,
                    bias: false
                )
            }
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
            let output = model(inputs, cache: cache)
            if let languageHead {
                return languageHead(output)
            }
            return model.wordEmbeddings.asLinear(output)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            OpenMedMapleWeightSanitizer.sanitize(
                weights,
                configuration: configuration
            )
        }

        func newCache(parameters _: GenerateParameters?) -> [KVCache] {
            layerUsesSlidingAttention.map { usesSliding in
                if usesSliding {
                    return RotatingKVCache(maxSize: configuration.slidingWindow)
                }
                return KVCacheSimple()
            }
        }

        var loraLayers: [Module] { model.layers }
    }

    enum OpenMedMapleTesting {
        static func parameterPaths(
            configuration: OpenMedMapleConfiguration
        ) -> [String] {
            MapleMLXModel(configuration).parameters().flattened().map(\.0)
        }
    }

    private struct OpenMedMapleTokenizerAdapter: MLXLMCommon.Tokenizer, @unchecked Sendable {
        let tokenizer: any Tokenizers.Tokenizer

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenizer.decode(tokens: tokenIds, skipSpecialTokens: skipSpecialTokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            tokenizer.convertTokenToId(token)
        }

        func convertIdToToken(_ id: Int) -> String? {
            tokenizer.convertIdToToken(id)
        }

        var bosToken: String? { tokenizer.bosToken }
        var eosToken: String? { tokenizer.eosToken }
        var unknownToken: String? { tokenizer.unknownToken }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            try tokenizer.applyChatTemplate(
                messages: messages.map(Self.foundationDictionary),
                tools: tools?.map(Self.foundationDictionary),
                additionalContext: additionalContext.map(Self.foundationDictionary)
            )
        }

        private static func foundationDictionary(
            _ value: [String: any Sendable]
        ) -> [String: Any] {
            value.reduce(into: [:]) { result, item in
                result[item.key] = item.value
            }
        }
    }

    private struct OpenMedMapleTokenizerLoader: TokenizerLoader {
        func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
            let tokenizer = try OpenMed.loadTokenizer(
                tokenizerName: directory.path,
                tokenizerFolderURL: directory
            )
            return OpenMedMapleTokenizerAdapter(tokenizer: tokenizer)
        }
    }

    /// Errors raised before or during local Maple inference.
    public enum OpenMedMapleRuntimeError: LocalizedError {
        case missingFiles([String])
        case unsupportedArchitecture(String)
        case invalidConfiguration(String)

        public var errorDescription: String? {
            switch self {
            case .missingFiles(let files):
                return "The Maple model cache is incomplete. Missing: \(files.joined(separator: ", "))."
            case .unsupportedArchitecture(let type):
                return "Unsupported Maple model_type: \(type)."
            case .invalidConfiguration(let detail):
                return "Invalid Maple model configuration: \(detail)."
            }
        }
    }

    /// Local-only Maple text generation backed by MLX on Apple silicon.
    ///
    /// The runtime never downloads, logs, or persists prompts. Callers must
    /// prepare a complete local model directory before initialization.
    public actor OpenMedMaple {
        public static let repositoryID = "deepgrove/maple-preview-2bit-mlx"
        public static let pinnedRevision = "361db5da5e74ff6fcdd852d478e1f266ce11013a"
        public static let estimatedDownloadBytes: Int64 = 5_324_164_826

        /// Files needed by the exact-head runtime. `model-flashhead.safetensors`
        /// is intentionally excluded because approximate-head generation is off.
        public static let requiredModelFiles = [
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ]

        public static let optionalTokenizerFiles = [
            "added_tokens.json",
            "chat_template.jinja",
            "merges.txt",
            "special_tokens_map.json",
            "vocab.json",
        ]

        private var container: ModelContainer?

        public init(modelDirectoryURL: URL) async throws {
            let missing = Self.missingRequiredFiles(in: modelDirectoryURL)
            guard missing.isEmpty else {
                throw OpenMedMapleRuntimeError.missingFiles(missing)
            }

            let registry = ModelTypeRegistry<LanguageModel>(creators: [
                "maple": { data in
                    let configuration = try JSONDecoder.json5().decode(
                        OpenMedMapleConfiguration.self,
                        from: data
                    )
                    try configuration.validate()
                    return MapleMLXModel(configuration)
                }
            ])
            let factory = LLMModelFactory(
                typeRegistry: registry,
                modelRegistry: LLMRegistry.shared
            )
            container = try await factory.loadContainer(
                from: modelDirectoryURL,
                using: OpenMedMapleTokenizerLoader()
            )
        }

        /// Returns whether all files needed for exact-head inference exist.
        public nonisolated static func isModelDirectoryReady(_ directory: URL) -> Bool {
            missingRequiredFiles(in: directory).isEmpty
        }

        /// Runs a deterministic task without sending document text off-device.
        public func complete(_ request: OpenMedMapleRequest) async throws
            -> OpenMedMapleResponse
        {
            guard let container else {
                throw OpenMedMapleRuntimeError.invalidConfiguration(
                    "the runtime has been unloaded"
                )
            }
            let messages = OpenMedMaplePrompt.messages(for: request).map { message in
                switch message.role {
                case .system:
                    return Chat.Message.system(message.content)
                case .user:
                    return Chat.Message.user(message.content)
                case .assistant:
                    return Chat.Message.assistant(message.content)
                }
            }
            let input = try await container.prepare(
                input: UserInput(prompt: .chat(messages))
            )
            let parameters = GenerateParameters(
                maxTokens: min(request.maximumTokens, 4_096),
                temperature: 0,
                topP: 1,
                topK: 0,
                prefillStepSize: 256,
                seed: 0
            )
            let stream = try await container.generate(
                input: input,
                parameters: parameters
            )
            var generatedText = ""
            for await event in stream {
                try Task.checkCancellation()
                if case .chunk(let text) = event {
                    generatedText.append(text)
                }
            }
            return try OpenMedMapleOutputParser.parse(
                generatedText,
                task: request.task,
                sourceDocument: request.document,
                allowedEntityLabels: request.entityLabels,
                allowedRelationLabels: request.relationLabels
            )
        }

        /// Releases model ownership. Call before switching to another large
        /// on-device model.
        public func unload() {
            container = nil
            OpenMed.clearRuntimeMemoryCache()
        }

        private nonisolated static func missingRequiredFiles(in directory: URL) -> [String] {
            requiredModelFiles.filter { file in
                let url = directory.appending(path: file)
                guard FileManager.default.fileExists(atPath: url.path) else { return true }
                let values = try? url.resourceValues(forKeys: [.fileSizeKey])
                return (values?.fileSize ?? 0) <= 0
            }
        }
    }
#endif
