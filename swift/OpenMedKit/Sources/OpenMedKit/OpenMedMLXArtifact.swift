import Foundation

struct OpenMedMLXManifest: Decodable, Sendable {
    struct Tokenizer: Decodable, Sendable {
        let path: String
        let files: [String]
    }

    struct Quantization: Decodable, Sendable {
        let bits: Int
    }

    struct PromptSpec: Decodable, Sendable {
        let kind: String?
        let entityToken: String?
        let relationToken: String?
        let labelToken: String?
        let separatorToken: String?
        let exampleToken: String?
        let promptFirst: Bool?
        let classTokenIndex: Int?
        let relTokenIndex: Int?
        let textTokenIndex: Int?
        let exampleTokenIndex: Int?
        let embedMarkerToken: Bool?
        let splitMode: String?

        enum CodingKeys: String, CodingKey {
            case kind
            case entityToken = "entity_token"
            case relationToken = "relation_token"
            case labelToken = "label_token"
            case separatorToken = "separator_token"
            case exampleToken = "example_token"
            case promptFirst = "prompt_first"
            case classTokenIndex = "class_token_index"
            case relTokenIndex = "rel_token_index"
            case textTokenIndex = "text_token_index"
            case exampleTokenIndex = "example_token_index"
            case embedMarkerToken = "embed_marker_token"
            case splitMode = "split_mode"
        }
    }

    let format: String
    let formatVersion: Int
    let task: String
    let family: String
    let sourceModelID: String
    let configPath: String
    let labelMapPath: String?
    let preferredWeights: String
    let fallbackWeights: [String]
    let availableWeights: [String]
    let weightsFormat: String
    let quantization: Quantization?
    let maxSequenceLength: Int?
    let promptSpec: PromptSpec?
    let tokenizer: Tokenizer

    enum CodingKeys: String, CodingKey {
        case format
        case formatVersion = "format_version"
        case task
        case family
        case sourceModelID = "source_model_id"
        case configPath = "config_path"
        case labelMapPath = "label_map_path"
        case preferredWeights = "preferred_weights"
        case fallbackWeights = "fallback_weights"
        case availableWeights = "available_weights"
        case weightsFormat = "weights_format"
        case quantization
        case maxSequenceLength = "max_sequence_length"
        case promptSpec = "prompt_spec"
        case tokenizer
    }
}

enum OpenMedMLXTask: String, Sendable {
    case tokenClassification = "token-classification"
    case zeroShotNER = "zero-shot-ner"
    case zeroShotSequenceClassification = "zero-shot-sequence-classification"
    case zeroShotRelationExtraction = "zero-shot-relation-extraction"

    init?(manifestValue: String) {
        self.init(rawValue: manifestValue.replacingOccurrences(of: "_", with: "-").lowercased())
    }
}

enum OpenMedMLXFamily: String, Sendable {
    case bert
    case distilbert
    case roberta
    case xlmRoberta = "xlm-roberta"
    case electra
    case debertaV2 = "deberta-v2"
    case openaiPrivacyFilter = "openai-privacy-filter"
    case glinerUniEncoderSpan = "gliner-uni-encoder-span"
    case gliclassUniEncoder = "gliclass-uni-encoder"
    case glinerUniEncoderTokenRelex = "gliner-uni-encoder-token-relex"

    init?(manifestValue: String) {
        self.init(rawValue: manifestValue.replacingOccurrences(of: "_", with: "-").lowercased())
    }
}

enum OpenMedMLXArtifactError: LocalizedError {
    case invalidManifestFormat(String)
    case missingTokenizerAssets(URL)
    case missingTokenizerReference(URL)
    case unsupportedArchitecture(String)
    case missingWeights(URL)

    var errorDescription: String? {
        switch self {
        case .invalidManifestFormat(let format):
            return "Unsupported MLX manifest format: \(format)"
        case .missingTokenizerAssets(let url):
            return "Missing tokenizer assets in \(url.path)"
        case .missingTokenizerReference(let url):
            return "No tokenizer assets or source tokenizer reference were found in \(url.path)"
        case .unsupportedArchitecture(let family):
            return "Unsupported MLX architecture for Swift runtime: \(family)"
        case .missingWeights(let url):
            return "No MLX weights were found in \(url.path)"
        }
    }
}

struct OpenMedMLXBertConfiguration: Decodable, Sendable {
    let modelType: String
    let vocabularySize: Int
    let hiddenSize: Int
    let numAttentionHeads: Int
    let numHiddenLayers: Int
    let intermediateSize: Int
    let maxPositionEmbeddings: Int
    let typeVocabularySize: Int
    let layerNormEps: Float
    let numLabels: Int
    let positionOffset: Int
    let weightsFormat: String?
    let quantizationBits: Int?
    let id2label: [Int: String]
    let sourceModelName: String?
    let encoderHiddenSize: Int
    let embeddingSize: Int
    let hiddenDropoutProb: Float
    let attentionDropoutProb: Float
    let hiddenAct: String
    let relativeAttention: Bool
    let positionBiasedInput: Bool
    let shareAttentionKey: Bool
    let positionAttentionTypes: [String]
    let positionBuckets: Int
    let maxRelativePositions: Int
    let normRelativeEmbedding: String
    let maxWidth: Int
    let classTokenIndex: Int?
    let relTokenIndex: Int?
    let textTokenIndex: Int?
    let exampleTokenIndex: Int?
    let numRNNLayers: Int
    let poolingStrategy: String
    let extractTextFeatures: Bool
    let useSegmentEmbeddings: Bool
    let normalizeFeatures: Bool
    let logitScaleInitValue: Float
    let embedEntityToken: Bool
    let embedClassToken: Bool
    let embedRelationToken: Bool?
    let headDim: Int
    let numKeyValueHeads: Int
    let numExperts: Int
    let expertsPerToken: Int
    let bidirectionalLeftContext: Int
    let bidirectionalRightContext: Int
    let initialContextLength: Int
    let ropeTheta: Float
    let ropeScalingFactor: Float
    let ropeNTKAlpha: Float
    let ropeNTKBeta: Float
    let parameterDType: String
    let encoding: String?
    let rmsNormEps: Float
    let swigluLimit: Float
    let viterbiBiases: [String: Float]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case encoderHiddenSize = "encoder_hidden_size"
        case embeddingSize = "embedding_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabularySize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
        case hiddenDropoutProb = "hidden_dropout_prob"
        case attentionDropoutProb = "attention_probs_dropout_prob"
        case hiddenAct = "hidden_act"
        case relativeAttention = "relative_attention"
        case positionBiasedInput = "position_biased_input"
        case shareAttentionKey = "share_att_key"
        case positionAttentionTypes = "pos_att_type"
        case positionBuckets = "position_buckets"
        case maxRelativePositions = "max_relative_positions"
        case normRelativeEmbedding = "norm_rel_ebd"
        case maxWidth = "max_width"
        case classTokenIndex = "class_token_index"
        case relTokenIndex = "rel_token_index"
        case textTokenIndex = "text_token_index"
        case exampleTokenIndex = "example_token_index"
        case numRNNLayers = "num_rnn_layers"
        case poolingStrategy = "pooling_strategy"
        case extractTextFeatures = "extract_text_features"
        case useSegmentEmbeddings = "use_segment_embeddings"
        case normalizeFeatures = "normalize_features"
        case logitScaleInitValue = "logit_scale_init_value"
        case embedEntityToken = "embed_ent_token"
        case embedClassToken = "embed_class_token"
        case embedRelationToken = "embed_rel_token"
        case headDim = "head_dim"
        case numKeyValueHeads = "num_key_value_heads"
        case numExperts = "num_experts"
        case expertsPerToken = "experts_per_token"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case bidirectionalLeftContext = "bidirectional_left_context"
        case bidirectionalRightContext = "bidirectional_right_context"
        case slidingWindow = "sliding_window"
        case initialContextLength = "initial_context_length"
        case ropeTheta = "rope_theta"
        case ropeScalingFactor = "rope_scaling_factor"
        case ropeNTKAlpha = "rope_ntk_alpha"
        case ropeNTKBeta = "rope_ntk_beta"
        case parameterDType = "param_dtype"
        case encoding
        case rmsNormEps = "rms_norm_eps"
        case swigluLimit = "swiglu_limit"
        case viterbiBiases = "_mlx_viterbi_biases"
        case numLabels = "num_labels"
        case positionOffset = "_mlx_position_offset"
        case weightsFormat = "_mlx_weights_format"
        case mlxModelType = "_mlx_model_type"
        case sourceModelName = "_name_or_path"
        case id2label
        case quantization = "_mlx_quantization"

        case dim
        case nHeads = "n_heads"
        case nLayers = "n_layers"
        case hiddenDim = "hidden_dim"
        case padTokenID = "pad_token_id"
        case dropout
    }

    struct Quantization: Decodable, Sendable {
        let bits: Int
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let rawModelType =
            try container.decodeIfPresent(String.self, forKey: .mlxModelType)
            ?? container.decode(String.self, forKey: .modelType)
        modelType = rawModelType

        vocabularySize = try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 30522
        numLabels = try container.decodeIfPresent(Int.self, forKey: .numLabels) ?? 2
        maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 512
        layerNormEps = try container.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-12
        weightsFormat = try container.decodeIfPresent(String.self, forKey: .weightsFormat)
        quantizationBits =
            try container.decodeIfPresent(Quantization.self, forKey: .quantization)?.bits
        sourceModelName = try container.decodeIfPresent(String.self, forKey: .sourceModelName)
        let configuredHiddenDropout =
            try container.decodeIfPresent(Float.self, forKey: .hiddenDropoutProb)
        let configuredDropout = try container.decodeIfPresent(Float.self, forKey: .dropout)
        hiddenDropoutProb = configuredHiddenDropout ?? configuredDropout ?? 0.0
        attentionDropoutProb =
            try container.decodeIfPresent(Float.self, forKey: .attentionDropoutProb)
            ?? hiddenDropoutProb
        hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu"
        relativeAttention = try container.decodeIfPresent(Bool.self, forKey: .relativeAttention) ?? false
        positionBiasedInput =
            try container.decodeIfPresent(Bool.self, forKey: .positionBiasedInput) ?? true
        shareAttentionKey =
            try container.decodeIfPresent(Bool.self, forKey: .shareAttentionKey) ?? false
        positionAttentionTypes =
            try container.decodeIfPresent([String].self, forKey: .positionAttentionTypes) ?? []
        positionBuckets = try container.decodeIfPresent(Int.self, forKey: .positionBuckets) ?? -1
        maxRelativePositions =
            try container.decodeIfPresent(Int.self, forKey: .maxRelativePositions) ?? -1
        normRelativeEmbedding =
            try container.decodeIfPresent(String.self, forKey: .normRelativeEmbedding) ?? "none"
        maxWidth = try container.decodeIfPresent(Int.self, forKey: .maxWidth) ?? 12
        classTokenIndex = try container.decodeIfPresent(Int.self, forKey: .classTokenIndex)
        relTokenIndex = try container.decodeIfPresent(Int.self, forKey: .relTokenIndex)
        textTokenIndex = try container.decodeIfPresent(Int.self, forKey: .textTokenIndex)
        exampleTokenIndex = try container.decodeIfPresent(Int.self, forKey: .exampleTokenIndex)
        numRNNLayers = try container.decodeIfPresent(Int.self, forKey: .numRNNLayers) ?? 0
        poolingStrategy =
            try container.decodeIfPresent(String.self, forKey: .poolingStrategy) ?? "first"
        extractTextFeatures =
            try container.decodeIfPresent(Bool.self, forKey: .extractTextFeatures) ?? false
        useSegmentEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .useSegmentEmbeddings) ?? false
        normalizeFeatures =
            try container.decodeIfPresent(Bool.self, forKey: .normalizeFeatures) ?? false
        logitScaleInitValue =
            try container.decodeIfPresent(Float.self, forKey: .logitScaleInitValue) ?? 1.0
        embedEntityToken =
            try container.decodeIfPresent(Bool.self, forKey: .embedEntityToken) ?? true
        embedClassToken =
            try container.decodeIfPresent(Bool.self, forKey: .embedClassToken) ?? true
        embedRelationToken = try container.decodeIfPresent(Bool.self, forKey: .embedRelationToken)
        headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? 64
        let configuredNumKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads)
        let configuredNumAttentionHeads =
            try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = configuredNumKeyValueHeads ?? configuredNumAttentionHeads ?? 12

        let configuredNumExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts)
        let configuredNumLocalExperts =
            try container.decodeIfPresent(Int.self, forKey: .numLocalExperts)
        numExperts = configuredNumExperts ?? configuredNumLocalExperts ?? 1

        let configuredExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .expertsPerToken)
        let configuredNumExpertsPerToken =
            try container.decodeIfPresent(Int.self, forKey: .numExpertsPerToken)
        expertsPerToken = configuredExpertsPerToken ?? configuredNumExpertsPerToken ?? 1
        let slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow)
        bidirectionalLeftContext =
            try container.decodeIfPresent(Int.self, forKey: .bidirectionalLeftContext)
            ?? slidingWindow.map { max(0, ($0 - 1) / 2) }
            ?? 0
        bidirectionalRightContext =
            try container.decodeIfPresent(Int.self, forKey: .bidirectionalRightContext)
            ?? slidingWindow.map { max(0, ($0 - 1) / 2) }
            ?? 0
        initialContextLength =
            try container.decodeIfPresent(Int.self, forKey: .initialContextLength) ?? 4096
        ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 150_000.0
        ropeScalingFactor =
            try container.decodeIfPresent(Float.self, forKey: .ropeScalingFactor) ?? 1.0
        ropeNTKAlpha = try container.decodeIfPresent(Float.self, forKey: .ropeNTKAlpha) ?? 1.0
        ropeNTKBeta = try container.decodeIfPresent(Float.self, forKey: .ropeNTKBeta) ?? 32.0
        parameterDType =
            try container.decodeIfPresent(String.self, forKey: .parameterDType) ?? "bfloat16"
        encoding = try container.decodeIfPresent(String.self, forKey: .encoding)
        rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        swigluLimit = try container.decodeIfPresent(Float.self, forKey: .swigluLimit) ?? 7.0
        viterbiBiases =
            try container.decodeIfPresent([String: Float].self, forKey: .viterbiBiases) ?? [:]

        let normalized = rawModelType.replacingOccurrences(of: "_", with: "-").lowercased()
        let resolvedPositionOffset: Int
        if normalized == "distilbert" {
            hiddenSize = try container.decodeIfPresent(Int.self, forKey: .dim) ?? 768
            numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .nHeads) ?? 12
            numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .nLayers) ?? 12
            intermediateSize = try container.decodeIfPresent(Int.self, forKey: .hiddenDim) ?? 3072
            typeVocabularySize = 0
            resolvedPositionOffset =
                try container.decodeIfPresent(Int.self, forKey: .positionOffset) ?? 0
        } else {
            hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768
            numAttentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 12
            numHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 12
            intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072

            if normalized == "roberta" || normalized == "xlm-roberta" {
                let padTokenID = try container.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 1
                typeVocabularySize = try container.decodeIfPresent(
                    Int.self,
                    forKey: .typeVocabularySize
                ) ?? 1
                resolvedPositionOffset =
                    try container.decodeIfPresent(Int.self, forKey: .positionOffset)
                    ?? (padTokenID + 1)
            } else {
                typeVocabularySize =
                    try container.decodeIfPresent(Int.self, forKey: .typeVocabularySize) ?? 2
                resolvedPositionOffset =
                    try container.decodeIfPresent(Int.self, forKey: .positionOffset) ?? 0
            }
        }
        positionOffset = resolvedPositionOffset
        encoderHiddenSize =
            try container.decodeIfPresent(Int.self, forKey: .encoderHiddenSize) ?? hiddenSize
        embeddingSize =
            try container.decodeIfPresent(Int.self, forKey: .embeddingSize) ?? encoderHiddenSize

        let rawLabels =
            try container.decodeIfPresent([String: String].self, forKey: .id2label) ?? [:]
        id2label = Dictionary(uniqueKeysWithValues: rawLabels.compactMap { key, value in
            Int(key).map { ($0, value) }
        })
    }
}

struct OpenMedMLXArtifact: Sendable {
    private static let knownTokenizerFiles = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    ]

    let directoryURL: URL
    let manifest: OpenMedMLXManifest
    let configuration: OpenMedMLXBertConfiguration
    let id2label: [Int: String]
    let tokenizerDirectoryURL: URL?
    let tokenizerName: String?
    let task: OpenMedMLXTask
    let family: OpenMedMLXFamily

    init(modelDirectoryURL: URL) throws {
        let manifestURL = modelDirectoryURL.appending(path: "openmed-mlx.json")

        let decoder = JSONDecoder()
        let configURL = modelDirectoryURL.appending(path: "config.json")
        let configData = try Data(contentsOf: configURL)
        let configuration = try decoder.decode(OpenMedMLXBertConfiguration.self, from: configData)

        let manifest: OpenMedMLXManifest
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            let manifestData = try Data(contentsOf: manifestURL)
            manifest = try decoder.decode(OpenMedMLXManifest.self, from: manifestData)
            guard manifest.format == "openmed-mlx" else {
                throw OpenMedMLXArtifactError.invalidManifestFormat(manifest.format)
            }
        } else {
            manifest = Self.makeLegacyManifest(
                modelDirectoryURL: modelDirectoryURL,
                configuration: configuration
            )
        }

        guard
            let task = OpenMedMLXTask(manifestValue: manifest.task),
            let family = OpenMedMLXFamily(manifestValue: manifest.family),
            Self.supports(task: task, family: family)
        else {
            throw OpenMedMLXArtifactError.unsupportedArchitecture(manifest.family)
        }

        let tokenizerDirectoryURL = modelDirectoryURL.appending(path: manifest.tokenizer.path)
        let listedTokenizerFiles = manifest.tokenizer.files
        let hasListedTokenizerAssets = listedTokenizerFiles.allSatisfy { fileName in
            FileManager.default.fileExists(
                atPath: tokenizerDirectoryURL.appending(path: fileName).path
            )
        }
        let discoveredTokenizerFiles = Self.discoverTokenizerFiles(in: tokenizerDirectoryURL)

        let resolvedTokenizerDirectoryURL: URL?
        let hasUsableListedTokenizer = hasListedTokenizerAssets
            && Self.hasUsableTokenizerAssets(listedTokenizerFiles)
        let hasUsableDiscoveredTokenizer = Self.hasUsableTokenizerAssets(discoveredTokenizerFiles)
        if !listedTokenizerFiles.isEmpty {
            guard hasListedTokenizerAssets else {
                throw OpenMedMLXArtifactError.missingTokenizerAssets(tokenizerDirectoryURL)
            }
            if hasUsableListedTokenizer {
                resolvedTokenizerDirectoryURL = tokenizerDirectoryURL
            } else if configuration.sourceModelName != nil {
                resolvedTokenizerDirectoryURL = nil
            } else {
                throw OpenMedMLXArtifactError.missingTokenizerAssets(tokenizerDirectoryURL)
            }
        } else if hasUsableDiscoveredTokenizer {
            resolvedTokenizerDirectoryURL = tokenizerDirectoryURL
        } else if configuration.sourceModelName != nil {
            resolvedTokenizerDirectoryURL = nil
        } else {
            throw OpenMedMLXArtifactError.missingTokenizerReference(modelDirectoryURL)
        }

        let labelMap: [Int: String]
        if let labelMapPath = manifest.labelMapPath {
            let labelURL = modelDirectoryURL.appending(path: labelMapPath)
            let data = try Data(contentsOf: labelURL)
            let raw = try decoder.decode([String: String].self, from: data)
            labelMap = Dictionary(uniqueKeysWithValues: raw.compactMap { key, value in
                Int(key).map { ($0, value) }
            })
        } else {
            labelMap = configuration.id2label
        }

        self.directoryURL = modelDirectoryURL
        self.manifest = manifest
        self.configuration = configuration
        self.id2label = labelMap
        self.tokenizerDirectoryURL = resolvedTokenizerDirectoryURL
        self.tokenizerName = configuration.sourceModelName
        self.task = task
        self.family = family
    }

    var weightCandidateURLs: [URL] {
        let candidates = Array(
            NSOrderedSet(array: manifest.availableWeights + [manifest.preferredWeights] + manifest.fallbackWeights)
        ) as? [String] ?? []
        return candidates.map { directoryURL.appending(path: $0) }
    }

    private static func discoverTokenizerFiles(in directoryURL: URL) -> [String] {
        knownTokenizerFiles.filter {
            FileManager.default.fileExists(atPath: directoryURL.appending(path: $0).path)
        }
    }

    private static func hasUsableTokenizerAssets(_ files: [String]) -> Bool {
        let fileSet = Set(files)
        if fileSet.contains("tokenizer.json") {
            return true
        }
        if fileSet.contains("vocab.txt") {
            return true
        }
        if fileSet.contains("vocab.json") && fileSet.contains("merges.txt") {
            return true
        }
        if fileSet.contains("spm.model") || fileSet.contains("sentencepiece.bpe.model") {
            return true
        }
        return false
    }

    private static func supports(task: OpenMedMLXTask, family: OpenMedMLXFamily) -> Bool {
        switch (task, family) {
        case (.tokenClassification, .bert),
             (.tokenClassification, .distilbert),
             (.tokenClassification, .roberta),
             (.tokenClassification, .xlmRoberta),
             (.tokenClassification, .electra),
             (.tokenClassification, .openaiPrivacyFilter):
            return true
        case (.zeroShotNER, .glinerUniEncoderSpan),
             (.zeroShotSequenceClassification, .gliclassUniEncoder),
             (.zeroShotRelationExtraction, .glinerUniEncoderTokenRelex):
            return true
        default:
            return false
        }
    }

    private static func makeLegacyManifest(
        modelDirectoryURL: URL,
        configuration: OpenMedMLXBertConfiguration
    ) -> OpenMedMLXManifest {
        let availableWeights = ["weights.safetensors", "weights.npz"].filter {
            FileManager.default.fileExists(atPath: modelDirectoryURL.appending(path: $0).path)
        }
        let preferredWeights: String
        if configuration.weightsFormat == "npz",
            availableWeights.contains("weights.npz")
        {
            preferredWeights = "weights.npz"
        } else if availableWeights.contains("weights.safetensors") {
            preferredWeights = "weights.safetensors"
        } else {
            preferredWeights = availableWeights.first ?? "weights.safetensors"
        }

        return OpenMedMLXManifest(
            format: "openmed-mlx",
            formatVersion: 1,
            task: "token-classification",
            family: configuration.modelType,
            sourceModelID: configuration.sourceModelName ?? modelDirectoryURL.lastPathComponent,
            configPath: "config.json",
            labelMapPath: FileManager.default.fileExists(
                atPath: modelDirectoryURL.appending(path: "id2label.json").path
            ) ? "id2label.json" : nil,
            preferredWeights: preferredWeights,
            fallbackWeights: availableWeights.filter { $0 != preferredWeights },
            availableWeights: availableWeights,
            weightsFormat: configuration.weightsFormat ?? "safetensors",
            quantization: configuration.quantizationBits.map(OpenMedMLXManifest.Quantization.init),
            maxSequenceLength: configuration.maxPositionEmbeddings,
            promptSpec: nil,
            tokenizer: .init(
                path: ".",
                files: discoverTokenizerFiles(in: modelDirectoryURL)
            )
        )
    }
}
