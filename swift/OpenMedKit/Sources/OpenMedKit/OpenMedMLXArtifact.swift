import Foundation

struct OpenMedMLXManifest: Decodable, Sendable {
    struct Tokenizer: Decodable, Sendable {
        let path: String
        let files: [String]
    }

    struct Quantization: Decodable, Sendable {
        let bits: Int
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
        case tokenizer
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

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numAttentionHeads = "num_attention_heads"
        case numHiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case typeVocabularySize = "type_vocab_size"
        case layerNormEps = "layer_norm_eps"
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

        let normalizedFamily = manifest.family.replacingOccurrences(of: "_", with: "-").lowercased()
        guard ["bert", "distilbert", "roberta", "xlm-roberta", "electra"].contains(normalizedFamily)
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
        if !listedTokenizerFiles.isEmpty {
            guard hasListedTokenizerAssets else {
                throw OpenMedMLXArtifactError.missingTokenizerAssets(tokenizerDirectoryURL)
            }
            resolvedTokenizerDirectoryURL = tokenizerDirectoryURL
        } else if !discoveredTokenizerFiles.isEmpty {
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
            tokenizer: .init(
                path: ".",
                files: discoverTokenizerFiles(in: modelDirectoryURL)
            )
        )
    }
}
