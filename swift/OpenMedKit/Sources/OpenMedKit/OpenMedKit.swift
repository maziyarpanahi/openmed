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
///     repoID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
/// )
/// let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: modelDirectory))
/// let entities = try openmed.extractPII("Patient John Doe, SSN 123-45-6789")
/// for entity in entities {
///     print(entity)  // [first_name] "John Doe" (8:16) conf=0.95
/// }
/// ```
public final class OpenMed {
    private enum Runtime {
        case coreML(NERPipeline)
        case mlx(MLXTokenClassificationPipeline)
        case privacyFilter(OpenMedPrivacyFilterPipeline)
    }

    private let runtime: Runtime
    private let tokenizer: (any Tokenizer)?
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
            let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
            if artifact.family == .openaiPrivacyFilter {
                let pipeline = try OpenMedPrivacyFilterPipeline(
                    artifact: artifact,
                    maxSeqLength: maxSeqLength
                )
                self.runtime = .privacyFilter(pipeline)
                self.tokenizer = nil
                self.maxSeqLength = pipeline.resolvedMaxSequenceLength
            } else {
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
        let entities: [EntityPrediction]
        switch runtime {
        case .coreML(let pipeline):
            let (inputIDs, attentionMask, _, offsets) = try tokenize(text)
            entities = try pipeline.predict(
                inputIds: inputIDs,
                attentionMask: attentionMask,
                offsets: offsets,
                text: text
            )
        case .mlx(let pipeline):
            let (inputIDs, attentionMask, tokenTypeIDs, offsets) = try tokenize(text)
            entities = try pipeline.predict(
                inputIDs: inputIDs,
                attentionMask: attentionMask,
                tokenTypeIDs: tokenTypeIDs,
                offsets: offsets,
                text: text
            )
        case .privacyFilter(let pipeline):
            entities = try pipeline.predict(text)
        }

        return entities.filter { $0.confidence >= confidenceThreshold }
    }

    /// Run PII detection on the given text with OpenMed's smart post-processing.
    ///
    /// This applies the same high-level PII cleanup used by the Python package:
    /// grouped BIO spans, span repair, and semantic-unit merging for items such
    /// as dates, SSNs, phone numbers, and emails.
    public func extractPII(
        _ text: String,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) throws -> [EntityPrediction] {
        let entities = try analyzeText(text, confidenceThreshold: confidenceThreshold)
        let repairedEntities = PostProcessing.repairEntitySpans(entities, text: text)

        guard useSmartMerging else {
            return repairedEntities
        }

        switch runtime {
        case .privacyFilter:
            return PostProcessing.mergePIIEntities(
                repairedEntities,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true,
                allowSemanticOnlyMatches: false,
                allowSemanticLabelExpansion: false
            )
        case .coreML, .mlx:
            return PostProcessing.mergePIIEntities(
                repairedEntities,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true
            )
        }
    }

    // MARK: - Private

    private func tokenize(_ text: String) throws -> ([Int], [Int], [Int], [(Int, Int)]) {
        // Use swift-transformers for tokenization
        // This ensures token IDs match the Python HuggingFace tokenizer
        guard let tokenizer else {
            throw TokenizerError.missingConfig
        }
        let inputIds = Array(tokenizer(text, addSpecialTokens: true).prefix(maxSeqLength))
        let tokens = tokenizer.convertIdsToTokens(inputIds).map { $0 ?? "" }
        let attentionMask = Array(repeating: 1, count: inputIds.count)
        let tokenTypeIDs = Array(repeating: 0, count: inputIds.count)
        let offsets = Self.buildOffsets(tokens: tokens, in: text)

        return (inputIds, attentionMask, tokenTypeIDs, offsets)
    }

    static func loadTokenizer(
        tokenizerName: String,
        tokenizerFolderURL: URL?
    ) throws -> any Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<any Tokenizer, Error>?

        Task.detached {
            do {
                let tokenizer = try await loadTokenizerAsync(
                    tokenizerName: tokenizerName,
                    tokenizerFolderURL: tokenizerFolderURL
                )
                result = .success(tokenizer)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let unwrapped = result else {
            throw TokenizerError.missingConfig
        }
        return try unwrapped.get()
    }

    private static func loadTokenizerAsync(
        tokenizerName: String,
        tokenizerFolderURL: URL?
    ) async throws -> any Tokenizer {
        if let tokenizerFolderURL {
            return try loadTokenizerFromDirectory(
                tokenizerFolderURL,
                fallbackTokenizerName: tokenizerName
            )
        }

        if tokenizerName.contains("/") {
            let localDirectory = try await ensureTokenizerAssets(modelID: tokenizerName)
            return try loadTokenizerFromDirectory(
                localDirectory,
                fallbackTokenizerName: tokenizerName
            )
        }

        return try await AutoTokenizer.from(pretrained: tokenizerName)
    }

    private static func loadTokenizerFromDirectory(
        _ directoryURL: URL,
        fallbackTokenizerName: String?
    ) throws -> any Tokenizer {
        let tokenizerDataURL = directoryURL.appending(path: "tokenizer.json")
        let tokenizerConfigURL = directoryURL.appending(path: "tokenizer_config.json")

        guard FileManager.default.fileExists(atPath: tokenizerDataURL.path),
              FileManager.default.fileExists(atPath: tokenizerConfigURL.path) else {
            if let fallbackTokenizerName {
                return try blockingPretrainedTokenizer(named: fallbackTokenizerName)
            }
            throw TokenizerError.missingConfig
        }

        let preparedDirectory = try prepareTokenizerDirectory(directoryURL)
        return try blockingLocalTokenizer(from: preparedDirectory)
    }

    static func patchTokenizerConfigDataIfNeeded(
        tokenizerConfigData: Data,
        tokenizerData: Data
    ) throws -> Data? {
        guard
            let tokenizerConfig = try JSONSerialization.jsonObject(with: tokenizerConfigData)
                as? [String: Any],
            let tokenizerDataObject = try JSONSerialization.jsonObject(with: tokenizerData)
                as? [String: Any]
        else {
            return nil
        }

        let modelType =
            ((tokenizerDataObject["model"] as? [String: Any])?["type"] as? String)?
            .lowercased()
        let tokenizerClass = tokenizerConfig["tokenizer_class"] as? String

        guard modelType == "unigram" else {
            return nil
        }

        let shouldForceUnigram =
            tokenizerClass == nil
            || tokenizerClass == "RobertaTokenizer"
            || tokenizerClass == "RobertaTokenizerFast"
            || tokenizerClass == "XLMRobertaTokenizer"
            || tokenizerClass == "XLMRobertaTokenizerFast"
            || tokenizerClass == "DebertaV2Tokenizer"
            || tokenizerClass == "DebertaV2TokenizerFast"
            || tokenizerClass == "PreTrainedTokenizer"

        let hasListShapedExtraSpecialTokens = tokenizerConfig["extra_special_tokens"] is [Any]

        guard shouldForceUnigram || hasListShapedExtraSpecialTokens else {
            return nil
        }

        var patchedConfig = tokenizerConfig
        if shouldForceUnigram {
            patchedConfig["tokenizer_class"] = "T5Tokenizer"
        }
        if let extraSpecialTokens = tokenizerConfig["extra_special_tokens"] as? [Any] {
            patchedConfig["extra_special_tokens"] = nil
            if patchedConfig["additional_special_tokens"] == nil {
                patchedConfig["additional_special_tokens"] = extraSpecialTokens
            }
        }
        return try JSONSerialization.data(
            withJSONObject: patchedConfig,
            options: [.prettyPrinted, .sortedKeys]
        )
    }

    static func prepareTokenizerDirectory(_ directoryURL: URL) throws -> URL {
        let tokenizerDataURL = directoryURL.appending(path: "tokenizer.json")
        let tokenizerConfigURL = directoryURL.appending(path: "tokenizer_config.json")
        let modelConfigURL = directoryURL.appending(path: "config.json")

        let tokenizerData = try Data(contentsOf: tokenizerDataURL)
        let tokenizerConfigData = try Data(contentsOf: tokenizerConfigURL)
        let patchedTokenizerConfigData =
            try patchTokenizerConfigDataIfNeeded(
                tokenizerConfigData: tokenizerConfigData,
                tokenizerData: tokenizerData
            ) ?? tokenizerConfigData

        if FileManager.default.fileExists(atPath: modelConfigURL.path),
           patchedTokenizerConfigData == tokenizerConfigData
        {
            return directoryURL
        }

        let preparedDirectory = try preparedTokenizerCacheDirectory(for: directoryURL)
        let fileManager = FileManager.default

        if fileManager.fileExists(atPath: preparedDirectory.path) {
            try fileManager.removeItem(at: preparedDirectory)
        }
        try fileManager.createDirectory(
            at: preparedDirectory,
            withIntermediateDirectories: true
        )

        for fileName in tokenizerAssetFileNames {
            let sourceURL = directoryURL.appending(path: fileName)
            let destinationURL = preparedDirectory.appending(path: fileName)
            guard fileManager.fileExists(atPath: sourceURL.path) else {
                continue
            }
            if fileName == "tokenizer_config.json" {
                continue
            }
            let fileData = try Data(contentsOf: sourceURL)
            try fileData.write(to: destinationURL, options: .atomic)
        }

        let preparedModelConfigURL = preparedDirectory.appending(path: "config.json")
        if fileManager.fileExists(atPath: modelConfigURL.path) {
            let modelConfigData = try Data(contentsOf: modelConfigURL)
            try modelConfigData.write(to: preparedModelConfigURL, options: .atomic)
        } else {
            try Data("{}".utf8).write(to: preparedModelConfigURL, options: .atomic)
        }

        try patchedTokenizerConfigData.write(
            to: preparedDirectory.appending(path: "tokenizer_config.json"),
            options: .atomic
        )
        return preparedDirectory
    }

    private static func preparedTokenizerCacheDirectory(for directoryURL: URL) throws -> URL {
        let base =
            try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        let leafName = sanitizedCacheComponent(directoryURL.lastPathComponent)
        let digest = stableDigest(for: directoryURL.path)
        return base
            .appending(path: "OpenMed", directoryHint: .isDirectory)
            .appending(path: "PreparedTokenizerAssets", directoryHint: .isDirectory)
            .appending(path: "\(leafName)-\(digest)", directoryHint: .isDirectory)
    }

    private static func blockingLocalTokenizer(from modelFolder: URL) throws -> any Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<any Tokenizer, Error>?

        Task.detached {
            do {
                result = .success(try await AutoTokenizer.from(modelFolder: modelFolder))
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let unwrapped = result else {
            throw TokenizerError.missingConfig
        }
        return try unwrapped.get()
    }

    private static func ensureTokenizerAssets(modelID: String) async throws -> URL {
        let directory = try tokenizerCacheDirectory(modelID: modelID)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        let requiredFiles = [
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        let optionalFiles = tokenizerAssetFileNames.filter { fileName in
            !requiredFiles.contains(fileName)
        }

        for fileName in requiredFiles {
            try await downloadTokenizerFile(
                modelID: modelID,
                relativePath: fileName,
                destinationURL: directory.appending(path: fileName),
                required: true
            )
        }

        for fileName in optionalFiles {
            try await downloadTokenizerFile(
                modelID: modelID,
                relativePath: fileName,
                destinationURL: directory.appending(path: fileName),
                required: false
            )
        }

        return directory
    }

    private static func tokenizerCacheDirectory(modelID: String) throws -> URL {
        let base =
            try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        let sanitized = modelID.replacingOccurrences(of: "/", with: "__")
        return base
            .appending(path: "OpenMed", directoryHint: .isDirectory)
            .appending(path: "TokenizerAssets", directoryHint: .isDirectory)
            .appending(path: sanitized, directoryHint: .isDirectory)
    }

    private static let tokenizerAssetFileNames = [
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

    private static func downloadTokenizerFile(
        modelID: String,
        relativePath: String,
        destinationURL: URL,
        required: Bool
    ) async throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return
        }

        let encodedModelID = modelID
            .split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let encodedPath = relativePath
            .split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")

        guard let url = URL(
            string: "https://huggingface.co/\(encodedModelID)/resolve/main/\(encodedPath)?download=1"
        ) else {
            throw TokenizerError.missingConfig
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw TokenizerError.missingConfig
        }
        if http.statusCode == 404 && !required {
            return
        }
        guard (200..<300).contains(http.statusCode) else {
            throw TokenizerError.missingConfig
        }

        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: destinationURL, options: .atomic)
    }

    private static func blockingPretrainedTokenizer(named name: String) throws -> any Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<any Tokenizer, Error>?

        Task.detached {
            do {
                result = .success(try await AutoTokenizer.from(pretrained: name))
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        guard let unwrapped = result else {
            throw TokenizerError.missingConfig
        }
        return try unwrapped.get()
    }

    private static func sanitizedCacheComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "__")
            .replacingOccurrences(of: ":", with: "_")
            .replacingOccurrences(of: " ", with: "_")
    }

    private static func stableDigest(for value: String) -> String {
        var hash: UInt64 = 0xcbf29ce484222325
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100000001b3
        }
        return String(format: "%016llx", hash)
    }

    static func buildOffsets(
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

            let searchSlice = text[searchStart...]
            let exactRange = searchSlice.range(of: piece)
            let insensitiveRange = searchSlice.range(
                of: piece,
                options: [.caseInsensitive, .diacriticInsensitive]
            )

            let range: Range<String.Index>?
            switch (exactRange, insensitiveRange) {
            case let (exact?, insensitive?):
                if exact.lowerBound <= insensitive.lowerBound {
                    range = exact
                } else {
                    range = insensitive
                }
            case let (exact?, nil):
                range = exact
            case let (nil, insensitive?):
                range = insensitive
            case (nil, nil):
                range = nil
            }

            if let range {
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
