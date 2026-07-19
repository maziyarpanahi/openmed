import CryptoKit
import Dispatch
import Foundation
import MLX
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

    /// Release cached MLX buffers that are no longer referenced by model objects.
    ///
    /// OpenMedKit runtimes own their model weights via ARC. Apps that swap between
    /// large on-device models can first drop their runtime references, then call
    /// this helper so MLX returns cached Metal buffers instead of carrying them
    /// into the next model load.
    public static func clearRuntimeMemoryCache() {
        Memory.clearCache()
    }

    /// Initialize OpenMed with an explicit backend.
    public init(
        backend: OpenMedBackend,
        maxSeqLength: Int = 512,
        allowNetworkAccess: Bool = true
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
                tokenizerFolderURL: tokenizerFolderURL,
                allowNetworkAccess: allowNetworkAccess
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
                    tokenizerFolderURL: pipeline.tokenizerDirectoryURL,
                    allowNetworkAccess: allowNetworkAccess
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
    ///   - allowNetworkAccess: Whether missing tokenizer assets may be downloaded.
    public convenience init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        tokenizerFolderURL: URL? = nil,
        maxSeqLength: Int = 512,
        allowNetworkAccess: Bool = true
    ) throws {
        try self.init(
            backend: .coreML(
                modelURL: modelURL,
                id2labelURL: id2labelURL,
                tokenizerName: tokenizerName,
                tokenizerFolderURL: tokenizerFolderURL
            ),
            maxSeqLength: maxSeqLength,
            allowNetworkAccess: allowNetworkAccess
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

    /// De-identify text under a bundled policy profile.
    ///
    /// Pass `Policy.defaultName` to use the default `hipaa_safe_harbor`
    /// posture. The policy argument is explicit so existing method-based
    /// `deidentify(_:)` call sites keep resolving to mask redaction. This path
    /// does not write the input text or detected span text to stdout, stderr,
    /// or logs.
    public func deidentify(
        _ text: String,
        policy: String,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) throws -> PolicyDeidentificationResult {
        let resolvedPolicy = try Policy(named: policy)
        return try deidentify(
            text,
            policy: resolvedPolicy,
            confidenceThreshold: confidenceThreshold,
            useSmartMerging: useSmartMerging
        )
    }

    /// De-identify text under an already loaded policy profile.
    ///
    /// Detected span offsets in the returned action records reference the
    /// original input text even when replacement lengths differ.
    public func deidentify(
        _ text: String,
        policy: Policy,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) throws -> PolicyDeidentificationResult {
        let entities = try extractPII(
            text,
            confidenceThreshold: confidenceThreshold,
            useSmartMerging: useSmartMerging
        )
        return Self.deidentify(text, entities: entities, policy: policy)
    }

    /// Detect and de-identify PII, returning a Python-schema-compatible result.
    public func deidentify(
        _ text: String,
        method: DeidentificationMethod = .mask,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) throws -> DeidentificationResult {
        let entities = try extractPII(
            text,
            confidenceThreshold: confidenceThreshold,
            useSmartMerging: useSmartMerging
        )
        let deidentifiedText = Self.deidentifiedText(
            text,
            entities: entities,
            method: method
        )
        return DeidentificationResult(
            originalText: text,
            deidentifiedText: deidentifiedText,
            entities: entities,
            method: method.rawValue
        )
    }

    /// Run PII detection over long text using overlapping token windows.
    ///
    /// The returned entity offsets always reference the original full text.
    /// Overlapping duplicate detections are merged before the final smart
    /// semantic merge pass.
    public func extractPIIChunked(
        _ text: String,
        confidenceThreshold: Float = 0.5,
        chunkTokenLimit: Int = 256,
        tokenOverlap: Int = 32,
        useSmartMerging: Bool = true
    ) throws -> [EntityPrediction] {
        let chunks = try makeTokenChunks(
            for: text,
            chunkTokenLimit: chunkTokenLimit,
            tokenOverlap: tokenOverlap
        )
        guard chunks.count > 1 else {
            return try extractPII(
                text,
                confidenceThreshold: confidenceThreshold,
                useSmartMerging: useSmartMerging
            )
        }

        var chunkEntities: [EntityPrediction] = []
        for chunk in chunks {
            let chunkText = Self.substring(text, start: chunk.start, end: chunk.end)
            let entities = try extractPII(
                chunkText,
                confidenceThreshold: confidenceThreshold,
                useSmartMerging: useSmartMerging
            )
            chunkEntities.append(
                contentsOf: entities.compactMap { entity in
                    Self.offset(entity, by: chunk.start, in: text)
                })
        }

        return mergeChunkedPIIEntities(
            chunkEntities,
            text: text,
            useSmartMerging: useSmartMerging
        )
    }

    // MARK: - Private

    struct TextChunk: Equatable {
        let start: Int
        let end: Int
        let tokenStart: Int
        let tokenEnd: Int
    }

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

    func makeTokenChunks(
        for text: String,
        chunkTokenLimit: Int,
        tokenOverlap: Int
    ) throws -> [TextChunk] {
        guard !text.isEmpty else {
            return []
        }

        let tokenOffsets = try tokenOffsets(in: text)
            .filter { $0.0 < $0.1 }

        let tokenLimit = max(1, chunkTokenLimit)
        guard tokenOffsets.count > tokenLimit else {
            return [
                TextChunk(
                    start: 0,
                    end: text.count,
                    tokenStart: 0,
                    tokenEnd: tokenOffsets.count
                )
            ]
        }

        let overlap = min(max(0, tokenOverlap), tokenLimit - 1)
        var chunks: [TextChunk] = []
        var tokenStart = 0

        while tokenStart < tokenOffsets.count {
            let tokenEnd = min(tokenStart + tokenLimit, tokenOffsets.count)
            chunks.append(
                TextChunk(
                    start: tokenOffsets[tokenStart].0,
                    end: tokenOffsets[tokenEnd - 1].1,
                    tokenStart: tokenStart,
                    tokenEnd: tokenEnd
                )
            )

            guard tokenEnd < tokenOffsets.count else {
                break
            }
            tokenStart = max(tokenStart + 1, tokenEnd - overlap)
        }

        return chunks
    }

    private func tokenOffsets(in text: String) throws -> [(Int, Int)] {
        switch runtime {
        case .coreML, .mlx:
            guard let tokenizer else {
                throw TokenizerError.missingConfig
            }
            let inputIDs = tokenizer(text, addSpecialTokens: false)
            let tokens = tokenizer.convertIdsToTokens(inputIDs).map { $0 ?? "" }
            return Self.buildOffsets(tokens: tokens, in: text)
        case .privacyFilter(let pipeline):
            return try pipeline.tokenOffsets(in: text)
        }
    }

    func mergeChunkedPIIEntities(
        _ entities: [EntityPrediction],
        text: String,
        useSmartMerging: Bool
    ) -> [EntityPrediction] {
        let repaired = PostProcessing.repairEntitySpans(
            Self.deduplicateOverlappingEntities(entities),
            text: text
        )

        guard useSmartMerging else {
            return Self.deduplicateOverlappingEntities(repaired)
        }

        let merged: [EntityPrediction]
        switch runtime {
        case .privacyFilter:
            merged = PostProcessing.mergePIIEntities(
                repaired,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true,
                allowSemanticOnlyMatches: false,
                allowSemanticLabelExpansion: false
            )
        case .coreML, .mlx:
            merged = PostProcessing.mergePIIEntities(
                repaired,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true
            )
        }

        return Self.deduplicateOverlappingEntities(merged)
    }

    static func deidentify(
        _ text: String,
        entities: [EntityPrediction],
        policy: Policy
    ) -> PolicyDeidentificationResult {
        let candidates = deidentificationCandidates(entities, in: text)
        var redactedText = text
        var actionRecords: [DeidentifiedSpanAction] = []

        for entity in candidates {
            let canonicalLabel = Policy.canonicalLabel(for: entity.label)
            let action = policy.action(for: entity.label)
            let original = substring(text, start: entity.start, end: entity.end)
            let replacement = replacementText(
                for: action,
                canonicalLabel: canonicalLabel,
                original: original
            )

            actionRecords.append(
                DeidentifiedSpanAction(
                    label: entity.label,
                    canonicalLabel: canonicalLabel,
                    action: action,
                    start: entity.start,
                    end: entity.end,
                    confidence: entity.confidence,
                    replacement: replacement
                )
            )
        }

        for record in actionRecords.reversed() {
            guard let replacement = record.replacement else {
                continue
            }
            let lowerBound = redactedText.index(
                redactedText.startIndex,
                offsetBy: record.start
            )
            let upperBound = redactedText.index(
                lowerBound,
                offsetBy: record.end - record.start
            )
            redactedText.replaceSubrange(lowerBound..<upperBound, with: replacement)
        }

        return PolicyDeidentificationResult(
            redactedText: redactedText,
            policyName: policy.name,
            actions: actionRecords
        )
    }

    private static func deidentificationCandidates(
        _ entities: [EntityPrediction],
        in text: String
    ) -> [EntityPrediction] {
        let textLength = text.count
        let sorted =
            entities
            .filter { $0.start >= 0 && $0.end > $0.start && $0.end <= textLength }
            .sorted {
                if $0.start == $1.start {
                    let lhsLength = entityLength($0)
                    let rhsLength = entityLength($1)
                    if lhsLength == rhsLength {
                        return $0.confidence > $1.confidence
                    }
                    return lhsLength > rhsLength
                }
                return $0.start < $1.start
            }

        var selected: [EntityPrediction] = []
        for entity in sorted {
            let overlapsSelected = selected.contains { existing in
                entity.start < existing.end && entity.end > existing.start
            }
            if !overlapsSelected {
                selected.append(entity)
            }
        }

        return selected.sorted {
            if $0.start == $1.start {
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }
    }

    private static func replacementText(
        for action: PolicyAction,
        canonicalLabel: String,
        original: String
    ) -> String? {
        switch action.redactionEquivalent {
        case .keep:
            return nil
        case .mask:
            return "[\(canonicalLabel)]"
        case .replace:
            return "[\(canonicalLabel)_REPLACED]"
        case .remove:
            return ""
        case .hash:
            return "\(canonicalLabel)_\(stableHash(original))"
        case .redact:
            return "[\(canonicalLabel)]"
        }
    }

    private static func stableHash(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.prefix(12).map { String(format: "%02x", $0) }.joined()
    }

    static func deduplicateOverlappingEntities(
        _ entities: [EntityPrediction]
    ) -> [EntityPrediction] {
        var selected: [EntityPrediction] = []

        for entity in entities.sorted(by: entitySort) {
            guard
                let existingIndex = selected.firstIndex(where: {
                    areDuplicateCandidates(entity, $0)
                })
            else {
                selected.append(entity)
                continue
            }

            if isBetterDuplicate(candidate: entity, existing: selected[existingIndex]) {
                selected[existingIndex] = entity
            }
        }

        return selected.sorted {
            if $0.start == $1.start {
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }
    }

    private static func offset(
        _ entity: EntityPrediction,
        by baseOffset: Int,
        in text: String
    ) -> EntityPrediction? {
        let start = entity.start + baseOffset
        let end = entity.end + baseOffset
        guard start >= 0, end > start, end <= text.count else {
            return nil
        }
        return EntityPrediction(
            label: entity.label,
            text: substring(text, start: start, end: end),
            confidence: entity.confidence,
            start: start,
            end: end
        )
    }

    private static func entitySort(
        lhs: EntityPrediction,
        rhs: EntityPrediction
    ) -> Bool {
        if lhs.start == rhs.start {
            if lhs.end == rhs.end {
                return lhs.confidence > rhs.confidence
            }
            return entityLength(lhs) > entityLength(rhs)
        }
        return lhs.start < rhs.start
    }

    private static func areDuplicateCandidates(
        _ lhs: EntityPrediction,
        _ rhs: EntityPrediction
    ) -> Bool {
        guard labelsAreCompatible(lhs.label, rhs.label) else {
            return false
        }
        let overlap = min(lhs.end, rhs.end) - max(lhs.start, rhs.start)
        guard overlap > 0 else {
            return false
        }
        let shorterLength = max(1, min(entityLength(lhs), entityLength(rhs)))
        return Double(overlap) / Double(shorterLength) >= 0.5
    }

    private static func labelsAreCompatible(_ lhs: String, _ rhs: String) -> Bool {
        if lhs == rhs {
            return true
        }

        let normalizedLHS = PostProcessing.normalizeLabel(lhs)
        let normalizedRHS = PostProcessing.normalizeLabel(rhs)
        if normalizedLHS == normalizedRHS {
            return true
        }

        let lowerLHS = lhs.lowercased()
        let lowerRHS = rhs.lowercased()
        let nameTokens = ["name", "person"]
        return nameTokens.contains { token in
            lowerLHS.contains(token) && lowerRHS.contains(token)
        }
    }

    private static func isBetterDuplicate(
        candidate: EntityPrediction,
        existing: EntityPrediction
    ) -> Bool {
        let candidateLength = entityLength(candidate)
        let existingLength = entityLength(existing)

        if candidate.start == existing.start && candidate.end == existing.end {
            return candidate.confidence > existing.confidence
        }

        if candidateLength > existingLength && candidate.confidence >= existing.confidence - 0.10 {
            return true
        }

        return candidate.confidence > existing.confidence + 0.05
    }

    private static func entityLength(_ entity: EntityPrediction) -> Int {
        max(0, entity.end - entity.start)
    }

    private static func substring(_ text: String, start: Int, end: Int) -> String {
        guard start >= 0, end >= start, end <= text.count else {
            return ""
        }
        let lowerBound = text.index(text.startIndex, offsetBy: start)
        let upperBound = text.index(lowerBound, offsetBy: end - start)
        return String(text[lowerBound..<upperBound])
    }

    private static func deidentifiedText(
        _ text: String,
        entities: [EntityPrediction],
        method: DeidentificationMethod
    ) -> String {
        var redacted = text
        for entity in entities.sorted(by: { $0.start > $1.start }) {
            guard let range = characterRange(in: redacted, start: entity.start, end: entity.end) else {
                continue
            }
            redacted.replaceSubrange(
                range,
                with: replacementText(for: entity, method: method)
            )
        }
        return redacted
    }

    private static func replacementText(
        for entity: EntityPrediction,
        method: DeidentificationMethod
    ) -> String {
        switch method {
        case .mask:
            return "[\(entity.entityType.uppercased())]"
        case .remove:
            return ""
        }
    }

    private static func characterRange(
        in text: String,
        start: Int,
        end: Int
    ) -> Range<String.Index>? {
        guard start >= 0, end >= start, end <= text.count else {
            return nil
        }
        let lowerBound = text.index(text.startIndex, offsetBy: start)
        let upperBound = text.index(lowerBound, offsetBy: end - start)
        return lowerBound..<upperBound
    }

    static func loadTokenizer(
        tokenizerName: String,
        tokenizerFolderURL: URL?,
        allowNetworkAccess: Bool = true
    ) throws -> any Tokenizer {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<any Tokenizer, Error>?

        Task.detached {
            do {
                let tokenizer = try await loadTokenizerAsync(
                    tokenizerName: tokenizerName,
                    tokenizerFolderURL: tokenizerFolderURL,
                    allowNetworkAccess: allowNetworkAccess
                )
                result = .success(tokenizer)
            } catch {
                result = .failure(error)
            }
            semaphore.signal()
        }

        semaphore.wait()
        return try result!.get()
    }

    private static func loadTokenizerAsync(
        tokenizerName: String,
        tokenizerFolderURL: URL?,
        allowNetworkAccess: Bool
    ) async throws -> any Tokenizer {
        if let tokenizerFolderURL {
            return try loadTokenizerFromDirectory(
                tokenizerFolderURL,
                fallbackTokenizerName: allowNetworkAccess ? tokenizerName : nil
            )
        }

        guard allowNetworkAccess else {
            throw TokenizerError.missingConfig
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
            FileManager.default.fileExists(atPath: tokenizerConfigURL.path)
        else {
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
        return
            base
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
        return try result!.get()
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
        return
            base
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

        let encodedModelID =
            modelID
            .split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let encodedPath =
            relativePath
            .split(separator: "/")
            .map { String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")

        guard
            let url = URL(
                string: "https://huggingface.co/\(encodedModelID)/resolve/main/\(encodedPath)?download=1"
            )
        else {
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
        return try result!.get()
    }

    private static func sanitizedCacheComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "__")
            .replacingOccurrences(of: ":", with: "_")
            .replacingOccurrences(of: " ", with: "_")
    }

    private static func stableDigest(for value: String) -> String {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in value.utf8 {
            hash ^= UInt64(byte)
            hash &*= 0x100_0000_01b3
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
            case (let exact?, let insensitive?):
                if exact.lowerBound <= insensitive.lowerBound {
                    range = exact
                } else {
                    range = insensitive
                }
            case (let exact?, nil):
                range = exact
            case (nil, let insensitive?):
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
            let endIndex =
                text.index(
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
