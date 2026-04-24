import Foundation
import MLX

enum OpenMedPrivacyFilterError: LocalizedError {
    case unsupportedArtifact(String)
    case missingTokenizer(URL)
    case invalidTokenizer(URL)
    case unknownToken(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedArtifact(let family):
            return "Expected OpenAI Privacy Filter MLX artifact, got \(family)."
        case .missingTokenizer(let url):
            return "Privacy Filter tokenizer.json was not found at \(url.path)."
        case .invalidTokenizer(let url):
            return "Privacy Filter tokenizer.json is not a supported byte-level BPE tokenizer: \(url.path)."
        case .unknownToken(let token):
            return "Privacy Filter tokenizer could not map token piece: \(token)."
        }
    }
}

struct OpenMedPrivacyFilterEncodedText {
    let tokenIDs: [Int]
    let charStarts: [Int]
    let charEnds: [Int]
    let decodedText: String
}

final class OpenMedPrivacyFilterTokenizer {
    private struct TokenizerJSON: Decodable {
        struct Model: Decodable {
            let type: String
            let vocab: [String: Int]
            let merges: [MergeEntry]
        }

        struct PreTokenizer: Decodable {
            struct SplitPattern: Decodable {
                let regex: String?

                enum CodingKeys: String, CodingKey {
                    case regex = "Regex"
                }
            }

            struct Entry: Decodable {
                let type: String
                let pattern: SplitPattern?
                let pretokenizers: [Entry]?
            }

            let type: String
            let pattern: SplitPattern?
            let pretokenizers: [Entry]?
        }

        let model: Model
        let preTokenizer: PreTokenizer?

        enum CodingKeys: String, CodingKey {
            case model
            case preTokenizer = "pre_tokenizer"
        }
    }

    private enum MergeEntry: Decodable {
        case pair(String, String)

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let parts = try? container.decode([String].self), parts.count == 2 {
                self = .pair(parts[0], parts[1])
                return
            }
            let text = try container.decode(String.self)
            let parts = text.split(separator: " ", maxSplits: 1).map(String.init)
            guard parts.count == 2 else {
                throw DecodingError.dataCorruptedError(
                    in: container,
                    debugDescription: "Expected BPE merge pair"
                )
            }
            self = .pair(parts[0], parts[1])
        }
    }

    private struct PairKey: Hashable {
        let left: String
        let right: String
    }

    private let vocab: [String: Int]
    private let idToToken: [Int: String]
    private let mergeRanks: [PairKey: Int]
    private let regex: NSRegularExpression?
    private let byteEncoder: [UInt8: String]
    private let byteDecoder: [UnicodeScalar: UInt8]
    private var bpeCache = [String: [String]]()

    init(directoryURL: URL) throws {
        let tokenizerURL = directoryURL.appending(path: "tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw OpenMedPrivacyFilterError.missingTokenizer(tokenizerURL)
        }

        let data = try Data(contentsOf: tokenizerURL)
        let tokenizer = try JSONDecoder().decode(TokenizerJSON.self, from: data)
        guard tokenizer.model.type.lowercased() == "bpe" else {
            throw OpenMedPrivacyFilterError.invalidTokenizer(tokenizerURL)
        }

        self.vocab = tokenizer.model.vocab
        self.idToToken = Dictionary(uniqueKeysWithValues: tokenizer.model.vocab.map { ($0.value, $0.key) })
        self.mergeRanks = Dictionary(
            uniqueKeysWithValues: tokenizer.model.merges.enumerated().map { rank, entry in
                switch entry {
                case .pair(let left, let right):
                    return (PairKey(left: left, right: right), rank)
                }
            }
        )

        let byteMaps = Self.makeByteMaps()
        self.byteEncoder = byteMaps.encoder
        self.byteDecoder = byteMaps.decoder

        if let pattern = Self.findRegexPattern(in: tokenizer.preTokenizer) {
            self.regex = try? NSRegularExpression(pattern: pattern)
        } else {
            self.regex = nil
        }
    }

    func encode(_ text: String, maxTokens: Int) throws -> OpenMedPrivacyFilterEncodedText {
        var tokenIDs = [Int]()
        tokenIDs.reserveCapacity(min(maxTokens, max(8, text.count / 3)))

        for piece in preTokenize(text) {
            let byteLevelPiece = piece.utf8.map { byteEncoder[$0, default: ""] }.joined()
            for token in bytePairEncode(byteLevelPiece) {
                guard let id = vocab[token] else {
                    throw OpenMedPrivacyFilterError.unknownToken(token)
                }
                tokenIDs.append(id)
                if tokenIDs.count >= maxTokens {
                    break
                }
            }
            if tokenIDs.count >= maxTokens {
                break
            }
        }

        let offsets = decodeOffsets(tokenIDs)
        return OpenMedPrivacyFilterEncodedText(
            tokenIDs: tokenIDs,
            charStarts: offsets.charStarts,
            charEnds: offsets.charEnds,
            decodedText: offsets.text
        )
    }

    func tokenBytes(tokenID: Int) -> [UInt8] {
        guard let token = idToToken[tokenID] else {
            return []
        }
        var bytes = [UInt8]()
        for scalar in token.unicodeScalars {
            if let byte = byteDecoder[scalar] {
                bytes.append(byte)
            } else {
                bytes.append(contentsOf: String(scalar).utf8)
            }
        }
        return bytes
    }

    private static func makeByteMaps() -> (
        encoder: [UInt8: String],
        decoder: [UnicodeScalar: UInt8]
    ) {
        var bytes = Array(33...126) + Array(161...172) + Array(174...255)
        var scalars = bytes
        var byteSet = Set(bytes)
        var next = 0
        for byte in 0..<256 where !byteSet.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + next)
            byteSet.insert(byte)
            next += 1
        }

        var encoder = [UInt8: String]()
        var decoder = [UnicodeScalar: UInt8]()
        for (byte, scalarValue) in zip(bytes, scalars) {
            guard let scalar = UnicodeScalar(scalarValue) else {
                continue
            }
            encoder[UInt8(byte)] = String(scalar)
            decoder[scalar] = UInt8(byte)
        }
        return (encoder, decoder)
    }

    private static func findRegexPattern(in preTokenizer: TokenizerJSON.PreTokenizer?) -> String? {
        guard let preTokenizer else {
            return nil
        }
        if preTokenizer.type == "Split", let regex = preTokenizer.pattern?.regex {
            return regex
        }
        for entry in preTokenizer.pretokenizers ?? [] {
            if entry.type == "Split", let regex = entry.pattern?.regex {
                return regex
            }
        }
        return nil
    }

    private func preTokenize(_ text: String) -> [String] {
        guard let regex else {
            return text.isEmpty ? [] : [text]
        }

        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let matches = regex.matches(in: text, options: [], range: fullRange)
        guard !matches.isEmpty else {
            return text.isEmpty ? [] : [text]
        }

        var pieces = [String]()
        var cursor = text.startIndex
        for match in matches {
            guard let range = Range(match.range, in: text) else {
                continue
            }
            if cursor < range.lowerBound {
                pieces.append(String(text[cursor..<range.lowerBound]))
            }
            pieces.append(String(text[range]))
            cursor = range.upperBound
        }
        if cursor < text.endIndex {
            pieces.append(String(text[cursor..<text.endIndex]))
        }
        return pieces.filter { !$0.isEmpty }
    }

    private func bytePairEncode(_ token: String) -> [String] {
        if let cached = bpeCache[token] {
            return cached
        }
        var word = token.map(String.init)
        guard word.count > 1 else {
            bpeCache[token] = word
            return word
        }

        while word.count > 1 {
            var bestRank = Int.max
            var bestPair: PairKey?
            for index in 0..<(word.count - 1) {
                let pair = PairKey(left: word[index], right: word[index + 1])
                if let rank = mergeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }
            guard let bestPair else {
                break
            }

            var merged = [String]()
            var index = 0
            while index < word.count {
                if index < word.count - 1,
                   word[index] == bestPair.left,
                   word[index + 1] == bestPair.right
                {
                    merged.append(bestPair.left + bestPair.right)
                    index += 2
                } else {
                    merged.append(word[index])
                    index += 1
                }
            }
            word = merged
        }

        bpeCache[token] = word
        return word
    }

    private func decodeOffsets(_ tokenIDs: [Int]) -> (
        text: String,
        charStarts: [Int],
        charEnds: [Int]
    ) {
        let decodedTokenBytes = tokenIDs.map { self.tokenBytes(tokenID: $0) }
        let allBytes = decodedTokenBytes.flatMap { $0 }
        let decodedText = String(decoding: allBytes, as: UTF8.self)

        var charByteStarts = [Int]()
        var charByteEnds = [Int]()
        var byteCursor = 0
        for character in decodedText {
            charByteStarts.append(byteCursor)
            byteCursor += String(character).utf8.count
            charByteEnds.append(byteCursor)
        }

        var charStarts = [Int]()
        var charEnds = [Int]()
        var tokenByteCursor = 0
        for rawBytes in decodedTokenBytes {
            let tokenByteStart = tokenByteCursor
            let tokenByteEnd = tokenByteStart + rawBytes.count
            tokenByteCursor = tokenByteEnd
            let start = Self.bisectRight(charByteEnds, value: tokenByteStart)
            var end = Self.bisectLeft(charByteStarts, value: tokenByteEnd)
            if end < start {
                end = start
            }
            charStarts.append(start)
            charEnds.append(end)
        }

        return (decodedText, charStarts, charEnds)
    }

    private static func bisectLeft(_ values: [Int], value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] < value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }

    private static func bisectRight(_ values: [Int], value: Int) -> Int {
        var low = 0
        var high = values.count
        while low < high {
            let mid = (low + high) / 2
            if values[mid] <= value {
                low = mid + 1
            } else {
                high = mid
            }
        }
        return low
    }
}

struct OpenMedPrivacyFilterLabelInfo {
    let spanClassNames: [String]
    let tokenToSpanLabel: [Int: Int]
    let tokenBoundaryTags: [Int: String]
    let backgroundTokenLabel: Int
    let backgroundSpanLabel: Int

    init(id2label: [Int: String]) {
        var spanClassNames = ["O"]
        var spanLabelLookup = ["O": 0]
        var tokenToSpanLabel = [Int: Int]()
        var tokenBoundaryTags = [Int: String]()
        var backgroundTokenLabel = 0

        for index in id2label.keys.sorted() {
            let label = id2label[index] ?? "O"
            if label == "O" {
                backgroundTokenLabel = index
                tokenToSpanLabel[index] = 0
                continue
            }

            let split = Self.splitBoundaryLabel(label)
            let spanLabel: Int
            if let existing = spanLabelLookup[split.baseLabel] {
                spanLabel = existing
            } else {
                spanLabel = spanClassNames.count
                spanClassNames.append(split.baseLabel)
                spanLabelLookup[split.baseLabel] = spanLabel
            }
            tokenToSpanLabel[index] = spanLabel
            tokenBoundaryTags[index] = split.boundary
        }

        self.spanClassNames = spanClassNames
        self.tokenToSpanLabel = tokenToSpanLabel
        self.tokenBoundaryTags = tokenBoundaryTags
        self.backgroundTokenLabel = backgroundTokenLabel
        self.backgroundSpanLabel = 0
    }

    private static func splitBoundaryLabel(_ label: String) -> (boundary: String, baseLabel: String) {
        guard label.count > 2 else {
            return ("B", label)
        }
        let boundary = String(label.prefix(1))
        let separatorIndex = label.index(label.startIndex, offsetBy: 1)
        let baseIndex = label.index(label.startIndex, offsetBy: 2)
        if label[separatorIndex] == "-", ["B", "I", "E", "S"].contains(boundary) {
            return (boundary, String(label[baseIndex...]))
        }
        return ("B", label)
    }
}

enum OpenMedPrivacyFilterViterbi {
    private static let negativeInfinity: Float = -1.0e9
    private static let biasKeys = [
        "transition_bias_background_stay",
        "transition_bias_background_to_start",
        "transition_bias_inside_to_continue",
        "transition_bias_inside_to_end",
        "transition_bias_end_to_background",
        "transition_bias_end_to_start",
    ]

    static func decode(
        tokenLogProbabilities: [[Float]],
        labelInfo: OpenMedPrivacyFilterLabelInfo,
        biases: [String: Float]
    ) -> [Int] {
        guard !tokenLogProbabilities.isEmpty else {
            return []
        }

        var resolvedBiases = Dictionary(uniqueKeysWithValues: biasKeys.map { ($0, Float(0.0)) })
        for (key, value) in biases where resolvedBiases[key] != nil {
            resolvedBiases[key] = value
        }

        let scores = buildScores(labelInfo: labelInfo, biases: resolvedBiases)
        let numClasses = labelInfo.tokenToSpanLabel.count
        var currentScores = (0..<numClasses).map {
            tokenLogProbabilities[0][$0] + scores.start[$0]
        }
        var backpointers = [[Int]]()

        for tokenScores in tokenLogProbabilities.dropFirst() {
            var nextScores = [Float]()
            var paths = [Int]()
            nextScores.reserveCapacity(numClasses)
            paths.reserveCapacity(numClasses)

            for nextIndex in 0..<numClasses {
                var bestIndex = 0
                var bestScore = -Float.infinity
                for previousIndex in 0..<numClasses {
                    let score = currentScores[previousIndex]
                        + scores.transition[previousIndex][nextIndex]
                    if score > bestScore {
                        bestScore = score
                        bestIndex = previousIndex
                    }
                }
                nextScores.append(bestScore + tokenScores[nextIndex])
                paths.append(bestIndex)
            }
            currentScores = nextScores
            backpointers.append(paths)
        }

        let finalScores = currentScores.enumerated().map { index, score in
            score + scores.end[index]
        }
        guard finalScores.contains(where: { $0.isFinite }) else {
            return tokenLogProbabilities.map { row in
                row.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
            }
        }

        var label = finalScores.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        var path = [label]
        for paths in backpointers.reversed() {
            label = paths[label]
            path.append(label)
        }
        return Array(path.reversed())
    }

    private static func buildScores(
        labelInfo: OpenMedPrivacyFilterLabelInfo,
        biases: [String: Float]
    ) -> (
        start: [Float],
        end: [Float],
        transition: [[Float]]
    ) {
        let numClasses = labelInfo.tokenToSpanLabel.count
        var startScores = Array(repeating: negativeInfinity, count: numClasses)
        var endScores = Array(repeating: negativeInfinity, count: numClasses)
        var transitionScores = Array(
            repeating: Array(repeating: negativeInfinity, count: numClasses),
            count: numClasses
        )

        for previousIndex in 0..<numClasses {
            let previousTag = labelInfo.tokenBoundaryTags[previousIndex]
            let previousSpan = labelInfo.tokenToSpanLabel[previousIndex]
            if previousTag == "B" || previousTag == "S" || previousIndex == labelInfo.backgroundTokenLabel {
                startScores[previousIndex] = 0.0
            }
            if previousTag == "E" || previousTag == "S" || previousIndex == labelInfo.backgroundTokenLabel {
                endScores[previousIndex] = 0.0
            }

            for nextIndex in 0..<numClasses {
                let nextTag = labelInfo.tokenBoundaryTags[nextIndex]
                let nextSpan = labelInfo.tokenToSpanLabel[nextIndex]
                if isValidTransition(
                    previousTag: previousTag,
                    previousSpan: previousSpan,
                    nextTag: nextTag,
                    nextSpan: nextSpan,
                    labelInfo: labelInfo,
                    nextIndex: nextIndex
                ) {
                    transitionScores[previousIndex][nextIndex] = transitionBias(
                        previousTag: previousTag,
                        previousSpan: previousSpan,
                        nextTag: nextTag,
                        nextSpan: nextSpan,
                        labelInfo: labelInfo,
                        previousIndex: previousIndex,
                        nextIndex: nextIndex,
                        biases: biases
                    )
                }
            }
        }

        return (startScores, endScores, transitionScores)
    }

    private static func isValidTransition(
        previousTag: String?,
        previousSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        labelInfo: OpenMedPrivacyFilterLabelInfo,
        nextIndex: Int
    ) -> Bool {
        let nextIsBackground =
            nextSpan == labelInfo.backgroundSpanLabel || nextIndex == labelInfo.backgroundTokenLabel
        if (nextSpan == nil || nextTag == nil) && !nextIsBackground {
            return false
        }

        guard let previousSpan, let previousTag else {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }

        if previousSpan == labelInfo.backgroundSpanLabel {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if previousTag == "E" || previousTag == "S" {
            return nextIsBackground || nextTag == "B" || nextTag == "S"
        }
        if previousTag == "B" || previousTag == "I" {
            return previousSpan == nextSpan && (nextTag == "I" || nextTag == "E")
        }
        return false
    }

    private static func transitionBias(
        previousTag: String?,
        previousSpan: Int?,
        nextTag: String?,
        nextSpan: Int?,
        labelInfo: OpenMedPrivacyFilterLabelInfo,
        previousIndex: Int,
        nextIndex: Int,
        biases: [String: Float]
    ) -> Float {
        let previousIsBackground =
            previousSpan == labelInfo.backgroundSpanLabel || previousIndex == labelInfo.backgroundTokenLabel
        let nextIsBackground =
            nextSpan == labelInfo.backgroundSpanLabel || nextIndex == labelInfo.backgroundTokenLabel

        if previousIsBackground {
            if nextIsBackground {
                return biases["transition_bias_background_stay"] ?? 0.0
            }
            if nextTag == "B" || nextTag == "S" {
                return biases["transition_bias_background_to_start"] ?? 0.0
            }
            return 0.0
        }

        if previousTag == "B" || previousTag == "I" {
            if nextTag == "I" && previousSpan == nextSpan {
                return biases["transition_bias_inside_to_continue"] ?? 0.0
            }
            if nextTag == "E" && previousSpan == nextSpan {
                return biases["transition_bias_inside_to_end"] ?? 0.0
            }
            return 0.0
        }

        if previousTag == "E" || previousTag == "S" {
            if nextIsBackground {
                return biases["transition_bias_end_to_background"] ?? 0.0
            }
            if nextTag == "B" || nextTag == "S" {
                return biases["transition_bias_end_to_start"] ?? 0.0
            }
        }
        return 0.0
    }
}

final class OpenMedPrivacyFilterPipeline {
    private let artifact: OpenMedMLXArtifact
    private let model: OpenMedPrivacyFilterForTokenClassification
    private let tokenizer: OpenMedPrivacyFilterTokenizer
    private let labelInfo: OpenMedPrivacyFilterLabelInfo
    private let maxSeqLength: Int

    convenience init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
        try self.init(artifact: artifact, maxSeqLength: maxSeqLength)
    }

    init(artifact: OpenMedMLXArtifact, maxSeqLength: Int = 512) throws {
        guard MLXTokenClassificationPipeline.isRuntimeSupported else {
            throw OpenMedMLXRuntimeError.unsupportedPlatform
        }
        guard artifact.task == .tokenClassification,
              artifact.family == .openaiPrivacyFilter
        else {
            throw OpenMedPrivacyFilterError.unsupportedArtifact(artifact.manifest.family)
        }

        self.artifact = artifact
        self.model = try OpenMedMLXModelLoader.loadPrivacyFilter(from: artifact)
        self.tokenizer = try OpenMedPrivacyFilterTokenizer(
            directoryURL: artifact.tokenizerDirectoryURL ?? artifact.directoryURL
        )
        self.labelInfo = OpenMedPrivacyFilterLabelInfo(id2label: artifact.id2label)
        self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
    }

    var resolvedMaxSequenceLength: Int {
        maxSeqLength
    }

    func predict(
        _ text: String,
        confidenceThreshold: Float = 0.0
    ) throws -> [EntityPrediction] {
        let encoded = try tokenizer.encode(text, maxTokens: maxSeqLength)
        guard !encoded.tokenIDs.isEmpty else {
            return []
        }

        let sequenceLength = encoded.tokenIDs.count
        let inputIDs = MLXArray(encoded.tokenIDs.map(Int32.init), [1, sequenceLength])
        let attentionMask = MLXArray.ones([1, sequenceLength], type: Bool.self)
        let logits = model(inputIDs, attentionMask: attentionMask).asType(.float32)
        let logProbabilities = (logits - logSumExp(logits, axis: -1, keepDims: true))[0]
        let probabilities = exp(logProbabilities)
        eval(logProbabilities, probabilities)

        let flatLogProbabilities = logProbabilities.asArray(Float.self)
        let flatProbabilities = probabilities.asArray(Float.self)
        let numLabels = artifact.configuration.numLabels
        let tokenLogProbabilities = stride(from: 0, to: flatLogProbabilities.count, by: numLabels)
            .map { offset in Array(flatLogProbabilities[offset..<(offset + numLabels)]) }
        let tokenProbabilities = stride(from: 0, to: flatProbabilities.count, by: numLabels)
            .map { offset in Array(flatProbabilities[offset..<(offset + numLabels)]) }

        let predictedIDs = OpenMedPrivacyFilterViterbi.decode(
            tokenLogProbabilities: tokenLogProbabilities,
            labelInfo: labelInfo,
            biases: artifact.configuration.viterbiBiases
        )

        let sourceText = encoded.decodedText == text ? text : encoded.decodedText
        return decodeGroupedEntities(
            predictedIDs: predictedIDs,
            probabilities: tokenProbabilities,
            charStarts: encoded.charStarts,
            charEnds: encoded.charEnds,
            text: sourceText,
            confidenceThreshold: confidenceThreshold
        )
    }

    private func decodeGroupedEntities(
        predictedIDs: [Int],
        probabilities: [[Float]],
        charStarts: [Int],
        charEnds: [Int],
        text: String,
        confidenceThreshold: Float
    ) -> [EntityPrediction] {
        let spans = labelsToTokenSpans(predictedIDs)
        var entities = [EntityPrediction]()
        for span in spans {
            guard span.tokenStart >= 0,
                  span.tokenStart < span.tokenEnd,
                  span.tokenEnd <= charStarts.count
            else {
                continue
            }

            var start = charStarts[span.tokenStart]
            var end = charEnds[span.tokenEnd - 1]
            trimWhitespace(start: &start, end: &end, text: text)
            guard end > start else {
                continue
            }

            let scores = (span.tokenStart..<span.tokenEnd).compactMap { index -> Float? in
                guard index < probabilities.count, predictedIDs[index] < probabilities[index].count else {
                    return nil
                }
                return probabilities[index][predictedIDs[index]]
            }
            let confidence = scores.isEmpty ? Float(0.0) : scores.reduce(0.0, +) / Float(scores.count)
            guard confidence >= confidenceThreshold else {
                continue
            }

            let label: String
            if span.label >= 0 && span.label < labelInfo.spanClassNames.count {
                label = labelInfo.spanClassNames[span.label]
            } else {
                label = "label_\(span.label)"
            }
            refineStructuredPIISpan(label: label, start: &start, end: &end, text: text)
            guard end > start else {
                continue
            }
            entities.append(
                EntityPrediction(
                    label: label,
                    text: substring(text, start: start, end: end),
                    confidence: confidence,
                    start: start,
                    end: end
                )
            )
        }
        return entities
    }

    private func refineStructuredPIISpan(
        label: String,
        start: inout Int,
        end: inout Int,
        text: String
    ) {
        let normalizedLabel = label.lowercased()
        let span = substring(text, start: start, end: end)
        let patterns: [(hint: String, pattern: String)] = [
            ("email", #"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"#),
            ("url", #"\b(?:https?://|www\.)[^\s,;)\]]+"#),
            ("phone", #"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}"#),
        ]

        for candidate in patterns where normalizedLabel.contains(candidate.hint) {
            guard let match = span.range(
                of: candidate.pattern,
                options: [.regularExpression, .caseInsensitive]
            ) else {
                continue
            }
            let lowerOffset = span.distance(from: span.startIndex, to: match.lowerBound)
            let upperOffset = span.distance(from: span.startIndex, to: match.upperBound)
            start += lowerOffset
            end = start + (upperOffset - lowerOffset)
            return
        }

        let lowercasedSpan = span.lowercased()
        for suffix in [" and", " or"] where lowercasedSpan.hasSuffix(suffix) {
            end -= suffix.count
            trimWhitespace(start: &start, end: &end, text: text)
            return
        }
    }

    private func labelsToTokenSpans(_ predictedIDs: [Int]) -> [(
        label: Int,
        tokenStart: Int,
        tokenEnd: Int
    )] {
        var spans = [(label: Int, tokenStart: Int, tokenEnd: Int)]()
        var currentLabel: Int?
        var startIndex: Int?
        var previousIndex: Int?

        for (tokenIndex, labelID) in predictedIDs.enumerated() {
            let spanLabel = labelInfo.tokenToSpanLabel[labelID]
            let boundaryTag = labelInfo.tokenBoundaryTags[labelID]

            if let previousIndex, tokenIndex != previousIndex + 1 {
                if let currentLabel, let startIndex {
                    spans.append((currentLabel, startIndex, previousIndex + 1))
                }
                currentLabel = nil
                startIndex = nil
            }

            if spanLabel == nil {
                previousIndex = tokenIndex
                continue
            }

            if spanLabel == labelInfo.backgroundSpanLabel {
                if let currentLabel, let startIndex {
                    spans.append((currentLabel, startIndex, tokenIndex))
                }
                currentLabel = nil
                startIndex = nil
                previousIndex = tokenIndex
                continue
            }

            switch boundaryTag {
            case "S":
                if let currentLabel, let startIndex, let previousIndex {
                    spans.append((currentLabel, startIndex, previousIndex + 1))
                }
                spans.append((spanLabel ?? 0, tokenIndex, tokenIndex + 1))
                currentLabel = nil
                startIndex = nil
            case "B":
                if let currentLabel, let startIndex, let previousIndex {
                    spans.append((currentLabel, startIndex, previousIndex + 1))
                }
                currentLabel = spanLabel
                startIndex = tokenIndex
            case "I":
                if currentLabel == nil || currentLabel != spanLabel {
                    if let currentLabel, let startIndex, let previousIndex {
                        spans.append((currentLabel, startIndex, previousIndex + 1))
                    }
                    currentLabel = spanLabel
                    startIndex = tokenIndex
                }
            case "E":
                if currentLabel == nil || currentLabel != spanLabel || startIndex == nil {
                    if let currentLabel, let startIndex, let previousIndex {
                        spans.append((currentLabel, startIndex, previousIndex + 1))
                    }
                    spans.append((spanLabel ?? 0, tokenIndex, tokenIndex + 1))
                    currentLabel = nil
                    startIndex = nil
                } else if let resolvedLabel = currentLabel, let resolvedStart = startIndex {
                    spans.append((resolvedLabel, resolvedStart, tokenIndex + 1))
                    currentLabel = nil
                    startIndex = nil
                }
            default:
                break
            }

            previousIndex = tokenIndex
        }

        if let currentLabel, let startIndex, let previousIndex {
            spans.append((currentLabel, startIndex, previousIndex + 1))
        }
        return spans
    }

    private func trimWhitespace(start: inout Int, end: inout Int, text: String) {
        while start < end, character(at: start, in: text)?.isWhitespace == true {
            start += 1
        }
        while end > start, character(at: end - 1, in: text)?.isWhitespace == true {
            end -= 1
        }
    }

    private func character(at offset: Int, in text: String) -> Character? {
        guard offset >= 0,
              let index = text.index(text.startIndex, offsetBy: offset, limitedBy: text.endIndex),
              index < text.endIndex
        else {
            return nil
        }
        return text[index]
    }

    private func substring(_ text: String, start: Int, end: Int) -> String {
        let lower = text.index(text.startIndex, offsetBy: max(0, start), limitedBy: text.endIndex)
            ?? text.endIndex
        let upper = text.index(text.startIndex, offsetBy: max(start, end), limitedBy: text.endIndex)
            ?? text.endIndex
        guard lower <= upper else {
            return ""
        }
        return String(text[lower..<upper])
    }
}
