import Foundation
import MLX
import MLXNN
import Tokenizers

public struct OpenMedZeroShotEntity: Sendable {
    public let text: String
    public let label: String
    public let score: Float
    public let start: Int
    public let end: Int

    public init(text: String, label: String, score: Float, start: Int, end: Int) {
        self.text = text
        self.label = label
        self.score = score
        self.start = start
        self.end = end
    }
}

public struct OpenMedClassification: Sendable {
    public let label: String
    public let score: Float

    public init(label: String, score: Float) {
        self.label = label
        self.score = score
    }
}

public struct OpenMedRelation: Sendable {
    public let label: String
    public let score: Float
    public let head: OpenMedZeroShotEntity
    public let tail: OpenMedZeroShotEntity

    public init(
        label: String,
        score: Float,
        head: OpenMedZeroShotEntity,
        tail: OpenMedZeroShotEntity
    ) {
        self.label = label
        self.score = score
        self.head = head
        self.tail = tail
    }
}

public struct OpenMedRelationResult: Sendable {
    public let entities: [OpenMedZeroShotEntity]
    public let relations: [OpenMedRelation]

    public init(entities: [OpenMedZeroShotEntity], relations: [OpenMedRelation]) {
        self.entities = entities
        self.relations = relations
    }
}

public enum OpenMedZeroShotError: LocalizedError {
    case unsupportedArtifact(expectedTask: String, expectedFamily: String, actualTask: String, actualFamily: String)
    case missingPromptSpec(String)
    case emptyInput

    public var errorDescription: String? {
        switch self {
        case let .unsupportedArtifact(expectedTask, expectedFamily, actualTask, actualFamily):
            return "Expected \(expectedTask)/\(expectedFamily) MLX artifact, got \(actualTask)/\(actualFamily)."
        case .missingPromptSpec(let field):
            return "GLiNER artifact is missing prompt_spec.\(field)."
        case .emptyInput:
            return "Input text did not contain any extractable words."
        }
    }
}

public final class OpenMedZeroShotNER {
    private let artifact: OpenMedMLXArtifact
    private let model: OpenMedGLiNERSpanModel
    private let tokenizer: any Tokenizer
    private let promptEncoder: OpenMedGLiNERPromptEncoder
    private let maxSeqLength: Int

    public init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
        guard MLXTokenClassificationPipeline.isRuntimeSupported else {
            throw OpenMedMLXRuntimeError.unsupportedPlatform
        }

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
        guard artifact.task == .zeroShotNER, artifact.family == .glinerUniEncoderSpan else {
            throw OpenMedZeroShotError.unsupportedArtifact(
                expectedTask: OpenMedMLXTask.zeroShotNER.rawValue,
                expectedFamily: OpenMedMLXFamily.glinerUniEncoderSpan.rawValue,
                actualTask: artifact.manifest.task,
                actualFamily: artifact.manifest.family
            )
        }

        self.artifact = artifact
        self.model = try OpenMedMLXModelLoader.loadGLiNERSpanModel(from: artifact)
        self.tokenizer = try OpenMed.loadTokenizer(
            tokenizerName: artifact.tokenizerName ?? modelDirectoryURL.path,
            tokenizerFolderURL: artifact.tokenizerDirectoryURL
        )
        self.promptEncoder = OpenMedGLiNERPromptEncoder(tokenizer: tokenizer)
        self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
    }

    public func extract(
        _ text: String,
        labels: [String],
        threshold: Float = 0.5,
        flatNER: Bool = true
    ) throws -> [OpenMedZeroShotEntity] {
        guard !labels.isEmpty else {
            return []
        }

        let split = OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(text)
        guard !split.words.isEmpty else {
            return []
        }

        let spec = artifact.manifest.promptSpec
        let entityToken = spec?.entityToken ?? "<<ENT>>"
        let separatorToken = spec?.separatorToken ?? "<<SEP>>"
        var promptWords = [String]()
        for label in labels {
            promptWords.append(entityToken)
            promptWords.append(label)
        }
        promptWords.append(separatorToken)

        let encoded = promptEncoder.encodeWords(
            promptWords + split.words,
            skipFirstWords: promptWords.count,
            maxSeqLength: maxSeqLength,
            specialTokenIDs: [
                entityToken: spec?.classTokenIndex ?? artifact.configuration.classTokenIndex,
                separatorToken: artifact.configuration.textTokenIndex,
            ].compactMapValues { $0 }
        )
        let spans = OpenMedGLiNERPromptEncoder.buildCandidateSpanBatch(
            wordCount: split.words.count,
            maxWidth: artifact.configuration.maxWidth
        )

        let inputIDs = MLXArray(encoded.inputIDs.map(Int32.init), [1, encoded.inputIDs.count])
        let attentionMask = MLXArray(encoded.attentionMask, [1, encoded.attentionMask.count])
            .asType(.float32)
        let wordsMask = MLXArray(encoded.wordsMask.map(Int32.init), [1, encoded.wordsMask.count])
        let output = model(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            wordsMask: wordsMask,
            spanIndex: spans.index,
            spanMask: spans.mask
        )
        let probabilities = sigmoid(output.logits)
        eval(probabilities, output.promptMask, output.spanIndex, output.spanMask)

        let promptMaskValues = output.promptMask[0].asArray(Bool.self)
        let validPromptCount = min(labels.count, promptMaskValues.filter { $0 }.count)
        let scores = probabilities[0].asArray(Float.self)
        let scoreLabelWidth = output.logits.dim(2)
        let spanIndex = output.spanIndex[0].asArray(Int32.self).map(Int.init)
        let spanMask = output.spanMask[0].asArray(Bool.self)
        let spanCount = spanMask.count

        var entities = [OpenMedZeroShotEntity]()
        for span in 0..<spanCount where spanMask[span] {
            let startWord = spanIndex[span * 2]
            let endWord = spanIndex[span * 2 + 1]
            guard endWord < split.offsets.count else {
                continue
            }
            let startChar = split.offsets[startWord].0
            let endChar = split.offsets[endWord].1
            for labelIndex in 0..<validPromptCount {
                let score = scores[span * scoreLabelWidth + labelIndex]
                guard score >= threshold else {
                    continue
                }
                entities.append(
                    OpenMedZeroShotEntity(
                        text: String(text.characterSlice(start: startChar, end: endChar)),
                        label: labels[labelIndex],
                        score: score,
                        start: startChar,
                        end: endChar
                    )
                )
            }
        }

        return flatNER
            ? OpenMedGLiNERPromptEncoder.suppressOverlaps(entities)
            : entities.sorted { ($0.start, $0.end, $0.label) < ($1.start, $1.end, $1.label) }
    }
}

public final class OpenMedZeroShotClassifier {
    private let artifact: OpenMedMLXArtifact
    private let model: OpenMedGLiClassUniEncoderModel
    private let tokenizer: any Tokenizer
    private let maxSeqLength: Int

    public init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
        guard MLXTokenClassificationPipeline.isRuntimeSupported else {
            throw OpenMedMLXRuntimeError.unsupportedPlatform
        }

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
        guard artifact.task == .zeroShotSequenceClassification,
              artifact.family == .gliclassUniEncoder else {
            throw OpenMedZeroShotError.unsupportedArtifact(
                expectedTask: OpenMedMLXTask.zeroShotSequenceClassification.rawValue,
                expectedFamily: OpenMedMLXFamily.gliclassUniEncoder.rawValue,
                actualTask: artifact.manifest.task,
                actualFamily: artifact.manifest.family
            )
        }

        self.artifact = artifact
        self.model = try OpenMedMLXModelLoader.loadGLiClassUniEncoderModel(from: artifact)
        self.tokenizer = try OpenMed.loadTokenizer(
            tokenizerName: artifact.tokenizerName ?? modelDirectoryURL.path,
            tokenizerFolderURL: artifact.tokenizerDirectoryURL
        )
        self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
    }

    public func classify(
        _ text: String,
        labels: [String],
        threshold: Float = 0.5,
        prompt: String? = nil
    ) throws -> [OpenMedClassification] {
        guard !labels.isEmpty else {
            return []
        }

        let spec = artifact.manifest.promptSpec
        let labelToken = spec?.labelToken ?? "<<LABEL>>"
        let separatorToken = spec?.separatorToken ?? "<<SEP>>"
        let promptFirst = spec?.promptFirst ?? true

        var packedWords = [String]()
        if !promptFirst {
            packedWords.append(contentsOf: OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(text).words)
        }
        for label in labels {
            packedWords.append(labelToken)
            packedWords.append(contentsOf: OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(label).words)
        }
        packedWords.append(separatorToken)
        if let prompt, !prompt.isEmpty {
            packedWords.append(contentsOf: OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(prompt).words)
            packedWords.append(separatorToken)
        }
        if promptFirst {
            packedWords.append(contentsOf: OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(text).words)
        }

        let inputIDs = OpenMedGLiNERPromptEncoder(tokenizer: tokenizer).encodeWords(
            packedWords,
            skipFirstWords: packedWords.count,
            maxSeqLength: maxSeqLength,
            specialTokenIDs: [
                labelToken: spec?.classTokenIndex ?? artifact.configuration.classTokenIndex,
                separatorToken: spec?.textTokenIndex ?? artifact.configuration.textTokenIndex,
                spec?.exampleToken ?? "<<EXAMPLE>>": spec?.exampleTokenIndex
                    ?? artifact.configuration.exampleTokenIndex,
            ].compactMapValues { $0 }
        ).inputIDs
        let attentionMask = Array(repeating: 1, count: inputIDs.count)

        let output = model(
            inputIDs: MLXArray(inputIDs.map(Int32.init), [1, inputIDs.count]),
            attentionMask: MLXArray(attentionMask, [1, attentionMask.count]).asType(.float32)
        )
        let probabilities = sigmoid(output.logits)
        eval(probabilities, output.classesMask)

        let scores = probabilities[0].asArray(Float.self)
        let classMask = output.classesMask[0].asArray(Bool.self)
        let validLabelCount = min(labels.count, classMask.filter { $0 }.count)
        var predictions = [OpenMedClassification]()
        for index in 0..<validLabelCount {
            let score = scores[index]
            if score >= threshold {
                predictions.append(OpenMedClassification(label: labels[index], score: score))
            }
        }
        return predictions.sorted { $0.score > $1.score }
    }
}

public final class OpenMedRelationExtractor {
    private let artifact: OpenMedMLXArtifact
    private let model: OpenMedGLiNERRelexModel
    private let tokenizer: any Tokenizer
    private let promptEncoder: OpenMedGLiNERPromptEncoder
    private let maxSeqLength: Int

    public init(modelDirectoryURL: URL, maxSeqLength: Int = 512) throws {
        guard MLXTokenClassificationPipeline.isRuntimeSupported else {
            throw OpenMedMLXRuntimeError.unsupportedPlatform
        }

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: modelDirectoryURL)
        guard artifact.task == .zeroShotRelationExtraction,
              artifact.family == .glinerUniEncoderTokenRelex else {
            throw OpenMedZeroShotError.unsupportedArtifact(
                expectedTask: OpenMedMLXTask.zeroShotRelationExtraction.rawValue,
                expectedFamily: OpenMedMLXFamily.glinerUniEncoderTokenRelex.rawValue,
                actualTask: artifact.manifest.task,
                actualFamily: artifact.manifest.family
            )
        }

        self.artifact = artifact
        self.model = try OpenMedMLXModelLoader.loadGLiNERRelexModel(from: artifact)
        self.tokenizer = try OpenMed.loadTokenizer(
            tokenizerName: artifact.tokenizerName ?? modelDirectoryURL.path,
            tokenizerFolderURL: artifact.tokenizerDirectoryURL
        )
        self.promptEncoder = OpenMedGLiNERPromptEncoder(tokenizer: tokenizer)
        self.maxSeqLength = min(maxSeqLength, artifact.manifest.maxSequenceLength ?? maxSeqLength)
    }

    public func extract(
        _ text: String,
        entityLabels: [String],
        relationLabels: [String],
        threshold: Float = 0.5,
        relationThreshold: Float = 0.9,
        flatNER: Bool = true
    ) throws -> OpenMedRelationResult {
        guard !entityLabels.isEmpty else {
            return OpenMedRelationResult(entities: [], relations: [])
        }

        let split = OpenMedGLiNERPromptEncoder.splitWordsWithOffsets(text)
        guard !split.words.isEmpty else {
            return OpenMedRelationResult(entities: [], relations: [])
        }

        let spec = artifact.manifest.promptSpec
        let entityToken = spec?.entityToken ?? "<<ENT>>"
        let relationToken = spec?.relationToken ?? "<<REL>>"
        let separatorToken = spec?.separatorToken ?? "<<SEP>>"
        var promptWords = [String]()
        for label in entityLabels {
            promptWords.append(entityToken)
            promptWords.append(label)
        }
        promptWords.append(separatorToken)
        for relation in relationLabels {
            promptWords.append(relationToken)
            promptWords.append(relation)
        }
        promptWords.append(separatorToken)

        let encodedInput = promptEncoder.encodeWords(
            promptWords + split.words,
            skipFirstWords: promptWords.count,
            maxSeqLength: maxSeqLength,
            specialTokenIDs: [
                entityToken: spec?.classTokenIndex ?? artifact.configuration.classTokenIndex,
                relationToken: spec?.relTokenIndex ?? artifact.configuration.relTokenIndex,
                separatorToken: artifact.configuration.textTokenIndex,
            ].compactMapValues { $0 }
        )
        let inputIDs = MLXArray(encodedInput.inputIDs.map(Int32.init), [1, encodedInput.inputIDs.count])
        let attentionMask = MLXArray(encodedInput.attentionMask, [1, encodedInput.attentionMask.count])
            .asType(.float32)
        let wordsMask = MLXArray(encodedInput.wordsMask.map(Int32.init), [1, encodedInput.wordsMask.count])

        let entityOutput = model(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            wordsMask: wordsMask
        )
        let entityScores = sigmoid(entityOutput.entityScores)
        eval(entityScores, entityOutput.entityPromptMask)

        let entityPromptCount = min(
            entityLabels.count,
            entityOutput.entityPromptMask[0].asArray(Bool.self).filter { $0 }.count
        )
        let decodedSpans = OpenMedGLiNERPromptEncoder.decodeTokenLevelSpans(
            scores: entityScores[0].asArray(Float.self),
            sequenceLength: entityScores.dim(1),
            numClasses: entityScores.dim(2),
            threshold: threshold,
            flatNER: flatNER
        ).filter { $0.labelIndex < entityPromptCount }

        let spanBatch = OpenMedGLiNERPromptEncoder.buildRaggedSpanBatch(
            decodedSpans.map { ($0.start, $0.end) }
        )
        let relationOutput = model.relationScores(
            encoded: entityOutput.encoded,
            spanIndex: spanBatch.index,
            spanMask: spanBatch.mask
        )
        let relationScores = sigmoid(relationOutput.pairScores)
        eval(relationScores, relationOutput.pairIndex, relationOutput.pairMask)

        let entities = decodedSpans.enumerated().compactMap { _, span -> OpenMedZeroShotEntity? in
            guard span.end < split.offsets.count else {
                return nil
            }
            let startChar = split.offsets[span.start].0
            let endChar = split.offsets[span.end].1
            return OpenMedZeroShotEntity(
                text: String(text.characterSlice(start: startChar, end: endChar)),
                label: entityLabels[span.labelIndex],
                score: span.score,
                start: startChar,
                end: endChar
            )
        }

        let relationPromptCount = min(
            relationLabels.count,
            relationOutput.relationPromptMask[0].asArray(Bool.self).filter { $0 }.count
        )
        let pairScores = relationScores[0].asArray(Float.self)
        let pairScoreWidth = relationOutput.pairScores.dim(2)
        let pairIndex = relationOutput.pairIndex[0].asArray(Int32.self).map(Int.init)
        let pairMask = relationOutput.pairMask[0].asArray(Bool.self)
        let pairCount = pairMask.count
        var relations = [OpenMedRelation]()

        for pair in 0..<pairCount where pairMask[pair] {
            let headIndex = pairIndex[pair * 2]
            let tailIndex = pairIndex[pair * 2 + 1]
            guard headIndex < entities.count, tailIndex < entities.count else {
                continue
            }
            for relationIndex in 0..<relationPromptCount {
                let score = pairScores[pair * pairScoreWidth + relationIndex]
                guard score >= relationThreshold else {
                    continue
                }
                relations.append(
                    OpenMedRelation(
                        label: relationLabels[relationIndex],
                        score: score,
                        head: entities[headIndex],
                        tail: entities[tailIndex]
                    )
                )
            }
        }

        return OpenMedRelationResult(entities: entities, relations: relations)
    }
}

struct OpenMedGLiNERPromptEncoder {
    struct EncodedWords {
        let inputIDs: [Int]
        let attentionMask: [Int]
        let wordsMask: [Int]
    }

    struct TokenLevelSpan {
        let start: Int
        let end: Int
        let labelIndex: Int
        let score: Float
    }

    let tokenizer: any Tokenizer

    func encodeWords(
        _ words: [String],
        skipFirstWords: Int,
        maxSeqLength: Int,
        specialTokenIDs: [String: Int] = [:]
    ) -> EncodedWords {
        let emptySpecialIDs = tokenizer.encode(text: "", addSpecialTokens: true)
        let prefix = emptySpecialIDs.first.map { [$0] } ?? []
        let suffix = emptySpecialIDs.count > 1 ? [emptySpecialIDs.last!] : []
        let contentLimit = max(0, maxSeqLength - prefix.count - suffix.count)

        var inputIDs = prefix
        var wordsMask = Array(repeating: 0, count: prefix.count)
        var emittedContentTokens = 0
        var textWordIndex = 0

        for (wordPosition, word) in words.enumerated() {
            let tokenIDs: [Int]
            if let specialTokenID = specialTokenIDs[word] {
                tokenIDs = [specialTokenID]
            } else {
                tokenIDs = tokenizer.encode(text: word, addSpecialTokens: false)
            }
            guard !tokenIDs.isEmpty else {
                continue
            }
            let remaining = contentLimit - emittedContentTokens
            guard remaining > 0 else {
                break
            }

            let truncated = Array(tokenIDs.prefix(remaining))
            let isTextWord = wordPosition >= skipFirstWords
            if isTextWord {
                textWordIndex += 1
            }
            for tokenIndex in 0..<truncated.count {
                wordsMask.append(isTextWord && tokenIndex == 0 ? textWordIndex : 0)
            }
            inputIDs.append(contentsOf: truncated)
            emittedContentTokens += truncated.count
        }

        inputIDs.append(contentsOf: suffix)
        wordsMask.append(contentsOf: Array(repeating: 0, count: suffix.count))
        return EncodedWords(
            inputIDs: inputIDs,
            attentionMask: Array(repeating: 1, count: inputIDs.count),
            wordsMask: wordsMask
        )
    }

    static func splitWordsWithOffsets(_ text: String) -> (words: [String], offsets: [(Int, Int)]) {
        let pattern = #"\w+(?:[-_]\w+)*|\S"#
        let regex = try! NSRegularExpression(pattern: pattern)
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        var words = [String]()
        var offsets = [(Int, Int)]()

        for match in regex.matches(in: text, range: range) {
            guard let matchRange = Range(match.range, in: text) else {
                continue
            }
            words.append(String(text[matchRange]))
            offsets.append((
                text.distance(from: text.startIndex, to: matchRange.lowerBound),
                text.distance(from: text.startIndex, to: matchRange.upperBound)
            ))
        }
        return (words, offsets)
    }

    static func buildCandidateSpanBatch(
        wordCount: Int,
        maxWidth: Int
    ) -> (index: MLXArray, mask: MLXArray) {
        var spans = [[Int]]()
        var mask = [Int]()
        for start in 0..<max(wordCount, 1) {
            for width in 0..<maxWidth {
                let end = start + width
                spans.append([start, end])
                mask.append(end < wordCount ? 1 : 0)
            }
        }
        let flatSpans = spans.flatMap { $0 }.map(Int32.init)
        return (
            MLXArray(flatSpans, [1, spans.count, 2]),
            MLXArray(mask, [1, mask.count]).asType(.bool)
        )
    }

    static func buildRaggedSpanBatch(
        _ spans: [(Int, Int)]
    ) -> (index: MLXArray, mask: MLXArray) {
        let width = max(spans.count, 1)
        let padded = spans.map { [$0.0, $0.1] }
            + Array(repeating: [0, 0], count: max(0, width - spans.count))
        let mask = Array(repeating: 1, count: spans.count)
            + Array(repeating: 0, count: max(0, width - spans.count))
        return (
            MLXArray(padded.flatMap { $0 }.map(Int32.init), [1, width, 2]),
            MLXArray(mask, [1, width]).asType(.bool)
        )
    }

    static func suppressOverlaps(_ entities: [OpenMedZeroShotEntity]) -> [OpenMedZeroShotEntity] {
        var selected = [OpenMedZeroShotEntity]()
        for entity in entities.sorted(by: {
            if $0.score != $1.score {
                return $0.score > $1.score
            }
            if $0.start != $1.start {
                return $0.start < $1.start
            }
            return $0.end < $1.end
        }) {
            let overlaps = selected.contains {
                !(entity.end <= $0.start || entity.start >= $0.end)
            }
            if !overlaps {
                selected.append(entity)
            }
        }
        return selected.sorted { ($0.start, $0.end, $0.label) < ($1.start, $1.end, $1.label) }
    }

    static func decodeTokenLevelSpans(
        scores: [Float],
        sequenceLength: Int,
        numClasses: Int,
        threshold: Float,
        flatNER: Bool
    ) -> [TokenLevelSpan] {
        var candidates = [TokenLevelSpan]()
        guard sequenceLength > 0, numClasses > 0 else {
            return []
        }

        func scoreAt(token: Int, label: Int, channel: Int) -> Float {
            scores[(token * numClasses + label) * 3 + channel]
        }

        for labelIndex in 0..<numClasses {
            var starts = [Int]()
            var ends = [Int]()
            for tokenIndex in 0..<sequenceLength {
                if scoreAt(token: tokenIndex, label: labelIndex, channel: 0) > threshold {
                    starts.append(tokenIndex)
                }
                if scoreAt(token: tokenIndex, label: labelIndex, channel: 1) > threshold {
                    ends.append(tokenIndex)
                }
            }

            for start in starts {
                for end in ends where end >= start {
                    var minimumScore = min(
                        scoreAt(token: start, label: labelIndex, channel: 0),
                        scoreAt(token: end, label: labelIndex, channel: 1)
                    )
                    var isValid = true
                    for tokenIndex in start...end {
                        let insideScore = scoreAt(token: tokenIndex, label: labelIndex, channel: 2)
                        if insideScore < threshold {
                            isValid = false
                            break
                        }
                        minimumScore = min(minimumScore, insideScore)
                    }
                    if isValid {
                        candidates.append(
                            TokenLevelSpan(
                                start: start,
                                end: end,
                                labelIndex: labelIndex,
                                score: minimumScore
                            )
                        )
                    }
                }
            }
        }

        var selected = [TokenLevelSpan]()
        for span in candidates.sorted(by: { $0.score > $1.score }) {
            let overlaps = selected.contains { existing in
                if span.start == existing.start && span.end == existing.end {
                    return true
                }
                let isDisjoint = span.start > existing.end || existing.start > span.end
                if flatNER {
                    return !isDisjoint
                }
                let isNested =
                    (span.start <= existing.start && span.end >= existing.end)
                    || (existing.start <= span.start && existing.end >= span.end)
                return !(isDisjoint || isNested)
            }
            if !overlaps {
                selected.append(span)
            }
        }
        return selected.sorted {
            ($0.start, $0.end, $0.labelIndex) < ($1.start, $1.end, $1.labelIndex)
        }
    }
}

private extension String {
    func characterSlice(start: Int, end: Int) -> Substring {
        let lower = index(startIndex, offsetBy: max(0, start), limitedBy: endIndex) ?? endIndex
        let upper = index(startIndex, offsetBy: max(0, end), limitedBy: endIndex) ?? endIndex
        return self[lower..<upper]
    }
}
