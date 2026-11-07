import Foundation
import OpenMedKit
import WhisperKit

final class StrictLocalWhisperKit: WhisperKit {
    override func loadTokenizerIfNeeded() async throws {
        guard tokenizer != nil else { throw StreamingSpeechError.unverifiedArtifact }
    }
}

final class LocalEnglishTokenizer: WhisperTokenizer {
    private let lock = NSLock()
    private var invalidAlignment = false
    var alignmentValid: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !invalidAlignment
    }
    func resetAlignment() {
        lock.lock()
        defer { lock.unlock() }
        invalidAlignment = false
    }
    private func markInvalidAlignment() {
        lock.lock()
        defer { lock.unlock() }
        invalidAlignment = true
    }
    private let base: TokenizerWrapper
    let specialTokens: SpecialTokens
    let allLanguageTokens: Set<Int>
    init(_ base: TokenizerWrapper) throws {
        self.base = base
        func token(_ text: String) throws -> Int {
            guard let value = base.convertTokenToId(text) else { throw StreamingSpeechError.artifactMismatch }
            return value
        }
        let whitespace = base.encode(text: " ", addSpecialTokens: false)
        guard whitespace.count == 1 else { throw StreamingSpeechError.artifactMismatch }
        let end = try token("<|endoftext|>")
        let english = try token("<|en|>")
        let translate = try token("<|translate|>")
        guard end == 50256, english == 50258, translate == 50357 else { throw StreamingSpeechError.artifactMismatch }
        specialTokens = try SpecialTokens(
            endToken: end, englishToken: english, noSpeechToken: token("<|nocaptions|>"),
            noTimestampsToken: token("<|notimestamps|>"), specialTokenBegin: end,
            startOfPreviousToken: token("<|startofprev|>"), startOfTranscriptToken: token("<|startoftranscript|>"),
            timeTokenBegin: token("<|0.00|>"), transcribeToken: token("<|transcribe|>"),
            translateToken: translate, whitespaceToken: whitespace[0])
        guard specialTokens.noSpeechToken == 50361, specialTokens.noTimestampsToken == 50362,
            specialTokens.startOfPreviousToken == 50360, specialTokens.startOfTranscriptToken == 50257,
            specialTokens.timeTokenBegin == 50363, specialTokens.transcribeToken == 50358,
            specialTokens.whitespaceToken == 220
        else { throw StreamingSpeechError.artifactMismatch }
        allLanguageTokens = Set(english..<translate)
    }
    func encode(text: String) -> [Int] { base.encode(text: text, addSpecialTokens: false) }
    func decode(tokens: [Int]) -> String { base.decode(tokens: tokens) }
    func convertTokenToId(_ token: String) -> Int? { base.convertTokenToId(token) }
    func convertIdToToken(_ id: Int) -> String? { base.convertIdToToken(id) }
    func splitToWordTokens(tokenIds: [Int]) -> (words: [String], wordTokens: [[Int]]) {
        guard !tokenIds.isEmpty else { return ([], []) }
        let complete = decode(tokens: tokenIds)
        var words = [String]()
        var groups = [[Int]]()
        var pending = [Int]()
        var consumed = ""
        for id in tokenIds {
            pending.append(id)
            let piece = decode(tokens: pending)
            guard !piece.isEmpty, complete.hasPrefix(consumed + piece) else { continue }
            let trimmed = piece.trimmingCharacters(in: .whitespacesAndNewlines)
            let punctuation = !trimmed.isEmpty && trimmed.unicodeScalars.allSatisfy { CharacterSet.punctuationCharacters.contains($0) }
            if words.isEmpty || piece.first?.isWhitespace == true || punctuation || pending[0] >= specialTokens.specialTokenBegin {
                words.append(piece)
                groups.append(pending)
            } else {
                words[words.count - 1] += piece
                groups[groups.count - 1] += pending
            }
            consumed += piece
            pending.removeAll(keepingCapacity: true)
        }
        guard pending.isEmpty, consumed == complete else {
            // Preserve the runtime's shape, but refuse its alignment before emission.
            markInvalidAlignment()
            return ([complete], [tokenIds])
        }
        return (words, groups)
    }
}
