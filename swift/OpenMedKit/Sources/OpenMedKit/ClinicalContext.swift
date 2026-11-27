import CryptoKit
import Foundation

/// Value-free clinical context failures. The raw value is the reason code.
public enum OpenMedClinicalContextError: String, Error, Equatable, Sendable, CaseIterable, LocalizedError {
    /// The bundled cue table resource is missing.
    case cueTableUnavailable = "cue_table_unavailable"
    /// The cue table is malformed or declares an unsupported schema.
    case cueTableInvalid = "cue_table_invalid"
    /// A span is outside `0 <= start <= end <= text length` (Unicode scalars).
    case invalidSpan = "invalid_span"
    /// A sentence is outside `0 <= start <= end <= text length`.
    case invalidSentence = "invalid_sentence"

    /// Stable reason code for logs and metrics.
    public var reason: String { rawValue }

    public var errorDescription: String? {
        switch self {
        case .cueTableUnavailable: return "The bundled clinical context cue table is unavailable."
        case .cueTableInvalid: return "The clinical context cue table is invalid."
        case .invalidSpan: return "A clinical span lies outside the source text."
        case .invalidSentence: return "A sentence lies outside the source text."
        }
    }
}

/// Advisory ConText assertion axes for one clinical span.
///
/// The raw values are the Python `assert_context` codes. The axes are
/// annotations for review: they are not diagnoses and trigger no action.
public struct OpenMedClinicalAssertion: Equatable, Hashable, Sendable {
    /// Whether the span is affirmed or negated (`denies chest pain`).
    public enum Negation: String, CaseIterable, Sendable {
        case affirmed
        case negated
    }

    /// Whether the span is asserted or hedged (`possible pneumonia`).
    public enum Uncertainty: String, CaseIterable, Sendable {
        case certain
        case uncertain
    }

    /// Whether the span is current, past or conditional.
    public enum Temporality: String, CaseIterable, Sendable {
        case recent
        case historical
        case hypothetical
    }

    /// Who the span is about.
    public enum Experiencer: String, CaseIterable, Sendable {
        case patient
        case family
        case other
    }

    public let negation: Negation
    public let uncertainty: Uncertainty
    public let temporality: Temporality
    public let experiencer: Experiencer

    public init(negation: Negation, uncertainty: Uncertainty, temporality: Temporality, experiencer: Experiencer) {
        self.negation = negation
        self.uncertainty = uncertainty
        self.temporality = temporality
        self.experiencer = experiencer
    }
}

/// Why no assertion was computed.
public enum OpenMedClinicalAssertionUnavailableReason: String, CaseIterable, Equatable, Sendable {
    /// The caller declared a language other than English. English cues are
    /// never applied to other languages.
    case unsupportedLanguage = "unsupported_language"
}

/// The assertions for a list of spans, or why there are none.
public enum OpenMedClinicalAssertionResult: Equatable, Sendable {
    /// One assertion per input span, in input order.
    case asserted([OpenMedClinicalAssertion])
    /// No assertion was computed.
    case unavailable(OpenMedClinicalAssertionUnavailableReason)

    /// The assertions, or `nil` when unavailable.
    public var assertions: [OpenMedClinicalAssertion]? {
        if case .asserted(let assertions) = self { return assertions }
        return nil
    }

    /// The reason no assertion was computed, or `nil`.
    public var unavailableReason: OpenMedClinicalAssertionUnavailableReason? {
        if case .unavailable(let reason) = self { return reason }
        return nil
    }
}

/// An offline, deterministic port of the English path of Python's
/// `openmed.clinical.context.assert_context` without sections.
///
/// The cues come from the bundled `Resources/clinical/context_cues_en_v1.json`,
/// a byte-identical copy of `tests/fixtures/parity/context_cues_en_v1.json`
/// exported from the Python lexicon; no cue is transcribed into Swift. For
/// each span the port:
///
/// - scopes cue matching to the span's clause, bounded by sentence punctuation
///   and the table's scope terminators (`and`, `but`, `however`, `or`);
/// - matches cues case-insensitively on token boundaries, longest cue first,
///   with any whitespace run between cue words;
/// - masks pseudo-negations (`cannot be excluded`) before counting negation
///   cues, so an odd count negates and an even count affirms;
/// - collects forward and backward modifier cues from the span's sentence
///   that no conjunction terminator separates from the span, using the
///   caller's sentence bounds or a regex sentence fallback;
/// - resolves the experiencer from the nearest family or other subject cue in
///   the span's clause, defaulting to the patient.
///
/// Offsets are half-open Unicode scalar offsets, like Python string indices.
/// The results match Python's recorded behaviour, including its known gaps:
/// a negation does not reach past `and` or `or` (`Denies fever and cough.`
/// leaves `cough` affirmed), exactly as the shared parity vectors record.
/// Without caller sentences Python segments with pySBD when installed; this
/// port always uses the regex fallback.
public struct OpenMedClinicalContext: Sendable {
    /// A half-open Unicode scalar range in a source text.
    public struct Span: Equatable, Hashable, Sendable {
        public let start: Int
        public let end: Int

        public init(start: Int, end: Int) {
            self.start = start
            self.end = end
        }
    }

    /// Bundled cue table file name.
    public static let cueTableFile = "context_cues_en_v1.json"
    /// Cue table schema name.
    public static let cueTableSchema = "openmed.tests.context_cues"
    /// Supported cue table version.
    public static let cueTableVersion = 1

    /// Lowercase hex SHA-256 of the cue table bytes this context was built from.
    public let cueTableSHA256: String

    private let historical: CueMatcher
    private let hypothetical: CueMatcher
    private let uncertainty: CueMatcher
    private let negation: CueMatcher
    private let pseudoNegation: CueMatcher
    private let contextCues: CueMatcher
    private let scopeTerminators: CueMatcher
    private let conjunctionTerminators: CueMatcher
    private let familyCues: CueMatcher
    private let otherCues: CueMatcher
    private let backwardCues: Set<String>

    /// Clause boundaries a subject cue may not cross. Python keeps these in
    /// `openmed/clinical/experiencer.py` (`_CLAUSE_BOUNDARY_RE`), not in the
    /// cue table: sentence punctuation plus contrastive conjunctions.
    private static let experiencerBoundaries = CueMatcher(
        cues: ["but", "however", "whereas"],
        punctuation: ".!?;"
    )

    /// Punctuation Python's ConText scope terminator pattern always includes.
    private static let scopePunctuation = ".!?;\u{3002}\u{FF01}\u{FF1F}\u{FF1B}"

    /// Load the context from the cue table bundled with OpenMedKit.
    ///
    /// - Throws: ``OpenMedClinicalContextError/cueTableUnavailable`` or
    ///   ``OpenMedClinicalContextError/cueTableInvalid``.
    public static func bundled() throws -> OpenMedClinicalContext {
        guard let data = bundledCueTableData() else {
            throw OpenMedClinicalContextError.cueTableUnavailable
        }
        return try OpenMedClinicalContext(cueTable: data)
    }

    /// Bytes of the bundled cue table, or `nil` when it is missing.
    public static func bundledCueTableData() -> Data? {
        let stem = (cueTableFile as NSString).deletingPathExtension
        let url =
            Bundle.module.url(forResource: stem, withExtension: "json", subdirectory: "clinical")
            ?? Bundle.module.url(forResource: stem, withExtension: "json", subdirectory: "Resources/clinical")
            ?? Bundle.module.url(forResource: stem, withExtension: "json")
        return url.flatMap { try? Data(contentsOf: $0) }
    }

    /// Build a context from cue table bytes.
    ///
    /// - Throws: ``OpenMedClinicalContextError/cueTableInvalid`` for another
    ///   schema, version or language, a missing cue list, or a table that does
    ///   not use token boundaries.
    public init(cueTable data: Data) throws {
        guard let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            root["schema"] as? String == Self.cueTableSchema,
            (root["version"] as? NSNumber)?.intValue == Self.cueTableVersion,
            root["language"] as? String == "en",
            let lexicon = root["lexicon"] as? [String: Any],
            lexicon["token_boundaries"] as? Bool == true,
            let experiencer = root["experiencer"] as? [String: Any]
        else {
            throw OpenMedClinicalContextError.cueTableInvalid
        }
        func cues(_ object: [String: Any], _ key: String) throws -> [String] {
            guard let values = object[key] as? [String], values.allSatisfy({ !$0.isEmpty }) else {
                throw OpenMedClinicalContextError.cueTableInvalid
            }
            return values
        }
        let historicalCues = try cues(lexicon, "historical")
        let hypotheticalCues = try cues(lexicon, "hypothetical")
        let uncertaintyCues = try cues(lexicon, "uncertainty")
        let negationCues = try cues(lexicon, "negation")
        historical = CueMatcher(cues: historicalCues)
        hypothetical = CueMatcher(cues: hypotheticalCues)
        uncertainty = CueMatcher(cues: uncertaintyCues)
        negation = CueMatcher(cues: negationCues)
        pseudoNegation = CueMatcher(cues: try cues(lexicon, "pseudo_negation"))
        contextCues = CueMatcher(cues: historicalCues + hypotheticalCues + uncertaintyCues + negationCues)
        scopeTerminators = CueMatcher(cues: try cues(lexicon, "scope_terminators"), punctuation: Self.scopePunctuation)
        conjunctionTerminators = CueMatcher(cues: try cues(lexicon, "conjunction_terminators"))
        familyCues = CueMatcher(cues: try cues(experiencer, "family_cues"))
        otherCues = CueMatcher(cues: try cues(experiencer, "other_cues"))
        backwardCues = Set(try cues(lexicon, "backward").map { CueMatcher.normalizedCue(Array($0.unicodeScalars)) })
        cueTableSHA256 = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Whether a declared language takes the English path.
    ///
    /// `nil` and an empty code mean English, like Python's default. Otherwise
    /// the primary subtag decides (`en`, `en-US` and `EN_gb` are English).
    public static func isEnglish(_ language: String?) -> Bool {
        guard let language else { return true }
        let code = language.lowercased().replacingOccurrences(of: "_", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let primary = code.split(separator: "-", maxSplits: 1, omittingEmptySubsequences: false).first ?? ""
        return primary.isEmpty || primary == "en"
    }

    /// Assert every span against its context in `text`.
    ///
    /// - Parameters:
    ///   - text: Source text the offsets index.
    ///   - spans: Target spans in Unicode scalar offsets.
    ///   - sentences: Optional sentence bounds that limit which modifier cues
    ///     reach a span. When `nil`, sentences come from a regex fallback.
    ///   - language: Declared language. Anything other than English returns
    ///     ``OpenMedClinicalAssertionResult/unavailable(_:)`` with
    ///     ``OpenMedClinicalAssertionUnavailableReason/unsupportedLanguage``.
    /// - Returns: One assertion per span, in input order.
    /// - Throws: ``OpenMedClinicalContextError/invalidSpan`` or
    ///   ``OpenMedClinicalContextError/invalidSentence`` for out-of-range
    ///   offsets. Errors carry no text.
    public func assertions(
        in text: String,
        spans: [Span],
        sentences: [Span]? = nil,
        language: String? = nil
    ) throws -> OpenMedClinicalAssertionResult {
        guard Self.isEnglish(language) else {
            return .unavailable(.unsupportedLanguage)
        }
        let scalars = Array(text.unicodeScalars)
        for span in spans where !(0 <= span.start && span.start <= span.end && span.end <= scalars.count) {
            throw OpenMedClinicalContextError.invalidSpan
        }
        let windows = try sentenceWindows(scalars, sentences)
        let hits = contextCues.matches(in: scalars).map { range in
            CueHit(
                range: range,
                isBackward: backwardCues.contains(CueMatcher.normalizedCue(Array(scalars[range])))
            )
        }
        return .asserted(spans.map { assertion(for: $0, in: scalars, windows: windows, hits: hits) })
    }

    // MARK: - Axes

    private struct CueHit {
        let range: Range<Int>
        let isBackward: Bool
    }

    private func assertion(
        for span: Span,
        in scalars: [Unicode.Scalar],
        windows: [Range<Int>],
        hits: [CueHit]
    ) -> OpenMedClinicalAssertion {
        let parts = textParts(for: span, in: scalars, windows: windows, hits: hits)
        let temporality: OpenMedClinicalAssertion.Temporality
        if parts.contains(where: { hypothetical.firstMatch(in: $0) != nil }) {
            temporality = .hypothetical
        } else if parts.contains(where: { historical.firstMatch(in: $0) != nil }) {
            temporality = .historical
        } else {
            temporality = .recent
        }
        let isUncertain = parts.contains { uncertainty.firstMatch(in: $0) != nil }
        let negationCount = parts.reduce(0) { count, part in
            count + negation.matches(in: pseudoNegation.masked(part)).count
        }
        return OpenMedClinicalAssertion(
            negation: negationCount % 2 == 1 ? .negated : .affirmed,
            uncertainty: isUncertain ? .uncertain : .certain,
            temporality: temporality,
            experiencer: experiencer(for: span, in: scalars)
        )
    }

    /// Python `_text_parts`: the span's clause plus reaching modifier cues the
    /// clause does not already contain.
    private func textParts(
        for span: Span,
        in scalars: [Unicode.Scalar],
        windows: [Range<Int>],
        hits: [CueHit]
    ) -> [[Unicode.Scalar]] {
        var scopeStart = 0
        for match in scopeTerminators.matches(in: scalars, to: span.start) {
            scopeStart = match.upperBound
        }
        let scopeEnd = scopeTerminators.firstMatch(in: scalars, from: span.end)?.lowerBound ?? scalars.count
        let clause = Array(scalars[scopeStart..<max(scopeStart, scopeEnd)])

        var parts = [clause]
        let foldedClause = CueMatcher.casefolded(clause)
        for text in modifierTexts(for: span, in: scalars, windows: windows, hits: hits) {
            if !foldedClause.isEmpty && CueMatcher.contains(foldedClause, CueMatcher.casefolded(text)) {
                continue
            }
            parts.append(text)
        }
        return parts.filter { !$0.isEmpty }
    }

    /// Python `scan_context_cues` followed by `_scoped_modifier_texts`.
    private func modifierTexts(
        for span: Span,
        in scalars: [Unicode.Scalar],
        windows: [Range<Int>],
        hits: [CueHit]
    ) -> [[Unicode.Scalar]] {
        guard let sentence = Self.windowIndex(windows, span.start, span.end) else { return [] }
        var texts: [[Unicode.Scalar]] = []
        for hit in hits {
            guard Self.windowIndex(windows, hit.range.lowerBound, hit.range.upperBound) == sentence else { continue }
            let between: [Unicode.Scalar]
            if hit.isBackward {
                guard span.end <= hit.range.lowerBound else { continue }
                between = Array(scalars[span.end..<hit.range.lowerBound])
            } else {
                guard hit.range.upperBound <= span.start else { continue }
                between = Array(scalars[hit.range.upperBound..<span.start])
            }
            guard conjunctionTerminators.firstMatch(in: between) == nil else { continue }

            let scopeGap: [Unicode.Scalar]
            if hit.range.upperBound <= span.start {
                scopeGap = Array(scalars[hit.range.upperBound..<span.start])
            } else if span.end <= hit.range.lowerBound {
                scopeGap = Array(scalars[span.end..<hit.range.lowerBound])
            } else {
                scopeGap = []
            }
            guard scopeTerminators.firstMatch(in: scopeGap) == nil else { continue }
            texts.append(Array(scalars[hit.range]))
        }
        return texts
    }

    /// Python `resolve_experiencer` without a section prior.
    private func experiencer(for span: Span, in scalars: [Unicode.Scalar]) -> OpenMedClinicalAssertion.Experiencer {
        var clauseStart = 0
        for match in Self.experiencerBoundaries.matches(in: scalars, to: span.start) {
            clauseStart = match.upperBound
        }
        var best: (gap: Int, rank: Int, experiencer: OpenMedClinicalAssertion.Experiencer)?
        for (experiencer, matcher, rank) in [
            (OpenMedClinicalAssertion.Experiencer.other, otherCues, 0), (.family, familyCues, 1),
        ] {
            for match in matcher.matches(in: scalars, from: clauseStart, to: span.start) {
                let gap = span.start - match.upperBound
                if best.map({ (gap, rank) < ($0.gap, $0.rank) }) ?? true {
                    best = (gap, rank, experiencer)
                }
            }
        }
        return best?.experiencer ?? .patient
    }

    // MARK: - Sentences

    private func sentenceWindows(_ scalars: [Unicode.Scalar], _ sentences: [Span]?) throws -> [Range<Int>] {
        var windows: [Range<Int>]
        if let sentences {
            windows = try sentences.map { sentence in
                guard 0 <= sentence.start, sentence.start <= sentence.end, sentence.end <= scalars.count else {
                    throw OpenMedClinicalContextError.invalidSentence
                }
                return sentence.start..<sentence.end
            }
        } else {
            windows = Self.regexSentenceWindows(scalars)
        }
        if windows.isEmpty && !scalars.isEmpty {
            windows = [0..<scalars.count]
        }
        return windows
    }

    /// Python `_regex_sentence_windows`: split after `[.!?]+(?:\s+|$)` or at
    /// `\n+`, then trim whitespace from each window.
    static func regexSentenceWindows(_ scalars: [Unicode.Scalar]) -> [Range<Int>] {
        guard !scalars.isEmpty else { return [] }
        var windows: [Range<Int>] = []
        func appendTrimmed(_ start: Int, _ end: Int) {
            var lower = start
            var upper = end
            while lower < upper && CueMatcher.isSpace(scalars[lower]) { lower += 1 }
            while upper > lower && CueMatcher.isSpace(scalars[upper - 1]) { upper -= 1 }
            if lower < upper { windows.append(lower..<upper) }
        }
        var start = 0
        var index = 0
        while index < scalars.count {
            var matchEnd: Int?
            if "\n" == scalars[index] {
                var end = index
                while end < scalars.count && scalars[end] == "\n" { end += 1 }
                matchEnd = end
            } else if ".!?".unicodeScalars.contains(scalars[index]) {
                var end = index
                while end < scalars.count && ".!?".unicodeScalars.contains(scalars[end]) { end += 1 }
                if end < scalars.count && CueMatcher.isSpace(scalars[end]) {
                    while end < scalars.count && CueMatcher.isSpace(scalars[end]) { end += 1 }
                    matchEnd = end
                } else if end == scalars.count {
                    matchEnd = end
                }
            }
            if let matchEnd {
                appendTrimmed(start, matchEnd)
                start = matchEnd
                index = matchEnd
            } else {
                index += 1
            }
        }
        appendTrimmed(start, scalars.count)
        return windows
    }

    private static func windowIndex(_ windows: [Range<Int>], _ start: Int, _ end: Int) -> Int? {
        windows.firstIndex { start >= $0.lowerBound && end <= $0.upperBound }
    }
}

/// A Python `re` compatible matcher for one ConText cue pattern:
/// `(?:[punctuation]|(?<!\w)(?:cue1|cue2|...)(?!\w))` with `re.IGNORECASE`,
/// cues sorted longest first and cue words joined by `\s+`.
///
/// Matching runs on Unicode scalars so offsets equal Python string indices.
/// `\w`, `\s` and case-insensitive letters follow Python's `str` semantics.
struct CueMatcher: Sendable {
    private let alternatives: [[[Unicode.Scalar]]]
    private let punctuation: Set<Unicode.Scalar>

    init(cues: [String], punctuation: String = "") {
        let unique = Set(cues)
        alternatives =
            unique
            .sorted { lhs, rhs in
                let left = lhs.unicodeScalars.count
                let right = rhs.unicodeScalars.count
                return left != right ? left > right : lhs < rhs
            }
            .map { cue in
                cue.unicodeScalars.split(whereSeparator: Self.isSpace).map { Array($0) }
            }
            .filter { !$0.isEmpty }
        self.punctuation = Set(punctuation.unicodeScalars)
    }

    /// Non-overlapping matches, like Python `pattern.finditer(text, pos, endpos)`:
    /// lookbehind may see scalars before `from`, and `to` acts as the end of
    /// the text for lookahead.
    func matches(in scalars: [Unicode.Scalar], from: Int = 0, to: Int? = nil) -> [Range<Int>] {
        let end = min(to ?? scalars.count, scalars.count)
        var result: [Range<Int>] = []
        var index = max(0, from)
        while index < end {
            if let match = match(in: scalars, at: index, end: end) {
                result.append(match)
                index = match.upperBound > index ? match.upperBound : index + 1
            } else {
                index += 1
            }
        }
        return result
    }

    /// The first match, like Python `pattern.search(text, pos)`.
    func firstMatch(in scalars: [Unicode.Scalar], from: Int = 0) -> Range<Int>? {
        var index = max(0, from)
        while index < scalars.count {
            if let match = match(in: scalars, at: index, end: scalars.count) {
                return match
            }
            index += 1
        }
        return nil
    }

    /// Python `pattern.sub(lambda m: " " * len(m.group(0)), text)`.
    func masked(_ scalars: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result = scalars
        for match in matches(in: scalars) {
            for index in match { result[index] = " " }
        }
        return result
    }

    private func match(in scalars: [Unicode.Scalar], at start: Int, end: Int) -> Range<Int>? {
        if punctuation.contains(scalars[start]) {
            return start..<(start + 1)
        }
        if start > 0 && Self.isWord(scalars[start - 1]) {
            return nil
        }
        for words in alternatives {
            if let matchEnd = matchWords(words, in: scalars, at: start, end: end),
                matchEnd == end || !Self.isWord(scalars[matchEnd])
            {
                return start..<matchEnd
            }
        }
        return nil
    }

    private func matchWords(_ words: [[Unicode.Scalar]], in scalars: [Unicode.Scalar], at start: Int, end: Int) -> Int? {
        var index = start
        for (position, word) in words.enumerated() {
            if position > 0 {
                let spaceStart = index
                while index < end && Self.isSpace(scalars[index]) { index += 1 }
                guard index > spaceStart else { return nil }
            }
            for scalar in word {
                guard index < end, Self.equalIgnoringCase(scalars[index], scalar) else { return nil }
                index += 1
            }
        }
        return index
    }

    // MARK: - Python `str` character classes

    /// Python `\w`: `str.isalnum()` or `_`.
    static func isWord(_ scalar: Unicode.Scalar) -> Bool {
        if scalar == "_" { return true }
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter:
            return true
        default:
            return scalar.properties.numericType != nil
        }
    }

    /// Python `\s` and `str.isspace()`.
    static func isSpace(_ scalar: Unicode.Scalar) -> Bool {
        scalar.properties.isWhitespace || (0x1C...0x1F).contains(scalar.value)
    }

    /// Python `re.IGNORECASE` for one pattern scalar against one text scalar,
    /// including the non-ASCII letters Python folds onto ASCII `i`, `k`, `s`.
    static func equalIgnoringCase(_ text: Unicode.Scalar, _ pattern: Unicode.Scalar) -> Bool {
        if text == pattern { return true }
        let folded = simpleLowercase(pattern)
        switch text.value {
        case 0x130, 0x131: return folded == "i"
        case 0x17F: return folded == "s"
        case 0x212A: return folded == "k"
        default: return simpleLowercase(text) == folded
        }
    }

    private static func simpleLowercase(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        let lowered = scalar.properties.lowercaseMapping.unicodeScalars
        return lowered.count == 1 ? lowered.first ?? scalar : scalar
    }

    /// Python `str.casefold()` for the characters cue matching can produce.
    static func casefolded(_ scalars: [Unicode.Scalar]) -> [Unicode.Scalar] {
        var result: [Unicode.Scalar] = []
        result.reserveCapacity(scalars.count)
        for scalar in scalars {
            switch scalar.value {
            case 0x17F: result.append("s")
            case 0x212A: result.append("k")
            case 0xDF: result.append(contentsOf: ["s", "s"])
            default: result.append(contentsOf: scalar.properties.lowercaseMapping.unicodeScalars)
            }
        }
        return result
    }

    /// Python `" ".join(text.casefold().split())`.
    static func normalizedCue(_ scalars: [Unicode.Scalar]) -> String {
        casefolded(scalars)
            .split(whereSeparator: isSpace)
            .map { String(String.UnicodeScalarView($0)) }
            .joined(separator: " ")
    }

    /// Whether `needle` occurs in `haystack`.
    static func contains(_ haystack: [Unicode.Scalar], _ needle: [Unicode.Scalar]) -> Bool {
        guard !needle.isEmpty else { return true }
        guard needle.count <= haystack.count else { return false }
        for start in 0...(haystack.count - needle.count) where haystack[start] == needle[0] {
            if haystack[start..<(start + needle.count)].elementsEqual(needle) { return true }
        }
        return false
    }
}
