import Foundation

/// Controlled failures; no transcript or backend exception is interpolated.
public enum TranscriptLanguageError: String, Error, LocalizedError {
    case invalidLanguageTag = "invalid_language_tag"
    case invalidConfidence = "invalid_confidence"
    case invalidConfidenceThreshold = "invalid_confidence_threshold"
    case invalidInstalledDetectors = "invalid_installed_detectors"
    case invalidTextIdentifier = "invalid_text_identifier"
    case invalidSegmentIndex = "invalid_segment_index"
    case invalidTranscript = "invalid_transcript"
    case invalidDetectorSpan = "invalid_detector_span"
    case reviewRequired = "review_required"
    case segmentWithheld = "segment_withheld"

    public var errorDescription: String? { rawValue }
}

/// A fixed provider or local text-side hypothesis; this type performs no ASR.
public struct TranscriptLanguageHypothesis: Sendable {
    public let tag: String
    public let confidence: Double

    /// Accept a bounded language[-Script][-REGION] BCP 47 tag and probability.
    public init(tag: String, confidence: Double) throws {
        self.tag = try transcriptTag(tag)
        guard confidence.isFinite, (0...1).contains(confidence) else {
            throw TranscriptLanguageError.invalidConfidence
        }
        self.confidence = confidence
    }
}

/// Half-open Unicode-scalar offsets from an installed local PHI detector.
public struct TranscriptPHISpan: Sendable, Equatable {
    public let start: Int
    public let end: Int

    /// Construct offset-only evidence without retaining any PHI value.
    public init(start: Int, end: Int) throws {
        guard start >= 0, end > start else { throw TranscriptLanguageError.invalidDetectorSpan }
        self.start = start
        self.end = end
    }
}

/// Routing metadata in original source Unicode-scalar coordinates.
public struct TranscriptLanguageRun: Codable, Sendable, Equatable {
    public let start: Int
    public let end: Int
    public let tag: String
    public let confidenceBucket: String

    enum CodingKeys: String, CodingKey {
        case start, end, tag
        case confidenceBucket = "confidence_bucket"
    }
}

/// Content-free per-segment decision, matching the Python diagnostic shape.
public struct TranscriptLanguageAudit: Codable, Sendable {
    public let segmentIndex: Int
    public let status: String
    public let reasonCode: String
    public let providerTag: String?
    public let providerConfidenceBucket: String
    public let runs: [TranscriptLanguageRun]
    public let phiSpanCount: Int

    enum CodingKeys: String, CodingKey {
        case segmentIndex = "segment_index"
        case status
        case reasonCode = "reason_code"
        case providerTag = "provider_tag"
        case providerConfidenceBucket = "provider_confidence_bucket"
        case runs
        case phiSpanCount = "phi_span_count"
    }

    /// Encode missing provider evidence as JSON null, matching Python.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(segmentIndex, forKey: .segmentIndex)
        try container.encode(status, forKey: .status)
        try container.encode(reasonCode, forKey: .reasonCode)
        try container.encode(providerTag, forKey: .providerTag)
        try container.encode(providerConfidenceBucket, forKey: .providerConfidenceBucket)
        try container.encode(runs, forKey: .runs)
        try container.encode(phiSpanCount, forKey: .phiSpanCount)
    }
}

/// Language coverage does not prove clinical validity or detector recall.
public struct TranscriptLanguageDecision: Sendable, CustomStringConvertible {
    public let audit: TranscriptLanguageAudit
    public let phiSpans: [TranscriptPHISpan]
    public let notice = "Non-diagnostic transcript; explicit reviewer confirmation required."
    fileprivate let reviewText: String?

    public var description: String {
        "TranscriptLanguageDecision(segment: \(audit.segmentIndex), status: \(audit.status), reason: \(audit.reasonCode))"
    }

    /// The shared release/draft boundary requires confirmation of this output.
    public func reviewedText(reviewerConfirmed: Bool = false) throws -> String {
        guard let reviewText else { throw TranscriptLanguageError.segmentWithheld }
        guard reviewerConfirmed else { throw TranscriptLanguageError.reviewRequired }
        return reviewText
    }
}

/// Combine fixed spoken-language evidence with an explicitly local token LID.
///
/// Applications supply metadata for actually installed primary-language PHI
/// packs and one offline detector per code. Catalog availability is insufficient.
/// No models, translation, networking or cloud fallback are invoked implicitly.
public final class TranscriptLanguageRouter {
    public typealias Detector = (String) throws -> [TranscriptPHISpan]
    public typealias TextIdentifier = (String, [String]) throws -> TranscriptLanguageHypothesis?

    private let detectors: [String: Detector]
    private let identifier: TextIdentifier
    private let candidates: [String]
    private let threshold: Double

    /// Snapshot installed metadata; candidates must include uninstalled languages.
    public init(
        installedPackCodes: [String], detectors: [String: Detector],
        candidateLanguages: [String], confidenceThreshold: Double = 0.8,
        textIdentifier: @escaping TextIdentifier
    ) throws {
        guard confidenceThreshold.isFinite, (0...1).contains(confidenceThreshold) else {
            throw TranscriptLanguageError.invalidConfidence
        }
        guard confidenceThreshold > 0 else { throw TranscriptLanguageError.invalidConfidenceThreshold }
        guard Set(installedPackCodes).count == installedPackCodes.count,
            Set(installedPackCodes) == Set(detectors.keys),
            try installedPackCodes.allSatisfy({ try transcriptTag($0) == $0 && !$0.contains("-") })
        else { throw TranscriptLanguageError.invalidInstalledDetectors }
        let candidates = try Set(candidateLanguages.map { primary(try transcriptTag($0)) }).sorted()
        guard !candidates.isEmpty else { throw TranscriptLanguageError.invalidTextIdentifier }
        self.detectors = detectors
        self.identifier = textIdentifier
        self.candidates = candidates
        self.threshold = confidenceThreshold
    }

    /// Route finalized protected text atomically; failures never enter drafts.
    /// All offsets count Unicode scalars, matching Python code-point offsets.
    public func route(
        segmentIndex: Int, text: String, providerLanguage: TranscriptLanguageHypothesis?,
        finalized: Bool = true
    ) throws -> TranscriptLanguageDecision {
        guard segmentIndex >= 0 else { throw TranscriptLanguageError.invalidSegmentIndex }
        let source = Array(text.unicodeScalars)
        guard source.count <= 65_536 else { throw TranscriptLanguageError.invalidTranscript }
        let providerTag = providerLanguage?.tag
        let providerBucket = bucket(providerLanguage?.confidence)
        func decision(
            _ status: String, _ reason: String, runs: [TranscriptLanguageRun] = [],
            spans: [TranscriptPHISpan] = [], output: String? = nil
        ) -> TranscriptLanguageDecision {
            TranscriptLanguageDecision(
                audit: TranscriptLanguageAudit(
                    segmentIndex: segmentIndex, status: status, reasonCode: reason,
                    providerTag: providerTag, providerConfidenceBucket: providerBucket,
                    runs: runs, phiSpanCount: spans.count),
                phiSpans: spans, reviewText: output)
        }
        guard finalized else { return decision("uncertain", "segment_not_finalized") }
        guard let providerLanguage else { return decision("uncertain", "provider_language_missing") }
        guard providerLanguage.confidence >= threshold else {
            return decision("uncertain", "provider_confidence_low")
        }
        let runs: [TranscriptLanguageRun]
        let languages: [String: Int]
        do {
            (runs, languages) = try textRuns(text, providerTag: providerLanguage.tag)
        } catch {
            return decision("uncertain", "text_identifier_failed")
        }
        guard !runs.isEmpty, let maximum = languages.values.max() else {
            return decision("uncertain", "text_language_uncertain")
        }
        guard !runs.contains(where: { ["low", "missing"].contains($0.confidenceBucket) }) else {
            return decision("uncertain", "text_confidence_low", runs: runs)
        }
        guard languages[primary(providerLanguage.tag)] == maximum else {
            return decision("uncertain", "language_disagreement", runs: runs)
        }
        guard runs.allSatisfy({ detectors[primary($0.tag)] != nil }) else {
            return decision("unsupported", "phi_pack_unavailable", runs: runs)
        }
        var spans: [TranscriptPHISpan] = []
        do {
            for run in runs {
                let value = String(String.UnicodeScalarView(source[run.start..<run.end]))
                for span in try detectors[primary(run.tag)]!(value) {
                    guard span.end <= run.end - run.start else {
                        return decision("uncertain", "detector_result_invalid", runs: runs)
                    }
                    spans.append(try TranscriptPHISpan(start: run.start + span.start, end: run.start + span.end))
                }
            }
        } catch {
            return decision("uncertain", "detector_failed", runs: runs)
        }
        var merged: [TranscriptPHISpan] = []
        for span in spans.sorted(by: { ($0.start, $0.end) < ($1.start, $1.end) }) {
            if let previous = merged.last, span.start <= previous.end {
                merged.removeLast()
                merged.append(try TranscriptPHISpan(start: previous.start, end: max(previous.end, span.end)))
            } else {
                merged.append(span)
            }
        }
        var output = source
        for span in merged {
            output.replaceSubrange(span.start..<span.end, with: repeatElement(Unicode.Scalar(0x2588)!, count: span.end - span.start))
        }
        return decision(
            languages.count > 1 ? "mixed" : "supported", "language_routed", runs: runs,
            spans: merged, output: String(String.UnicodeScalarView(output)))
    }

    private func bucket(_ value: Double?) -> String {
        guard let value else { return "missing" }
        if value < threshold { return "low" }
        return value >= 0.9 ? "high" : "accepted"
    }

    private func textRuns(_ text: String, providerTag: String) throws -> ([TranscriptLanguageRun], [String: Int]) {
        // Same token boundaries as the Python code-mix tokenizer for Latin
        // text. Neutral tokens, whitespace and leading prefixes are retained.
        let tokenizer = try NSRegularExpression(pattern: "\\d+(?:[./:-]\\d+)*|[\\u0900-\\u097f]+|[\\p{L}\\p{Nl}\\p{No}]+(?:[’'-][\\p{L}\\p{Nl}\\p{No}]+)*|\\S")
        let options = Array(Set(candidates + [primary(providerTag)])).sorted()
        var evidence: [(Int, String, Double)] = []
        var counts: [String: Int] = [:]
        for match in tokenizer.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text) else { throw TranscriptLanguageError.invalidTranscript }
            let surface = String(text[range])
            guard surface.unicodeScalars.contains(where: { CharacterSet.letters.contains($0) }) else { continue }
            guard let prediction = try identifier(surface, options) else { return ([], counts) }
            counts[primary(prediction.tag), default: 0] += 1
            evidence.append((text[..<range.lowerBound].unicodeScalars.count, prediction.tag, prediction.confidence))
        }
        guard let first = evidence.first else { return ([], counts) }
        var runs: [TranscriptLanguageRun] = []
        var start = 0
        var tag = first.1
        var confidence = first.2
        for (boundary, nextTag, nextConfidence) in evidence.dropFirst() {
            if nextTag != tag {
                runs.append(TranscriptLanguageRun(start: start, end: boundary, tag: tag, confidenceBucket: bucket(confidence)))
                (start, tag, confidence) = (boundary, nextTag, nextConfidence)
            } else {
                confidence = min(confidence, nextConfidence)
            }
        }
        runs.append(TranscriptLanguageRun(start: start, end: text.unicodeScalars.count, tag: tag, confidenceBucket: bucket(confidence)))
        return (runs, counts)
    }
}

/// Counts-only evidence; no transcript, identifier values or segment IDs.
public func transcriptLanguageReport(_ decisions: [TranscriptLanguageDecision]) -> [String: [String: Int]] {
    var statuses: [String: Int] = [:]
    var reasons: [String: Int] = [:]
    var tags: [String: Int] = [:]
    var providerTags: [String: Int] = [:]
    var buckets: [String: Int] = [:]
    for decision in decisions {
        statuses[decision.audit.status, default: 0] += 1
        reasons[decision.audit.reasonCode, default: 0] += 1
        buckets[decision.audit.providerConfidenceBucket, default: 0] += 1
        if let tag = decision.audit.providerTag { providerTags[tag, default: 0] += 1 }
        for run in decision.audit.runs { tags[run.tag, default: 0] += 1 }
    }
    return ["status_counts": statuses, "reason_counts": reasons, "tag_counts": tags, "provider_tag_counts": providerTags, "confidence_bucket_counts": buckets]
}

private func primary(_ tag: String) -> String { String(tag.split(separator: "-")[0]) }

private func transcriptTag(_ value: String) throws -> String {
    let regex = try NSRegularExpression(pattern: "\\A([a-zA-Z]{2})(?:-([a-zA-Z]{4}))?(?:-([a-zA-Z]{2}|[0-9]{3}))?\\z")
    guard let match = regex.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)) else {
        throw TranscriptLanguageError.invalidLanguageTag
    }
    func part(_ index: Int) -> String? {
        guard let range = Range(match.range(at: index), in: value) else { return nil }
        return String(value[range])
    }
    let scripts = Set("Arab Armn Beng Cyrl Deva Ethi Geor Grek Gujr Guru Hang Hani Hans Hant Hebr Jpan Kana Khmr Knda Kore Laoo Latn Mlym Mymr Orya Sinh Taml Telu Thaa Thai Tibt".split(separator: " ").map(String.init))
    if let script = part(2), !scripts.contains(script.capitalized) {
        throw TranscriptLanguageError.invalidLanguageTag
    }
    return [part(1)?.lowercased(), part(2)?.capitalized, part(3)?.uppercased()].compactMap { $0 }.joined(separator: "-")
}
