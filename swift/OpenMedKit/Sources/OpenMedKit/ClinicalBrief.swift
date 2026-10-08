import CoreFoundation
import CryptoKit
import Foundation

/// Value-free failures from the on-device brief boundary.
public enum ClinicalBriefError: String, Error, LocalizedError, Sendable {
    case reviewRequired = "review_required"
    case modelUnavailable = "model_unavailable"
    case invalidPacket = "invalid_packet"
    case privacy = "privacy"
    case unsupportedClaim = "unsupported_claim"

    public var errorDescription: String? { rawValue }
}

// Native mirror of the existing script_detect.py defenses, using the same
// curated Unicode UTS #39 confusables 17.0.0 (Unicode-3.0) mappings.
// Source: https://www.unicode.org/Public/17.0.0/security/confusables.txt
// Ported 2026-10-08; shared fixtures cover every retained mapping.
enum ClinicalBriefLeakageMatcher {
    private static let confusableCharacters = Array("ΑΒΕΗΙΚΜΝΟΡΤΧαεηικμορτυχАВЕКМНОРСТХаеорсхі〇".unicodeScalars)
    private static let confusableValues = Array("ABEHIKMNOPTXaenikuoptuxABEKMHOPCTXaeopcxiO".unicodeScalars)
    private static let zeroWidth: Set<UInt32> = [0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF]
    private static let unspacedRanges: [ClosedRange<UInt32>] = [0x3400...0x4DBF, 0x4E00...0x9FFF, 0xF900...0xFAFF, 0x20000...0x2A6DF, 0x2A700...0x2B73F, 0x2B740...0x2B81F, 0x2B820...0x2CEAF, 0x2CEB0...0x2EBEF, 0x30000...0x3134F, 0x31350...0x323AF, 0x3040...0x309F, 0x30A0...0x30FF, 0x31F0...0x31FF, 0x1B000...0x1B16F, 0xFF65...0xFF9F, 0xE00...0xE7F]
    private static let hangulRanges: [ClosedRange<UInt32>] = [0x1100...0x11FF, 0x3130...0x318F, 0xA960...0xA97F, 0xAC00...0xD7AF, 0xD7B0...0xD7FF]
    private static let particles = ["은", "는", "이", "가", "을", "를", "의", "에", "에서", "에게", "에게서", "께", "께서", "한테", "한테서", "와", "과", "랑", "이랑", "하고", "도", "만", "부터", "까지", "보다", "처럼", "으로", "로", "으로서", "로서", "으로써", "로써", "이라고", "라고", "이나", "나", "든지", "이든지", "조차", "마저", "밖에", "뿐", "님", "씨"]
    private static let indicDigitBases: [UInt32] = [0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66]

    static func normalize(_ text: String) -> String {
        let visible = String(String.UnicodeScalarView(text.unicodeScalars.filter { !zeroWidth.contains($0.value) }))
        let scalars = Array(visible.decomposedStringWithCanonicalMapping.unicodeScalars)
        var result = ""
        for (index, scalar) in scalars.enumerated() {
            let code = scalar.value
            if scalar.properties.generalCategory == .nonspacingMark {
                let previous = index > 0 ? scalars[index - 1] : nil
                let attachedIndic = isIndic(code) && previous.map { isIndic($0.value) && isLetterOrMark($0) } == true
                let attachedEthiopic = isEthiopic(code) && previous.map { isEthiopic($0.value) && isLetterOrMark($0) } == true
                if !attachedIndic && !attachedEthiopic { continue }
            }
            if let folded = confusableCharacters.firstIndex(of: scalar) {
                result.unicodeScalars.append(confusableValues[folded])
            } else if (0xFF01...0xFF5E).contains(code) {
                result.unicodeScalars.append(Unicode.Scalar(code - 0xFEE0)!)
            } else if code == 0x3000 {
                result.append(" ")
            } else if let base = indicDigitBases.first(where: { ($0...($0 + 9)).contains(code) }) {
                result.unicodeScalars.append(Unicode.Scalar(0x30 + code - base)!)
            } else {
                result.unicodeScalars.append(scalar)
            }
        }
        // Preserve the prior Python IGNORECASE i-family equivalence separately
        // from the detector's curated confusable inventory.
        return result.precomposedStringWithCanonicalMapping
            .folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
            .replacingOccurrences(of: "\u{0131}", with: "i")
    }

    static func contains(_ identifier: String, in normalizedCandidate: String) throws -> Bool {
        // Retain native punctuation-component protection for compound names
        // and identifiers. Python's existing contract uses whitespace parts.
        let components = identifier.components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count >= 3 }
        let rawParts = [identifier] + whitespaceParts(identifier) + components
        for part in rawParts {
            let words = whitespaceParts(normalize(part))
            if words.isEmpty { continue }
            let literal = words.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "[\\s\\x{001c}-\\x{001f}]+")
            let unspaced = part.unicodeScalars.contains { scalar in
                scalar.value == 0x3007 || (isLetterOrMark(scalar) && unspacedRanges.contains { $0.contains(scalar.value) })
            }
            let otherUnspaced = part.unicodeScalars.contains { scalar in
                let code = scalar.value
                return (0x0E80...0x0EFF).contains(code) || (0x1780...0x17FF).contains(code)
                    || (0x1000...0x109F).contains(code) || (0xA9E0...0xA9FF).contains(code)
                    || (0xAA60...0xAA7F).contains(code)
            }
            let hangul = part.unicodeScalars.contains { scalar in
                isLetterOrMark(scalar) && hangulRanges.contains { $0.contains(scalar.value) }
            }
            let word = "[\\p{L}\\p{N}_]"
            let pattern: String
            if unspaced || otherUnspaced {
                pattern = literal
            } else if hangul {
                let suffix = "([\\x{1100}-\\x{11ff}\\x{3130}-\\x{318f}\\x{a960}-\\x{a97f}\\x{ac00}-\\x{d7af}\\x{d7b0}-\\x{d7ff}]*)"
                pattern = "(?<!" + word + ")" + literal + suffix + "(?!" + word + ")"
            } else {
                pattern = "(?<!" + word + ")" + literal + "(?!" + word + ")"
            }
            guard let expression = try? NSRegularExpression(pattern: pattern) else { throw ClinicalBriefError.invalidPacket }
            let range = NSRange(normalizedCandidate.startIndex..<normalizedCandidate.endIndex, in: normalizedCandidate)
            for match in expression.matches(in: normalizedCandidate, range: range) {
                if hangul && !unspaced && !otherUnspaced {
                    guard let suffixRange = Range(match.range(at: 1), in: normalizedCandidate) else { continue }
                    if !allowedHangulSuffix(String(normalizedCandidate[suffixRange])) { continue }
                }
                return true
            }
        }
        return false
    }

    // Segment suffixes without a repeated ambiguous regex alternative.
    private static func allowedHangulSuffix(_ suffix: String) -> Bool {
        let scalars = Array(suffix.unicodeScalars)
        var reachable = Array(repeating: false, count: scalars.count + 1)
        reachable[0] = true
        for index in scalars.indices where reachable[index] {
            for particle in particles {
                let value = Array(particle.unicodeScalars)
                let end = index + value.count
                if end <= scalars.count && scalars[index..<end].elementsEqual(value) { reachable[end] = true }
            }
        }
        return reachable[scalars.count]
    }

    private static func whitespaceParts(_ text: String) -> [String] {
        text.unicodeScalars.split { CharacterSet.whitespacesAndNewlines.contains($0) || (0x1C...0x1F).contains($0.value) }
            .map { String(String.UnicodeScalarView($0)) }
    }

    private static func isLetterOrMark(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
            .nonspacingMark, .spacingMark, .enclosingMark:
            return true
        default: return false
        }
    }

    private static func isIndic(_ code: UInt32) -> Bool { (0x0900...0x0D7F).contains(code) }
    private static func isEthiopic(_ code: UInt32) -> Bool {
        (0x1200...0x137F).contains(code) || (0x1380...0x139F).contains(code)
            || (0x2D80...0x2DDF).contains(code) || (0xAB00...0xAB2F).contains(code)
            || (0x1E7E0...0x1E7FF).contains(code)
    }
}

/// One source/output Unicode-scalar citation in the shared Python wire schema.
public struct ClinicalBriefCitation: Codable, Sendable, Equatable {
    public let claimIndex: Int
    public let sourceStart: Int
    public let sourceEnd: Int
    public let outputStart: Int
    public let outputEnd: Int

    enum CodingKeys: String, CodingKey {
        case claimIndex = "claim_index"
        case sourceStart = "source_start"
        case sourceEnd = "source_end"
        case outputStart = "output_start"
        case outputEnd = "output_end"
    }
}

/// One outcome emitted by the configured local evidence verifier.
public struct ClinicalBriefVerdict: Codable, Sendable, Equatable {
    public let claimIndex: Int
    public let label: String

    enum CodingKeys: String, CodingKey {
        case claimIndex = "claim_index"
        case label
    }
}

/// A fully verified local evaluation packet, not an unguarded model answer.
///
/// Applications supply an on-device verifier for the complete evidence/NLI
/// pipeline. This boundary independently checks leakage, the safety envelope,
/// citations and content digests. It never mints review history or NLI scores.
/// Protected summary text is omitted from debug descriptions and audit JSON.
public struct ClinicalBrief: Sendable, CustomStringConvertible {
    public let summary: String
    public let citations: [ClinicalBriefCitation]
    public let verdicts: [ClinicalBriefVerdict]
    public let digest: String
    public let refusalReason: String?
    private let response: Data
    private let audit: Data

    public var description: String { "ClinicalBrief(citations: \(citations.count), requiresReview: true)" }

    /// Explicit protected output. Never log this data.
    public func responseJSON() -> Data { response }

    /// Value-free audit packet; generated/source text is excluded.
    public func auditJSON() -> Data { audit }

    /// Verify a packet returned by trusted local evidence/NLI application code.
    ///
    /// `privacyCheck` must scan the complete rendered response, not only the
    /// summary. No server, remote model or cloud fallback is called here.
    public static func validate(
        evaluationJSON: Data,
        source: String,
        generatedSummary: String,
        originalIdentifiers: [String],
        privacyCheck: (String) throws -> Bool
    ) throws -> ClinicalBrief {
        guard evaluationJSON.count <= 1_048_576, originalIdentifiers.count <= 1024,
            originalIdentifiers.reduce(0, { $0 + $1.utf8.count }) <= 16_384,
            source.utf8.count <= 16_384, generatedSummary.utf8.count <= 16_384,
            var payload = try? JSONSerialization.jsonObject(with: evaluationJSON) as? [String: Any],
            let summary = payload["summary"] as? String,
            summary == generatedSummary, !summary.isEmpty,
            payload["schema_version"] as? Int == 1,
            payload["status"] as? String == "needs_review",
            payload["refusal_reason"] is NSNull,
            let envelope = payload["envelope"] as? [String: Any],
            envelope["status"] as? String == "ready",
            envelope["requires_human_review"] as? Bool == true,
            envelope["human_review_mode"] as? Bool == true,
            envelope["is_diagnostic"] as? Bool == false,
            let provenance = envelope["provenance"] as? [String: Any],
            provenance["content_hash"] as? String == hash(Data(source.utf8)),
            let recordedDigest = payload["digest"] as? String,
            let summaryDigest = payload["summary_digest"] as? String,
            payload["summary_characters"] as? Int == summary.unicodeScalars.count,
            let citationRows = payload["citations"] as? [[String: Any]],
            let verdicts = payload["verdicts"] as? [[String: Any]],
            !citationRows.isEmpty, citationRows.count <= 64,
            verdicts.count == citationRows.count
        else { throw ClinicalBriefError.invalidPacket }

        let normalizedSummary = ClinicalBriefLeakageMatcher.normalize(summary)
        for identifier in originalIdentifiers {
            if try ClinicalBriefLeakageMatcher.contains(identifier, in: normalizedSummary) {
                throw ClinicalBriefError.privacy
            }
        }
        guard summaryDigest == hash(try canonical(summary)) else { throw ClinicalBriefError.invalidPacket }
        payload.removeValue(forKey: "summary")
        payload.removeValue(forKey: "digest")
        guard recordedDigest == hash(try canonical(payload)) else { throw ClinicalBriefError.invalidPacket }

        guard let citationData = try? JSONSerialization.data(withJSONObject: citationRows),
            let citations = try? JSONDecoder().decode([ClinicalBriefCitation].self, from: citationData)
        else { throw ClinicalBriefError.invalidPacket }
        let input = Array(source.unicodeScalars)
        let output = Array(summary.unicodeScalars)
        var end = 0
        for (index, citation) in citations.enumerated() {
            guard citation.claimIndex == index,
                verdicts[index]["claim_index"] as? Int == index,
                verdicts[index]["label"] as? String == "entailment",
                citation.sourceStart >= 0, citation.sourceEnd <= input.count,
                citation.sourceStart < citation.sourceEnd,
                citation.outputStart >= end, citation.outputEnd <= output.count,
                citation.outputStart < citation.outputEnd,
                input[citation.sourceStart..<citation.sourceEnd].elementsEqual(output[citation.outputStart..<citation.outputEnd]),
                output[end..<citation.outputStart].allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) })
            else { throw ClinicalBriefError.unsupportedClaim }
            end = citation.outputEnd
        }
        guard output[end...].allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }) else {
            throw ClinicalBriefError.unsupportedClaim
        }
        guard let rendered = String(data: evaluationJSON, encoding: .utf8) else { throw ClinicalBriefError.invalidPacket }
        var clean = false
        do { clean = try privacyCheck(rendered) } catch { throw ClinicalBriefError.privacy }
        guard clean else { throw ClinicalBriefError.privacy }
        payload["digest"] = recordedDigest
        let typedVerdicts = citations.map { ClinicalBriefVerdict(claimIndex: $0.claimIndex, label: "entailment") }
        return ClinicalBrief(
            summary: summary, citations: citations, verdicts: typedVerdicts, digest: recordedDigest,
            refusalReason: nil, response: evaluationJSON, audit: try canonical(payload))
    }

    private static func hash(_ data: Data) -> String {
        "sha256:" + SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Python's sorted, compact, ASCII JSON convention for the shared digest.
    private static func canonical(_ value: Any, depth: Int = 0) throws -> Data {
        guard depth < 64 else { throw ClinicalBriefError.invalidPacket }
        func render(_ item: Any) throws -> String {
            String(decoding: try canonical(item, depth: depth + 1), as: UTF8.self)
        }
        let text: String
        if value is NSNull {
            text = "null"
        } else if let string = value as? String {
            var escaped = "\""
            for scalar in string.unicodeScalars {
                switch scalar.value {
                case 34: escaped += "\\\""
                case 92: escaped += "\\\\"
                case 8: escaped += "\\b"
                case 9: escaped += "\\t"
                case 10: escaped += "\\n"
                case 12: escaped += "\\f"
                case 13: escaped += "\\r"
                case 32...126: escaped.unicodeScalars.append(scalar)
                case 0...65535: escaped += String(format: "\\u%04x", scalar.value)
                default:
                    let code = scalar.value - 65536
                    escaped += String(format: "\\u%04x\\u%04x", 0xD800 + (code >> 10), 0xDC00 + (code & 1023))
                }
            }
            text = escaped + "\""
        } else if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                text = number.boolValue ? "true" : "false"
            } else {
                guard number.doubleValue.isFinite else { throw ClinicalBriefError.invalidPacket }
                text = String(cString: number.objCType) == "d" ? String(number.doubleValue) : number.stringValue
            }
        } else if let array = value as? [Any] {
            text = "[" + (try array.map(render)).joined(separator: ",") + "]"
        } else if let object = value as? [String: Any] {
            text = "{" + (try object.keys.sorted().map { try render($0) + ":" + render(object[$0]!) }).joined(separator: ",") + "}"
        } else {
            throw ClinicalBriefError.invalidPacket
        }
        return Data(text.utf8)
    }
}
