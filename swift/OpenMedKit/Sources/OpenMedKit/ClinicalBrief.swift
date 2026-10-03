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

        let folded = summary.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))
        for identifier in originalIdentifiers {
            let tokens = identifier.components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count >= 3 }
            for token in tokens {
                if folded.contains(token.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX"))) {
                    throw ClinicalBriefError.privacy
                }
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
