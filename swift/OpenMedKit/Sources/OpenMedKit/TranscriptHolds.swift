import CryptoKit
import Foundation

/// Controlled, value-free failures in the fixed-evidence token review gate.
public enum TranscriptHoldError: String, Error {
    case invalidIdentity = "invalid_identity"
    case invalidToken = "invalid_token"
    case invalidConfidence = "invalid_confidence"
    case missingCitations = "missing_citations"
    case duplicateCitation = "duplicate_citation"
    case duplicateIdentity = "duplicate_identity"
    case unknownCitation = "unknown_citation"
    case unknownDoseToken = "unknown_dose_token"
    case unknownStatement = "unknown_statement"
    case staleConfirmation = "stale_confirmation"
    case invalidConfirmation = "invalid_confirmation"
    case reviewerDenied = "reviewer_denied"
    case unresolvedTokenHold = "unresolved_token_hold"
    case reviewRequired = "review_required"
}

/// Deterministic English critical-token categories. Medication words are caller supplied.
public enum CriticalTokenClass: String, Codable, CaseIterable, Hashable {
    case number, unit, dose, negation, uncertainty, laterality, medication, modifier

    /// Policy v1 requires 0.95, except uncertainty cues require 0.90.
    public var confidenceThreshold: Double { self == .uncertainty ? 0.90 : 0.95 }
}

/// Opaque numeric segment/token references, never patient or source strings.
public struct TranscriptTokenIdentity: Hashable, Comparable {
    public let segmentID: Int
    public let tokenID: Int

    public init(segmentID: Int, tokenID: Int) throws {
        guard segmentID >= 0, tokenID >= 0 else { throw TranscriptHoldError.invalidIdentity }
        self.segmentID = segmentID
        self.tokenID = tokenID
    }

    public static func < (lhs: Self, rhs: Self) -> Bool {
        lhs.segmentID == rhs.segmentID ? lhs.tokenID < rhs.tokenID : lhs.segmentID < rhs.segmentID
    }
}

/// Private fixed token evidence. Empty alternatives represent possible dropped tokens.
public struct FixedTranscriptToken: CustomStringConvertible, CustomDebugStringConvertible {
    public let identity: TranscriptTokenIdentity
    public let text: String
    public let confidence: Double?
    public let alternatives: [String]
    public var description: String { "FixedTranscriptToken()" }
    public var debugDescription: String { description }

    public init(
        identity: TranscriptTokenIdentity, text: String, confidence: Double?, alternatives: [String] = []
    ) throws {
        if let confidence {
            guard confidence.isFinite, (0...1).contains(confidence) else {
                throw TranscriptHoldError.invalidConfidence
            }
        }
        self.identity = identity
        self.text = text
        self.confidence = confidence
        self.alternatives = alternatives
    }
}

/// Fixed draft statement identity and its complete exact-token dependency set.
public struct DraftTokenCitation {
    public let statementID: Int
    public let tokens: [TranscriptTokenIdentity]

    public init(statementID: Int, tokens: [TranscriptTokenIdentity]) throws {
        guard statementID >= 0 else { throw TranscriptHoldError.invalidIdentity }
        guard !tokens.isEmpty else { throw TranscriptHoldError.missingCitations }
        guard Set(tokens).count == tokens.count else { throw TranscriptHoldError.duplicateCitation }
        self.statementID = statementID
        self.tokens = tokens
    }
}

/// Controlled outcomes supplied by the existing offline dosing checker adapter.
public enum TranscriptDoseCheckStatus: String {
    case inRange = "in_range"
    case flagged
    case notChecked = "not_checked"

    fileprivate var score: Double {
        switch self {
        case .inRange: return 0
        case .notChecked: return 0.5
        case .flagged: return 1
        }
    }
}

/// Value-free dosing evidence. OpenMedKit does not replace the Python dose-range checker.
public struct TokenDoseEvidence {
    public let identity: TranscriptTokenIdentity
    public let status: TranscriptDoseCheckStatus

    public init(identity: TranscriptTokenIdentity, status: TranscriptDoseCheckStatus) {
        self.identity = identity
        self.status = status
    }
}

/// Value-free hold record with matching Python JSON field names.
public struct TokenHoldRecord: Encodable, Equatable {
    public let identity: TranscriptTokenIdentity
    public let classes: [CriticalTokenClass]
    public let confidence: Double?
    public let threshold: Double
    public let disagreementScore: Int
    public let doseFlagScore: Double
    public let policyVersion = 1

    enum CodingKeys: String, CodingKey {
        case segmentID = "segment_id"
        case tokenID = "token_id"
        case classes, confidence, threshold
        case disagreementScore = "disagreement_score"
        case doseFlagScore = "dose_flag_score"
        case policyVersion = "policy_version"
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(identity.segmentID, forKey: .segmentID)
        try container.encode(identity.tokenID, forKey: .tokenID)
        try container.encode(classes, forKey: .classes)
        try container.encode(confidence, forKey: .confidence)
        try container.encode(threshold, forKey: .threshold)
        try container.encode(disagreementScore, forKey: .disagreementScore)
        try container.encode(doseFlagScore, forKey: .doseFlagScore)
        try container.encode(policyVersion, forKey: .policyVersion)
    }
}

/// Confirmation from a separate correction workflow, bound to this exact snapshot.
public struct TranscriptHoldConfirmation {
    public let identity: TranscriptTokenIdentity
    public let evidenceDigest: String
    public let reviewerID: Int
    public let confirmed: Bool

    public init(
        identity: TranscriptTokenIdentity, evidenceDigest: String, reviewerID: Int, confirmed: Bool
    ) throws {
        guard reviewerID >= 0 else { throw TranscriptHoldError.invalidIdentity }
        guard evidenceDigest.count == 64,
            evidenceDigest.allSatisfy({ "0123456789abcdef".contains($0) })
        else { throw TranscriptHoldError.invalidConfirmation }
        self.identity = identity
        self.evidenceDigest = evidenceDigest
        self.reviewerID = reviewerID
        self.confirmed = confirmed
    }
}

/// Consequential export permit; contains no note text and requires current evidence binding.
public struct TranscriptReviewedPermit {
    public let statementID: Int
    public let revision: Int
    public let evidenceDigest: String
    public let notice: String
}

/// Local hold classification and propagation over fixed evidence, without ASR or note assembly.
public final class TranscriptHoldGate {
    public static let policyVersion = 1
    public static let notice =
        "Non-diagnostic transcript review aid. A clinician must independently verify "
        + "the cited evidence and confirm before consequential use. No doses or orders "
        + "are changed."

    public let holds: [TokenHoldRecord]
    public let evidenceDigest: String
    private let revision: Int
    private let statements: [Int: [TranscriptTokenIdentity]]
    private var resolved = Set<TranscriptTokenIdentity>()

    private static func words(_ text: String) -> Set<String> {
        Set(text.split(separator: " ").map(String.init))
    }
    private static let numbers = words(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
            + "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty "
            + "fifty sixty seventy eighty ninety hundred thousand million half quarter")
    private static let units = words(
        "mg g kg mcg ug µg μg ml l mmol meq iu unit units tablet tablets capsule "
            + "capsules mg/kg mg/ml mcg/kg percent %")
    private static let negations = words("no not never without denies denied neither nor absent")
    private static let uncertainty = words("possible possibly probable probably maybe uncertain suspected")
    private static let laterality = words("left right bilateral unilateral ipsilateral contralateral")
    private static let modifiers = words("hypo hyper hypoglycemia hyperglycemia hypotension hypertension")

    private static func normalize(_ text: String) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            .trimmingCharacters(in: CharacterSet(charactersIn: ";:!?"))
        while value.last == "." || value.last == "," { value.removeLast() }
        return value
    }

    /// Classify fixed English token text and caller-owned medication-token lexicons.
    public static func classify(_ text: String, medicationNames: [String] = []) -> [CriticalTokenClass] {
        let value = normalize(text)
        var result = Set<CriticalTokenClass>()
        let parts = value.split(separator: "-", omittingEmptySubsequences: false).map(String.init)
        if !value.isEmpty,
            parts.allSatisfy({ numbers.contains($0) })
                || value.range(of: #"^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?:/\d+)?%?$"#, options: .regularExpression) != nil
        {
            result.insert(.number)
        }
        let dosePattern = #"^([+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+))\s*([^\d\s]+)$"#
        if let regex = try? NSRegularExpression(pattern: dosePattern),
            let match = regex.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)),
            let range = Range(match.range(at: 2), in: value), units.contains(String(value[range]))
        {
            result.formUnion([.number, .unit, .dose])
        }
        if units.contains(value) { result.insert(.unit) }
        for (words, kind) in [
            (negations, CriticalTokenClass.negation), (uncertainty, .uncertainty),
            (laterality, .laterality), (modifiers, .modifier),
        ] {
            if words.contains(value) { result.insert(kind) }
        }
        if !value.isEmpty, Set(medicationNames.map(normalize)).contains(value) { result.insert(.medication) }
        return result.sorted { $0.rawValue < $1.rawValue }
    }

    /// Build a fixed finalized snapshot; reject duplicate identities and unknown dependencies.
    public init(
        tokens: [FixedTranscriptToken], statements: [DraftTokenCitation], revision: Int,
        medicationNames: [String] = [], doseEvidence: [TokenDoseEvidence] = []
    ) throws {
        guard revision >= 0 else { throw TranscriptHoldError.invalidIdentity }
        var tokenMap = [TranscriptTokenIdentity: FixedTranscriptToken]()
        var statementMap = [Int: [TranscriptTokenIdentity]]()
        for token in tokens {
            guard tokenMap[token.identity] == nil else { throw TranscriptHoldError.duplicateIdentity }
            tokenMap[token.identity] = token
        }
        for statement in statements {
            guard statementMap[statement.statementID] == nil else { throw TranscriptHoldError.duplicateIdentity }
            guard statement.tokens.allSatisfy({ tokenMap[$0] != nil }) else { throw TranscriptHoldError.unknownCitation }
            statementMap[statement.statementID] = statement.tokens
        }
        var doseScores = [TranscriptTokenIdentity: Double]()
        for finding in doseEvidence {
            guard tokenMap[finding.identity] != nil else { throw TranscriptHoldError.unknownDoseToken }
            doseScores[finding.identity] = max(doseScores[finding.identity] ?? 0, finding.status.score)
        }
        let medications = Set(medicationNames.map(Self.normalize)).sorted()
        var classes = [TranscriptTokenIdentity: Set<CriticalTokenClass>]()
        for token in tokens {
            classes[token.identity] = Set(
                ([token.text] + token.alternatives).flatMap {
                    Self.classify($0, medicationNames: medications)
                })
        }
        let ordered = tokenMap.keys.sorted()
        for (left, right) in zip(ordered, ordered.dropFirst()) {
            if left.segmentID == right.segmentID, right.tokenID > left.tokenID,
                right.tokenID - left.tokenID == 1,
                classes[left]!.contains(.number), classes[right]!.contains(.unit)
            {
                classes[left]!.insert(.dose)
                classes[right]!.insert(.dose)
            }
        }
        var records = [TokenHoldRecord]()
        for ref in ordered {
            let token = tokenMap[ref]!
            if doseScores[ref] != nil { classes[ref]!.insert(.dose) }
            let kinds = classes[ref]!.sorted { $0.rawValue < $1.rawValue }
            guard let threshold = kinds.map(\.confidenceThreshold).max() else { continue }
            let disagreement = token.alternatives.contains { Self.normalize($0) != Self.normalize(token.text) } ? 1 : 0
            let doseScore = doseScores[ref] ?? 0
            if token.confidence == nil || token.confidence! < threshold || disagreement != 0 || doseScore != 0 {
                records.append(
                    TokenHoldRecord(
                        identity: ref, classes: kinds, confidence: token.confidence,
                        threshold: threshold, disagreementScore: disagreement, doseFlagScore: doseScore))
            }
        }
        // Private payload exists only in memory. Digests are local snapshot identities,
        // not a wire format shared with the future correction ledger.
        let payload: [Any] = [
            Self.policyVersion, revision, medications,
            ordered.map { ref -> [Any] in
                let token = tokenMap[ref]!
                return [
                    ref.segmentID, ref.tokenID, token.text, token.confidence as Any? ?? NSNull(),
                    token.alternatives, classes[ref]!.map(\.rawValue).sorted(), doseScores[ref] as Any? ?? NSNull(),
                ]
            },
            statementMap.keys.sorted().map { key -> [Any] in
                [key, statementMap[key]!.map { [$0.segmentID, $0.tokenID] }]
            },
        ]
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .fragmentsAllowed])
        self.evidenceDigest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        self.holds = records
        self.revision = revision
        self.statements = statementMap
    }

    /// Return all unresolved holds cited by a statement.
    public func statementHolds(_ statementID: Int) throws -> [TokenHoldRecord] {
        guard let refs = statements[statementID] else { throw TranscriptHoldError.unknownStatement }
        return holds.filter { refs.contains($0.identity) && !resolved.contains($0.identity) }
    }

    /// Resolve via an explicitly confirmed, authorized current-evidence correction receipt.
    public func resolve(
        _ confirmation: TranscriptHoldConfirmation, authorize: (Int) throws -> Bool
    ) throws {
        guard confirmation.evidenceDigest == evidenceDigest else { throw TranscriptHoldError.staleConfirmation }
        guard confirmation.confirmed, holds.contains(where: { $0.identity == confirmation.identity }) else {
            throw TranscriptHoldError.invalidConfirmation
        }
        let allowed = (try? authorize(confirmation.reviewerID)) ?? false
        guard allowed else { throw TranscriptHoldError.reviewerDenied }
        resolved.insert(confirmation.identity)
    }

    /// Return a permit only after current holds clear and the clinician explicitly reviews.
    public func exportReviewed(_ statementID: Int, reviewerConfirmed: Bool) throws -> TranscriptReviewedPermit {
        guard try statementHolds(statementID).isEmpty else { throw TranscriptHoldError.unresolvedTokenHold }
        guard reviewerConfirmed else { throw TranscriptHoldError.reviewRequired }
        return TranscriptReviewedPermit(
            statementID: statementID, revision: revision,
            evidenceDigest: evidenceDigest, notice: Self.notice)
    }
}
