import CoreFoundation
import Foundation

/// Controlled, value-free admission refusals shared with Python.
public enum ReviewAdmissionRefusal: String, Error, LocalizedError, Sendable {
    case invalid = "invalid_reviewed_evidence"
    case missing = "review_receipt_missing"
    case expired = "review_receipt_expired"
    case mismatched = "review_receipt_mismatched"
    case revoked = "review_receipt_revoked"
    case authorityUnavailable = "review_authority_unavailable"
    case sourceUnavailable = "review_source_unavailable"
    case sourceChanged = "review_source_changed"
    case policyChanged = "review_policy_changed"

    public var errorDescription: String? { rawValue }
}

/// Only the independently held local registry can return current authority.
public enum ReviewAuthorityStatus: Sendable {
    case current, revoked, mismatched
}

/// Trusted on-device custody lookup; must authorize the caller's source access.
public protocol CurrentLocalSource: Sendable {
    func currentDigest(sourceID: String) throws -> String?
}

/// Trusted on-device review registry; compare the exact record and revocation.
/// A serialized receipt or an approval marker is never sufficient authority.
public protocol ReviewAuthorityVerifier: Sendable {
    func verify(_ receipt: LocalReviewReceipt, evidenceDigest: String, now: Int) throws -> ReviewAuthorityStatus
}

/// Public opaque receipt locator and bindings. Contains no credentials.
public struct LocalReviewReceipt: Codable, Sendable, Equatable {
    public let receiptID: String
    public let authorityID: String
    public let evidenceDigest: String
    public let issuedAt: Int
    public let expiresAt: Int

    enum CodingKeys: String, CodingKey {
        case receiptID = "receipt_id"
        case authorityID = "authority_id"
        case evidenceDigest = "evidence_digest"
        case issuedAt = "issued_at"
        case expiresAt = "expires_at"
    }
}

/// Half-open Unicode-scalar offsets into the digest-bound de-identified source.
public struct ReviewedLocalReference: Codable, Sendable, Equatable {
    public let referenceID: String
    public let start: Int
    public let end: Int

    enum CodingKeys: String, CodingKey {
        case referenceID = "reference_id"
        case start, end
    }
}

/// Separate v1 wire contract; parsing never grants authority to generate.
public struct ReviewedLocalEvidence: Codable, Sendable, Equatable {
    public let schemaVersion: Int
    public let kind: String
    public let provenanceClass: String
    public let offsetConvention: String
    public let sourceID: String
    public let sourceDigest: String
    public let sourceLength: Int
    public let policyDigest: String
    public let references: [ReviewedLocalReference]
    public let reviewReceipt: LocalReviewReceipt?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case provenanceClass = "provenance_class"
        case offsetConvention = "offset_convention"
        case sourceID = "source_id"
        case sourceDigest = "source_digest"
        case sourceLength = "source_length"
        case policyDigest = "policy_digest"
        case references
        case reviewReceipt = "review_receipt"
    }

    private func wireObject() throws -> [String: Any] {
        let data = try JSONEncoder().encode(self)
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ReviewAdmissionRefusal.invalid
        }
        // JSONEncoder omits a nil optional; the wire contract requires null.
        if reviewReceipt == nil { object["review_receipt"] = NSNull() }
        return object
    }

    /// Canonical, value-free JSON matching the Python wire representation.
    public func toJSON() throws -> Data {
        do {
            let data = try ClinicalBrief.canonical(wireObject())
            _ = try Self.fromJSON(data)
            return data
        } catch { throw ReviewAdmissionRefusal.invalid }
    }

    /// Digest binds version, provenance, source, policy and every offset.
    public var evidenceDigest: String {
        get throws {
            do {
                var object = try wireObject()
                object.removeValue(forKey: "review_receipt")
                return ClinicalBrief.hash(try ClinicalBrief.canonical(object))
            } catch { throw ReviewAdmissionRefusal.invalid }
        }
    }

    /// SHA-256 over exact UTF-8 source bytes, not the JSON string representation.
    public static func sourceDigest(_ source: String) -> String {
        ClinicalBrief.hash(Data(source.utf8))
    }

    private static func opaque(_ value: Any?, prefix: String) -> Bool {
        guard let string = value as? String else { return false }
        return string.range(of: "^" + prefix + ":[0-9a-f]{64}$", options: .regularExpression) != nil
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber,
            CFGetTypeID(number) != CFBooleanGetTypeID(),
            String(cString: number.objCType) != "d",
            String(cString: number.objCType) != "f",
            number.doubleValue.isFinite,
            number.doubleValue >= 0,
            number.doubleValue < Double(Int.max),
            number.doubleValue.rounded(.towardZero) == number.doubleValue
        else { return nil }
        return number.intValue
    }

    /// Decode only bounded, value-free metadata with exact object keys.
    /// Generic Codable decoding is also revalidated by every admission call.
    public static func fromJSON(_ data: Data) throws -> ReviewedLocalEvidence {
        guard data.count <= 65536,
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            Set(object.keys)
                == Set([
                    "schema_version", "kind", "provenance_class", "offset_convention", "source_id",
                    "source_digest", "source_length", "policy_digest", "references", "review_receipt",
                ]),
            integer(object["schema_version"]) == 1,
            object["kind"] as? String == "reviewed_local_evidence",
            object["provenance_class"] as? String == "reviewed_local",
            object["offset_convention"] as? String == "unicode_scalar_half_open",
            opaque(object["source_id"], prefix: "source"),
            opaque(object["source_digest"], prefix: "sha256"),
            opaque(object["policy_digest"], prefix: "sha256"),
            let sourceLength = integer(object["source_length"]),
            let refs = object["references"] as? [[String: Any]],
            !refs.isEmpty, refs.count <= 64
        else { throw ReviewAdmissionRefusal.invalid }
        var seen = Set<String>()
        for ref in refs {
            guard Set(ref.keys) == Set(["reference_id", "start", "end"]),
                opaque(ref["reference_id"], prefix: "ref"),
                let id = ref["reference_id"] as? String,
                seen.insert(id).inserted,
                let start = integer(ref["start"]), let end = integer(ref["end"]),
                start < end, end <= sourceLength
            else { throw ReviewAdmissionRefusal.invalid }
        }
        if !(object["review_receipt"] is NSNull) {
            guard let receipt = object["review_receipt"] as? [String: Any],
                Set(receipt.keys) == Set(["receipt_id", "authority_id", "evidence_digest", "issued_at", "expires_at"]),
                opaque(receipt["receipt_id"], prefix: "receipt"),
                opaque(receipt["authority_id"], prefix: "authority"),
                opaque(receipt["evidence_digest"], prefix: "sha256"),
                let issued = integer(receipt["issued_at"]), let expires = integer(receipt["expires_at"]),
                issued < expires
            else { throw ReviewAdmissionRefusal.invalid }
        }
        guard let normalized = try? JSONSerialization.data(withJSONObject: object),
            let packet = try? JSONDecoder().decode(ReviewedLocalEvidence.self, from: normalized)
        else {
            throw ReviewAdmissionRefusal.invalid
        }
        return packet
    }

    /// Check current source and review authority immediately before generation.
    /// This result is not a reusable authorization token; call on every use.
    public func admit(
        source: String,
        policyDigest currentPolicy: String,
        currentSource: any CurrentLocalSource,
        authority: any ReviewAuthorityVerifier,
        clock: () -> TimeInterval = { Date().timeIntervalSince1970 }
    ) throws {
        let packet = try Self.fromJSON(toJSON())
        guard packet.sourceDigest == Self.sourceDigest(source), sourceLength == source.unicodeScalars.count else {
            throw ReviewAdmissionRefusal.sourceChanged
        }
        guard policyDigest == currentPolicy else { throw ReviewAdmissionRefusal.policyChanged }
        guard let receipt = reviewReceipt else { throw ReviewAdmissionRefusal.missing }
        let digest = try evidenceDigest
        guard receipt.evidenceDigest == digest else { throw ReviewAdmissionRefusal.mismatched }
        let instant = clock()
        guard instant.isFinite, instant >= 0, instant < Double(Int.max) else {
            throw ReviewAdmissionRefusal.authorityUnavailable
        }
        let now = Int(instant)
        guard now >= receipt.issuedAt else { throw ReviewAdmissionRefusal.mismatched }
        guard now < receipt.expiresAt else { throw ReviewAdmissionRefusal.expired }
        let current: String?
        do { current = try currentSource.currentDigest(sourceID: sourceID) } catch { throw ReviewAdmissionRefusal.sourceUnavailable }
        guard let current else { throw ReviewAdmissionRefusal.sourceUnavailable }
        guard current == sourceDigest else { throw ReviewAdmissionRefusal.sourceChanged }
        let status: ReviewAuthorityStatus
        do { status = try authority.verify(receipt, evidenceDigest: digest, now: now) } catch { throw ReviewAdmissionRefusal.authorityUnavailable }
        switch status {
        case .current: return
        case .revoked: throw ReviewAdmissionRefusal.revoked
        case .mismatched: throw ReviewAdmissionRefusal.mismatched
        }
    }
}

extension ClinicalBrief {
    /// Admit reviewed-local evidence before invoking trusted on-device generation.
    /// Only reviewed spans enter `generate`; `evaluate` must execute the complete
    /// guarded evidence/NLI pipeline. No cloud fallback or new backend is added.
    public static func reviewedLocal(
        evidence: ReviewedLocalEvidence,
        source: String,
        policyDigest: String,
        currentSource: any CurrentLocalSource,
        authority: any ReviewAuthorityVerifier,
        originalIdentifiers: [String],
        clock: () -> TimeInterval = { Date().timeIntervalSince1970 },
        generate: (String) async throws -> String,
        evaluate: (String, String) async throws -> Data,
        privacyCheck: (String) throws -> Bool
    ) async throws -> ClinicalBrief {
        guard source.utf8.count <= 16384 else { throw ReviewAdmissionRefusal.invalid }
        try evidence.admit(source: source, policyDigest: policyDigest, currentSource: currentSource, authority: authority, clock: clock)
        let scalars = Array(source.unicodeScalars)
        let admitted = evidence.references.sorted { ($0.start, $0.end) < ($1.start, $1.end) }.map {
            String(String.UnicodeScalarView(scalars[$0.start..<$0.end]))
        }.joined(separator: " ")
        try evidence.admit(source: source, policyDigest: policyDigest, currentSource: currentSource, authority: authority, clock: clock)
        do {
            let summary = try await generate(admitted)
            let evaluation = try await evaluate(source, summary)
            let brief = try validate(
                evaluationJSON: evaluation, source: source, generatedSummary: summary,
                originalIdentifiers: originalIdentifiers, privacyCheck: privacyCheck)
            guard
                brief.citations.allSatisfy({ citation in
                    evidence.references.contains { $0.start == citation.sourceStart && $0.end == citation.sourceEnd }
                })
            else { throw ClinicalBriefError.unsupportedClaim }
            return brief
        } catch let error as ClinicalBriefError { throw error } catch is CancellationError { throw CancellationError() } catch { throw ClinicalBriefError.invalidPacket }
    }
}
