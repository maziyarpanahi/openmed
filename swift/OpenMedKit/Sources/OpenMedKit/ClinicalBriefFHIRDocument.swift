import CoreFoundation
import CryptoKit
import Foundation

/// Value-free failures from the passive R4 document boundary.
public enum ClinicalBriefFHIRDocumentError: String, Error, LocalizedError, Sendable {
    case invalidDocument = "invalid_document"
    case missingProvenance = "missing_provenance"
    case unsupportedFinalization = "unsupported_finalization"
    case privacy = "privacy"

    public var errorDescription: String? { rawValue }
}

/// Closed, preliminary R4 document projection. No EHR transport or attestation.
public struct ClinicalBriefFHIRDocument: Sendable, CustomStringConvertible {
    public static let subset = "openmed.clinical.brief.fhir-r4.v1"
    private static let extensionURL = "https://openmed.ai/fhir/StructureDefinition/clinical-brief-document"
    private static let losses = [
        "profile_sections_unavailable", "source_payloads_omitted", "evaluation_metrics_omitted",
        "review_packet_details_omitted", "model_details_omitted",
    ]
    private static let disclaimer =
        "This summary is for human review only. It is not a diagnosis, medical advice, or a substitute for qualified clinical judgment."
    private let response: Data
    private let audit: Data
    public let briefDigest: String
    public let citationCount: Int
    public var description: String { "ClinicalBriefFHIRDocument(citations: \(citationCount), status: preliminary)" }

    /// Protected narrative and attachment metadata; never log this response.
    public func responseJSON() -> Data { response }
    /// Counts, digests, controlled losses and review status only.
    public func auditJSON() -> Data { audit }

    /// Export a verified brief with an explicit export time and final privacy gate.
    /// Original identifier tokens are excluded from all rendered string surfaces.
    /// The existing brief contract requires review, so finalization is rejected.
    public static func export(
        _ brief: ClinicalBrief, recordedAt: Date, status: String = "preliminary",
        originalIdentifiers: [String] = [], privacyCheck: (String) throws -> Bool
    ) throws -> Self {
        guard status == "preliminary" else { throw ClinicalBriefFHIRDocumentError.unsupportedFinalization }
        guard recordedAt.timeIntervalSince1970.isFinite,
            let packet = try? JSONSerialization.jsonObject(with: brief.auditJSON()) as? [String: Any],
            let envelope = packet["envelope"] as? [String: Any],
            let source = envelope["provenance"] as? [String: Any], isTrue(source["verified"]),
            let sourceDigest = source["content_hash"] as? String,
            let provenance = packet["provenance"] as? [String: Any],
            let evidence = provenance["evidence"] as? [[String: Any]],
            let provenanceDigest = provenance["record_hash"] as? String,
            let policyDigest = provenance["policy_fingerprint"] as? String,
            provenance["review_status"] as? String == "queued",
            isTrue(provenance["review_required"]),
            let review = packet["review_packet"] as? [String: Any],
            review["review_status"] as? String == "review_required"
        else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
        let keys = [
            "schema_version", "record_type", "record_id", "output_kind", "input_hash", "output_hash", "evidence_ids",
            "evidence", "model", "policy_fingerprint", "review_required", "review_status", "review_transitions", "integrity",
        ]
        var material: [String: Any] = [:]
        for key in keys {
            guard let value = provenance[key] else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
            material[key] = value
        }
        guard provenanceDigest == (try domainHash(material, "guarded-clinical-record")),
            provenance["output_hash"] as? String == (try domainHash(brief.summary, "guarded-clinical-output")),
            let integrity = provenance["integrity"] as? [String: Any],
            integrity["reason_codes"] as? [String] == [],
            integrity["missing_evidence_count"] as? Int == 0,
            integrity["changed_evidence_count"] as? Int == 0,
            isTrue(integrity["input_present"]), isTrue(integrity["output_present"]),
            isTrue(integrity["model_present"]), isTrue(integrity["policy_present"]),
            isTrue(integrity["record_hash_valid"]), isTrue(integrity["manifest_hash_valid"]),
            (integrity["input_changed"] as? Bool) == false,
            (integrity["output_changed"] as? Bool) == false
        else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
        var citations: [[String: Any]] = []
        for citation in brief.citations {
            let matches = evidence.filter {
                let offsets = $0["source_offsets"] as? [String: Int]
                return offsets == ["start": citation.sourceStart, "end": citation.sourceEnd]
            }
            guard matches.count == 1,
                let evidenceID = matches[0]["evidence_id"] as? String,
                let evidenceHash = matches[0]["evidence_hash"] as? String
            else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
            citations.append([
                "claim_index": citation.claimIndex, "source_start": citation.sourceStart, "source_end": citation.sourceEnd,
                "output_start": citation.outputStart, "output_end": citation.outputEnd,
                "evidence_id": evidenceID, "evidence_hash": evidenceHash,
            ])
        }
        let metadata: [String: Any] = [
            "subset": subset, "brief_digest": brief.digest,
            "summary_digest": try ClinicalBrief.hash(ClinicalBrief.canonical(brief.summary)),
            "source_digest": sourceDigest, "provenance_digest": provenanceDigest, "policy_digest": policyDigest,
            "review_status": "queued", "requires_human_review": true, "citations": citations, "conversion_loss": losses,
        ]
        let date = ISO8601DateFormatter().string(from: recordedAt).replacingOccurrences(of: "Z", with: "+00:00")
        guard ISO8601DateFormatter().date(from: date) != nil else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
        return try finish(build(metadata, brief.summary, date), metadata, brief.summary, originalIdentifiers, privacyCheck)
    }

    /// Validate a round trip of this exact subset without reconstructing approval.
    /// Unknown fields, final status, external links and altered section order fail.
    public static func validate(
        documentJSON: Data, originalIdentifiers: [String] = [], privacyCheck: (String) throws -> Bool
    ) throws -> Self {
        guard documentJSON.count <= 262_144,
            let document = try? JSONSerialization.jsonObject(with: documentJSON) as? [String: Any],
            let bundle = document["bundle"] as? [String: Any],
            let entries = bundle["entry"] as? [[String: Any]],
            let composition = entries.first?["resource"] as? [String: Any],
            let extensions = composition["extension"] as? [[String: Any]],
            let encoded = extensions.first?["valueString"] as? String,
            let metadata = try? JSONSerialization.jsonObject(with: Data(encoded.utf8)) as? [String: Any],
            let narrative = composition["text"] as? [String: String], let div = narrative["div"],
            let date = bundle["timestamp"] as? String,
            let parsed = ISO8601DateFormatter().date(from: date),
            ISO8601DateFormatter().string(from: parsed).replacingOccurrences(of: "Z", with: "+00:00") == date
        else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
        let prefix = "<div xmlns=\"http://www.w3.org/1999/xhtml\">"
        guard div.hasPrefix(prefix), div.hasSuffix("</div>") else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
        let summary = unescape(String(div.dropFirst(prefix.count).dropLast(6)))
        let expected = try build(metadata, summary, date)
        guard try ClinicalBrief.canonical(expected) == ClinicalBrief.canonical(document) else {
            throw ClinicalBriefFHIRDocumentError.invalidDocument
        }
        return try finish(expected, metadata, summary, originalIdentifiers, privacyCheck)
    }

    private static func finish(
        _ document: [String: Any], _ metadata: [String: Any], _ summary: String,
        _ identifiers: [String], _ privacyCheck: (String) throws -> Bool
    ) throws -> Self {
        guard identifiers.count <= 1024, identifiers.reduce(0, { $0 + $1.utf8.count }) <= 16_384 else {
            throw ClinicalBriefFHIRDocumentError.privacy
        }
        let data = try ClinicalBrief.canonical(document)
        guard data.count <= 262_144 else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
        let rendered = String(decoding: data, as: UTF8.self)
        let decoded = ([summary] + strings(document).map(unescape)).joined(separator: "\n")
        let locale = Locale(identifier: "en_US_POSIX")
        let folded = decoded.folding(options: [.caseInsensitive], locale: locale)
        for identifier in identifiers {
            let original = identifier.trimmingCharacters(in: .whitespacesAndNewlines).folding(options: [.caseInsensitive], locale: locale)
            if !original.isEmpty && folded.contains(original) { throw ClinicalBriefFHIRDocumentError.privacy }
            for token in identifier.components(separatedBy: CharacterSet.alphanumerics.inverted) where token.count >= 3 {
                if folded.contains(token.folding(options: [.caseInsensitive], locale: locale)) {
                    throw ClinicalBriefFHIRDocumentError.privacy
                }
            }
        }
        var clean = false
        do { clean = try privacyCheck(decoded) && privacyCheck(rendered) } catch { clean = false }
        guard clean else { throw ClinicalBriefFHIRDocumentError.privacy }
        let digest = metadata["brief_digest"] as! String
        let count = (metadata["citations"] as! [[String: Any]]).count
        let audit: [String: Any] = [
            "subset": subset, "brief_digest": digest, "citation_count": count,
            "status": "preliminary", "review_status": "queued", "requires_human_review": true, "conversion_loss": losses,
        ]
        return Self(response: data, audit: try ClinicalBrief.canonical(audit), briefDigest: digest, citationCount: count)
    }

    private static func build(_ metadata: [String: Any], _ summary: String, _ date: String) throws -> [String: Any] {
        guard
            Set(metadata.keys)
                == Set([
                    "subset", "brief_digest", "summary_digest", "source_digest", "provenance_digest", "policy_digest",
                    "review_status", "requires_human_review", "citations", "conversion_loss",
                ]), metadata["subset"] as? String == subset, metadata["review_status"] as? String == "queued",
            isTrue(metadata["requires_human_review"]), metadata["conversion_loss"] as? [String] == losses,
            metadata["brief_digest"] is String,
            let citations = metadata["citations"] as? [[String: Any]], !citations.isEmpty, citations.count <= 64,
            !summary.isEmpty, summary.utf8.count <= 16_384,
            metadata["summary_digest"] as? String == (try ClinicalBrief.hash(ClinicalBrief.canonical(summary)))
        else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
        for key in ["brief_digest", "summary_digest", "source_digest", "provenance_digest", "policy_digest"] {
            guard isDigest(metadata[key]) else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
        }
        let seed = try ClinicalBrief.hash(ClinicalBrief.canonical(["metadata": metadata, "recorded_at": date]))
        let extensionRows: [[String: Any]] = [
            [
                "url": extensionURL, "valueString": String(decoding: try ClinicalBrief.canonical(metadata), as: UTF8.self),
            ]
        ]
        let output = Array(summary.unicodeScalars)
        var end = 0
        var sections: [[String: Any]] = []
        var resources: [[String: Any]] = [[:], ["resourceType": "Device", "id": "brief-exporter", "status": "active"]]
        for (index, citation) in citations.enumerated() {
            guard
                Set(citation.keys)
                    == Set([
                        "claim_index", "source_start", "source_end", "output_start", "output_end", "evidence_id", "evidence_hash",
                    ]), integer(citation["claim_index"]) == index,
                let start = integer(citation["output_start"]), let stop = integer(citation["output_end"]),
                let sourceStart = integer(citation["source_start"]), let sourceEnd = integer(citation["source_end"]),
                sourceStart >= 0, sourceEnd > sourceStart, sourceEnd <= 16_384,
                start >= end, stop > start, stop <= output.count,
                output[end..<start].allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }),
                isDigest(citation["evidence_id"]), isDigest(citation["evidence_hash"])
            else { throw ClinicalBriefFHIRDocumentError.invalidDocument }
            let evidenceID = citation["evidence_id"] as! String
            let evidenceHash = citation["evidence_hash"] as! String
            resources.append([
                "resourceType": "DocumentReference", "id": "evidence-\(index)", "status": "current",
                "description": "Source evidence commitment",
                "identifier": [["system": "urn:openmed:evidence", "value": evidenceID]],
                "content": [
                    [
                        "attachment": [
                            "contentType": "text/plain", "title": "Source evidence (payload omitted)",
                            "url": "urn:sha256:" + evidenceHash.dropFirst(7),
                        ]
                    ]
                ],
            ])
            sections.append([
                "title": "Claim \(index + 1)", "text": narrative(String(String.UnicodeScalarView(output[start..<stop]))),
                "entry": [["reference": fullURL(seed, index + 2)]],
            ])
            end = stop
        }
        guard output[end...].allSatisfy({ CharacterSet.whitespacesAndNewlines.contains($0) }) else {
            throw ClinicalBriefFHIRDocumentError.invalidDocument
        }
        sections.append(["title": "Limitations", "text": narrative(disclaimer)])
        resources[0] = [
            "resourceType": "Composition", "id": "brief", "status": "preliminary", "type": ["text": "Clinical brief"],
            "date": date, "author": [["reference": fullURL(seed, 1)]], "title": "Clinical brief for human review",
            "text": narrative(summary), "extension": extensionRows, "section": sections,
        ]
        let bundle: [String: Any] = [
            "resourceType": "Bundle", "type": "document", "id": String(seed.dropFirst(7)),
            "identifier": ["system": "urn:openmed:brief-document", "value": seed], "timestamp": date,
            "entry": resources.enumerated().map { ["fullUrl": fullURL(seed, $0.offset), "resource": $0.element] },
        ]
        let reference: [String: Any] = [
            "resourceType": "DocumentReference", "status": "current", "docStatus": "preliminary",
            "type": ["text": "Clinical brief"], "date": date, "extension": extensionRows,
            "content": [
                [
                    "attachment": [
                        "contentType": "application/fhir+json", "url": fullURL(seed, -1), "title": "Preliminary clinical brief",
                    ]
                ]
            ],
        ]
        return ["bundle": bundle, "document_reference": reference, "document_url": fullURL(seed, -1)]
    }

    private static func domainHash(_ value: Any, _ domain: String) throws -> String {
        // Guarded provenance uses type-tagged canonical values, including a
        // second tagging pass over its domain envelope. Match that contract.
        try ClinicalBrief.hash(ClinicalBrief.canonical(tagged(["domain": domain, "value": try tagged(value)])))
    }
    private static func tagged(_ value: Any, depth: Int = 0) throws -> Any {
        guard depth <= 32 else { throw ClinicalBriefFHIRDocumentError.missingProvenance }
        if value is NSNull { return ["null"] }
        if let text = value as? String { return ["string", text] }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return ["bool", number.boolValue] as [Any] }
            let type = String(cString: number.objCType)
            return [type == "d" || type == "f" ? "float" : "int", number] as [Any]
        }
        if let object = value as? [String: Any] {
            return ["mapping", try object.keys.sorted().map { [$0, try tagged(object[$0]!, depth: depth + 1)] }] as [Any]
        }
        if let array = value as? [Any] { return ["sequence", try array.map { try tagged($0, depth: depth + 1) }] as [Any] }
        throw ClinicalBriefFHIRDocumentError.missingProvenance
    }
    private static func isDigest(_ value: Any?) -> Bool {
        guard let text = value as? String else { return false }
        return text.utf8.count == 71 && text.range(of: "^sha256:[0-9a-f]{64}$", options: .regularExpression) != nil
    }
    private static func isTrue(_ value: Any?) -> Bool {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return false }
        return number.boolValue
    }
    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
            String(cString: number.objCType) != "d", String(cString: number.objCType) != "f"
        else { return nil }
        return number.intValue
    }
    private static func strings(_ value: Any) -> [String] {
        if let object = value as? [String: Any] { return object.keys.sorted().flatMap { [$0] + strings(object[$0]!) } }
        if let array = value as? [Any] { return array.flatMap(strings) }
        if let text = value as? String { return [text] }
        return []
    }
    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&#x27;")
    }
    private static func unescape(_ text: String) -> String {
        text.replacingOccurrences(of: "&#x27;", with: "'").replacingOccurrences(of: "&quot;", with: "\"")
            .replacingOccurrences(of: "&gt;", with: ">").replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&amp;", with: "&")
    }
    private static func narrative(_ text: String) -> [String: String] {
        ["status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">" + escape(text) + "</div>"]
    }
    /// UUIDv5 matches the existing Python FHIR Bundle helper's namespace/seed.
    private static func fullURL(_ seed: String, _ index: Int) -> String {
        let namespace: [UInt8] = [0x9e, 0xee, 0xc1, 0x44, 0xaf, 0xaa, 0x5c, 0xa6, 0x94, 0x65, 0x09, 0x1c, 0xbb, 0xcd, 0x46, 0x3f]
        var bytes = Array(Insecure.SHA1.hash(data: Data(namespace) + Data("\(seed):\(index)".utf8)).prefix(16))
        bytes[6] = (bytes[6] & 0x0f) | 0x50
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        let hex = bytes.map { String(format: "%02x", $0) }.joined()
        let chunks = [0..<8, 8..<12, 12..<16, 16..<20, 20..<32].map { range in
            String(hex.dropFirst(range.lowerBound).prefix(range.count))
        }
        return "urn:uuid:" + chunks.joined(separator: "-")
    }
}
