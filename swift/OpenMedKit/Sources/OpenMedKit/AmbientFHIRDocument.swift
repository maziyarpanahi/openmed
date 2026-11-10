import CoreFoundation
import CryptoKit
import Foundation

/// Controlled errors; protected note/source values never appear in diagnostics.
public enum AmbientFHIRDocumentError: String, Error, LocalizedError {
    case invalidDraft = "invalid_draft"
    case invalidDocument = "invalid_document"
    case invalidRecordedTime = "invalid_recorded_time"
    case unreviewedDraft = "unreviewed_draft"
    case staleReview = "stale_review"
    case staleEvidence = "stale_evidence"
    case correctionPending = "correction_pending"
    case confirmationMismatch = "confirmation_mismatch"

    public var errorDescription: String? { rawValue }
}

/// Counts-only loss category declared by the caller; no omitted payload is accepted.
public struct AmbientConversionLoss: Sendable, Equatable {
    public let category: String
    public let count: Int
}

/// Protected clinical output, with a value-free description suitable for diagnostics.
public struct AmbientFHIRExport: Sendable, CustomStringConvertible {
    public let documentJSON: Data
    public let losses: [AmbientConversionLoss]
    public let draftDigest: String
    public var description: String { "AmbientFHIRExport(losses: \(losses.count))" }
}

/// Offline exporter over fixed reviewed fixtures; no assembly, provider or EHR transport.
///
/// Review receipts are caller assertions. Authenticate reviewer authority and obtain
/// current evidence/correction state from the local ledger before every export.
public enum AmbientFHIRDocument {
    private static let base = "https://openmed.dev/fhir/StructureDefinition/ambient-"
    private static let notice = "Non-diagnostic ambient note. Explicit clinician confirmation is required before consequential use. This export performs no EHR write."
    private static let divStart = "<div xmlns=\"http://www.w3.org/1999/xhtml\"><p>"
    private static let divEnd = "</p></div>"
    private static let sectionCodes: Set<String> = ["history", "exam", "subjective", "objective", "assessment", "plan"]
    private static let lossCategories: Set<String> = ["audio_payload", "transcript_payload", "model_metadata", "review_history", "unsupported_elements"]

    /// Compute the shared UTF-8 framed SHA-256 commitment before local clinician review.
    public static func draftDigest(draftJSON: Data) throws -> String {
        commitment(try snapshot(parse(draftJSON)))
    }

    /// Export the v1 fixture as Composition/Provenance. Final status requires explicit
    /// confirmation of this exact digest; the default remains preliminary.
    public static func export(
        draftJSON: Data,
        currentEvidenceDigest: String,
        recordedAt: String,
        confirmedDigest: String? = nil
    ) throws -> AmbientFHIRExport {
        let draft = try snapshot(parse(draftJSON))
        try digest(currentEvidenceDigest)
        try instant(recordedAt)
        let hash = commitment(draft)
        guard draft["correction_pending"] as? Bool == false else { throw AmbientFHIRDocumentError.correctionPending }
        guard draft["evidence_digest"] as? String == currentEvidenceDigest else { throw AmbientFHIRDocumentError.staleEvidence }
        guard let review = draft["review"] as? [String: Any] else { throw AmbientFHIRDocumentError.unreviewedDraft }
        guard review["digest"] as? String == hash else { throw AmbientFHIRDocumentError.staleReview }
        guard confirmedDigest == nil || confirmedDigest == hash else { throw AmbientFHIRDocumentError.confirmationMismatch }
        let rows = draft["loss_counts"] as! [[String: Any]]
        let losses = rows.map { AmbientConversionLoss(category: $0["category"] as! String, count: $0["count"] as! Int) }
        return AmbientFHIRExport(documentJSON: try encode(render(draft, hash, recordedAt, confirmedDigest != nil)), losses: losses, draftDigest: hash)
    }

    /// Check the exact closed subset offline. This is not full FHIR or clinical
    /// validation and does not authenticate transported review assertions.
    public static func validate(documentJSON: Data) -> Bool {
        (try? importDocument(documentJSON: documentJSON)) != nil
    }

    /// Round-trip section order, note text and opaque citations. The returned review
    /// receipt is untrusted transported metadata and grants no clinical permission.
    public static func importDocument(documentJSON: Data) throws -> Data {
        do {
            let bundle = try parse(documentJSON)
            guard let entries = bundle["entry"] as? [[String: Any]], entries.count == 2,
                let composition = entries[0]["resource"] as? [String: Any],
                let ext = composition["extension"] as? [[String: Any]], ext.count >= 5,
                let identifier = composition["identifier"] as? [String: Any],
                let authors = composition["author"] as? [[String: Any]], authors.count == 1,
                let reviewer = authors[0]["identifier"] as? [String: Any],
                let rows = composition["section"] as? [[String: Any]],
                let status = composition["status"] as? String, ["preliminary", "final"].contains(status),
                let recorded = composition["date"] as? String,
                let hash = ext[1]["valueString"] as? String,
                let evidenceDigest = ext[2]["valueString"] as? String
            else { throw AmbientFHIRDocumentError.invalidDocument }
            var sections: [[String: Any]] = []
            for row in rows {
                guard let code = row["code"] as? [String: Any],
                    let coding = code["coding"] as? [[String: Any]], coding.count == 1,
                    let text = row["text"] as? [String: Any], let div = text["div"] as? String,
                    div.hasPrefix(divStart), div.hasSuffix(divEnd),
                    let evidence = row["extension"] as? [[String: Any]]
                else { throw AmbientFHIRDocumentError.invalidDocument }
                var citations: [[String: Any]] = []
                for item in evidence {
                    guard let e = item["extension"] as? [[String: Any]], e.count == 4 else { throw AmbientFHIRDocumentError.invalidDocument }
                    citations.append(["reference": e[0]["valueUri"] as Any, "speaker": e[1]["valueUri"] as Any, "start": e[2]["valueUnsignedInt"] as Any, "end": e[3]["valueUnsignedInt"] as Any])
                }
                let body = String(div.dropFirst(divStart.count).dropLast(divEnd.count))
                sections.append(["code": coding[0]["code"] as Any, "note": unescape(body), "evidence": citations])
            }
            var losses: [[String: Any]] = []
            for row in ext.dropFirst(5) {
                guard let e = row["extension"] as? [[String: Any]], e.count == 2 else { throw AmbientFHIRDocumentError.invalidDocument }
                losses.append(["category": e[0]["valueCode"] as Any, "count": e[1]["valueUnsignedInt"] as Any])
            }
            let draft: [String: Any] = [
                "schema_version": ext[0]["valueUnsignedInt"] as Any,
                "draft_ref": identifier["value"] as Any, "evidence_digest": evidenceDigest,
                "review": ["digest": hash, "reviewer": reviewer["value"] as Any],
                "correction_pending": false, "loss_counts": losses, "sections": sections,
            ]
            let data = try encode(draft)
            let expected = try export(draftJSON: data, currentEvidenceDigest: evidenceDigest, recordedAt: recorded, confirmedDigest: status == "final" ? hash : nil)
            guard try encode(bundle) == expected.documentJSON else { throw AmbientFHIRDocumentError.invalidDocument }
            return data
        } catch {
            throw AmbientFHIRDocumentError.invalidDocument
        }
    }

    private static func parse(_ data: Data) throws -> [String: Any] {
        guard data.count <= 1_048_576, let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw AmbientFHIRDocumentError.invalidDraft }
        return value
    }

    private static func encode(_ value: [String: Any]) throws -> Data {
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes]) else { throw AmbientFHIRDocumentError.invalidDraft }
        return data
    }

    private static func keys(_ value: [String: Any], _ names: Set<String>) throws {
        guard Set(value.keys) == names else { throw AmbientFHIRDocumentError.invalidDraft }
    }

    private static func opaque(_ value: Any?) throws {
        guard let value = value as? String, value.hasPrefix("urn:uuid:"),
            let uuid = UUID(uuidString: String(value.dropFirst(9))),
            value == "urn:uuid:" + uuid.uuidString.lowercased()
        else { throw AmbientFHIRDocumentError.invalidDraft }
    }

    private static func digest(_ value: Any?) throws {
        guard let value = value as? String, value.count == 64, value.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw AmbientFHIRDocumentError.invalidDraft }
    }

    private static func integer(_ value: Any?, _ maximum: Int = 2_147_483_647) throws -> Int {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
            let value = value as? Int, value >= 0, value <= maximum,
            String(cString: number.objCType) != "d", String(cString: number.objCType) != "f"
        else { throw AmbientFHIRDocumentError.invalidDraft }
        return value
    }

    private static func snapshot(_ draft: [String: Any]) throws -> [String: Any] {
        try keys(draft, ["schema_version", "draft_ref", "evidence_digest", "sections", "loss_counts", "review", "correction_pending"])
        guard try integer(draft["schema_version"]) == 1,
            let pending = draft["correction_pending"] as? NSNumber, CFGetTypeID(pending) == CFBooleanGetTypeID(),
            let sections = draft["sections"] as? [[String: Any]], (1...32).contains(sections.count),
            let losses = draft["loss_counts"] as? [[String: Any]], losses.count <= lossCategories.count
        else { throw AmbientFHIRDocumentError.invalidDraft }
        try opaque(draft["draft_ref"])
        try digest(draft["evidence_digest"])
        var total = 0
        for section in sections {
            try keys(section, ["code", "note", "evidence"])
            guard let code = section["code"] as? String, sectionCodes.contains(code),
                let note = section["note"] as? String, (1...16384).contains(note.unicodeScalars.count),
                note.unicodeScalars.allSatisfy({ [9, 10, 13].contains($0.value) || (0x20...0xD7FF).contains($0.value) || (0xE000...0xFFFD).contains($0.value) || (0x10000...0x10FFFF).contains($0.value) }),
                let evidence = section["evidence"] as? [[String: Any]], (1...128).contains(evidence.count)
            else { throw AmbientFHIRDocumentError.invalidDraft }
            total += note.utf8.count
            for item in evidence {
                try keys(item, ["reference", "speaker", "start", "end"])
                try opaque(item["reference"])
                try opaque(item["speaker"])
                guard try integer(item["start"]) < integer(item["end"]) else { throw AmbientFHIRDocumentError.invalidDraft }
            }
        }
        guard total <= 262144 else { throw AmbientFHIRDocumentError.invalidDraft }
        var seen: Set<String> = []
        for item in losses {
            try keys(item, ["category", "count"])
            guard let category = item["category"] as? String, lossCategories.contains(category), seen.insert(category).inserted,
                try integer(item["count"]) > 0
            else { throw AmbientFHIRDocumentError.invalidDraft }
        }
        if !(draft["review"] is NSNull) {
            guard let review = draft["review"] as? [String: Any] else { throw AmbientFHIRDocumentError.invalidDraft }
            try keys(review, ["digest", "reviewer"])
            try digest(review["digest"])
            try opaque(review["reviewer"])
        }
        return draft
    }

    private static func commitment(_ draft: [String: Any]) -> String {
        let sections = draft["sections"] as! [[String: Any]]
        let losses = draft["loss_counts"] as! [[String: Any]]
        var values = [draft["draft_ref"] as! String, draft["evidence_digest"] as! String, String(sections.count)]
        for section in sections {
            let evidence = section["evidence"] as! [[String: Any]]
            values += [section["code"] as! String, section["note"] as! String, String(evidence.count)]
            for item in evidence {
                values += [item["reference"] as! String, item["speaker"] as! String, String(item["start"] as! Int), String(item["end"] as! Int)]
            }
        }
        values.append(String(losses.count))
        for item in losses { values += [item["category"] as! String, String(item["count"] as! Int)] }
        var data = Data("openmed-ambient-v1\n".utf8)
        for value in values { data.append(Data("\(value.utf8.count):\(value)".utf8)) }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func instant(_ value: String) throws {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss'Z'"
        formatter.isLenient = false
        guard let date = formatter.date(from: value), formatter.string(from: date) == value else { throw AmbientFHIRDocumentError.invalidRecordedTime }
    }

    private static func ext(_ name: String, _ kind: String, _ value: Any) -> [String: Any] {
        ["url": base + name, kind: value]
    }

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;").replacingOccurrences(of: "'", with: "&#x27;")
    }

    private static func unescape(_ text: String) -> String {
        text.replacingOccurrences(of: "&#x27;", with: "'").replacingOccurrences(of: "&quot;", with: "\"").replacingOccurrences(of: "&gt;", with: ">").replacingOccurrences(of: "&lt;", with: "<").replacingOccurrences(of: "&amp;", with: "&")
    }

    private static func fullURL(_ hash: String, _ index: Int) -> String {
        var namespace = UUID(uuidString: "9eeec144-afaa-5ca6-9465-091cbbcd463f")!.uuid
        var data = withUnsafeBytes(of: &namespace) { Data($0) }
        data.append(Data("\(hash):\(index)".utf8))
        var bytes = Array(Insecure.SHA1.hash(data: data).prefix(16))
        bytes[6] = (bytes[6] & 0x0f) | 0x50
        bytes[8] = (bytes[8] & 0x3f) | 0x80
        let hex = bytes.map { String(format: "%02x", $0) }.joined()
        let chars = Array(hex)
        let uuid = [String(chars[0..<8]), String(chars[8..<12]), String(chars[12..<16]), String(chars[16..<20]), String(chars[20..<32])].joined(separator: "-")
        return "urn:uuid:" + uuid
    }

    private static func render(_ draft: [String: Any], _ hash: String, _ recorded: String, _ final: Bool) -> [String: Any] {
        let review = draft["review"] as! [String: Any]
        let reviewer: [String: Any] = ["identifier": ["system": base + "reviewer", "value": review["reviewer"]!]]
        let sections = draft["sections"] as! [[String: Any]]
        let losses = draft["loss_counts"] as! [[String: Any]]
        var extensions = [ext("schema", "valueUnsignedInt", 1), ext("draft-digest", "valueString", hash), ext("evidence-digest", "valueString", draft["evidence_digest"]!), ext("review-status", "valueCode", "reviewed"), ext("clinician-confirmed", "valueBoolean", final)]
        extensions += losses.map { ["url": base + "conversion-loss", "extension": [ext("category", "valueCode", $0["category"]!), ext("count", "valueUnsignedInt", $0["count"]!)]] }
        let sectionRows: [[String: Any]] = sections.map { section in
            let evidence = section["evidence"] as! [[String: Any]]
            return [
                "code": ["coding": [["system": "https://openmed.dev/fhir/CodeSystem/ambient-section", "code": section["code"]!]]],
                "text": ["status": "additional", "div": divStart + escape(section["note"] as! String) + divEnd],
                "extension": evidence.map { ["url": base + "evidence", "extension": [ext("reference", "valueUri", $0["reference"]!), ext("speaker", "valueUri", $0["speaker"]!), ext("start", "valueUnsignedInt", $0["start"]!), ext("end", "valueUnsignedInt", $0["end"]!)]] },
            ]
        }
        let composition: [String: Any] = [
            "resourceType": "Composition", "id": "ambient-note",
            "identifier": ["system": base + "draft", "value": draft["draft_ref"]!],
            "status": final ? "final" : "preliminary",
            "type": ["coding": [["system": "https://openmed.dev/fhir/CodeSystem/document-type", "code": "ambient-note"]]],
            "date": recorded, "author": [reviewer], "title": "Reviewed ambient note",
            "text": ["status": "additional", "div": divStart + notice + divEnd],
            "extension": extensions, "section": sectionRows,
        ]
        let sources = Set(sections.flatMap { ($0["evidence"] as! [[String: Any]]).map { $0["reference"] as! String } }).sorted()
        let provenance: [String: Any] = [
            "resourceType": "Provenance", "id": "ambient-provenance",
            "target": [["reference": fullURL(hash, 0)]], "recorded": recorded,
            "activity": ["coding": [["system": "https://openmed.dev/fhir/CodeSystem/ambient-activity", "code": "reviewed-export"]]],
            "agent": [["who": reviewer]],
            "entity": sources.map { ["role": "source", "what": ["identifier": ["system": base + "transcript-evidence", "value": $0]]] },
        ]
        return [
            "resourceType": "Bundle", "type": "document",
            "entry": [["fullUrl": fullURL(hash, 0), "resource": composition], ["fullUrl": fullURL(hash, 1), "resource": provenance]],
            "identifier": ["system": base + "draft-digest", "value": hash], "timestamp": recorded,
        ]
    }
}
