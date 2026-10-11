import Foundation
import XCTest

@testable import OpenMedKit

final class ClinicalBriefFHIRDocumentTests: XCTestCase {
    private func fixture() throws -> (ClinicalBrief, [String: Any]) {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let row = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "tests/fixtures/clinical/brief_parity/verified.json")))
                as? [String: Any])
        let source = try XCTUnwrap(row["source"] as? String)
        let summary = try XCTUnwrap(row["generator_output"] as? String)
        let packet = Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
        let brief = try ClinicalBrief.validate(
            evaluationJSON: packet, source: source, generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true })
        return (brief, try XCTUnwrap(row["fhir_document"] as? [String: Any]))
    }

    func testPythonAndNativeDocumentSubsetsAreByteIdentical() throws {
        let (brief, expected) = try fixture()
        let result = try ClinicalBriefFHIRDocument.export(
            brief, recordedAt: Date(timeIntervalSince1970: 1_767_225_600), privacyCheck: { _ in true })
        XCTAssertEqual(result.responseJSON(), try ClinicalBrief.canonical(expected))
        let restored = try ClinicalBriefFHIRDocument.validate(documentJSON: result.responseJSON(), privacyCheck: { _ in true })
        XCTAssertEqual(restored.responseJSON(), result.responseJSON())
        XCTAssertEqual(restored.auditJSON(), result.auditJSON())
        XCTAssertEqual(result.briefDigest, brief.digest)
        XCTAssertEqual(result.citationCount, 3)
        XCTAssertFalse(String(decoding: result.auditJSON(), as: UTF8.self).contains("dehydration"))
        XCTAssertFalse(result.description.contains("dehydration"))
    }

    func testFinalizationAndIdentifiersAreRejected() throws {
        let (brief, expected) = try fixture()
        XCTAssertThrowsError(
            try ClinicalBriefFHIRDocument.export(brief, recordedAt: Date(), status: "final", privacyCheck: { _ in true })
        ) { XCTAssertEqual($0 as? ClinicalBriefFHIRDocumentError, .unsupportedFinalization) }
        for identifier in ["dehydration", "DEHYDRATION", "dehydration@example.invalid", "Th"] {
            XCTAssertThrowsError(
                try ClinicalBriefFHIRDocument.validate(
                    documentJSON: JSONSerialization.data(withJSONObject: expected),
                    originalIdentifiers: [identifier], privacyCheck: { _ in true })
            ) { XCTAssertEqual($0 as? ClinicalBriefFHIRDocumentError, .privacy) }
        }
    }

    func testClosedSubsetRejectsNarrativeExtensionAttachmentAndWriteRequests() throws {
        let (_, fixture) = try fixture()
        for mutation in ["narrative", "extension", "attachment", "request", "final", "section_order", "reference"] {
            var document = fixture
            var bundle = try XCTUnwrap(document["bundle"] as? [String: Any])
            var entries = try XCTUnwrap(bundle["entry"] as? [[String: Any]])
            var composition = try XCTUnwrap(entries[0]["resource"] as? [String: Any])
            switch mutation {
            case "narrative": composition["text"] = ["status": "generated", "div": "PRIVATE_SENTINEL"]
            case "extension": composition["extension"] = [["url": "urn:private", "valueString": "PRIVATE_SENTINEL"]]
            case "attachment": document["document_reference"] = ["content": [["attachment": ["title": "/private/PRIVATE_SENTINEL"]]]]
            case "request": entries[0]["request"] = ["method": "POST", "url": "Composition"]
            case "final": composition["status"] = "final"
            case "section_order": composition["section"] = (composition["section"] as! [[String: Any]]).reversed().map { $0 }
            default: composition["author"] = [["reference": "https://ehr.invalid/PRIVATE_SENTINEL"]]
            }
            entries[0]["resource"] = composition
            bundle["entry"] = entries
            document["bundle"] = bundle
            XCTAssertThrowsError(
                try ClinicalBriefFHIRDocument.validate(documentJSON: JSONSerialization.data(withJSONObject: document), privacyCheck: { _ in true })
            ) { XCTAssertFalse($0.localizedDescription.contains("PRIVATE_SENTINEL")) }
        }
    }

    func testFinalPrivacyCheckSeesNarrativeExtensionsAndAttachmentsAndSanitizesErrors() throws {
        let (brief, _) = try fixture()
        var seen: [String] = []
        _ = try ClinicalBriefFHIRDocument.export(brief, recordedAt: Date()) { text in
            seen.append(text)
            return true
        }
        XCTAssertTrue(seen.contains { $0.contains("dehydration") })
        XCTAssertTrue(seen.contains { $0.contains("provenance_digest") })
        XCTAssertTrue(seen.contains { $0.contains("application/fhir+json") })
        XCTAssertThrowsError(
            try ClinicalBriefFHIRDocument.export(
                brief, recordedAt: Date(),
                privacyCheck: { _ in
                    throw NSError(domain: "PRIVATE_SENTINEL", code: 1)
                })
        ) { XCTAssertEqual($0 as? ClinicalBriefFHIRDocumentError, .privacy) }
        XCTAssertThrowsError(try ClinicalBriefFHIRDocument.export(brief, recordedAt: Date(), privacyCheck: { _ in false }))
    }

    func testIncompleteProvenanceCannotExportEvenIfBriefWireValidationPassed() throws {
        let (brief, _) = try fixture()
        var packet = try XCTUnwrap(JSONSerialization.jsonObject(with: brief.responseJSON()) as? [String: Any])
        packet["provenance"] = [:]
        packet.removeValue(forKey: "summary")
        packet.removeValue(forKey: "digest")
        packet["digest"] = try ClinicalBrief.hash(ClinicalBrief.canonical(packet))
        packet["summary"] = brief.summary
        let acceptedWire = try ClinicalBrief.validate(
            evaluationJSON: ClinicalBrief.canonical(packet), source: brief.summary,
            generatedSummary: brief.summary, originalIdentifiers: [], privacyCheck: { _ in true })
        XCTAssertThrowsError(try ClinicalBriefFHIRDocument.export(acceptedWire, recordedAt: Date(), privacyCheck: { _ in true })) {
            XCTAssertEqual($0 as? ClinicalBriefFHIRDocumentError, .missingProvenance)
        }
    }
}
