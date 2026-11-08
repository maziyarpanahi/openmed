import Foundation
import XCTest

@testable import OpenMedKit

final class AmbientFHIRDocumentTests: XCTestCase {
    private func fixture() throws -> [String: Any] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/clinical/ambient_fhir.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func encode(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .withoutEscapingSlashes])
    }

    func testSharedPythonGoldenAndRoundtrip() throws {
        let fixture = try fixture()
        let draft = try XCTUnwrap(fixture["draft"] as? [String: Any])
        let hash = try XCTUnwrap(fixture["digest"] as? String)
        let data = try encode(draft)
        XCTAssertEqual(try AmbientFHIRDocument.draftDigest(draftJSON: data), hash)
        let result = try AmbientFHIRDocument.export(draftJSON: data, currentEvidenceDigest: String(repeating: "1", count: 64), recordedAt: "2026-01-02T03:04:05Z", confirmedDigest: hash)
        XCTAssertEqual(result.documentJSON, try encode(fixture["final_document"]!))
        XCTAssertTrue(AmbientFHIRDocument.validate(documentJSON: result.documentJSON))
        XCTAssertEqual(try AmbientFHIRDocument.importDocument(documentJSON: result.documentJSON), data)
        XCTAssertEqual(result.losses.map(\.category), ["transcript_payload", "audio_payload", "review_history"])
        XCTAssertFalse(result.description.contains("Synthetic patient"))
        let preliminary = try AmbientFHIRDocument.export(draftJSON: data, currentEvidenceDigest: String(repeating: "1", count: 64), recordedAt: "2026-01-02T03:04:05Z")
        XCTAssertTrue(AmbientFHIRDocument.validate(documentJSON: preliminary.documentJSON))
        XCTAssertEqual(try AmbientFHIRDocument.importDocument(documentJSON: preliminary.documentJSON), data)
    }

    func testRefusesUnreviewedStaleAndPendingDrafts() throws {
        let fixture = try fixture()
        let original = try XCTUnwrap(fixture["draft"] as? [String: Any])
        for final in [false, true] {
            for state in ["unreviewed", "stale_review", "stale_evidence", "pending", "confirmation"] {
                var draft = original
                var current = String(repeating: "1", count: 64)
                var confirmation = final ? fixture["digest"] as? String : nil
                var expected = AmbientFHIRDocumentError.unreviewedDraft
                switch state {
                case "unreviewed": draft["review"] = NSNull()
                case "stale_review":
                    var sections = draft["sections"] as! [[String: Any]]
                    sections[0]["note"] = "Synthetic changed note"
                    draft["sections"] = sections
                    expected = .staleReview
                case "stale_evidence":
                    current = String(repeating: "2", count: 64)
                    expected = .staleEvidence
                case "pending":
                    draft["correction_pending"] = true
                    expected = .correctionPending
                default:
                    confirmation = String(repeating: "2", count: 64)
                    expected = .confirmationMismatch
                }
                XCTAssertThrowsError(try AmbientFHIRDocument.export(draftJSON: encode(draft), currentEvidenceDigest: current, recordedAt: "2026-01-02T03:04:05Z", confirmedDigest: confirmation)) {
                    XCTAssertEqual($0 as? AmbientFHIRDocumentError, expected)
                }
            }
        }
    }

    func testClosedSubsetRejectsSourcePayloadAndTampering() throws {
        let fixture = try fixture()
        let secret = "SyntheticTranscriptOnlyIdentifier-7395"
        var draft = fixture["draft"] as! [String: Any]
        draft["transcript"] = secret
        XCTAssertThrowsError(try AmbientFHIRDocument.draftDigest(draftJSON: encode(draft))) {
            XCTAssertEqual($0 as? AmbientFHIRDocumentError, .invalidDraft)
            XCTAssertFalse($0.localizedDescription.contains(secret))
        }
        for kind in ["attachment", "extension", "narrative", "status", "request", "confirmation", "source"] {
            var bundle = fixture["final_document"] as! [String: Any]
            var entries = bundle["entry"] as! [[String: Any]]
            var composition = entries[0]["resource"] as! [String: Any]
            switch kind {
            case "attachment": composition["content"] = [["attachment": ["data": secret, "title": secret]]]
            case "extension":
                var extensions = composition["extension"] as! [[String: Any]]
                extensions.append(["url": secret, "valueString": secret])
                composition["extension"] = extensions
            case "narrative": composition["text"] = ["status": "additional", "div": secret]
            case "status": composition["status"] = "preliminary"
            case "request": entries[0]["request"] = ["method": "POST", "url": secret]
            case "confirmation":
                var extensions = composition["extension"] as! [[String: Any]]
                extensions[4]["valueBoolean"] = 1
                composition["extension"] = extensions
            default:
                var provenance = entries[1]["resource"] as! [String: Any]
                provenance["entity"] = [["role": "source", "what": ["display": secret]]]
                entries[1]["resource"] = provenance
            }
            entries[0]["resource"] = composition
            bundle["entry"] = entries
            let data = try encode(bundle)
            XCTAssertFalse(AmbientFHIRDocument.validate(documentJSON: data), kind)
            XCTAssertThrowsError(try AmbientFHIRDocument.importDocument(documentJSON: data)) {
                XCTAssertEqual($0 as? AmbientFHIRDocumentError, .invalidDocument)
                XCTAssertFalse($0.localizedDescription.contains(secret))
            }
        }
    }

    func testRejectsMalformedEvidenceOffsetsAndTime() throws {
        let fixture = try fixture()
        for value in [true, -1, 1.5, "/private/secret.wav"] as [Any] {
            var draft = fixture["draft"] as! [String: Any]
            var sections = draft["sections"] as! [[String: Any]]
            var evidence = sections[0]["evidence"] as! [[String: Any]]
            evidence[0]["start"] = value
            sections[0]["evidence"] = evidence
            draft["sections"] = sections
            XCTAssertThrowsError(try AmbientFHIRDocument.draftDigest(draftJSON: encode(draft)))
        }
        XCTAssertThrowsError(try AmbientFHIRDocument.export(draftJSON: encode(fixture["draft"]!), currentEvidenceDigest: String(repeating: "1", count: 64), recordedAt: "2026-02-30T00:00:00Z")) {
            XCTAssertEqual($0 as? AmbientFHIRDocumentError, .invalidRecordedTime)
        }
    }
}
