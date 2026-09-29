import Foundation
import XCTest

@testable import OpenMedKit

final class ClinicalBriefTests: XCTestCase {
    private func fixture() throws -> (String, String, Data) {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/clinical/brief_parity/verified.json"))
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (
            try XCTUnwrap(row["source"] as? String),
            try XCTUnwrap(row["generator_output"] as? String),
            Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
        )
    }

    func testSharedPythonPacketIsByteIdenticalAfterNativeGuards() throws {
        let (source, summary, data) = try fixture()
        let brief = try ClinicalBrief.validate(
            evaluationJSON: data, source: source,
            generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true })
        XCTAssertEqual(brief.responseJSON(), data)
        XCTAssertEqual(brief.summary, summary)
        XCTAssertEqual(brief.citations.count, 3)
        XCTAssertFalse(String(decoding: brief.auditJSON(), as: UTF8.self).contains("dehydration"))
        XCTAssertFalse(brief.description.contains("dehydration"))
    }

    func testChangedSourceAndOutputAreRejected() throws {
        let (source, summary, data) = try fixture()
        for (input, output) in [(source + " changed", summary), (source, summary + " fabricated")] {
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: data,
                    source: input, generatedSummary: output, originalIdentifiers: [], privacyCheck: { _ in true }))
        }
    }

    func testSourceIdentifierAndPrivacyDetectorFailuresAreRejected() throws {
        let (source, summary, data) = try fixture()
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: data, source: source,
                generatedSummary: summary, originalIdentifiers: ["dehydration"], privacyCheck: { _ in true })
        ) {
            XCTAssertEqual($0 as? ClinicalBriefError, .privacy)
        }
        XCTAssertThrowsError(
            try ClinicalBrief.validate(
                evaluationJSON: data, source: source,
                generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in false }))
    }

    func testMalformedAndUnsafePacketsAreRejected() throws {
        let (source, summary, data) = try fixture()
        let packet = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for key in ["citations", "verdicts", "envelope", "digest"] {
            var mutated = packet
            mutated.removeValue(forKey: key)
            XCTAssertThrowsError(
                try ClinicalBrief.validate(
                    evaluationJSON: JSONSerialization.data(withJSONObject: mutated), source: source,
                    generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true }))
        }
    }

    func testBriefTaskIsStructuredAndNeverStreamsUncheckedText() throws {
        let request = OpenMedMapleRequest(task: .brief, document: "Synthetic evidence.")
        let messages = OpenMedMaplePrompt.messages(for: request)
        XCTAssertTrue(messages.last?.content.contains("verbatim") == true)
        let result = try OpenMedMapleOutputParser.parse(
            "{\"answer\":\"Synthetic evidence.\"}",
            task: .brief, sourceDocument: request.document)
        XCTAssertEqual(result.answer, "Synthetic evidence.")
    }
}
