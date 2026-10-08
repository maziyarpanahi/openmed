import Foundation
import XCTest

@testable import OpenMedKit

final class ClinicalBriefTests: XCTestCase {
    private func fixtureObject() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/clinical/brief_parity/verified.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func fixture() throws -> (String, String, Data) {
        let row = try fixtureObject()
        return (
            try XCTUnwrap(row["source"] as? String),
            try XCTUnwrap(row["generator_output"] as? String),
            Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
        )
    }

    func testSharedUnicodeLeakageCasesHavePythonParityAndValueFreeRefusals() throws {
        let rows = try XCTUnwrap(fixtureObject()["leakage_cases"] as? [[String: Any]])
        for row in rows {
            let id = try XCTUnwrap(row["id"] as? String)
            let identifier = try XCTUnwrap(row["surface"] as? String)
            let candidate = try XCTUnwrap(row["candidate"] as? String)
            let leaked = try XCTUnwrap((row["native_leaked"] ?? row["leaked"]) as? Bool)
            let data = Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
            XCTAssertEqual(
                try ClinicalBriefLeakageMatcher.contains(
                    identifier,
                    in: ClinicalBriefLeakageMatcher.normalize(candidate)), leaked, id)
            var detectorCalls = 0
            if leaked {
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: data,
                        source: candidate, generatedSummary: candidate, originalIdentifiers: [identifier],
                        privacyCheck: { _ in
                            detectorCalls += 1
                            return true
                        }), id
                ) {
                    XCTAssertEqual($0 as? ClinicalBriefError, .privacy, id)
                    XCTAssertEqual($0.localizedDescription, "privacy", id)
                }
                XCTAssertEqual(detectorCalls, 0, id)
            } else {
                let brief = try ClinicalBrief.validate(
                    evaluationJSON: data,
                    source: candidate, generatedSummary: candidate, originalIdentifiers: [identifier],
                    privacyCheck: { _ in
                        detectorCalls += 1
                        return true
                    })
                XCTAssertEqual(brief.summary, candidate, id)
                XCTAssertEqual(detectorCalls, 1, id)
            }
        }
    }

    func testSharedNormalizationIncludesEveryRetainedConfusableMapping() throws {
        let rows = try XCTUnwrap(fixtureObject()["normalization_cases"] as? [[String: String]])
        for row in rows {
            let input = try XCTUnwrap(row["input"])
            let expected = try XCTUnwrap(row["normalized"])
            XCTAssertEqual(Array(ClinicalBriefLeakageMatcher.normalize(input).utf8), Array(expected.utf8))
        }
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
