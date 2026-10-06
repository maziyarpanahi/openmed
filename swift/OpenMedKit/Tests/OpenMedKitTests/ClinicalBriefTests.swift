import Foundation
import XCTest

@testable import OpenMedKit

final class ClinicalBriefTests: XCTestCase {
    func testMultilingualSyntheticComposerPacketsPreserveScalarOffsetsAndRefusals() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/eval/summaries/multilingual_brief_packets.json"))
        let corpus = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(corpus["schema_version"] as? Int, 1)
        XCTAssertEqual(corpus["synthetic_only"] as? Bool, true)
        let rows = try XCTUnwrap(corpus["cases"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 10)
        for row in rows {
            let source = try XCTUnwrap(row["source"] as? String)
            let summary = try XCTUnwrap(row["generator_output"] as? String)
            let wire = Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8)
            if row["scenario"] as? String == "unsupported_provider" {
                // Native validation admits successful reviewed packets only.
                // An unavailable language provider must never become output.
                XCTAssertEqual(summary, "")
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source, generatedSummary: summary,
                        originalIdentifiers: [], privacyCheck: { _ in true })
                ) {
                    XCTAssertEqual($0 as? ClinicalBriefError, .invalidPacket)
                }
                continue
            }
            let brief = try ClinicalBrief.validate(
                evaluationJSON: wire, source: source, generatedSummary: summary,
                originalIdentifiers: [], privacyCheck: { _ in true })
            XCTAssertEqual(brief.responseJSON(), wire)
            let audit = String(decoding: brief.auditJSON(), as: UTF8.self)
            XCTAssertFalse(audit.contains(source))
            if row["scenario"] as? String == "preserved" {
                XCTAssertEqual(brief.citations.count, 3)
                let input = Array(source.unicodeScalars)
                let output = Array(summary.unicodeScalars)
                for citation in brief.citations {
                    XCTAssertEqual(
                        Array(input[citation.sourceStart..<citation.sourceEnd]),
                        Array(output[citation.outputStart..<citation.outputEnd]))
                }
                // The source has a non-BMP prefix. Swift Character/UTF-16
                // offsets cannot be substituted for the wire's scalar offsets.
                XCTAssertNotEqual(source.utf16.count, input.count)
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source + " changed",
                        generatedSummary: summary, originalIdentifiers: [], privacyCheck: { _ in true }))
                XCTAssertThrowsError(
                    try ClinicalBrief.validate(
                        evaluationJSON: wire, source: source, generatedSummary: summary,
                        originalIdentifiers: [summary], privacyCheck: { _ in true }))
            }
        }
    }

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
