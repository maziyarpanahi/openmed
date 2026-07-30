import Foundation
import XCTest

@testable import OpenMedKit

final class CoreMLParitySmokeTests: XCTestCase {
    func testLoadsPythonParityCorpusAndAcceptsIdenticalSwiftSpans() throws {
        let text = "Patient John Doe arrived."
        let corpusURL = try writeCorpus(
            textSHA256: CoreMLParitySmokeHarness.sha256Hex(text)
        )

        let fixtures = try CoreMLParitySmokeHarness.loadCorpus(from: corpusURL)
        XCTAssertEqual(fixtures.count, 1)
        XCTAssertTrue(
            CoreMLParitySmokeHarness.validateTextHash(text, fixture: fixtures[0])
        )

        let predictions = [
            "stub-note": [
                EntityPrediction(
                    label: "PERSON",
                    text: "John Doe",
                    confidence: 0.99,
                    start: 8,
                    end: 16
                )
            ]
        ]
        let result = CoreMLParitySmokeHarness.validate(
            fixtures: fixtures,
            predictions: predictions
        )

        XCTAssertTrue(result.passed)
        XCTAssertEqual(result.checkedFixtureIDs, ["stub-note"])
        XCTAssertTrue(result.failures.isEmpty)
    }

    func testReportsSpanMismatch() throws {
        let corpusURL = try writeCorpus(textSHA256: "fixture-hash")
        let fixtures = try CoreMLParitySmokeHarness.loadCorpus(from: corpusURL)
        let predictions = [
            "stub-note": [
                EntityPrediction(
                    label: "DATE",
                    text: "today",
                    confidence: 0.90,
                    start: 17,
                    end: 22
                )
            ]
        ]

        let result = CoreMLParitySmokeHarness.validate(
            fixtures: fixtures,
            predictions: predictions
        )

        XCTAssertFalse(result.passed)
        XCTAssertEqual(result.failures.first?.fixtureID, "stub-note")
    }

    private func writeCorpus(textSHA256: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("json")
        let payload = """
            {
              "schema_version": 1,
              "fixtures": [
                {
                  "fixture_id": "stub-note",
                  "text_sha256": "\(textSHA256)",
                  "python_reference_spans": [
                    {
                      "label": "PERSON",
                      "start": 8,
                      "end": 16,
                      "text": "John Doe"
                    }
                  ]
                }
              ]
            }
            """
        try payload.write(to: url, atomically: true, encoding: .utf8)
        return url
    }
}
