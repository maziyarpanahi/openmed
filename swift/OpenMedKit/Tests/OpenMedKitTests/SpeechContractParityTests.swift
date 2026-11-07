import Foundation
import XCTest

@testable import OpenMedKit

final class SpeechContractParityTests: XCTestCase {
    private func corpus() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "tests/fixtures/parity/speech_contract_v1.json"))) as? [String: Any])
    }

    func testSharedTokenTimingUnitsAndProtectedText() throws {
        let fixture = try corpus()
        let segment = try XCTUnwrap(fixture["segment"] as? [String: Any])
        let rows = try XCTUnwrap(segment["token_times"] as? [[String: Any]])
        func token(_ row: [String: Any]) throws -> SpeechTokenTiming {
            try SpeechTokenTiming(tokenIndex: XCTUnwrap(row["token_index"] as? Int), startMilliseconds: XCTUnwrap(row["start_ms"] as? Int64), endMilliseconds: XCTUnwrap(row["end_ms"] as? Int64), confidencePPM: XCTUnwrap(row["confidence_ppm"] as? Int), text: XCTUnwrap(row["token"] as? String))
        }
        let result = try SpeechRecognition(text: XCTUnwrap(segment["text"] as? String), tokenTimes: rows.map(token))
        XCTAssertEqual(result.text.unicodeScalars.count, 6)
        XCTAssertEqual(result.tokenTimes.map(\.startMilliseconds), [500, 650])
        XCTAssertEqual(result.tokenTimes.map(\.confidencePPM), [750_000, 250_000])
        XCTAssertFalse(String(reflecting: result).contains("Café"))
        for patch in try XCTUnwrap(fixture["invalid_tokens"] as? [[String: Any]]) {
            XCTAssertThrowsError(try token(rows[0].merging(patch) { _, new in new }))
        }
    }

    func testSharedProviderProvenanceGrammar() throws {
        let fixture = try corpus()
        let row = try XCTUnwrap(fixture["provenance"] as? [String: String])
        func validate(_ row: [String: String]) throws {
            _ = try SpeechEngineProvenance(providerID: XCTUnwrap(row["provider_id"]), providerRevision: XCTUnwrap(row["provider_revision"]), modelFingerprint: XCTUnwrap(row["model_fingerprint"]), language: XCTUnwrap(row["language"]))
        }
        try validate(row)
        for patch in try XCTUnwrap(fixture["invalid_provenance"] as? [[String: String]]) {
            var invalid = row
            invalid[try XCTUnwrap(patch["field"])] = try XCTUnwrap(patch["value"])
            XCTAssertThrowsError(try validate(invalid))
        }
    }
}
