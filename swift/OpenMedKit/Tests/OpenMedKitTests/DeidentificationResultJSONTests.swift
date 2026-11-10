import Foundation
import XCTest

@testable import OpenMedKit

final class DeidentificationResultJSONTests: XCTestCase {
    func testDeidentificationResultEncodesPythonSchemaKeys() throws {
        let original = "Alex Example has ID TEST-123."
        let entities = [
            EntityPrediction(
                label: "full_name",
                text: "Alex Example",
                confidence: 1.0,
                start: 0,
                end: 12
            ),
            EntityPrediction(
                label: "medical_record_number",
                text: "TEST-123",
                confidence: 0.5,
                start: 20,
                end: 28
            ),
        ]
        let result = DeidentificationResult(
            originalText: original,
            deidentifiedText: "[FULL_NAME] has ID [MEDICAL_RECORD_NUMBER].",
            entities: entities,
            method: "mask",
            timestamp: Date(timeIntervalSince1970: 0)
        )

        let json = try result.toJSON()
        let expected = """
            {"deidentified_text":"[FULL_NAME] has ID [MEDICAL_RECORD_NUMBER].","metadata":{},"method":"mask","num_entities_redacted":2,"original_text":"Alex Example has ID TEST-123.","pii_entities":[{"action":"mask","confidence":1,"end":12,"entity_type":"full_name","label":"full_name","start":0,"text":"Alex Example"},{"action":"mask","confidence":0.5,"end":28,"entity_type":"medical_record_number","label":"medical_record_number","start":20,"text":"TEST-123"}],"timestamp":"1970-01-01T00:00:00Z"}
            """
        XCTAssertEqual(json, expected)

        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any]
        )
        let records = try XCTUnwrap(object["pii_entities"] as? [[String: Any]])
        XCTAssertEqual(
            Set(records[0].keys),
            ["action", "confidence", "end", "entity_type", "label", "start", "text"]
        )
        XCTAssertNil(records[0]["entityType"])
    }

    func testDeidentificationResultJSONIsDeterministic() throws {
        let result = DeidentificationResult(
            originalText: "Sample patient token.",
            deidentifiedText: "[FULL_NAME] token.",
            piiEntities: [
                .init(
                    text: "Sample patient",
                    label: "full_name",
                    start: 0,
                    end: 14,
                    confidence: 1.0,
                    action: "mask"
                )
            ],
            method: "mask",
            timestamp: Date(timeIntervalSince1970: 0)
        )

        XCTAssertEqual(try result.toJSON(), try result.toJSON())
    }
}
