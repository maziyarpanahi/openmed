import XCTest
@testable import OpenMedKit

final class PostProcessingTests: XCTestCase {

    func testDecodeSingleEntity() {
        let text = "Patient John Doe visited"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 1, label: "B-first_name", score: 0.95, startOffset: 8, endOffset: 12),
            .init(labelId: 2, label: "I-first_name", score: 0.90, startOffset: 13, endOffset: 16),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text)

        XCTAssertEqual(entities.count, 1)
        XCTAssertEqual(entities[0].label, "first_name")
        XCTAssertEqual(entities[0].text, "John Doe")
        XCTAssertEqual(entities[0].start, 8)
        XCTAssertEqual(entities[0].end, 16)
    }

    func testDecodeMultipleEntities() {
        let text = "John Doe 555-1234"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 1, label: "B-first_name", score: 0.95, startOffset: 0, endOffset: 4),
            .init(labelId: 2, label: "I-first_name", score: 0.90, startOffset: 5, endOffset: 8),
            .init(labelId: 0, label: "O", score: 0.99, startOffset: 8, endOffset: 9),
            .init(labelId: 3, label: "B-phone", score: 0.85, startOffset: 9, endOffset: 17),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text)

        XCTAssertEqual(entities.count, 2)
        XCTAssertEqual(entities[0].label, "first_name")
        XCTAssertEqual(entities[1].label, "phone")
    }

    func testAverageAggregation() {
        let text = "John Doe"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 1, label: "B-NAME", score: 0.90, startOffset: 0, endOffset: 4),
            .init(labelId: 2, label: "I-NAME", score: 0.80, startOffset: 5, endOffset: 8),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text, strategy: .average)

        XCTAssertEqual(entities.count, 1)
        XCTAssertEqual(entities[0].confidence, 0.85, accuracy: 0.01)
    }

    func testMaxAggregation() {
        let text = "John Doe"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 1, label: "B-NAME", score: 0.90, startOffset: 0, endOffset: 4),
            .init(labelId: 2, label: "I-NAME", score: 0.80, startOffset: 5, endOffset: 8),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text, strategy: .max)

        XCTAssertEqual(entities.count, 1)
        XCTAssertEqual(entities[0].confidence, 0.90, accuracy: 0.01)
    }

    func testFirstAggregation() {
        let text = "John Doe"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 1, label: "B-NAME", score: 0.90, startOffset: 0, endOffset: 4),
            .init(labelId: 2, label: "I-NAME", score: 0.80, startOffset: 5, endOffset: 8),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text, strategy: .first)

        XCTAssertEqual(entities.count, 1)
        XCTAssertEqual(entities[0].confidence, 0.90, accuracy: 0.01)
    }

    func testEmptyTokens() {
        let entities = PostProcessing.decodeEntities(tokens: [], text: "")
        XCTAssertTrue(entities.isEmpty)
    }

    func testAllOTokens() {
        let text = "Hello world"
        let tokens: [PostProcessing.TokenPrediction] = [
            .init(labelId: 0, label: "O", score: 0.99, startOffset: 0, endOffset: 5),
            .init(labelId: 0, label: "O", score: 0.99, startOffset: 6, endOffset: 11),
        ]
        let entities = PostProcessing.decodeEntities(tokens: tokens, text: text)
        XCTAssertTrue(entities.isEmpty)
    }

    func testRepairEntitySpansExtendsTruncatedEnd() {
        let text = "Patient Maria Garcia"
        let entities = [
            EntityPrediction(label: "NAME", text: "Mari", confidence: 0.9, start: 8, end: 12),
        ]

        let repaired = PostProcessing.repairEntitySpans(entities, text: text)

        XCTAssertEqual(repaired.count, 1)
        XCTAssertEqual(repaired[0].text, "Maria")
        XCTAssertEqual(repaired[0].start, 8)
        XCTAssertEqual(repaired[0].end, 13)
    }

    func testMergePIIEntitiesCombinesFragmentedSSN() {
        let text = "Patient SSN: 123-45-6789"
        let entities = [
            EntityPrediction(label: "ssn", text: "123", confidence: 0.90, start: 13, end: 16),
            EntityPrediction(label: "ssn", text: "45", confidence: 0.85, start: 17, end: 19),
            EntityPrediction(label: "ssn", text: "6789", confidence: 0.88, start: 20, end: 24),
        ]

        let merged = PostProcessing.mergePIIEntities(entities, text: text)

        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(merged[0].label, "ssn")
        XCTAssertEqual(merged[0].text, "123-45-6789")
        XCTAssertEqual(merged[0].start, 13)
        XCTAssertEqual(merged[0].end, 24)
        XCTAssertEqual(merged[0].confidence, 0.918, accuracy: 0.02)
    }

    func testMergePIIEntitiesPrefersSpecificDOBLabel() {
        let text = "DOB: 01/15/1970"
        let entities = [
            EntityPrediction(label: "date", text: "01", confidence: 0.70, start: 5, end: 7),
            EntityPrediction(label: "date_of_birth", text: "/15/1970", confidence: 0.90, start: 7, end: 15),
        ]

        let merged = PostProcessing.mergePIIEntities(entities, text: text)

        XCTAssertEqual(merged.count, 1)
        XCTAssertEqual(merged[0].label, "date_of_birth")
        XCTAssertEqual(merged[0].text, "01/15/1970")
        XCTAssertEqual(merged[0].start, 5)
        XCTAssertEqual(merged[0].end, 15)
        XCTAssertEqual(merged[0].confidence, 0.876, accuracy: 0.03)
    }

    func testMergePIIEntitiesKeepsNonSemanticEntities() {
        let text = "Patient John Doe arrived"
        let entities = [
            EntityPrediction(label: "first_name", text: "John Doe", confidence: 0.95, start: 8, end: 16),
        ]

        let merged = PostProcessing.mergePIIEntities(entities, text: text)

        XCTAssertEqual(merged, entities)
    }

    func testMergePIIEntitiesAddsSemanticOnlyShowcaseMatches() {
        let text = """
        Patient: John Doe, DOB: 01/15/1970, SSN: 000-00-0000, MRN: MRN-TEST-88421, Address: 123 Example Street, Apt 4B, Springfield, CA 90000, Phone: (555) 010-2244, Email: john.doe@example.test, Insurance ID: TEST-POLICY-778291, Driver License: DLT-TEST-442190, Passport: P-TEST-998877, Emergency Contact: Jane Doe, (555) 010-7788, Employer: Example Manufacturing LLC, Employee ID: EMP-20481, Bank Account: ACCT-TEST-55667788, Routing: 000000000.
        """

        let merged = PostProcessing.mergePIIEntities([], text: text)

        XCTAssertTrue(merged.contains { $0.label == "full_name" && $0.text == "John Doe" })
        XCTAssertTrue(merged.contains { $0.label == "full_name" && $0.text == "Jane Doe" })
        XCTAssertTrue(merged.contains { $0.label == "date_of_birth" && $0.text == "01/15/1970" && $0.confidence >= 0.95 })
        XCTAssertTrue(merged.contains { $0.label == "ssn" && $0.text == "000-00-0000" && $0.confidence >= 0.95 })
        XCTAssertTrue(merged.contains { $0.label == "medical_record_number" && $0.text == "MRN-TEST-88421" })
        XCTAssertTrue(merged.contains { $0.label == "street_address" && $0.text == "123 Example Street, Apt 4B, Springfield, CA 90000" })
        XCTAssertEqual(merged.filter { $0.label == "phone_number" }.map(\.text).sorted(), ["(555) 010-2244", "(555) 010-7788"])
        XCTAssertTrue(merged.contains { $0.label == "email" && $0.text == "john.doe@example.test" })
        XCTAssertTrue(merged.contains { $0.label == "insurance_id" && $0.text == "TEST-POLICY-778291" })
        XCTAssertTrue(merged.contains { $0.label == "driver_license" && $0.text == "DLT-TEST-442190" })
        XCTAssertTrue(merged.contains { $0.label == "passport_number" && $0.text == "P-TEST-998877" })
        XCTAssertTrue(merged.contains { $0.label == "organization" && $0.text == "Example Manufacturing LLC" })
        XCTAssertTrue(merged.contains { $0.label == "employee_id" && $0.text == "EMP-20481" })
        XCTAssertTrue(merged.contains { $0.label == "account_number" && $0.text == "ACCT-TEST-55667788" })
        XCTAssertTrue(merged.contains { $0.label == "routing_number" && $0.text == "000000000" })
    }
}
