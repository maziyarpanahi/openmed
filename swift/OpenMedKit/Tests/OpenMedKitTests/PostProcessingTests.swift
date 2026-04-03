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
}
