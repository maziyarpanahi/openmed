import Foundation
import XCTest

@testable import OpenMedKit

final class OffsetContractParityTests: XCTestCase {
    private struct Fixture: Decodable {
        let version: Int
        let offsetUnit: String
        let cases: [Case]

        struct Case: Decodable {
            let id: String
            let category: String
            let script: String
            let text: String
            let inputStart: Int
            let inputEnd: Int
            let expectedStart: Int
            let expectedEnd: Int
            let replacement: String
            let expectedRedacted: String
        }
    }

    func testSharedUnicodeScalarOffsetContract() throws {
        let fixture = try loadFixture()

        XCTAssertEqual(fixture.version, 1)
        XCTAssertEqual(fixture.offsetUnit, "unicode_scalar")
        XCTAssertGreaterThanOrEqual(fixture.cases.count, 40)
        XCTAssertEqual(
            Set(
                fixture.cases
                    .filter { $0.category == "indic" }
                    .map(\.script)
            ),
            Set(["Deva", "Beng", "Guru", "Gujr", "Orya", "Taml", "Telu", "Knda", "Mlym"])
        )

        for fixtureCase in fixture.cases {
            let snapped = PostProcessing.snapScalarSpanToGraphemeBoundaries(
                start: fixtureCase.inputStart,
                end: fixtureCase.inputEnd,
                in: fixtureCase.text
            )
            XCTAssertEqual(
                snapped.start,
                fixtureCase.expectedStart,
                fixtureCase.id
            )
            XCTAssertEqual(
                snapped.end,
                fixtureCase.expectedEnd,
                fixtureCase.id
            )
            XCTAssertTrue(
                PostProcessing.isGraphemeBoundary(
                    snapped.start,
                    in: fixtureCase.text
                ),
                fixtureCase.id
            )
            XCTAssertTrue(
                PostProcessing.isGraphemeBoundary(
                    snapped.end,
                    in: fixtureCase.text
                ),
                fixtureCase.id
            )

            let decoded = PostProcessing.decodeEntities(
                tokens: [
                    .init(
                        labelId: 1,
                        label: "B-NAME",
                        score: 0.99,
                        startOffset: fixtureCase.inputStart,
                        endOffset: fixtureCase.inputEnd
                    )
                ],
                text: fixtureCase.text
            )
            XCTAssertEqual(decoded.count, 1, fixtureCase.id)
            XCTAssertEqual(
                decoded.first?.start,
                fixtureCase.expectedStart,
                fixtureCase.id
            )
            XCTAssertEqual(
                decoded.first?.end,
                fixtureCase.expectedEnd,
                fixtureCase.id
            )

            let entity = EntityPrediction(
                label: "NAME",
                text: "partial",
                confidence: 0.99,
                start: fixtureCase.inputStart,
                end: fixtureCase.inputEnd
            )
            let normalized = try XCTUnwrap(
                entity.snappedToGraphemeBoundaries(in: fixtureCase.text),
                fixtureCase.id
            )
            XCTAssertEqual(normalized.start, fixtureCase.expectedStart, fixtureCase.id)
            XCTAssertEqual(normalized.end, fixtureCase.expectedEnd, fixtureCase.id)
            XCTAssertEqual(
                normalized.text,
                PostProcessing.scalarSubstring(
                    fixtureCase.text,
                    start: fixtureCase.expectedStart,
                    end: fixtureCase.expectedEnd
                ),
                fixtureCase.id
            )

            let redacted = try XCTUnwrap(
                PostProcessing.replacingScalarSpan(
                    in: fixtureCase.text,
                    start: normalized.start,
                    end: normalized.end,
                    with: fixtureCase.replacement
                ),
                fixtureCase.id
            )
            XCTAssertEqual(redacted, fixtureCase.expectedRedacted, fixtureCase.id)
            XCTAssertEqual(
                Data(redacted.utf8),
                Data(fixtureCase.expectedRedacted.utf8),
                fixtureCase.id
            )

            let result = PlatformModel.redact(
                fixtureCase.text,
                entities: decoded,
                method: .mask
            )
            XCTAssertEqual(
                result.deidentifiedText,
                fixtureCase.expectedRedacted,
                fixtureCase.id
            )
            XCTAssertEqual(result.piiEntities.count, 1, fixtureCase.id)
            XCTAssertEqual(
                result.piiEntities.first?.start,
                fixtureCase.expectedStart,
                fixtureCase.id
            )
            XCTAssertEqual(
                result.piiEntities.first?.end,
                fixtureCase.expectedEnd,
                fixtureCase.id
            )
        }
    }

    private func loadFixture() throws -> Fixture {
        let url = repositoryRoot()
            .appending(path: "tests/fixtures/parity/offset_contract.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func repositoryRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return url
    }
}
