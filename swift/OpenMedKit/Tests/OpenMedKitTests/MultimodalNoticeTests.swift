import Foundation
import XCTest

@testable import OpenMedKit

final class MultimodalNoticeTests: XCTestCase {
    private func fixtures() throws -> [[String: Any]] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/notices_v1.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    func testSharedPythonSwiftCatalogAndWireFixtures() throws {
        for fixture in try fixtures() {
            let object = try XCTUnwrap(fixture["result"] as? [String: Any])
            let input = try JSONSerialization.data(withJSONObject: object)
            let output: Data
            if fixture["kind"] as? String == "vision_generation" {
                let generation = try JSONDecoder().decode(OpenMedVisionLanguageGeneration.self, from: input)
                XCTAssertEqual(generation.notice, MultimodalNoticeKind.visualDescription.notice)
                XCTAssertTrue(generation.description.contains(generation.notice.text))
                XCTAssertTrue(generation.description.contains(generation.notice.identifier))
                XCTAssertThrowsError(try generation.requireReviewerConfirmation(reviewerConfirmed: false))
                try generation.requireReviewerConfirmation(reviewerConfirmed: true)
                output = try JSONEncoder().encode(generation)
            } else {
                let result = try JSONDecoder().decode(MultimodalReviewResult.self, from: input)
                let kind = try XCTUnwrap(MultimodalNoticeKind(rawValue: fixture["kind"] as? String ?? ""))
                XCTAssertEqual(result.notice, kind.notice)
                XCTAssertTrue(result.description.contains(result.notice.text))
                XCTAssertTrue(result.description.contains(result.notice.identifier))
                XCTAssertThrowsError(try result.requireReviewerConfirmation(reviewerConfirmed: false))
                try result.requireReviewerConfirmation(reviewerConfirmed: true)
                output = try JSONEncoder().encode(result)
            }
            XCTAssertEqual(try JSONSerialization.jsonObject(with: output) as? NSDictionary, object as NSDictionary)
        }
    }

    func testMissingOrAlteredNoticesAndSafetyFlagsFailDecoding() throws {
        for fixture in try fixtures() {
            let original = try XCTUnwrap(fixture["result"] as? [String: Any])
            for mutation in ["notice", "identifier", "text", "wrongText", "wrongID", "review", "diagnostic", "extra"] {
                var object = original
                var notice = try XCTUnwrap(object["notice"] as? [String: Any])
                switch mutation {
                case "notice": object.removeValue(forKey: "notice")
                case "identifier", "text":
                    notice.removeValue(forKey: mutation)
                    object["notice"] = notice
                case "wrongText":
                    notice["text"] = "Synthetic Ada at 72 bpm."
                    object["notice"] = notice
                case "wrongID":
                    notice["identifier"] = "synthetic-mrn-98765"
                    object["notice"] = notice
                case "review": object["requires_reviewer_confirmation"] = false
                case "diagnostic": object["is_diagnostic"] = true
                default: object["unexpected"] = "Synthetic Ada"
                }
                let data = try JSONSerialization.data(withJSONObject: object)
                if fixture["kind"] as? String == "vision_generation" {
                    XCTAssertThrowsError(try JSONDecoder().decode(OpenMedVisionLanguageGeneration.self, from: data))
                } else {
                    XCTAssertThrowsError(try JSONDecoder().decode(MultimodalReviewResult.self, from: data))
                }
            }
        }
    }

    func testConstructorsRejectWrongKindAndPatientValueInterpolation() throws {
        for kind in MultimodalNoticeKind.allCases {
            let wrong = kind == .draft ? MultimodalNoticeKind.measurement.notice : MultimodalNoticeKind.draft.notice
            XCTAssertThrowsError(try MultimodalReviewResult(kind: kind, outputDigest: String(repeating: "a", count: 64), notice: wrong))
            XCTAssertThrowsError(try MultimodalReviewResult(kind: kind, outputDigest: "Synthetic Ada", notice: kind.notice))
            XCTAssertThrowsError(try MultimodalNotice(identifier: kind.notice.identifier, text: "Synthetic Ada at 72 bpm."))
            XCTAssertFalse(kind.notice.text.contains("72 bpm"))
            XCTAssertFalse(kind.notice.text.contains("Ada"))
        }
        XCTAssertThrowsError(
            try OpenMedVisionLanguageGeneration(
                text: "Synthetic geometric shapes", promptTokenCount: 1, generationTokenCount: 1,
                promptTime: 0, generationTime: 0, notice: MultimodalNoticeKind.draft.notice
            ))
    }

    func testVisionDebugRenderingExcludesProtectedContent() throws {
        let result = try OpenMedVisionLanguageGeneration(
            text: "Synthetic Ada synthetic-mrn-98765", tokenIDs: [98765],
            promptTokenCount: 1, generationTokenCount: 1,
            promptTime: 0, generationTime: 0, notice: MultimodalNoticeKind.visualDescription.notice
        )
        XCTAssertFalse(String(reflecting: result).contains("Ada"))
        XCTAssertFalse(String(reflecting: result).contains("98765"))
        XCTAssertTrue(result.description.contains(result.notice.text))
    }
}
