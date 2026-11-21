import Foundation
import XCTest

@testable import OpenMedKit

final class OpenMedUseErrorTests: XCTestCase {
    func testSharedVocabularyAndScopedRefusals() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let vector = try JSONDecoder().decode([String].self, from: Data(contentsOf: root.appending(path: "tests/fixtures/parity/use_error_codes_v1.json")))
        XCTAssertEqual(OpenMedUseErrorCode.allCases.map(\.rawValue).sorted(), vector)
        let recorder = OpenMedUseErrorRecorder()
        let notice = try OpenMedDisclaimer.forTier(.restricted)
        notice.didLeave(with: notice.acknowledge(at: Date()), recorder: recorder)
        XCTAssertTrue(recorder.snapshot().isEmpty)
        notice.didLeave(with: nil, recorder: recorder)
        try OpenMedUseErrorRecorder.$current.withValue(recorder) {
            XCTAssertThrowsError(try OpenMedClinicianMode.require("decisions"))
            try OpenMedClinicianMode.require("analyze_text")
        }
        XCTAssertThrowsError(try OpenMedClinicianMode.require("decisions"))
        let before = recorder.snapshot()
        DispatchQueue.concurrentPerform(iterations: 50) { _ in recorder.record(.exportBeforeReview) }
        XCTAssertEqual(before, ["acknowledgement_skipped": 1, "restricted_capability_refused": 1])
        XCTAssertEqual(recorder.snapshot()["export_before_review"], 50)
    }
}
