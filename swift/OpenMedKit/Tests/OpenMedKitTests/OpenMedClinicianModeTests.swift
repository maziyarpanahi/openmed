import Foundation
import XCTest

@testable import OpenMedKit

func withSyntheticClinicianReview<T>(_ operation: () throws -> T) async throws -> T {
    let mode = OpenMedClinicianMode()
    let acknowledgement = try OpenMedDisclaimer.forTier(.restricted).acknowledge(at: Date(timeIntervalSince1970: 0))
    try await mode.enable(purpose: .qualityAssurance, acknowledgement: acknowledgement)
    let scope = try await mode.reviewScope()
    let value = try scope.perform(operation)
    await mode.disable()
    return value
}

final class OpenMedClinicianModeTests: XCTestCase {
    func testMissingAndInvalidAcknowledgementsFail() async throws {
        let mode = OpenMedClinicianMode()
        do {
            _ = try await mode.enable(purpose: .review, acknowledgement: nil)
            XCTFail("missing acknowledgement")
        } catch {
            XCTAssertEqual(error as? OpenMedClinicianModeError, .acknowledgementRequired)
        }
        let lower = try OpenMedDisclaimer.forTier(.assistiveReview).acknowledge(at: Date())
        do {
            _ = try await mode.enable(purpose: .review, acknowledgement: lower)
            XCTFail("lower tier")
        } catch {
            XCTAssertEqual(error as? OpenMedClinicianModeError, .invalidAcknowledgement)
        }
    }

    func testRestrictedCallNeedsExplicitScopeAndRevocationIsImmediate() async throws {
        let mode = OpenMedClinicianMode()
        XCTAssertNoThrow(try OpenMedClinicianMode.require("deidentify"))
        XCTAssertThrowsError(try PhysiologicalRecordingQuality.evaluate([])) {
            XCTAssertEqual($0 as? OpenMedClinicianModeError, .clinicianModeRequired)
        }
        let ack = try OpenMedDisclaimer.forTier(.restricted).acknowledge(at: Date())
        let receipt = try await mode.enable(purpose: .research, acknowledgement: ack)
        let wire = try JSONDecoder().decode([String: String].self, from: JSONEncoder().encode(receipt))
        XCTAssertEqual(Set(wire.keys), ["notice_id", "notice_digest", "purpose"])
        let scope = try await mode.reviewScope()
        try scope.perform {
            try OpenMedClinicianMode.require("decisions")
            XCTAssertThrowsError(try PhysiologicalRecordingQuality.evaluate([])) {
                XCTAssertEqual($0 as? PhysiologicalWaveformError, .invalidRecording)
            }
        }
        XCTAssertThrowsError(try OpenMedClinicianMode.require("decisions"))
        try await scope.performAsync { try OpenMedClinicianMode.require("visual_qa") }
        await mode.disable()
        XCTAssertThrowsError(try scope.perform { try OpenMedClinicianMode.require("decisions") })
        let fresh = OpenMedClinicianMode()
        do {
            _ = try await fresh.reviewScope()
            XCTFail("fresh mode must be disabled")
        } catch {
            XCTAssertEqual(error as? OpenMedClinicianModeError, .clinicianModeRequired)
        }
    }
}
