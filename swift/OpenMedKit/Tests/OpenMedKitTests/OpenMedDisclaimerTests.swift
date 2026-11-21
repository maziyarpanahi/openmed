import Foundation
import XCTest

@testable import OpenMedKit

final class OpenMedDisclaimerTests: XCTestCase {
    func testAllSharedNoticeAndTierVectors() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/safety_primitives_v1.json"))
        let vectors = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let timestamp = try XCTUnwrap(vectors["acknowledged_at"] as? String)
        let instant = try XCTUnwrap(ISO8601DateFormatter().date(from: timestamp))
        for row in try XCTUnwrap(vectors["tiers"] as? [[String: Any]]) {
            let tier = try XCTUnwrap(OpenMedRegulatoryTier(rawValue: try XCTUnwrap(row["tier"] as? String)))
            XCTAssertEqual(tier.obligations, Set(try XCTUnwrap(row["obligations"] as? [String])))
            XCTAssertEqual(try OpenMedDisclaimer.forTier(tier).id, row["notice_id"] as? String)
        }
        for row in try XCTUnwrap(vectors["notices"] as? [[String: Any]]) {
            let notice = try XCTUnwrap(OpenMedDisclaimer.notice(id: try XCTUnwrap(row["id"] as? String)))
            XCTAssertEqual(notice.version, row["version"] as? String)
            XCTAssertEqual(notice.versionDigest, row["version_digest"] as? String)
            XCTAssertEqual(notice.locale, row["locale"] as? String)
            XCTAssertEqual(notice.text, row["text"] as? String)
            XCTAssertEqual(notice.tier.rawValue, row["tier"] as? String)
            XCTAssertEqual(notice.requiresAcknowledgement, row["requires_acknowledgement"] as? Bool)
            XCTAssertEqual(notice.permitsResults(with: nil), !notice.requiresAcknowledgement)
            let receipt = notice.acknowledge(at: instant)
            XCTAssertTrue(notice.permitsResults(with: receipt))
            if let expected = row["acknowledgement"] as? [String: String] {
                let encoded = try JSONEncoder().encode(XCTUnwrap(receipt))
                XCTAssertEqual(try JSONDecoder().decode([String: String].self, from: encoded), expected)
            } else {
                XCTAssertNil(receipt)
            }
        }
        for key in ["capability_notices", "entry_point_overrides"] {
            for (capability, id) in try XCTUnwrap(vectors[key] as? [String: String]) {
                XCTAssertEqual(try OpenMedDisclaimer.forCapability(capability)?.id, id)
            }
        }
    }

    func testRequiredGateRejectsUnboundOrMalformedReceipts() throws {
        let notice = try OpenMedDisclaimer.forTier(.restricted)
        XCTAssertEqual(try OpenMedDisclaimer.forTier(.unknown), notice)
        let foreign = try OpenMedDisclaimer.forTier(.assistiveReview).acknowledge(at: Date())
        XCTAssertFalse(notice.permitsResults(with: foreign))
        let receipt = try XCTUnwrap(notice.acknowledge(at: Date(timeIntervalSince1970: 0.001234)))
        XCTAssertEqual(receipt.acknowledgedAt, "1970-01-01T00:00:00.001234+00:00")
        let bad = OpenMedNoticeAcknowledgement(noticeID: notice.id, versionDigest: notice.versionDigest, acknowledgedAt: "invalid")
        XCTAssertFalse(notice.permitsResults(with: bad))
    }
}
