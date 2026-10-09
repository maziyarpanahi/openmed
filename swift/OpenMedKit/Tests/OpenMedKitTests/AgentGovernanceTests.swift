import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

final class AgentGovernanceTests: XCTestCase {
    private func fixture() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/agent/governance_parity/v1.json"))
        let values = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(values["schema_version"] as? String, "openmed.tests.agent_governance_parity.v1")
        XCTAssertEqual(values["synthetic"] as? Bool, true)
        return values
    }

    private func canonical(kind: String, data: Data, now: Int64) throws -> Data {
        switch kind {
        case "artifact": return try AgentArtifactReference.parse(data).canonicalJSON()
        case "handoff":
            let value = try AgentReviewerHandoff.parse(data, now: Date(timeIntervalSince1970: Double(now)))
            XCTAssertFalse(value.authorizesClinicalAction)
            XCTAssertTrue(value.requiresHumanReview)
            return value.canonicalJSON()
        case "receipt":
            let value = try AgentApprovalReceipt.parse(data)
            XCTAssertFalse(value.authorizesClinicalAction)
            return value.canonicalJSON()
        case "run":
            let value = try AgentRunEvidence.parse(data)
            XCTAssertFalse(value.authorizesClinicalAction)
            return value.canonicalJSON()
        case "preview":
            let value = try AgentOMOPPreview.parse(data)
            XCTAssertFalse(value.authorizesClinicalAction)
            return value.canonicalJSON()
        case "verification":
            let value = try AgentApprovalEvidenceResult.parse(data)
            XCTAssertFalse(value.authorizesClinicalAction)
            return value.canonicalJSON()
        default: throw AgentGovernanceError.invalidMetadata
        }
    }

    func testSharedNativePythonVectorsHaveIdenticalCanonicalJSONAndDigests() throws {
        let values = try fixture()
        let now = try XCTUnwrap(values["now"] as? Int64)
        let rows = try XCTUnwrap(values["valid"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 16)
        for row in rows {
            let kind = try XCTUnwrap(row["kind"] as? String)
            let raw = Data(try XCTUnwrap(row["json"] as? String).utf8)
            let result = try canonical(kind: kind, data: raw, now: now)
            XCTAssertEqual(String(decoding: result, as: UTF8.self), row["canonical_json"] as? String, kind)
            let digest = "sha256:" + SHA256.hash(data: result).map { String(format: "%02x", $0) }.joined()
            XCTAssertEqual(digest, row["digest"] as? String, kind)
        }
    }

    func testSharedMalformedUnknownVersionAndContentVectorsFailClosed() throws {
        let values = try fixture()
        let now = try XCTUnwrap(values["now"] as? Int64)
        let rows = try XCTUnwrap(values["negative"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 65)
        for row in rows {
            let kind = try XCTUnwrap(row["kind"] as? String)
            let raw = Data(try XCTUnwrap(row["json"] as? String).utf8)
            XCTAssertThrowsError(try canonical(kind: kind, data: raw, now: now), "\(kind) \(row["name"] ?? "")") { error in
                XCTAssertEqual((error as? AgentGovernanceError)?.rawValue, row["swift_error"] as? String)
                XCTAssertFalse(String(describing: error).contains("SYNTHETIC_SECRET_SENTINEL"))
            }
        }
    }

    func testSharedLocalVerificationRefusalsAndBurnedPresentationsHavePythonParity() throws {
        let values = try fixture()
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        for row in try XCTUnwrap(values["verification_scenarios"] as? [[String: Any]]) {
            let mode = try XCTUnwrap(row["authority"] as? String)
            let now = try XCTUnwrap(row["now"] as? Int64)
            let actions = try XCTUnwrap(row["actions"] as? [String])
            let roles = try XCTUnwrap(row["roles"] as? [String])
            let expected = try XCTUnwrap(row["expected_json"] as? [String])
            let verifier = AgentLocalApprovalEvidenceVerifier(
                authority: { _ in
                    switch mode {
                    case "recognized": return .recognized
                    case "unrecognized": return .unrecognized
                    case "unsupported": return .unsupported
                    default: throw TestFailure.privateText
                    }
                }, clock: { now })
            for index in actions.indices {
                let result = try verifier.verify(receipt, actionDigest: actions[index], reviewerRole: roles[index])
                XCTAssertEqual(String(decoding: result.canonicalJSON(), as: UTF8.self), expected[index])
                XCTAssertFalse(result.authorizesClinicalAction)
                XCTAssertFalse(result.description.contains(receipt.actionDigest))
                XCTAssertFalse(result.description.contains("SYNTHETIC_SECRET_SENTINEL"))
            }
        }
    }

    func testUnconfiguredCustodyIsExplicitlyUnsupported() throws {
        let values = try fixture()
        let now = try XCTUnwrap(values["now"] as? Int64)
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        let verifier = AgentLocalApprovalEvidenceVerifier(clock: { now })
        let result = try verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole)
        XCTAssertEqual(result.reasonCode, .unsupportedAuthority)
        XCTAssertFalse(result.authorizesClinicalAction)
    }

    func testNativePreviewReceiptBindingRejectsChangedBatchAndBurnsPresentation() throws {
        let values = try fixture()
        let rows = try XCTUnwrap(values["valid"] as? [[String: Any]])
        let row = try XCTUnwrap(rows.first { $0["kind"] as? String == "preview" })
        let preview = try AgentOMOPPreview.parse(Data(try XCTUnwrap(row["canonical_json"] as? String).utf8))
        let changed = try AgentOMOPPreview.parse(Data(try XCTUnwrap(values["changed_preview_json"] as? String).utf8))
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["bound_preview_receipt_json"] as? String).utf8))
        XCTAssertEqual(receipt.actionDigest, preview.previewDigest)
        let digest = receipt.receiptDigest
        let now = receipt.consumedAt
        let verifier = AgentLocalApprovalEvidenceVerifier(authority: { $0 == digest ? .recognized : .unrecognized }, clock: { now })
        XCTAssertEqual(try verifier.verify(receipt, actionDigest: changed.previewDigest, reviewerRole: receipt.reviewerRole).reasonCode, .actionMismatch)
        XCTAssertEqual(try verifier.verify(receipt, actionDigest: preview.previewDigest, reviewerRole: receipt.reviewerRole).reasonCode, .replayed)
    }

    func testCustodyReceivesOnlyCanonicalDigestAndRejectsChangedReceipt() throws {
        let values = try fixture()
        let now = try XCTUnwrap(values["now"] as? Int64)
        let data = Data(try XCTUnwrap(values["receipt_json"] as? String).utf8)
        let receipt = try AgentApprovalReceipt.parse(data)
        let known = receipt.receiptDigest
        let verifier = AgentLocalApprovalEvidenceVerifier(authority: { $0 == known ? .recognized : .unrecognized }, clock: { now })
        var fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        fields["consumed_at"] = now - 1
        let changed = try AgentApprovalReceipt.parse(JSONSerialization.data(withJSONObject: fields))
        XCTAssertEqual(try verifier.verify(changed, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole).reasonCode, .unrecognizedReceipt)
        XCTAssertEqual(try verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole).reasonCode, .verified)
    }

    func testExpiryIsCheckedAgainAfterCustodyLookup() throws {
        let values = try fixture()
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        let clock = StepClock([receipt.consumedAt, receipt.expiresAt])
        let verifier = AgentLocalApprovalEvidenceVerifier(authority: { _ in .recognized }, clock: { clock.next() })
        XCTAssertEqual(try verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole).reasonCode, .expired)
    }

    func testReplayStoreFailuresDoNotExposePrivateExceptions() throws {
        let values = try fixture()
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        let now = receipt.consumedAt
        let verifier = AgentLocalApprovalEvidenceVerifier(authority: { _ in .recognized }, replayStore: FailedStore(), clock: { now })
        let result = try verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole)
        XCTAssertEqual(result.reasonCode, .nonceStoreUnavailable)
        XCTAssertFalse(result.description.contains("SYNTHETIC_SECRET_SENTINEL"))
    }

    func testClockFailuresRemainTyped() throws {
        let values = try fixture()
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        for clock: @Sendable () throws -> Int64 in [{ -1 }, { throw TestFailure.privateText }] {
            let verifier = AgentLocalApprovalEvidenceVerifier(clock: clock)
            XCTAssertEqual(try verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole).reasonCode, .clockUnavailable)
        }
    }

    func testConcurrentPresentationHasOneObservation() throws {
        let values = try fixture()
        let receipt = try AgentApprovalReceipt.parse(Data(try XCTUnwrap(values["receipt_json"] as? String).utf8))
        let now = receipt.consumedAt
        let verifier = AgentLocalApprovalEvidenceVerifier(authority: { _ in .recognized }, clock: { now })
        let results = Results()
        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            if let result = try? verifier.verify(receipt, actionDigest: receipt.actionDigest, reviewerRole: receipt.reviewerRole) { results.append(result.reasonCode) }
        }
        XCTAssertEqual(results.values.filter { $0 == .verified }.count, 1)
        XCTAssertEqual(results.values.filter { $0 == .replayed }.count, 63)
    }

    func testEveryTypedReasonParsesWithoutGainingAuthority() throws {
        for reason in AgentApprovalEvidenceReason.allCases {
            let result = AgentApprovalEvidenceResult(reasonCode: reason, actionDigest: "sha256:" + String(repeating: "a", count: 64), receiptDigest: "sha256:" + String(repeating: "b", count: 64))
            XCTAssertEqual(try AgentApprovalEvidenceResult.parse(result.canonicalJSON()), result)
            XCTAssertFalse(result.authorizesClinicalAction)
        }
    }

    func testBoundedJSONDepthNodesAndSizesFailWithoutInputDiagnostics() throws {
        XCTAssertThrowsError(try AgentRunEvidence.parse(Data(repeating: 32, count: 1_048_577))) { XCTAssertEqual($0 as? AgentGovernanceError, .tooLarge) }
        XCTAssertThrowsError(try AgentApprovalReceipt.parse(Data(repeating: 32, count: 65_537))) { XCTAssertEqual($0 as? AgentGovernanceError, .tooLarge) }
        let nested = String(repeating: "[", count: 17) + "0" + String(repeating: "]", count: 17)
        XCTAssertThrowsError(try AgentJSON.parse(Data(nested.utf8)))
        let nodes = "[" + Array(repeating: "0", count: 200_001).joined(separator: ",") + "]"
        XCTAssertThrowsError(try AgentJSON.parse(Data(nodes.utf8)))
    }

    func testNestedDuplicateKeysAndTrailingDataAreRejected() throws {
        for raw in ["{\"x\":{\"a\":1,\"\\u0061\":2}}", "{}{}", "{\"x\":true,}", "[1,]", "{\"x\":1e999}", "{\"x\":9223372036854775808}", "{\"x\":\"\\q\"}"] {
            XCTAssertThrowsError(try AgentJSON.parse(Data(raw.utf8)))
        }
    }

    func testDescriptionsExcludeOpaqueMetadataAndRejectedValues() throws {
        let values = try fixture()
        for row in try XCTUnwrap(values["valid"] as? [[String: Any]]) {
            let raw = Data(try XCTUnwrap(row["json"] as? String).utf8)
            let description: String
            switch row["kind"] as? String {
            case "artifact": description = String(reflecting: try AgentArtifactReference.parse(raw))
            case "receipt": description = String(reflecting: try AgentApprovalReceipt.parse(raw))
            case "run": description = String(reflecting: try AgentRunEvidence.parse(raw))
            case "preview": description = String(reflecting: try AgentOMOPPreview.parse(raw))
            default: continue
            }
            XCTAssertFalse(description.contains("sha256:"))
            XCTAssertFalse(description.contains("role:"))
            XCTAssertFalse(description.contains("art_"))
        }
    }

    private enum TestFailure: Error, CustomStringConvertible {
        case privateText
        var description: String { "SYNTHETIC_SECRET_SENTINEL" }
    }

    private struct FailedStore: AgentReceiptReplayStore {
        func claim(tokenDigest: String, expiresAt: Int64, now: Int64) throws -> Bool { throw TestFailure.privateText }
    }

    private final class StepClock: @unchecked Sendable {
        private let lock = NSLock()
        private var values: [Int64]
        init(_ values: [Int64]) { self.values = values }
        func next() -> Int64 {
            lock.lock()
            defer { lock.unlock() }
            return values.removeFirst()
        }
    }

    private final class Results: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [AgentApprovalEvidenceReason] = []
        func append(_ value: AgentApprovalEvidenceReason) {
            lock.lock()
            defer { lock.unlock() }
            storage.append(value)
        }
        var values: [AgentApprovalEvidenceReason] {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }
}
