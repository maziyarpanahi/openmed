import Foundation
import XCTest

@testable import OpenMedKit

final class ReviewedLocalEvidenceTests: XCTestCase {
    private struct Source: CurrentLocalSource {
        let digest: String?
        func currentDigest(sourceID: String) throws -> String? { digest }
    }

    private struct Authority: ReviewAuthorityVerifier {
        let registered: LocalReviewReceipt?
        var status: ReviewAuthorityStatus = .current
        func verify(_ receipt: LocalReviewReceipt, evidenceDigest: String, now: Int) throws -> ReviewAuthorityStatus {
            if status == .revoked { return .revoked }
            return receipt == registered && receipt.evidenceDigest == evidenceDigest ? .current : .mismatched
        }
    }

    private func fixture() throws -> (ReviewedLocalEvidence, String, String, Data) {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let directory = root.appending(path: "tests/fixtures/clinical/brief_parity")
        let packet = try ReviewedLocalEvidence.fromJSON(Data(contentsOf: directory.appending(path: "reviewed_local.json")))
        let row = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: directory.appending(path: "verified.json"))) as? [String: Any])
        return (packet, try XCTUnwrap(row["source"] as? String), try XCTUnwrap(row["generator_output"] as? String), Data(try XCTUnwrap(row["evaluation_json"] as? String).utf8))
    }

    private func mutate(_ packet: ReviewedLocalEvidence, _ change: (inout [String: Any]) -> Void) throws -> ReviewedLocalEvidence {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: packet.toJSON()) as? [String: Any])
        change(&object)
        return try ReviewedLocalEvidence.fromJSON(JSONSerialization.data(withJSONObject: object))
    }

    func testPythonWireRoundTripDigestAndUnicodeCoordinates() throws {
        let (packet, source, _, _) = try fixture()
        XCTAssertEqual(try ReviewedLocalEvidence.fromJSON(packet.toJSON()), packet)
        XCTAssertEqual(try packet.evidenceDigest, packet.reviewReceipt?.evidenceDigest)
        XCTAssertEqual(ReviewedLocalEvidence.sourceDigest(source), packet.sourceDigest)
        XCTAssertEqual(ReviewedLocalEvidence.sourceDigest("A😀क。"), "sha256:9da25197118f0b51f8a6d65c45ccffdefee37003d3d8cb37ef6693bed7a91050")
        try packet.admit(source: source, policyDigest: packet.policyDigest, currentSource: Source(digest: packet.sourceDigest), authority: Authority(registered: packet.reviewReceipt), clock: { 1000 })
    }

    func testReceiptSourcePolicyAndAuthorityRefusals() throws {
        let (packet, source, _, _) = try fixture()
        let missing = try mutate(packet) { $0["review_receipt"] = NSNull() }
        let changedReceipt = try mutate(packet) {
            var receipt = $0["review_receipt"] as! [String: Any]
            receipt["expires_at"] = 9999
            $0["review_receipt"] = receipt
        }
        let changedOffset = try mutate(packet) {
            var refs = $0["references"] as! [[String: Any]]
            refs[0]["end"] = 2
            $0["references"] = refs
        }
        let cases: [(ReviewedLocalEvidence, String, String, String?, Authority, Double, ReviewAdmissionRefusal)] = [
            (missing, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1000, .missing),
            (packet, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1100, .expired),
            (packet, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 899, .mismatched),
            (changedReceipt, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1000, .mismatched),
            (changedOffset, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1000, .mismatched),
            (packet, source, packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt, status: .revoked), 1000, .revoked),
            (packet, source + " changed", packet.policyDigest, packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1000, .sourceChanged),
            (packet, source, "sha256:" + String(repeating: "d", count: 64), packet.sourceDigest, Authority(registered: packet.reviewReceipt), 1000, .policyChanged),
            (packet, source, packet.policyDigest, nil, Authority(registered: packet.reviewReceipt), 1000, .sourceUnavailable),
            (packet, source, packet.policyDigest, "sha256:" + String(repeating: "d", count: 64), Authority(registered: packet.reviewReceipt), 1000, .sourceChanged),
        ]
        for (evidence, text, policy, current, authority, now, reason) in cases {
            XCTAssertThrowsError(try evidence.admit(source: text, policyDigest: policy, currentSource: Source(digest: current), authority: authority, clock: { now })) {
                XCTAssertEqual($0 as? ReviewAdmissionRefusal, reason)
            }
        }
    }

    func testUnsafeFieldsAndSpanIntegrityFailClosed() throws {
        let (packet, _, _, _) = try fixture()
        for (key, value) in [("source_id", "private-patient"), ("offset_convention", "utf8_bytes"), ("text", "SYNTHETIC_PRIVATE")] {
            XCTAssertThrowsError(try mutate(packet) { $0[key] = value }) {
                XCTAssertEqual($0 as? ReviewAdmissionRefusal, .invalid)
            }
        }
        XCTAssertThrowsError(try mutate(packet) { $0["schema_version"] = true })
        XCTAssertThrowsError(
            try mutate(packet) {
                var refs = $0["references"] as! [[String: Any]]
                refs[0]["end"] = packet.sourceLength + 1
                $0["references"] = refs
            })
    }

    func testGenericDecoderCannotSerializeUnsafeMetadata() throws {
        let (packet, _, _, _) = try fixture()
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: packet.toJSON()) as? [String: Any])
        object["source_id"] = "SYNTHETIC_PRIVATE_PATH"
        let decoded = try JSONDecoder().decode(ReviewedLocalEvidence.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertThrowsError(try decoded.toJSON()) {
            XCTAssertEqual($0 as? ReviewAdmissionRefusal, .invalid)
        }
    }

    func testAdapterAdmitsOnlyReviewedSpansAndValidatesOutput() async throws {
        let (packet, source, summary, evaluation) = try fixture()
        let brief = try await ClinicalBrief.reviewedLocal(
            evidence: packet, source: source, policyDigest: packet.policyDigest,
            currentSource: Source(digest: packet.sourceDigest), authority: Authority(registered: packet.reviewReceipt),
            originalIdentifiers: [], clock: { 1000 },
            generate: { admitted in
                XCTAssertEqual(admitted, source)
                return summary
            }, evaluate: { _, _ in evaluation }, privacyCheck: { _ in true })
        XCTAssertEqual(brief.summary, summary)
    }

    func testAdapterRejectsCitationsOutsideReviewedEvidence() async throws {
        let (packet, source, summary, evaluation) = try fixture()
        var subset = try mutate(packet) {
            var refs = $0["references"] as! [[String: Any]]
            refs.removeLast()
            $0["references"] = refs
        }
        let digest = try subset.evidenceDigest
        subset = try mutate(subset) {
            var receipt = $0["review_receipt"] as! [String: Any]
            receipt["evidence_digest"] = digest
            $0["review_receipt"] = receipt
        }
        do {
            _ = try await ClinicalBrief.reviewedLocal(
                evidence: subset, source: source, policyDigest: subset.policyDigest,
                currentSource: Source(digest: subset.sourceDigest), authority: Authority(registered: subset.reviewReceipt),
                originalIdentifiers: [], clock: { 1000 },
                generate: { _ in summary }, evaluate: { _, _ in evaluation }, privacyCheck: { _ in true })
            XCTFail("unreviewed citation admitted")
        } catch { XCTAssertEqual(error as? ClinicalBriefError, .unsupportedClaim) }
    }

    func testExpiryAtGenerationNeverInvokesProvider() async throws {
        let (packet, source, _, _) = try fixture()
        var calls = 0
        do {
            _ = try await ClinicalBrief.reviewedLocal(
                evidence: packet, source: source, policyDigest: packet.policyDigest,
                currentSource: Source(digest: packet.sourceDigest), authority: Authority(registered: packet.reviewReceipt),
                originalIdentifiers: [],
                clock: {
                    calls += 1
                    return calls == 1 ? 1000 : 1100
                },
                generate: { _ in
                    XCTFail("generation reached")
                    return ""
                },
                evaluate: { _, _ in
                    XCTFail("evaluation reached")
                    return Data()
                }, privacyCheck: { _ in true })
            XCTFail("expired review admitted")
        } catch { XCTAssertEqual(error as? ReviewAdmissionRefusal, .expired) }
    }
    private final class TestAdmissionClock: @unchecked Sendable {
        var instant: TimeInterval = 1000
        var verifications = 0
        var generated = 0
    }

    private struct LaggingAuthority: ReviewAuthorityVerifier {
        let registered: LocalReviewReceipt
        let clock: TestAdmissionClock
        func verify(_ receipt: LocalReviewReceipt, evidenceDigest: String, now: Int) throws -> ReviewAuthorityStatus {
            guard receipt == registered, evidenceDigest == registered.evidenceDigest else { return .mismatched }
            clock.verifications += 1
            if clock.verifications == 2 { clock.instant = TimeInterval(receipt.expiresAt) }
            return .current
        }
    }

    func testSlowReviewCallbackCannotExpireReceiptBeforeGeneration() async throws {
        let (packet, source, _, _) = try fixture()
        let clock = TestAdmissionClock()
        do {
            _ = try await ClinicalBrief.reviewedLocal(
                evidence: packet, source: source, policyDigest: packet.policyDigest,
                currentSource: Source(digest: packet.sourceDigest),
                authority: LaggingAuthority(registered: try XCTUnwrap(packet.reviewReceipt), clock: clock),
                originalIdentifiers: [], clock: { clock.instant },
                generate: { _ in
                    clock.generated += 1
                    return ""
                },
                evaluate: { _, _ in Data() }, privacyCheck: { _ in true })
            XCTFail("expired review admitted")
        } catch { XCTAssertEqual(error as? ReviewAdmissionRefusal, .expired) }
        XCTAssertEqual(clock.generated, 0)
    }

    func testSharedWirePreservesExactSigned64BitBoundary() throws {
        let (packet, _, _, _) = try fixture()
        let changed = try mutate(packet) {
            var receipt = $0["review_receipt"] as! [String: Any]
            receipt["expires_at"] = Int.max
            $0["review_receipt"] = receipt
        }
        XCTAssertEqual(changed.reviewReceipt?.expiresAt, Int.max)
        XCTAssertEqual(try ReviewedLocalEvidence.fromJSON(changed.toJSON()), changed)
    }

}
