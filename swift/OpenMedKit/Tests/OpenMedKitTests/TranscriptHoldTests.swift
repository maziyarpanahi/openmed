import Foundation
import XCTest

@testable import OpenMedKit

final class TranscriptHoldTests: XCTestCase {
    private func ref(_ segment: Int = 1, _ token: Int = 0) throws -> TranscriptTokenIdentity {
        try TranscriptTokenIdentity(segmentID: segment, tokenID: token)
    }

    private func gate(_ token: FixedTranscriptToken, revision: Int = 1, doseEvidence: [TokenDoseEvidence] = []) throws -> TranscriptHoldGate {
        try TranscriptHoldGate(tokens: [token], statements: [DraftTokenCitation(statementID: 10, tokens: [token.identity]), DraftTokenCitation(statementID: 11, tokens: [token.identity])], revision: revision, medicationNames: ["syntheticmed"], doseEvidence: doseEvidence)
    }

    func testSharedThresholdAndAlternativesTable() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/multimodal/transcript_hold_cases.json"))
        let fixture = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(fixture["policy_version"] as? Int, TranscriptHoldGate.policyVersion)
        for row in try XCTUnwrap(fixture["cases"] as? [[String: Any]]) {
            let text = try XCTUnwrap(row["text"] as? String)
            let alternatives = try XCTUnwrap(row["alternatives"] as? [String])
            let threshold = try XCTUnwrap(row["threshold"] as? Double)
            for delta in [-0.000001, 0, 0.000001] {
                let result = try gate(FixedTranscriptToken(identity: ref(), text: text, confidence: threshold + delta, alternatives: alternatives))
                let expected = delta < 0 || alternatives.contains { $0 != text }
                XCTAssertEqual(!result.holds.isEmpty, expected)
                XCTAssertEqual(try result.statementHolds(10).isEmpty, !expected)
                XCTAssertEqual(try result.statementHolds(11).isEmpty, !expected)
                if expected {
                    XCTAssertEqual(result.holds[0].classes.map(\.rawValue), row["classes"] as? [String])
                    XCTAssertEqual(result.holds[0].threshold, threshold)
                }
            }
        }
        for kind in CriticalTokenClass.allCases {
            XCTAssertEqual(kind.confidenceThreshold, kind == .uncertainty ? 0.90 : 0.95)
        }
    }

    func testResolutionExportAndStaleCorrection() throws {
        let token = try FixedTranscriptToken(identity: ref(), text: "fifteen", confidence: 0.1, alternatives: ["fifty"])
        let result = try gate(token)
        let receipt = try TranscriptHoldConfirmation(identity: ref(), evidenceDigest: result.evidenceDigest, reviewerID: 7, confirmed: true)
        XCTAssertThrowsError(try result.exportReviewed(10, reviewerConfirmed: true)) { XCTAssertEqual($0 as? TranscriptHoldError, .unresolvedTokenHold) }
        XCTAssertThrowsError(try result.resolve(receipt, authorize: { _ in false })) { XCTAssertEqual($0 as? TranscriptHoldError, .reviewerDenied) }
        let changed = try gate(token, revision: 2)
        XCTAssertThrowsError(try changed.resolve(receipt, authorize: { _ in true })) { XCTAssertEqual($0 as? TranscriptHoldError, .staleConfirmation) }
        try result.resolve(receipt, authorize: { $0 == 7 })
        XCTAssertTrue(try result.statementHolds(10).isEmpty)
        XCTAssertTrue(try result.statementHolds(11).isEmpty)
        XCTAssertEqual(result.holds.count, 1)
        XCTAssertThrowsError(try result.exportReviewed(10, reviewerConfirmed: false))
        XCTAssertTrue(try result.exportReviewed(10, reviewerConfirmed: true).notice.hasPrefix("Non-diagnostic"))
    }

    func testDoseStatusAndAdjacentDoseShape() throws {
        for (status, score) in [(TranscriptDoseCheckStatus.inRange, 0.0), (.notChecked, 0.5), (.flagged, 1.0)] {
            let result = try gate(FixedTranscriptToken(identity: ref(), text: "50", confidence: 1), doseEvidence: [TokenDoseEvidence(identity: ref(), status: status)])
            XCTAssertEqual(result.holds.isEmpty, score == 0)
            if score != 0 { XCTAssertEqual(result.holds[0].doseFlagScore, score) }
        }
        let result = try TranscriptHoldGate(tokens: [FixedTranscriptToken(identity: ref(), text: "fifteen", confidence: 0.1), FixedTranscriptToken(identity: ref(1, 1), text: "mg", confidence: 1)], statements: [DraftTokenCitation(statementID: 1, tokens: [ref()])], revision: 1)
        XCTAssertTrue(result.holds[0].classes.contains(.dose))
    }

    func testMissingConfidenceInvalidInputsAndNegativeControls() throws {
        XCTAssertFalse(try gate(FixedTranscriptToken(identity: ref(), text: "no", confidence: nil)).holds.isEmpty)
        XCTAssertTrue(try gate(FixedTranscriptToken(identity: ref(), text: "Fifteen.", confidence: 1, alternatives: ["fifteen"])).holds.isEmpty)
        XCTAssertTrue(try gate(FixedTranscriptToken(identity: ref(), text: "ordinary", confidence: 0.1)).holds.isEmpty)
        XCTAssertThrowsError(try TranscriptTokenIdentity(segmentID: -1, tokenID: 0))
        XCTAssertThrowsError(try FixedTranscriptToken(identity: ref(), text: "PRIVATE", confidence: .nan))
        XCTAssertThrowsError(try DraftTokenCitation(statementID: 1, tokens: []))
        XCTAssertThrowsError(try DraftTokenCitation(statementID: 1, tokens: [ref(), ref()]))
        let token = try FixedTranscriptToken(identity: ref(), text: "no", confidence: 0.1)
        XCTAssertThrowsError(try TranscriptHoldGate(tokens: [token, token], statements: [], revision: 1))
        XCTAssertThrowsError(try TranscriptHoldGate(tokens: [token], statements: [DraftTokenCitation(statementID: 1, tokens: [ref(2)])], revision: 1))
    }

    func testContentFreeRecordsForPrivateAndMultilingualInputs() throws {
        for text in ["Invented Person", "patient@example.invalid", "رقم-خيالي", "/private/synthetic", "secret-token"] {
            let token = try FixedTranscriptToken(identity: ref(), text: text, confidence: 0.1, alternatives: ["no"])
            let result = try gate(token)
            let data = try JSONEncoder().encode(result.holds[0])
            XCTAssertFalse(String(decoding: data, as: UTF8.self).contains(text))
            XCTAssertFalse(String(describing: token).contains(text))
            let record = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(Set(record.keys), ["segment_id", "token_id", "classes", "confidence", "threshold", "disagreement_score", "dose_flag_score", "policy_version"])
        }
    }

    func testDeterministicSegmentIsolation() throws {
        let tokens = try (0..<8).map { try FixedTranscriptToken(identity: ref($0), text: "no", confidence: 0.1) }
        let statements = try tokens.enumerated().map { try DraftTokenCitation(statementID: $0.offset, tokens: [$0.element.identity]) }
        let first = try TranscriptHoldGate(tokens: tokens, statements: statements, revision: 1)
        let second = try TranscriptHoldGate(tokens: tokens.reversed(), statements: statements.reversed(), revision: 1)
        XCTAssertEqual(first.holds, second.holds)
        XCTAssertEqual(first.evidenceDigest, second.evidenceDigest)
        for index in 0..<8 { XCTAssertEqual(try first.statementHolds(index).map { $0.identity.segmentID }, [index]) }
    }
}
