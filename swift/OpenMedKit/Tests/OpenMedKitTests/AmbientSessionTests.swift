import XCTest

@testable import OpenMedKit

private final class AmbientTestClock {
    var value = 0.0
}

final class AmbientSessionTests: XCTestCase {
    private let participant = String(repeating: "1", count: 32)

    private func session(
        clock: AmbientTestClock = AmbientTestClock(), maxBytes: Int = 1024,
        maxOperations: Int = 4096, export: Bool = false
    ) throws -> AmbientSession {
        let stream = try AmbientSession(
            sessionRef: String(repeating: "0", count: 32), participants: [participant],
            policy: AmbientSessionPolicy(
                revision: String(repeating: "a", count: 64), maxLifetimeMs: 1000,
                maxBytes: maxBytes, maxOperations: maxOperations, allowReviewExport: export),
            clock: { clock.value }, automaticExpiry: false)
        try stream.consent(participant)
        try stream.transition("start")
        return stream
    }

    func testSharedTransitionFixture() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let fixture = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/ambient_session_transitions.json")))
                as? [String: Any])
        let participants = try XCTUnwrap(fixture["participants"] as? [String])
        let sessionRef = try XCTUnwrap(fixture["session_ref"] as? String)
        let revision = try XCTUnwrap(fixture["policy_revision"] as? String)
        for scenario in try XCTUnwrap(fixture["scenarios"] as? [[String: Any]]) {
            let clock = AmbientTestClock()
            let stream = try AmbientSession(
                sessionRef: sessionRef, participants: participants,
                policy: AmbientSessionPolicy(revision: revision, maxLifetimeMs: 10_000, allowReviewExport: true),
                clock: { clock.value }, automaticExpiry: false)
            var expected: [String: Any] = [
                "schema": "openmed.ambient.session.v1", "session_ref": sessionRef,
                "policy_revision": revision, "state": "awaiting_consent", "generation": 0,
                "participants": 2, "consents": 0, "owned_artifacts": 0, "owned_bytes": 0,
                "unaccounted_external_copies": 0, "deletion_verified": false, "reason": "none",
                "draft_only": true, "requires_review": true,
            ]
            var reference: AmbientReference?
            for (index, step) in try XCTUnwrap(scenario["steps"] as? [[String: Any]]).enumerated() {
                var actualError: String?
                do {
                    let action = try XCTUnwrap(step["action"] as? String)
                    switch action {
                    case "consent":
                        try stream.consent(participants[try XCTUnwrap(step["participant"] as? Int)])
                    case "retain":
                        reference = try stream.retain(
                            artifactRef: String(format: "%032x", index + 100),
                            kind: XCTUnwrap(AmbientArtifactKind(rawValue: XCTUnwrap(step["kind"] as? String))),
                            content: Data([1, 2, 3]))
                    case "use": _ = try stream.use(XCTUnwrap(reference)) { $0.count }
                    case "participants": try stream.changeParticipants([participants[0]])
                    case "expire":
                        clock.value = 10
                        stream.expire()
                    case "withdraw": stream.withdraw()
                    case "delete": stream.delete()
                    case "export":
                        let copy = try stream.exportForReview(XCTUnwrap(reference))
                        XCTAssertTrue(copy.draftOnly && copy.requiresReview)
                    default: try stream.transition(action)
                    }
                } catch let error as AmbientSessionError {
                    actualError = error.rawValue
                }
                XCTAssertEqual(actualError, step["error"] as? String, "step \(index)")
                for (key, value) in try XCTUnwrap(step["expect"] as? [String: Any]) { expected[key] = value }
                let actual = try XCTUnwrap(JSONSerialization.jsonObject(with: stream.toJSON()) as? [String: Any])
                XCTAssertEqual(NSDictionary(dictionary: actual), NSDictionary(dictionary: expected), "step \(index)")
            }
            stream.delete()
        }
    }

    func testRevocationOverwritesAllOwnedKindsBeforeBorrowReturns() throws {
        for ending in ["withdraw", "delete", "participants", "expire"] {
            for kind in [AmbientArtifactKind.audio, .transcript, .cache, .draft] {
                let clock = AmbientTestClock()
                let stream = try session(clock: clock)
                let reference = try stream.retain(
                    artifactRef: String(repeating: "f", count: 32), kind: kind, content: Data([1, 2, 3]))
                XCTAssertThrowsError(
                    try stream.use(reference) { bytes in
                        switch ending {
                        case "withdraw": stream.withdraw()
                        case "delete": stream.delete()
                        case "participants": try stream.changeParticipants([String(repeating: "2", count: 32)])
                        default:
                            clock.value = 1
                            stream.expire()
                        }
                        XCTAssertTrue(bytes.allSatisfy { $0 == 0 })
                        return bytes.count
                    })
                XCTAssertEqual(stream.snapshot().ownedBytes, 0)
                XCTAssertThrowsError(try stream.use(reference) { $0.count })
            }
        }
    }

    func testForeignSessionReferenceIsRejected() throws {
        let stream = try session()
        let other = try session()
        let reference = try stream.retain(artifactRef: String(repeating: "f", count: 32), kind: .draft, content: Data([1]))
        let foreign = try other.retain(artifactRef: String(repeating: "f", count: 32), kind: .draft, content: Data([2]))
        XCTAssertThrowsError(try stream.use(foreign) { $0.count }) { error in
            XCTAssertEqual(error as? AmbientSessionError, .staleReference)
        }
        XCTAssertEqual(try stream.use(reference) { $0.count }, 1)
        stream.delete()
        other.delete()
    }

    func testInvalidClockAndOriginalDeadline() throws {
        for value in [-1.0, Double.nan, Double.infinity, 1.0] {
            let clock = AmbientTestClock()
            let stream = try session(clock: clock)
            _ = try stream.retain(artifactRef: String(repeating: "f", count: 32), kind: .cache, content: Data([1]))
            clock.value = 0.5
            try stream.transition("pause")
            try stream.transition("resume")
            clock.value = value
            XCTAssertTrue(stream.expire())
            XCTAssertTrue(stream.snapshot().deletionVerified)
            XCTAssertEqual(stream.snapshot().reason, value == 1 ? "expired" : "clock_invalid")
        }
    }

    func testReviewCopyCannotBecomeVerifiedDeleted() throws {
        let stream = try session(export: true)
        let reference = try stream.retain(artifactRef: String(repeating: "f", count: 32), kind: .draft, content: Data([7, 8]))
        XCTAssertThrowsError(try stream.exportForReview(reference))
        try stream.transition("stop")
        let copy = try stream.exportForReview(reference)
        XCTAssertTrue(copy.draftOnly && copy.requiresReview)
        stream.delete()
        XCTAssertEqual(copy.content, Data([7, 8]))
        XCTAssertEqual(stream.snapshot().state, .deletionIncomplete)
        XCTAssertFalse(stream.snapshot().deletionVerified)
        let previous = stream.snapshot()
        stream.delete()
        XCTAssertEqual(stream.snapshot(), previous)
    }

    func testDefaultPolicyAndAudioRefuseExport() throws {
        let stream = try session()
        let reference = try stream.retain(artifactRef: String(repeating: "f", count: 32), kind: .draft, content: Data([1]))
        try stream.transition("stop")
        XCTAssertThrowsError(try stream.exportForReview(reference)) { error in
            XCTAssertEqual(error as? AmbientSessionError, .exportNotPermitted)
        }
        stream.delete()
        let enabled = try session(export: true)
        let audio = try enabled.retain(artifactRef: String(repeating: "e", count: 32), kind: .audio, content: Data([1]))
        try enabled.transition("stop")
        XCTAssertThrowsError(try enabled.exportForReview(audio))
        enabled.delete()
    }

    func testBoundsAndCallbackFailureClearPendingContent() throws {
        let limited = try session(maxBytes: 1)
        _ = try limited.retain(artifactRef: String(repeating: "f", count: 32), kind: .cache, content: Data([1]))
        XCTAssertThrowsError(try limited.retain(artifactRef: String(repeating: "e", count: 32), kind: .draft, content: Data([2])))
        XCTAssertTrue(limited.snapshot().deletionVerified)
        let operations = try session(maxOperations: 3)
        _ = try operations.retain(artifactRef: String(repeating: "f", count: 32), kind: .cache, content: Data([1]))
        XCTAssertThrowsError(try operations.transition("pause"))
        XCTAssertEqual(operations.snapshot().reason, "operation_limit")
        let callback = try session()
        let reference = try callback.retain(artifactRef: String(repeating: "f", count: 32), kind: .draft, content: Data([1]))
        XCTAssertThrowsError(
            try callback.use(reference) { _ -> Int in
                throw NSError(domain: "synthetic private text", code: 1)
            }
        ) { error in
            XCTAssertEqual(error as? AmbientSessionError, .readerFailed)
        }
        XCTAssertTrue(callback.snapshot().deletionVerified)
        XCTAssertFalse(String(decoding: try callback.toJSON(), as: UTF8.self).contains("synthetic"))
    }

    func testExternalCopiesHoldNewCohort() throws {
        let stream = try session()
        try stream.declareExternalCopy()
        try stream.changeParticipants([String(repeating: "2", count: 32)])
        XCTAssertEqual(stream.snapshot().state, .deletionIncomplete)
        XCTAssertFalse(stream.snapshot().deletionVerified)
        XCTAssertThrowsError(try stream.consent(String(repeating: "2", count: 32)))
    }

    func testMalformedPolicyAndParticipants() throws {
        XCTAssertThrowsError(try AmbientSessionPolicy(revision: "private"))
        XCTAssertThrowsError(try AmbientSessionPolicy(revision: String(repeating: "a", count: 64), maxBytes: 0))
        XCTAssertThrowsError(
            try AmbientSession(
                sessionRef: String(repeating: "0", count: 32), participants: [participant, participant],
                policy: AmbientSessionPolicy(revision: String(repeating: "a", count: 64))))
    }
}
