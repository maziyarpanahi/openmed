import Foundation
import XCTest

@testable import OpenMedKit

/// A local engine whose transcript is planted text that must never reach an
/// error or a description.
private actor LifecycleSpeechEngine: LocalSpeechEngine {
    static let transcript = "SYNTHETIC_PRIVATE_TRANSCRIPT_5355"
    nonisolated let provenance = try! SpeechEngineProvenance(
        providerID: "synthetic", providerRevision: "v1",
        modelFingerprint: "sha256:" + String(repeating: "c", count: 64), language: "en")
    nonisolated let localOnly = true

    func decode(
        _ window: SpeechPCMBuffer,
        progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool
    ) async throws -> SpeechRecognition {
        try SpeechRecognition(text: Self.transcript)
    }

    func shutdown() async {}
}

/// Replays `tests/fixtures/parity/session_lifecycle_v1.json` against the Swift
/// lifecycle vocabulary, the `SpeechSessionState` projection and the guards of
/// a real `StreamingSpeechSession`.
final class SessionLifecycleParityTests: XCTestCase {
    private typealias State = OpenMedSessionLifecycleState

    func testVocabularyMatchesTheSharedVectors() throws {
        let fixture = try loadFixture()
        XCTAssertEqual(fixture["schema"] as? String, "openmed.tests.session_lifecycle_parity")
        XCTAssertEqual(fixture["lifecycle_schema_version"] as? String, State.version)

        XCTAssertEqual(State.allCases.map(\.rawValue), fixture["states"] as? [String])
        XCTAssertEqual(State.allCases.filter(\.isTerminal).map(\.rawValue), fixture["terminal_states"] as? [String])
        let transitions = try XCTUnwrap(fixture["transitions"] as? [String: [String]])
        XCTAssertEqual(Set(transitions.keys), Set(State.allCases.map(\.rawValue)))
        for state in State.allCases {
            XCTAssertEqual(state.allowedTargets.map(\.rawValue), transitions[state.rawValue], state.rawValue)
        }
        XCTAssertEqual(
            OpenMedSessionLifecycleError.allCases.map(\.rawValue).sorted(), (fixture["reason_codes"] as? [String])?.sorted())
    }

    func testSwiftProjectionMatchesTheVectorsAndKeepsRawValues() throws {
        let fixture = try loadFixture()
        let projections = try XCTUnwrap(fixture["projections"] as? [String: [String: Any]])
        let swift = try XCTUnwrap(projections["swift.streaming_speech"])
        XCTAssertEqual(swift["native_enum"] as? String, "OpenMedKit.SpeechSessionState")
        let map = try XCTUnwrap(swift["map"] as? [String: String])

        let natives: [SpeechSessionState] = [.open, .finished, .cancelled, .failed]
        XCTAssertEqual(natives.map(\.rawValue), ["open", "finished", "cancelled", "failed"])
        XCTAssertEqual(Set(map.keys), Set(natives.map(\.rawValue)))
        for native in natives {
            XCTAssertEqual(native.lifecycleState.rawValue, map[native.rawValue], native.rawValue)
            XCTAssertEqual(SpeechSessionState(rawValue: native.rawValue), native)
        }
    }

    func testAllowedSequencesPassAndRejectedTransitionsGiveTheirReason() throws {
        let fixture = try loadFixture()
        let allowed = try XCTUnwrap(fixture["allowed_sequences"] as? [[String: Any]])
        XCTAssertFalse(allowed.isEmpty)
        for sequence in allowed {
            let id = try XCTUnwrap(sequence["id"] as? String)
            let states = try XCTUnwrap(sequence["states"] as? [String], id)
            XCTAssertEqual(states.first, State.open.rawValue, id)
            for (source, target) in zip(states, states.dropFirst()) {
                XCTAssertNil(State.transitionReason(from: source, to: target), "\(id) \(source)->\(target)")
                let typed = try XCTUnwrap(State(rawValue: source)).transitionReason(to: try XCTUnwrap(State(rawValue: target)))
                XCTAssertNil(typed, "\(id) \(source)->\(target)")
            }
        }

        let rejected = try XCTUnwrap(fixture["rejected_transitions"] as? [[String: Any]])
        XCTAssertFalse(rejected.isEmpty)
        for transition in rejected {
            let id = try XCTUnwrap(transition["id"] as? String)
            let source = try XCTUnwrap(transition["from"] as? String, id)
            let target = try XCTUnwrap(transition["to"] as? String, id)
            let expected = try XCTUnwrap(transition["reason"] as? String, id)
            XCTAssertEqual(State.transitionReason(from: source, to: target)?.rawValue, expected, id)
            if let start = State(rawValue: source), let end = State(rawValue: target) {
                XCTAssertEqual(start.transitionReason(to: end)?.rawValue, expected, id)
            }
        }
    }

    func testEveryPairFollowsThePythonReasonOrder() {
        // unknown_state first, then the terminal source code, then the table.
        for source in State.allCases {
            for target in State.allCases {
                let reason = source.transitionReason(to: target)
                switch source {
                case .finalized, .closed: XCTAssertEqual(reason, .sessionClosed)
                case .cancelled: XCTAssertEqual(reason, .sessionCancelled)
                case .failed: XCTAssertEqual(reason, .sessionFailed)
                case .open, .cancelling:
                    XCTAssertEqual(reason, source.allowedTargets.contains(target) ? nil : .transitionNotAllowed)
                }
                XCTAssertEqual(State.transitionReason(from: source.rawValue, to: "paused"), .unknownState)
                XCTAssertEqual(State.transitionReason(from: "finished", to: target.rawValue), .unknownState)
            }
        }
    }

    // MARK: - Session guards

    func testFinishedSessionIsFinalizedAndRefusesFurtherTransitions() async throws {
        let session = try makeSession()
        XCTAssertEqual(session.lifecycleState, .open)
        XCTAssertNoThrow(try session.checkLifecycleTransition(to: .finalized))
        try await session.append(buffer(), endOfUtterance: true)
        try await session.finish()

        // finalize_stream: open -> finalized.
        XCTAssertEqual(session.snapshot().state, .finished)
        XCTAssertEqual(session.lifecycleState, .finalized)
        XCTAssertEqual(session.snapshot().lifecycleState, .finalized)
        // finalized_finalizes_again: the native guard refuses with its closed code.
        await assertThrows(.closed) { try await session.finish() }
        await assertThrows(.closed) { try await session.append(self.buffer(start: 160)) }
        // finalized_cancels: cancel() stays an idempotent no-op on a terminal session.
        await session.cancel()
        session.requestCancellation()
        XCTAssertEqual(session.snapshot().state, .finished)
        XCTAssertEqual(session.lifecycleState, .finalized)
        for target in State.allCases {
            assertLifecycleRejects(session, target, .sessionClosed)
        }
    }

    func testCancelledSessionIsCancelledAndRefusesFurtherTransitions() async throws {
        let session = try makeSession()
        await session.cancel()

        // cancel_idle: open -> cancelled.
        XCTAssertEqual(session.lifecycleState, .cancelled)
        XCTAssertEqual(session.snapshot().lifecycleState, .cancelled)
        // cancelled_finalizes: the native guard refuses with its cancelled code.
        await assertThrows(.cancelled) { try await session.finish() }
        await assertThrows(.cancelled) { try await session.append(self.buffer()) }
        // cancelled_cancels_again: an idempotent no-op.
        await session.cancel()
        XCTAssertEqual(session.lifecycleState, .cancelled)
        for target in State.allCases {
            assertLifecycleRejects(session, target, .sessionCancelled)
        }
    }

    func testFailedSessionStaysFailedEvenWhenCancelledAfterwards() async throws {
        let session = try makeSession()
        try await session.append(buffer())
        await assertThrows(.sequence) { try await session.append(self.buffer(start: 0)) }

        // fail_while_open: open -> failed.
        XCTAssertEqual(session.snapshot().state, .failed)
        XCTAssertEqual(session.lifecycleState, .failed)
        await assertThrows(.sequence) { try await session.finish() }
        // The native state keeps its existing cancel() behaviour, but the
        // lifecycle state of a terminal session never changes.
        await session.cancel()
        XCTAssertEqual(session.snapshot().state, .cancelled)
        XCTAssertEqual(session.lifecycleState, .failed)
        XCTAssertEqual(session.snapshot().lifecycleState, .failed)
        for target in State.allCases {
            assertLifecycleRejects(session, target, .sessionFailed)
        }
    }

    func testOpenSessionChecksTransitionsWithoutMakingThem() throws {
        let session = try makeSession()
        for target in State.allCases {
            if State.open.allowedTargets.contains(target) {
                XCTAssertNoThrow(try session.checkLifecycleTransition(to: target), target.rawValue)
            } else {
                assertLifecycleRejects(session, target, .transitionNotAllowed)
            }
        }
        XCTAssertEqual(session.lifecycleState, .open)
        XCTAssertEqual(session.snapshot().state, .open)
    }

    func testLifecycleErrorsCarryReasonCodesOnly() async throws {
        let session = try makeSession()
        try await session.append(buffer(), endOfUtterance: true)
        try await session.finish()
        do {
            try session.checkLifecycleTransition(to: .open)
            XCTFail("terminal session reopened")
        } catch {
            XCTAssertEqual(error as? OpenMedSessionLifecycleError, .sessionClosed)
            XCTAssertEqual(error.localizedDescription, "session_closed")
            for text in [String(describing: error), String(reflecting: error), error.localizedDescription] {
                XCTAssertFalse(text.contains(LifecycleSpeechEngine.transcript))
                XCTAssertFalse(text.contains("SYNTHETIC"))
            }
        }
        for reason in OpenMedSessionLifecycleError.allCases {
            XCTAssertEqual(reason.errorDescription, reason.rawValue)
            XCTAssertNotNil(reason.rawValue.range(of: "^[a-z][a-z_]*$", options: .regularExpression))
        }
        XCTAssertFalse(String(describing: session.snapshot()).contains(LifecycleSpeechEngine.transcript))
    }

    // MARK: - Helpers

    private func makeSession() throws -> StreamingSpeechSession {
        try StreamingSpeechSession(engine: LifecycleSpeechEngine(), limits: SpeechSessionLimits(decodeStrideFrames: 160)) { _ in }
    }

    private func buffer(_ count: Int = 160, start: Int64 = 0) throws -> SpeechPCMBuffer {
        try [Float](repeating: 0.25, count: count).withUnsafeBufferPointer { try SpeechPCMBuffer(samples: $0, startSample: start) }
    }

    private func assertThrows(
        _ expected: StreamingSpeechError, file: StaticString = #filePath, line: UInt = #line,
        _ body: () async throws -> Void
    ) async {
        do {
            try await body()
            XCTFail("transition admitted", file: file, line: line)
        } catch {
            XCTAssertEqual(error as? StreamingSpeechError, expected, file: file, line: line)
        }
    }

    private func assertLifecycleRejects(
        _ session: StreamingSpeechSession, _ target: State, _ expected: OpenMedSessionLifecycleError,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try session.checkLifecycleTransition(to: target), target.rawValue, file: file, line: line) {
            XCTAssertEqual($0 as? OpenMedSessionLifecycleError, expected, target.rawValue, file: file, line: line)
        }
    }

    private func loadFixture() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/session_lifecycle_v1.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
