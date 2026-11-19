import Foundation
import WhisperKit
import XCTest
import os

@testable import OpenMedSpeech

// WhisperKit may deliver decoder progress from detached tasks, so the
// `@unchecked Sendable` progress gate is exercised from many threads at once.
// Run under Thread Sanitizer with:
//
//     swift test --sanitize=thread --filter SpeechDecoderProgressConcurrencyTests

private func progress(tokens: Int) -> TranscriptionProgress {
    TranscriptionProgress(
        timings: TranscriptionTimings(), text: "synthetic",
        tokens: Array(repeating: 1, count: tokens))
}

final class SpeechDecoderProgressConcurrencyTests: XCTestCase {
    func testOrdinalsStayStrictlyIncreasingAcrossConcurrentCallbacks() {
        let delivered = OSAllocatedUnfairLock(initialState: [Int]())
        let gate = SpeechDecoderProgress { ordinal, _ in
            delivered.withLock { $0.append(ordinal) }
            return true
        }
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            _ = gate.receive(progress(tokens: index + 1))
        }
        let ordinals = delivered.withLock { $0 }
        XCTAssertFalse(ordinals.isEmpty)
        XCTAssertEqual(ordinals, ordinals.sorted())
        XCTAssertEqual(Set(ordinals).count, ordinals.count)
        XCTAssertFalse(gate.failed)
        XCTAssertFalse(gate.stopped)
        gate.close()
        XCTAssertFalse(gate.receive(progress(tokens: 100)))
        XCTAssertEqual(delivered.withLock { $0 }, ordinals)
    }

    func testStopRequestFencesEveryLaterCallback() {
        let calls = OSAllocatedUnfairLock(initialState: 0)
        let gate = SpeechDecoderProgress { _, _ in
            calls.withLock { $0 += 1 }
            return false
        }
        let accepted = OSAllocatedUnfairLock(initialState: 0)
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            if gate.receive(progress(tokens: index + 1)) { accepted.withLock { $0 += 1 } }
        }
        XCTAssertEqual(calls.withLock { $0 }, 1)
        XCTAssertEqual(accepted.withLock { $0 }, 0)
        XCTAssertTrue(gate.stopped)
        XCTAssertFalse(gate.failed)
    }
}
