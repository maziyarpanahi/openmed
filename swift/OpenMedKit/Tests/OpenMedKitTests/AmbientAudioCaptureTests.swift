import AVFAudio
import Foundation
import XCTest

@testable import OpenMedKit

final class AmbientAudioCaptureTests: XCTestCase {
    private let participant = String(repeating: "a", count: 32)
    private let otherParticipant = String(repeating: "b", count: 32)

    private final class Source: AmbientAudioSource, @unchecked Sendable {
        var permission: CapturePermission = .granted
        var starts = 0
        var stops = 0
        var cleanup = true
        var failStart = false
        var earlyBuffer: CapturePCMBuffer?
        var callbacks: [@Sendable (CapturePCMBuffer) -> Void] = []
        var events: [@Sendable (CaptureInterruption) -> Void] = []
        func requestPermission(_ completion: @escaping @Sendable (CapturePermission) -> Void) { completion(permission) }
        func start(
            receive: @escaping @Sendable (CapturePCMBuffer) -> Void,
            interrupted: @escaping @Sendable (CaptureInterruption) -> Void
        ) throws {
            starts += 1
            callbacks.append(receive)
            events.append(interrupted)
            if let earlyBuffer { receive(earlyBuffer) }
            if failStart { throw AmbientAudioCaptureError.sourceFailed }
        }
        func stop() -> Bool {
            stops += 1
            return cleanup
        }
        func emit(_ buffer: CapturePCMBuffer, attempt: Int? = nil) { callbacks[attempt ?? (callbacks.count - 1)](buffer) }
    }

    private final class Consumer: AmbientPCMConsumer, @unchecked Sendable {
        var frames: [Int] = []
        var starts: [Int64] = []
        var epochs: [UInt64] = []
        var fail = false
        var entered: DispatchSemaphore?
        var release: DispatchSemaphore?
        func consume(_ samples: UnsafeBufferPointer<Float>, sampleRateHz: Int, startSample: Int64, captureEpoch: UInt64) throws {
            XCTAssertEqual(sampleRateHz, 16_000)
            XCTAssertEqual(Array(samples), [0.125, -0.25, 0.5])
            entered?.signal()
            if let release { XCTAssertEqual(release.wait(timeout: .now() + 5), .success) }
            if fail { throw NSError(domain: "private source content", code: 1) }
            frames.append(samples.count)
            starts.append(startSample)
            epochs.append(captureEpoch)
        }
    }

    private final class Clock: @unchecked Sendable { var now: Double = 0 }

    private func buffer(_ start: Int64 = 0) throws -> CapturePCMBuffer {
        try [Float(0.125), -0.25, 0.5].withUnsafeBufferPointer {
            try CapturePCMBuffer(samples: $0, sampleRateHz: 16_000, startSample: start)
        }
    }

    private func capture(
        _ source: Source, _ consumer: Consumer, clock: Clock = Clock(), maximumBytes: Int = 1024
    ) throws -> AmbientAudioCapture {
        try AmbientAudioCapture(
            sessionRef: String(repeating: "c", count: 32), participants: [participant],
            policy: AmbientSessionPolicy(revision: String(repeating: "d", count: 64), maxLifetimeMs: 1000, maxBytes: maximumBytes),
            source: source, consumer: consumer, clock: { clock.now }, automaticExpiry: false)
    }

    func testConsentAndPermissionBothRequiredWithoutAutomaticStart() throws {
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        XCTAssertThrowsError(try capture.start())
        XCTAssertEqual(source.starts, 0)
        try capture.consent(participant)
        source.permission = .denied
        XCTAssertThrowsError(try capture.start()) { XCTAssertEqual($0 as? AmbientAudioCaptureError, .permissionRequired) }
        capture.requestPermission { XCTAssertEqual($0, .denied) }
        XCTAssertEqual(capture.snapshot().session.state, .ready)
        source.permission = .granted
        capture.requestPermission { XCTAssertEqual($0, .granted) }
        XCTAssertEqual(source.starts, 0)
        try capture.start()
        let value = try buffer(20)
        source.emit(value)
        XCTAssertTrue(value.isCleared)
        XCTAssertEqual(consumer.starts, [20])
        XCTAssertEqual(capture.snapshot().ownedBuffers, 0)
        XCTAssertEqual(capture.snapshot().session.ownedBytes, 0)
    }

    func testPauseResumeRejectsPriorCaptureCallbacksAndPreservesSampleGap() throws {
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        source.emit(try buffer())
        try capture.pause()
        let paused = try buffer(3)
        source.emit(paused)
        XCTAssertTrue(paused.isCleared)
        XCTAssertThrowsError(try capture.start())
        try capture.start(resuming: true)
        let stale = try buffer(6)
        source.emit(stale, attempt: 0)
        source.emit(try buffer(100))
        XCTAssertTrue(stale.isCleared)
        XCTAssertEqual(consumer.starts, [0, 100])
        XCTAssertNotEqual(consumer.epochs[0], consumer.epochs[1])
        // A delayed event from an old engine cannot stop a newly admitted capture.
        source.events[0](.routeChanged)
        XCTAssertTrue(capture.snapshot().running)
    }

    func testInterruptionRouteAndBackgroundRequireExplicitResume() throws {
        for event in [CaptureInterruption.interruption, .routeChanged, .background, .sourceFailure] {
            let source = Source()
            let consumer = Consumer()
            let capture = try capture(source, consumer)
            try capture.consent(participant)
            try capture.start()
            source.events[0](event)
            XCTAssertFalse(capture.snapshot().running)
            XCTAssertEqual(capture.snapshot().session.state, .interrupted)
            source.emit(try buffer())
            XCTAssertEqual(consumer.frames.count, 0)
            XCTAssertThrowsError(try capture.start())
            try capture.start(resuming: true)
            source.emit(try buffer(10))
            XCTAssertEqual(consumer.starts, [10])
        }
    }

    func testParticipantChangeRevokesOldCallbacksAndRequiresAllNewConsents() throws {
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        try capture.changeParticipants([participant, otherParticipant])
        source.emit(try buffer())
        XCTAssertEqual(consumer.frames.count, 0)
        XCTAssertEqual(capture.snapshot().session.state, .awaitingConsent)
        try capture.consent(participant)
        XCTAssertThrowsError(try capture.start())
        try capture.consent(otherParticipant)
        try capture.start()
        source.emit(try buffer(), attempt: 0)
        source.emit(try buffer(3))
        XCTAssertEqual(consumer.starts, [3])
    }

    func testWithdrawalIsTerminalAndAccountsLateBufferDisposal() throws {
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        source.emit(try buffer())
        capture.withdraw()
        let late = try buffer(3)
        source.emit(late)
        XCTAssertTrue(late.isCleared)
        XCTAssertThrowsError(try capture.start(resuming: true))
        let snapshot = capture.snapshot()
        XCTAssertEqual(snapshot.session.state, .withdrawn)
        XCTAssertEqual(snapshot.receivedBuffers, 2)
        XCTAssertEqual(snapshot.clearedBuffers, 2)
        XCTAssertEqual(snapshot.admittedBuffers, 1)
        XCTAssertTrue(snapshot.cleanupComplete)
        XCTAssertEqual(snapshot.reason, "consent_withdrawn")
    }

    func testExpiryAndPermissionRevocationFenceAdmission() throws {
        for expiry in [false, true] {
            let source = Source()
            let consumer = Consumer()
            let clock = Clock()
            let capture = try capture(source, consumer, clock: clock)
            try capture.consent(participant)
            try capture.start()
            if expiry { clock.now = 1 } else { source.permission = .denied }
            let value = try buffer()
            source.emit(value)
            XCTAssertTrue(value.isCleared)
            XCTAssertFalse(capture.snapshot().running)
            XCTAssertEqual(consumer.frames.count, 0)
            XCTAssertEqual(capture.snapshot().session.state, expiry ? .expired : .interrupted)
        }
    }

    func testFailedStartAndSynchronousEarlyBufferNeverReachConsumer() throws {
        let source = Source()
        let consumer = Consumer()
        source.earlyBuffer = try buffer()
        source.failStart = true
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        XCTAssertThrowsError(try capture.start())
        XCTAssertTrue(source.earlyBuffer!.isCleared)
        XCTAssertEqual(consumer.frames.count, 0)
        XCTAssertFalse(capture.snapshot().running)
        XCTAssertEqual(capture.snapshot().session.state, .interrupted)
        XCTAssertGreaterThan(source.stops, 0)
    }

    func testConsumerFailureAndPolicyBufferLimitAreTerminal() throws {
        for limit in [false, true] {
            let source = Source()
            let consumer = Consumer()
            consumer.fail = !limit
            let capture = try capture(source, consumer, maximumBytes: limit ? 4 : 1024)
            try capture.consent(participant)
            try capture.start()
            let value = try buffer()
            source.emit(value)
            XCTAssertTrue(value.isCleared)
            XCTAssertEqual(capture.snapshot().session.state, .withdrawn)
            XCTAssertEqual(capture.snapshot().reason, limit ? "buffer_limit" : "consumer_failed")
            XCTAssertTrue(capture.snapshot().cleanupComplete)
        }
    }

    func testSourceCleanupFailureCannotBeReportedCompleteOrRestarted() throws {
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        source.cleanup = false
        try capture.pause()
        XCTAssertFalse(capture.snapshot().cleanupComplete)
        source.cleanup = true
        XCTAssertThrowsError(try capture.start(resuming: true))
        capture.delete()
        XCTAssertFalse(capture.snapshot().cleanupComplete)
    }

    func testWithdrawalWaitsForBorrowAndReturnsWithEveryOwnedBufferCleared() throws {
        let source = Source()
        let consumer = Consumer()
        consumer.entered = DispatchSemaphore(value: 0)
        consumer.release = DispatchSemaphore(value: 0)
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        let value = try buffer()
        let delivery = expectation(description: "delivery completed")
        DispatchQueue.global().async {
            source.emit(value)
            delivery.fulfill()
        }
        XCTAssertEqual(consumer.entered!.wait(timeout: .now() + 5), .success)
        let withdrawal = expectation(description: "withdrawal completed")
        DispatchQueue.global().async {
            capture.withdraw()
            XCTAssertTrue(value.isCleared)
            XCTAssertTrue(capture.snapshot().cleanupComplete)
            withdrawal.fulfill()
        }
        consumer.release!.signal()
        wait(for: [delivery, withdrawal], timeout: 5)
        source.emit(try buffer(3))
        XCTAssertEqual(consumer.frames.count, 1)
    }

    func testStopIsNotRestartableAndDeletionDisposesLateCallbacks() throws {
        let source = Source()
        let consumer = Consumer()
        var capture: AmbientAudioCapture? = try capture(source, consumer)
        try capture!.consent(participant)
        try capture!.start()
        try capture!.stop()
        XCTAssertThrowsError(try capture!.start(resuming: true))
        capture = nil
        let late = try buffer()
        source.emit(late)
        XCTAssertTrue(late.isCleared)
        XCTAssertEqual(consumer.frames.count, 0)
    }

    func testInvalidPCMIsRefusedBeforeAllocationAndDescriptionsContainNoSamples() throws {
        for samples in [[], [Float.nan], [Float.infinity]] {
            XCTAssertThrowsError(
                try samples.withUnsafeBufferPointer {
                    try CapturePCMBuffer(samples: $0, sampleRateHz: 16_000, startSample: 0)
                })
        }
        let value = try buffer()
        XCTAssertEqual(String(describing: value), "CapturePCMBuffer(samples=<private>)")
        value.clear()
        value.clear()
        XCTAssertTrue(value.isCleared)
        // Merely constructing the native source opens no engine and prompts for nothing.
        let native = AppleMicrophoneSource()
        XCTAssertTrue(native.stop())
    }

    func testNativePCMConversionPreservesSampleTimeAndExplicitMonoMix() throws {
        let format = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 2, interleaved: false))
        let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 3))
        pcm.frameLength = 3
        let channels = try XCTUnwrap(pcm.floatChannelData)
        for (index, sample) in [Float(0.25), -0.5, 1].enumerated() {
            channels[0][index] = sample
            channels[1][index] = 0
        }
        let value = try AppleMicrophoneSource.copyInput(pcm, time: AVAudioTime(sampleTime: 123, atRate: 16_000))
        let source = Source()
        let consumer = Consumer()
        let capture = try capture(source, consumer)
        try capture.consent(participant)
        try capture.start()
        source.emit(value)
        XCTAssertEqual(consumer.starts, [123])
        XCTAssertTrue(value.isCleared)
        // Original OS/caller-owned buffers are explicitly outside the overwrite claim.
        XCTAssertEqual(channels[0][0], 0.25)
    }

    func testNativePCMRejectsMissingTimingNonfiniteSamplesAndOversizedFrames() throws {
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(CapturePCMBuffer.maximumFrames + 1)))
        pcm.frameLength = 1
        pcm.floatChannelData![0][0] = 0
        XCTAssertThrowsError(try AppleMicrophoneSource.copyInput(pcm, time: AVAudioTime(hostTime: 1)))
        XCTAssertThrowsError(try AppleMicrophoneSource.copyInput(pcm, time: AVAudioTime(sampleTime: 0, atRate: 8000)))
        pcm.floatChannelData![0][0] = .nan
        XCTAssertThrowsError(try AppleMicrophoneSource.copyInput(pcm, time: AVAudioTime(sampleTime: 0, atRate: 16_000)))
        pcm.frameLength = AVAudioFrameCount(CapturePCMBuffer.maximumFrames + 1)
        XCTAssertThrowsError(try AppleMicrophoneSource.copyInput(pcm, time: AVAudioTime(sampleTime: 0, atRate: 16_000)))
    }

    func testInvalidCohortAndExpiredPauseStopTheProducer() throws {
        let source = Source()
        let consumer = Consumer()
        let clock = Clock()
        let capture = try capture(source, consumer, clock: clock)
        try capture.consent(participant)
        try capture.start()
        XCTAssertThrowsError(try capture.changeParticipants([]))
        XCTAssertEqual(capture.snapshot().session.state, .interrupted)
        XCTAssertFalse(capture.snapshot().running)
        try capture.start(resuming: true)
        clock.now = 1
        XCTAssertThrowsError(try capture.pause())
        XCTAssertFalse(capture.snapshot().running)
        XCTAssertEqual(capture.snapshot().session.state, .expired)
        XCTAssertTrue(capture.snapshot().cleanupComplete)
    }
}
