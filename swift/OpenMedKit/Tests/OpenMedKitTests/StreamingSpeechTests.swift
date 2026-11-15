import Foundation
import XCTest

@testable import OpenMedKit

private actor SpeechLatch {
    private var opened = false
    private var waiting = [CheckedContinuation<Void, Never>]()
    func wait() async {
        if !opened { await withCheckedContinuation { waiting.append($0) } }
    }
    func open() {
        opened = true
        let pending = waiting
        waiting.removeAll()
        for item in pending { item.resume() }
    }
}

private final class SpeechEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = [SpeechStreamEvent]()
    var values: [SpeechStreamEvent] {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
    func append(_ event: SpeechStreamEvent) {
        lock.lock()
        defer { lock.unlock() }
        stored.append(event)
    }
}

private struct SyntheticEngineError: Error, CustomStringConvertible {
    var description: String { "SYNTHETIC_PRIVATE_SENTINEL" }
}

private actor FakeSpeechEngine: LocalSpeechEngine {
    nonisolated let provenance = try! SpeechEngineProvenance(providerID: "synthetic", providerRevision: "v1", modelFingerprint: "sha256:" + String(repeating: "a", count: 64), language: "en")
    nonisolated let localOnly: Bool
    let started = SpeechLatch()
    let release = SpeechLatch()
    let hold: Bool
    let fail: Bool
    var shutdownCount = 0
    var windows = [SpeechPCMBuffer]()
    var callback: (@Sendable (Int, SpeechRecognition) -> Bool)?
    init(localOnly: Bool = true, hold: Bool = false, fail: Bool = false) {
        self.localOnly = localOnly
        self.hold = hold
        self.fail = fail
    }
    func recognition() throws -> SpeechRecognition {
        try SpeechRecognition(text: "synthetic", tokenTimes: [SpeechTokenTiming(tokenIndex: 0, startMilliseconds: 0, endMilliseconds: 1, confidencePPM: 800_000, text: "synthetic")], decoderConfidencePPM: 700_000)
    }
    func decode(_ window: SpeechPCMBuffer, progress: @escaping @Sendable (Int, SpeechRecognition) -> Bool) async throws -> SpeechRecognition {
        windows.append(window)
        callback = progress
        _ = progress(2, try recognition())
        _ = progress(1, try SpeechRecognition(text: "stale synthetic"))
        await started.open()
        if hold { await release.wait() }
        if fail { throw SyntheticEngineError() }
        return try recognition()
    }
    func lateCallback() throws -> Bool { callback?(3, try recognition()) ?? false }
    func windowsCleared() -> Bool { windows.allSatisfy(\.isCleared) }
    func shutdown() async { shutdownCount += 1 }
}

final class StreamingSpeechTests: XCTestCase {
    private func buffer(_ count: Int = 160, start: Int64 = 0) throws -> SpeechPCMBuffer {
        try [Float](repeating: 0.25, count: count).withUnsafeBufferPointer { try SpeechPCMBuffer(samples: $0, startSample: start) }
    }

    func testPartialFinalSequenceAndStreamRelativeTimes() async throws {
        let engine = FakeSpeechEngine()
        let events = SpeechEvents()
        let session = try StreamingSpeechSession(engine: engine, limits: SpeechSessionLimits(decodeStrideFrames: 160)) { events.append($0) }
        let first = try buffer()
        try await session.append(first)
        XCTAssertTrue(first.isCleared)
        try await session.append(buffer(start: 160), endOfUtterance: true)
        try await session.append(buffer(start: 320), endOfUtterance: true)
        try await session.finish()
        let partials = events.values.compactMap {
            if case .partial(let value) = $0 { return value }
            return nil
        }
        let finals = events.values.compactMap {
            if case .finalized(let value) = $0 { return value }
            return nil
        }
        XCTAssertEqual(partials.map(\.revision), [0, 1, 2, 3])
        XCTAssertTrue(partials.allSatisfy { $0.text == "synthetic" && !$0.isFinal })
        XCTAssertEqual(finals.map(\.segmentIndex), [0, 1])
        XCTAssertEqual(finals.map(\.startMilliseconds), [0, 20])
        XCTAssertEqual(finals.map(\.endMilliseconds), [20, 30])
        XCTAssertEqual(finals.last?.tokenTimes.first?.startMilliseconds, 20)
        XCTAssertEqual(finals.last?.tokenTimes.first?.confidencePPM, 800_000)
        XCTAssertFalse(finals[0].allowsAutomaticClinicalFinalization)
        XCTAssertEqual(session.snapshot().state, .finished)
        XCTAssertEqual(session.snapshot().bufferedFrames, 0)
        let cleared = await engine.windowsCleared()
        XCTAssertTrue(cleared)
        let count = events.values.count
        let late = try await engine.lateCallback()
        XCTAssertFalse(late)
        XCTAssertEqual(events.values.count, count)
    }

    func testBusyInputIsDisposedWithoutDisturbingActiveDecode() async throws {
        let engine = FakeSpeechEngine(hold: true)
        let session = try StreamingSpeechSession(engine: engine) { _ in }
        let active = try buffer()
        let pending = Task { try await session.append(active, endOfUtterance: true) }
        await engine.started.wait()
        let refused = try buffer(start: 160)
        do {
            try await session.append(refused)
            XCTFail("busy input admitted")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .busy) }
        XCTAssertTrue(refused.isCleared)
        do {
            try await session.finish()
            XCTFail("busy finish admitted")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .busy) }
        let shutdowns = await engine.shutdownCount
        XCTAssertEqual(shutdowns, 0)
        await engine.release.open()
        try await pending.value
        try await session.finish()
    }

    func testCancellationFencesCallbacksThenDrainsWindow() async throws {
        let engine = FakeSpeechEngine(hold: true)
        let events = SpeechEvents()
        let session = try StreamingSpeechSession(engine: engine) { events.append($0) }
        let input = try buffer()
        let pending = Task { try await session.append(input, endOfUtterance: true) }
        await engine.started.wait()
        session.requestCancellation()
        let before = events.values.count
        let late = try await engine.lateCallback()
        XCTAssertFalse(late)
        let draining = Task { await session.cancel() }
        await engine.release.open()
        await draining.value
        do {
            try await pending.value
            XCTFail("cancelled decode succeeded")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .cancelled) }
        XCTAssertEqual(events.values.count, before)
        XCTAssertTrue(input.isCleared)
        let cleared = await engine.windowsCleared()
        XCTAssertTrue(cleared)
        XCTAssertEqual(session.snapshot().bufferedFrames, 0)
        XCTAssertFalse(session.snapshot().decoding)
    }

    func testParentTaskCancellationPropagatesToChildDecoder() async throws {
        let engine = FakeSpeechEngine(hold: true)
        let session = try StreamingSpeechSession(engine: engine) { _ in }
        let input = try buffer()
        let pending = Task { try await session.append(input, endOfUtterance: true) }
        await engine.started.wait()
        pending.cancel()
        await engine.release.open()
        do {
            try await pending.value
            XCTFail("parent cancellation ignored")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .cancelled) }
        XCTAssertEqual(session.snapshot().state, .cancelled)
        XCTAssertTrue(input.isCleared)
        let cleared = await engine.windowsCleared()
        XCTAssertTrue(cleared)
    }

    func testEngineErrorIsSanitizedAndDisposesAllAudio() async throws {
        let engine = FakeSpeechEngine(fail: true)
        let session = try StreamingSpeechSession(engine: engine) { _ in }
        let input = try buffer()
        do {
            try await session.append(input, endOfUtterance: true)
            XCTFail("error ignored")
        } catch {
            XCTAssertEqual(error as? StreamingSpeechError, .unavailableEngine)
            XCTAssertFalse(String(describing: error).contains("SENTINEL"))
        }
        XCTAssertTrue(input.isCleared)
        XCTAssertEqual(session.snapshot().state, .failed)
        XCTAssertEqual(session.snapshot().bufferedFrames, 0)
        let cleared = await engine.windowsCleared()
        let shutdowns = await engine.shutdownCount
        XCTAssertTrue(cleared)
        XCTAssertGreaterThan(shutdowns, 0)
    }

    func testBoundedAccumulationAndSequenceRefusalsDisposeInput() async throws {
        for invalidSequence in [false, true] {
            let session = try StreamingSpeechSession(engine: FakeSpeechEngine(), limits: SpeechSessionLimits(maximumWindowFrames: 240, decodeStrideFrames: 240)) { _ in }
            try await session.append(buffer())
            let input = try buffer(start: invalidSequence ? 0 : 160)
            do {
                try await session.append(input)
                XCTFail("invalid append admitted")
            } catch { XCTAssertEqual(error as? StreamingSpeechError, invalidSequence ? .sequence : .bufferLimit) }
            XCTAssertTrue(input.isCleared)
            XCTAssertEqual(session.snapshot().bufferedFrames, 0)
        }
    }

    func testPartialBudgetFailsClosed() async throws {
        let session = try StreamingSpeechSession(engine: FakeSpeechEngine(), limits: SpeechSessionLimits(decodeStrideFrames: 160, maximumPartialHypotheses: 1)) { _ in }
        do {
            try await session.append(buffer())
            XCTFail("partial budget exceeded")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .bufferLimit) }
        XCTAssertEqual(session.snapshot().state, .failed)
        XCTAssertEqual(session.snapshot().bufferedFrames, 0)
    }

    func testClosedSessionClearsNewInputAndEmptyFinishIsValid() async throws {
        let session = try StreamingSpeechSession(engine: FakeSpeechEngine()) { _ in }
        try await session.finish()
        let input = try buffer()
        do {
            try await session.append(input)
            XCTFail("closed append admitted")
        } catch { XCTAssertEqual(error as? StreamingSpeechError, .closed) }
        XCTAssertTrue(input.isCleared)
        await session.cancel()
        XCTAssertEqual(session.snapshot().state, .finished)
    }

    func testNonlocalEngineAndInvalidPCMRefuse() throws {
        XCTAssertThrowsError(try StreamingSpeechSession(engine: FakeSpeechEngine(localOnly: false)) { _ in }) {
            XCTAssertEqual($0 as? StreamingSpeechError, .requiresNetwork)
        }
        for samples: [Float] in [[], [.nan], [.infinity], [1.1]] {
            XCTAssertThrowsError(try samples.withUnsafeBufferPointer { try SpeechPCMBuffer(samples: $0, startSample: 0) })
        }
        let valid: [Float] = [0]
        XCTAssertThrowsError(try valid.withUnsafeBufferPointer { try SpeechPCMBuffer(samples: $0, startSample: 0, sampleRateHz: 44_100) })
    }

    func testProtectedDescriptionsAndTimingValidation() throws {
        let token = try SpeechTokenTiming(tokenIndex: 0, startMilliseconds: 0, endMilliseconds: 10, confidencePPM: 900_000, text: "SYNTHETIC_PRIVATE_SENTINEL")
        let result = try SpeechRecognition(text: "SYNTHETIC_PRIVATE_SENTINEL", tokenTimes: [token])
        XCTAssertFalse(String(reflecting: token).contains("SENTINEL"))
        XCTAssertFalse(String(reflecting: result).contains("SENTINEL"))
        XCTAssertThrowsError(try result.validate(windowFrames: 16))
        XCTAssertThrowsError(try SpeechRecognition(text: "x", tokenTimes: [token, token]))
        XCTAssertThrowsError(try SpeechTokenTiming(tokenIndex: 0, startMilliseconds: 2, endMilliseconds: 1, confidencePPM: 0, text: "x"))
    }
}
