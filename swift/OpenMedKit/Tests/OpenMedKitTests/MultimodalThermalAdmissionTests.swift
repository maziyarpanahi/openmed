import Foundation
import XCTest

@testable import OpenMedKit

private final class ThermalTestSignals: MultimodalThermalSource, @unchecked Sendable {
    private let lock = NSLock()
    private var now: UInt64 = 100
    private var state: MultimodalThermalState? = .nominal
    private var observer: (@Sendable () -> Void)?
    private var calls = 0
    private(set) var cancelled = false
    func time() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        return now
    }
    func setTime(_ value: UInt64) {
        lock.lock()
        now = value
        lock.unlock()
    }
    func advance(_ delta: UInt64) {
        lock.lock()
        now += delta
        lock.unlock()
    }
    func call() {
        lock.lock()
        calls += 1
        lock.unlock()
    }
    func callCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return calls
    }
    func current() -> MultimodalThermalState? {
        lock.lock()
        defer { lock.unlock() }
        return state
    }
    func change(_ value: MultimodalThermalState?) {
        lock.lock()
        state = value
        let notify = observer
        lock.unlock()
        notify?()
    }
    func observe(_ change: @escaping @Sendable () -> Void) -> any ThermalObservationSubscription {
        lock.lock()
        observer = change
        lock.unlock()
        return TestSubscription { [weak self] in self?.unsubscribe() }
    }
    private func unsubscribe() {
        lock.lock()
        observer = nil
        cancelled = true
        lock.unlock()
    }
}

private struct TestSubscription: ThermalObservationSubscription {
    let action: @Sendable () -> Void
    init(_ action: @escaping @Sendable () -> Void) { self.action = action }
    func cancel() { action() }
}

private struct TestThermalWork: LocalMultimodalWork {
    let requiresNetwork: Bool
    let action: @Sendable (MultimodalWorkProfile) async throws -> String
    init(remote: Bool = false, _ action: @escaping @Sendable (MultimodalWorkProfile) async throws -> String) {
        requiresNetwork = remote
        self.action = action
    }
    func run(profile: MultimodalWorkProfile) async throws -> String { try await action(profile) }
}

final class MultimodalThermalAdmissionTests: XCTestCase {
    private func gate(_ signals: ThermalTestSignals, operations: Int = 100, recovery: UInt64 = 100) throws -> MultimodalThermalSession {
        try MultimodalThermalSession(budget: MultimodalThermalBudget(maxQueueMs: 20, maxLatencyMs: 50, maxOutputAgeMs: 60, recoveryMs: recovery, historyAgeMs: 500, historySize: 3, reducedCadenceMs: 10, normalWindowMs: 5000, reducedWindowMs: 1000, maxOperations: operations), source: signals, monotonicMilliseconds: { signals.time() })
    }
    private func work(_ signals: ThermalTestSignals, time: UInt64 = 0) -> TestThermalWork {
        TestThermalWork { _ in
            signals.call()
            signals.advance(time)
            return "synthetic protected content"
        }
    }

    func testNativeStyleSignalsGateAllFourStatesWithoutRemoteFallback() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s)
        let normal = await g.run(work(s), queuedAtMilliseconds: s.time(), requestedWindowMs: 5000)
        XCTAssertEqual(normal.reason, .ready)
        XCTAssertEqual(normal.mode, .normal)
        s.change(.fair)
        let tooSoon = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertEqual(tooSoon.reason, .resources)
        s.advance(10)
        let fair = await g.run(work(s), queuedAtMilliseconds: s.time(), requestedWindowMs: 5000)
        XCTAssertEqual(fair.mode, .reducedCadence)
        XCTAssertNotNil(fair.value)
        s.change(.serious)
        s.advance(10)
        let oversized = await g.run(work(s), queuedAtMilliseconds: s.time(), requestedWindowMs: 1001)
        XCTAssertNil(oversized.value)
        XCTAssertEqual(oversized.reason, .resources)
        let serious = await g.run(work(s), queuedAtMilliseconds: s.time(), requestedWindowMs: 1000)
        XCTAssertEqual(serious.mode, .reducedWindow)
        XCTAssertNotNil(serious.value)
        s.change(.critical)
        s.advance(100)
        let critical = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertEqual(critical.mode, .paused)
        XCTAssertEqual(critical.reason, .resources)
        XCTAssertNil(critical.value)
        XCTAssertEqual(s.callCount(), 3)
    }

    func testRecoveryNeedsContinuousLowerStateAndUnknownPauses() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s)
        s.change(.critical)
        s.change(.nominal)
        s.advance(99)
        XCTAssertEqual(g.evidence().mode, .paused)
        s.change(.critical)
        s.change(.nominal)
        s.advance(99)
        XCTAssertEqual(g.evidence().mode, .paused)
        s.advance(1)
        XCTAssertEqual(g.evidence().mode, .normal)
        s.change(nil)
        let missing = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertEqual(missing.reason, .observationUnavailable)
        XCTAssertEqual(s.callCount(), 0)
        s.change(.fair)
        s.advance(100)
        XCTAssertEqual(g.evidence().mode, .reducedCadence)
        g.close()
        XCTAssertTrue(s.cancelled)
        let closed = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertEqual(closed.reason, .closed)
    }

    func testHotPulseDuringInferenceInvalidatesItsResultEvenAfterRecovery() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s, recovery: 5)
        let result = await g.run(
            TestThermalWork { _ in
                s.change(.critical)
                s.change(.nominal)
                s.advance(5)
                _ = g.evidence()
                return "SYNTHETIC_PRIVATE"
            }, queuedAtMilliseconds: s.time())
        XCTAssertEqual(result.reason, .resources)
        XCTAssertNil(result.value)
        XCTAssertFalse(String(reflecting: result).contains("SYNTHETIC_PRIVATE"))
        XCTAssertEqual(g.evidence().withheld, 1)
    }

    func testMeasuredLatencyDegradesAndHistoryExpiresBeforeRecovery() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s)
        for _ in 0..<4 { _ = await g.run(work(s, time: 1), queuedAtMilliseconds: s.time()) }
        XCTAssertEqual(g.evidence().latencySamples, 3)
        let slow = await g.run(work(s, time: 51), queuedAtMilliseconds: s.time())
        XCTAssertEqual(slow.reason, .latency)
        XCTAssertNil(slow.value)
        XCTAssertEqual(g.evidence().latencyP95Ms, 51)
        XCTAssertEqual(g.evidence().mode, .reducedCadence)
        s.advance(501)
        XCTAssertEqual(g.evidence().latencySamples, 0)
        XCTAssertEqual(g.evidence().mode, .reducedCadence)
        s.advance(100)
        XCTAssertEqual(g.evidence().mode, .normal)
    }

    func testQueueAgeOutputDeadlineAndFutureQueueAreWithheld() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s)
        let old = await g.run(work(s), queuedAtMilliseconds: 79)
        let future = await g.run(work(s), queuedAtMilliseconds: 101)
        XCTAssertEqual(old.reason, .queueExpired)
        XCTAssertEqual(future.reason, .queueExpired)
        let late = await g.run(work(s, time: 41), queuedAtMilliseconds: 80)
        XCTAssertEqual(late.reason, .outputExpired)
        XCTAssertNil(late.value)
        XCTAssertEqual(g.evidence().queueAgeMs, 20)
        XCTAssertEqual(s.callCount(), 1)
    }

    func testClockFailureRemoteAndProviderErrorsNeverReleaseWork() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s)
        let remote = await g.run(
            TestThermalWork(remote: true) { _ in
                s.call()
                return "unused"
            }, queuedAtMilliseconds: s.time())
        XCTAssertEqual(remote.reason, .providerRejected)
        XCTAssertEqual(s.callCount(), 0)
        struct PrivateFailure: Error { let privateData = "SYNTHETIC_PRIVATE" }
        let failure = await g.run(TestThermalWork { _ in throw PrivateFailure() }, queuedAtMilliseconds: s.time())
        XCTAssertEqual(failure.reason, .providerRejected)
        XCTAssertFalse(String(reflecting: failure).contains("SYNTHETIC_PRIVATE"))
        let reversed = await g.run(
            TestThermalWork { _ in
                s.setTime(90)
                return "unused"
            }, queuedAtMilliseconds: 100)
        XCTAssertEqual(reversed.reason, .clockInvalid)
        XCTAssertNil(reversed.value)
        s.setTime(120)
        XCTAssertEqual(g.evidence().reason, .clockInvalid)
    }

    func testLimitIsBoundedAndEvidenceDoesNotExposeReturnedContent() async throws {
        let s = ThermalTestSignals()
        let g = try gate(s, operations: 1)
        let ok = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertNotNil(ok.value)
        XCTAssertFalse(String(reflecting: ok).contains("synthetic protected"))
        let refused = await g.run(work(s), queuedAtMilliseconds: s.time())
        XCTAssertEqual(refused.reason, .operationLimit)
        XCTAssertEqual(g.evidence().attempts, 1)
        XCTAssertEqual(g.evidence().completed, 1)
        XCTAssertEqual(g.evidence().withheld, 0)
        XCTAssertFalse(String(reflecting: g.evidence()).contains("synthetic protected"))
    }

    #if canImport(MLXLMCommon) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testVisionAdapterRefusesRemoteImagesBeforeOpeningThem() async throws {
            let signals = ThermalTestSignals()
            let session = try gate(signals)
            for url in [URL(string: "https://example.invalid/synthetic.png")!, URL(string: "file://remote.example/synthetic.png")!] {
                let request = ThermalVisionWork(imageURL: url) {
                    signals.call()
                    throw MultimodalThermalError.invalidBudget
                }
                let response = await session.run(request, queuedAtMilliseconds: signals.time())
                XCTAssertEqual(response.reason, .providerRejected)
            }
            XCTAssertEqual(signals.callCount(), 0)
            let local = ThermalVisionWork(imageURL: URL(fileURLWithPath: "/synthetic.png")) {
                signals.call()
                return OpenMedVisionLanguageGeneration(text: "protected", promptTokenCount: 1, generationTokenCount: 1, promptTime: 0, generationTime: 0)
            }
            let admitted = await session.run(local, queuedAtMilliseconds: signals.time())
            XCTAssertEqual(admitted.value?.text, "protected")
            XCTAssertEqual(signals.callCount(), 1)
        }
    #endif

    func testSystemNotificationSourceReadsWithoutRecordingDeviceIdentity() {
        let source = AppleSystemThermalSource()
        let signals = ThermalTestSignals()
        let token = source.observe { signals.call() }
        _ = source.current()
        NotificationCenter.default.post(name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
        XCTAssertGreaterThan(signals.callCount(), 0)
        token.cancel()
    }
}
