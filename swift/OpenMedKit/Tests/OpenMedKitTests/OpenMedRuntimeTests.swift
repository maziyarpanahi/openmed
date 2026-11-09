import Foundation
import XCTest

@testable import OpenMedKit

// Synthetic, offline fixtures only. Sentinels mark values that must never reach
// lifecycle errors, evidence or descriptions.
private let phiSentinel = "OMSENTINEL-PHI-7Q3"
private let pathSentinel = "/private/var/OMSENTINEL-PATH/model.mlx"
private let syntheticNote = "Patient Jane Roe 🧑‍⚕️ called 555-0100 on 2024-01-02 (\(phiSentinel))."

private func withLock<T>(_ lock: NSLock, _ body: () -> T) -> T {
    lock.lock()
    defer { lock.unlock() }
    return body()
}

// MARK: - Fakes

/// Manual monotonic clock: sleepers resume only when the test advances time.
private final class ManualRuntimeClock: OpenMedRuntimeClock, @unchecked Sendable {
    private struct Sleeper {
        let id: Int
        let deadline: UInt64
        let continuation: CheckedContinuation<Void, Error>
    }

    private let lock = NSLock()
    private var now: UInt64 = 10_000
    private var nextID = 0
    private var sleepers: [Sleeper] = []
    private var cancelledIDs: Set<Int> = []
    private var sleeperWaiters: [(count: Int, continuation: CheckedContinuation<Void, Never>)] = []

    func nowMilliseconds() -> UInt64 { withLock(lock) { now } }

    var pendingSleepers: Int { withLock(lock) { sleepers.count } }

    func sleep(milliseconds: UInt64) async throws {
        let id = withLock(lock) { () -> Int in
            nextID += 1
            return nextID
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.lock()
                if cancelledIDs.remove(id) != nil {
                    lock.unlock()
                    continuation.resume(throwing: CancellationError())
                    return
                }
                let deadline = now + milliseconds
                if deadline <= now {
                    lock.unlock()
                    continuation.resume()
                    return
                }
                sleepers.append(Sleeper(id: id, deadline: deadline, continuation: continuation))
                let count = sleepers.count
                let ready = sleeperWaiters.filter { $0.count <= count }
                sleeperWaiters.removeAll { $0.count <= count }
                lock.unlock()
                for waiter in ready { waiter.continuation.resume() }
            }
        } onCancel: {
            lock.lock()
            let index = sleepers.firstIndex { $0.id == id }
            let sleeper = index.map { sleepers.remove(at: $0) }
            if sleeper == nil { cancelledIDs.insert(id) }
            lock.unlock()
            sleeper?.continuation.resume(throwing: CancellationError())
        }
    }

    func advance(by milliseconds: UInt64) {
        lock.lock()
        now += milliseconds
        let due = sleepers.filter { $0.deadline <= now }
        sleepers.removeAll { $0.deadline <= now }
        lock.unlock()
        for sleeper in due { sleeper.continuation.resume() }
    }

    func waitForSleepers(_ count: Int) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            if sleepers.count >= count {
                lock.unlock()
                continuation.resume()
            } else {
                sleeperWaiters.append((count, continuation))
                lock.unlock()
            }
        }
    }
}

private final class FakeObservation: OpenMedRuntimeObservation, @unchecked Sendable {
    private let action: @Sendable () -> Void
    init(_ action: @escaping @Sendable () -> Void) { self.action = action }
    func cancel() { action() }
}

private final class FakeMemoryPressure: OpenMedMemoryPressureSource, @unchecked Sendable {
    private let lock = NSLock()
    private var handlers: [Int: @Sendable (OpenMedMemoryPressureLevel) -> Void] = [:]
    private var nextID = 0

    var observerCount: Int { withLock(lock) { handlers.count } }

    func observe(
        _ handler: @escaping @Sendable (OpenMedMemoryPressureLevel) -> Void
    ) -> any OpenMedRuntimeObservation {
        let id = withLock(lock) { () -> Int in
            nextID += 1
            handlers[nextID] = handler
            return nextID
        }
        return FakeObservation { [weak self] in
            guard let self else { return }
            withLock(self.lock) { self.handlers[id] = nil }
        }
    }

    func emit(_ level: OpenMedMemoryPressureLevel) {
        let current = withLock(lock) { Array(handlers.values) }
        for handler in current { handler(level) }
    }
}

private final class FakeBackgroundExecution: OpenMedBackgroundExecutionProvider, @unchecked Sendable {
    private let lock = NSLock()
    private var active: [Int: @Sendable () -> Void] = [:]
    private var foregroundHandlers: [Int: @Sendable () -> Void] = [:]
    private var nextID = 0
    private(set) var begun = 0
    private(set) var ended = 0

    var activeAssertions: Int { withLock(lock) { active.count } }
    var beginCount: Int { withLock(lock) { begun } }

    func beginBackgroundExecution(
        onExpiration: @escaping @Sendable () -> Void
    ) -> any OpenMedBackgroundExecutionAssertion {
        let id = withLock(lock) { () -> Int in
            nextID += 1
            begun += 1
            active[nextID] = onExpiration
            return nextID
        }
        return FakeAssertion { [weak self] in self?.end(id) }
    }

    func observeForeground(
        _ handler: @escaping @Sendable () -> Void
    ) -> any OpenMedRuntimeObservation {
        let id = withLock(lock) { () -> Int in
            nextID += 1
            foregroundHandlers[nextID] = handler
            return nextID
        }
        return FakeObservation { [weak self] in
            guard let self else { return }
            withLock(self.lock) { self.foregroundHandlers[id] = nil }
        }
    }

    private func end(_ id: Int) {
        withLock(lock) {
            if active.removeValue(forKey: id) != nil { ended += 1 }
        }
    }

    /// Simulate the platform expiring every outstanding background request.
    func expire() {
        let handlers = withLock(lock) { () -> [@Sendable () -> Void] in
            let current = Array(active.values)
            ended += active.count
            active.removeAll()
            return current
        }
        for handler in handlers { handler() }
    }

    func enterForeground() {
        let handlers = withLock(lock) { Array(foregroundHandlers.values) }
        for handler in handlers { handler() }
    }
}

private struct FakeAssertion: OpenMedBackgroundExecutionAssertion {
    let action: @Sendable () -> Void
    func end() { action() }
}

/// A single-use pause point. Interruptible gates throw when the waiting task
/// is cancelled, like async Core ML prediction; non-interruptible gates model
/// synchronous work that only notices cancellation at its next checkpoint.
private final class TestGate: @unchecked Sendable {
    private let lock = NSLock()
    private var armed = false
    private var interruptible = true
    private var arrived = false
    private var released = false
    private var held: CheckedContinuation<Void, Error>?
    private var arrivalWaiters: [CheckedContinuation<Void, Never>] = []

    func arm(interruptible: Bool = true) {
        withLock(lock) {
            armed = true
            self.interruptible = interruptible
            arrived = false
            released = false
        }
    }

    func pass() async throws {
        let shouldPause = withLock(lock) { () -> (Bool, Bool, [CheckedContinuation<Void, Never>]) in
            guard armed, !arrived else { return (false, false, []) }
            arrived = true
            armed = false
            let waiters = arrivalWaiters
            arrivalWaiters.removeAll()
            return (true, interruptible, waiters)
        }
        for waiter in shouldPause.2 { waiter.resume() }
        guard shouldPause.0 else { return }
        if shouldPause.1 {
            try await withTaskCancellationHandler {
                try await suspend(throwOnCancel: true)
            } onCancel: {
                let continuation = withLock(lock) { () -> CheckedContinuation<Void, Error>? in
                    let current = held
                    held = nil
                    return current
                }
                continuation?.resume(throwing: CancellationError())
            }
        } else {
            try await suspend(throwOnCancel: false)
        }
    }

    private func suspend(throwOnCancel: Bool) async throws {
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            lock.lock()
            if released {
                lock.unlock()
                continuation.resume()
            } else if throwOnCancel && Task.isCancelled {
                lock.unlock()
                continuation.resume(throwing: CancellationError())
            } else {
                held = continuation
                lock.unlock()
            }
        }
    }

    func waitUntilArrived() async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            lock.lock()
            if arrived {
                lock.unlock()
                continuation.resume()
            } else {
                arrivalWaiters.append(continuation)
                lock.unlock()
            }
        }
    }

    func release() {
        let continuation = withLock(lock) { () -> CheckedContinuation<Void, Error>? in
            released = true
            let current = held
            held = nil
            return current
        }
        continuation?.resume()
    }
}

private struct SentinelFailure: Error, CustomStringConvertible {
    var description: String { "failed reading \(pathSentinel) for \(phiSentinel)" }
}

/// Per-test model controls and leak ledger.
private final class FakeModelControls: @unchecked Sendable {
    private let lock = NSLock()
    let stageGate = TestGate()
    let loadGate = TestGate()
    private var pauseStage: OpenMedInferenceStage?
    private var predictionCalls = 0
    private var loaderCalls = 0
    private var liveModels = 0
    private var peakLiveModels = 0
    private var liveScratch = 0
    private var modelsCreated = 0
    private var nextLoadFailure: Error?
    private var nextPredictionFailure: Error?

    var calls: Int { withLock(lock) { predictionCalls } }
    var loads: Int { withLock(lock) { loaderCalls } }
    var live: Int { withLock(lock) { liveModels } }
    var peakLive: Int { withLock(lock) { peakLiveModels } }
    var scratch: Int { withLock(lock) { liveScratch } }
    var created: Int { withLock(lock) { modelsCreated } }

    func pause(at stage: OpenMedInferenceStage, interruptible: Bool = true) {
        withLock(lock) { pauseStage = stage }
        stageGate.arm(interruptible: interruptible)
    }

    func failNextLoad(_ error: Error) { withLock(lock) { nextLoadFailure = error } }
    func failNextPrediction(_ error: Error) { withLock(lock) { nextPredictionFailure = error } }

    func pass(_ stage: OpenMedInferenceStage) async throws {
        let matches = withLock(lock) { pauseStage == stage }
        if matches { try await stageGate.pass() }
    }

    func recordPrediction() throws {
        let failure = withLock(lock) { () -> Error? in
            predictionCalls += 1
            let failure = nextPredictionFailure
            nextPredictionFailure = nil
            return failure
        }
        if let failure { throw failure }
    }

    func recordLoad() throws {
        let failure = withLock(lock) { () -> Error? in
            loaderCalls += 1
            let failure = nextLoadFailure
            nextLoadFailure = nil
            return failure
        }
        if let failure { throw failure }
    }

    func modelCreated() -> Int {
        withLock(lock) {
            liveModels += 1
            modelsCreated += 1
            peakLiveModels = max(peakLiveModels, liveModels)
            return modelsCreated
        }
    }

    func modelDestroyed() { withLock(lock) { liveModels -= 1 } }
    func scratchAllocated() { withLock(lock) { liveScratch += 1 } }
    func scratchFreed() { withLock(lock) { liveScratch -= 1 } }
}

/// Records how many fake sessions were still alive when the runtime purged caches.
private enum CachePurgeLedger {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var controls: FakeModelControls?
    nonisolated(unsafe) private static var liveCountsAtPurge: [Int] = []

    static func reset(_ current: FakeModelControls) {
        withLock(lock) {
            controls = current
            liveCountsAtPurge = []
        }
    }

    static func recordPurge() {
        withLock(lock) { liveCountsAtPurge.append(controls?.live ?? -1) }
    }

    static var purges: [Int] { withLock(lock) { liveCountsAtPurge } }
}

private final class ScratchBuffer {
    private let controls: FakeModelControls
    private var storage = [UInt8](repeating: 0, count: 4_096)
    init(_ controls: FakeModelControls) {
        self.controls = controls
        controls.scratchAllocated()
    }
    func touch() { storage[0] &+= 1 }
    deinit { controls.scratchFreed() }
}

private final class FakeTextModel: OpenMedRuntimeModel, @unchecked Sendable {
    let generation: Int
    private let controls: FakeModelControls

    init(controls: FakeModelControls) {
        self.controls = controls
        self.generation = controls.modelCreated()
    }

    deinit { controls.modelDestroyed() }

    func predictEntities(
        in text: String,
        checkpoint: OpenMedInferenceCheckpoint
    ) async throws -> [EntityPrediction] {
        try controls.recordPrediction()
        let scratch = ScratchBuffer(controls)
        try checkpoint.check(.tokenization)
        try await controls.pass(.tokenization)
        scratch.touch()
        try checkpoint.check(.tokenization)
        try checkpoint.check(.windowing)
        try await controls.pass(.windowing)
        for _ in 0..<2 {
            try checkpoint.check(.modelPrediction)
            try await controls.pass(.modelPrediction)
            scratch.touch()
        }
        try checkpoint.check(.modelPrediction)
        return Self.entities(for: text)
    }

    static func releaseUnreferencedCaches() {
        CachePurgeLedger.recordPurge()
    }

    /// Deterministic detections with Unicode scalar offsets.
    static func entities(for text: String) -> [EntityPrediction] {
        let spans: [(String, String, Float)] = [
            ("Jane", "first_name", 0.97),
            ("Roe", "last_name", 0.95),
            ("555-0100", "phone_number", 0.91),
            ("2024-01-02", "date", 0.88),
            ("called", "occupation", 0.20),
        ]
        return spans.compactMap { needle, label, confidence in
            guard let range = text.range(of: needle) else { return nil }
            let start = PostProcessing.scalarOffset(in: text, at: range.lowerBound)
            let end = PostProcessing.scalarOffset(in: text, at: range.upperBound)
            return EntityPrediction(
                label: label,
                text: needle,
                confidence: confidence,
                start: start,
                end: end
            )
        }
    }
}

/// Selects the stage at which the in-band probe acts.
private final class ProbeTarget: @unchecked Sendable {
    enum Action { case cancelTask, criticalPressure, expireBackground }
    private let lock = NSLock()
    private var stage: OpenMedInferenceStage?
    private var action: Action = .cancelTask

    func set(_ stage: OpenMedInferenceStage?, _ action: Action = .cancelTask) {
        withLock(lock) {
            self.stage = stage
            self.action = action
        }
    }

    /// Fire once at the first checkpoint of the selected stage.
    func take(_ stage: OpenMedInferenceStage) -> Action? {
        withLock(lock) {
            guard self.stage == stage else { return nil }
            self.stage = nil
            return action
        }
    }
}

private struct HangGuardTimeout: Error {}

// MARK: - Tests

final class OpenMedRuntimeTests: XCTestCase {
    private var controls: FakeModelControls!
    private var clock: ManualRuntimeClock!
    private var pressure: FakeMemoryPressure!
    private var background: FakeBackgroundExecution!
    private var probe: ProbeTarget!

    override func setUp() {
        super.setUp()
        controls = FakeModelControls()
        clock = ManualRuntimeClock()
        pressure = FakeMemoryPressure()
        background = FakeBackgroundExecution()
        probe = ProbeTarget()
        CachePurgeLedger.reset(controls)
    }

    private func makeRuntime(
        maxConcurrentInferences: Int = 1,
        maxQueuedRequests: Int = 8,
        idleReleaseMilliseconds: UInt64? = nil,
        memoryWarningResponse: OpenMedPressureResponse = .drainInFlight(timeoutMilliseconds: nil),
        memoryCriticalResponse: OpenMedPressureResponse = .cancelInFlight
    ) -> OpenMedRuntime {
        var configuration = OpenMedRuntimeConfiguration(
            maxConcurrentInferences: maxConcurrentInferences,
            maxQueuedRequests: maxQueuedRequests,
            idleReleaseMilliseconds: idleReleaseMilliseconds,
            memoryWarningResponse: memoryWarningResponse,
            memoryCriticalResponse: memoryCriticalResponse
        )
        let probe = self.probe!
        let pressure = self.pressure!
        let background = self.background!
        configuration.stageProbe = { stage in
            switch probe.take(stage) {
            case .cancelTask?: withUnsafeCurrentTask { $0?.cancel() }
            case .criticalPressure?: pressure.emit(.critical)
            case .expireBackground?: background.expire()
            case nil: break
            }
        }
        let controls = self.controls!
        return OpenMedRuntime(
            configuration: configuration,
            memoryPressure: pressure,
            backgroundExecution: background,
            clock: clock,
            loader: {
                try controls.recordLoad()
                try await controls.loadGate.pass()
                return FakeTextModel(controls: controls)
            }
        )
    }

    // MARK: Helpers

    private func guarded<T: Sendable>(
        _ operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T?.self) { group in
            group.addTask { try await operation() }
            group.addTask {
                try await Task.sleep(nanoseconds: 30_000_000_000)
                return nil
            }
            defer { group.cancelAll() }
            guard let first = try await group.next(), let value = first else {
                throw HangGuardTimeout()
            }
            return value
        }
    }

    @discardableResult
    private func waitFor(
        _ runtime: OpenMedRuntime,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ predicate: @escaping @Sendable (OpenMedRuntimeSnapshot) -> Bool
    ) async throws -> OpenMedRuntimeSnapshot {
        let snapshot = try await guarded { await runtime.awaitSnapshot(where: predicate) }
        XCTAssertTrue(predicate(snapshot), "runtime did not reach the expected snapshot", file: file, line: line)
        return snapshot
    }

    private func arrival(_ gate: TestGate) async throws {
        try await guarded { await gate.waitUntilArrived() }
    }

    private func waitForSleepers(_ count: Int) async throws {
        let clock = self.clock!
        try await guarded { await clock.waitForSleepers(count) }
    }

    private func expectError<T: Sendable>(
        _ expected: OpenMedRuntimeError,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ operation: @escaping @Sendable () async throws -> T
    ) async {
        do {
            _ = try await guarded(operation)
            XCTFail("expected \(expected.reasonCode)", file: file, line: line)
        } catch let error as OpenMedRuntimeError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("unexpected error type \(type(of: error))", file: file, line: line)
        }
    }

    private func expectedPII(_ text: String = syntheticNote) throws -> [EntityPrediction] {
        let analyzed = OpenMedInferenceCore.applyConfidenceThreshold(
            FakeTextModel.entities(for: text),
            0.5
        )
        return try OpenMedInferenceCore.mergePII(
            analyzed,
            text: text,
            useSmartMerging: true,
            profile: .tokenClassification,
            checkpoint: OpenMedInferenceCheckpoint()
        )
    }

    private func bytes(_ entities: [EntityPrediction]) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(entities)
    }

    private func assertNoLeaks(
        _ runtime: OpenMedRuntime,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.activeLeases, 0, file: file, line: line)
        XCTAssertEqual(snapshot.pendingAdmissions, 0, file: file, line: line)
        XCTAssertEqual(controls.scratch, 0, "inference buffers retained", file: file, line: line)
        XCTAssertEqual(background.activeAssertions, 0, "background assertion leaked", file: file, line: line)
    }

    // MARK: State machine

    func testTransitionTableMatchesDocumentedLifecycle() {
        let valid: [(OpenMedRuntimeState, OpenMedRuntimeState)] = [
            (.idle, .loading), (.loading, .ready), (.loading, .idle), (.ready, .busy(1)),
            (.busy(1), .busy(2)), (.busy(2), .busy(1)), (.busy(1), .ready), (.busy(3), .draining),
            (.draining, .idle), (.draining, .released), (.ready, .idle), (.idle, .released),
            (.loading, .released), (.ready, .released),
        ]
        for (from, to) in valid {
            XCTAssertTrue(OpenMedRuntime.isValidTransition(from: from, to: to), "\(from) -> \(to)")
        }
        let invalid: [(OpenMedRuntimeState, OpenMedRuntimeState)] = [
            (.idle, .ready), (.idle, .busy(1)), (.ready, .busy(2)), (.busy(2), .ready),
            (.busy(1), .busy(3)), (.draining, .busy(1)), (.draining, .ready), (.released, .idle),
            (.released, .loading), (.loading, .busy(1)), (.ready, .draining),
        ]
        for (from, to) in invalid {
            XCTAssertFalse(OpenMedRuntime.isValidTransition(from: from, to: to), "\(from) -> \(to)")
        }
    }

    func testConcurrentCallersShareOneLoadAndMatchSynchronousCore() async throws {
        let runtime = makeRuntime(maxConcurrentInferences: 2)
        controls.loadGate.arm()
        let first = Task { try await runtime.extractPII(syntheticNote) }
        let second = Task { try await runtime.extractPII(syntheticNote) }
        let third = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.loadGate)
        try await waitFor(runtime) { $0.state == .loading && $0.pendingAdmissions == 3 }
        controls.loadGate.release()

        let results = try await guarded {
            [try await first.value, try await second.value, try await third.value]
        }
        let expected = try bytes(try expectedPII())
        for result in results { XCTAssertEqual(try bytes(result), expected) }
        XCTAssertEqual(controls.loads, 1)
        XCTAssertEqual(controls.created, 1)

        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.state, .ready)
        XCTAssertEqual(snapshot.admitted, 3)
        XCTAssertEqual(snapshot.completed, 3)
        XCTAssertEqual(snapshot.loads, 1)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testAsyncDeidentifyMatchesSynchronousCoreBytes() async throws {
        let runtime = makeRuntime()
        let policy = try Policy(named: Policy.defaultName)
        let entities = try expectedPII()
        let expectedPolicy = try OpenMedInferenceCore.policyResult(
            syntheticNote,
            entities: entities,
            policy: policy,
            checkpoint: OpenMedInferenceCheckpoint()
        )
        let expectedMethod = try OpenMedInferenceCore.methodResult(
            syntheticNote,
            entities: entities,
            method: .mask,
            checkpoint: OpenMedInferenceCheckpoint()
        )

        let byPolicy = try await runtime.deidentify(syntheticNote, policy: policy)
        let byName = try await runtime.deidentify(syntheticNote, policy: Policy.defaultName)
        let byMethod = try await runtime.deidentify(syntheticNote, method: .mask)
        let analyzed = try await runtime.analyzeText(syntheticNote)

        XCTAssertEqual(byPolicy, expectedPolicy)
        XCTAssertEqual(byName, expectedPolicy)
        XCTAssertEqual(Data(byPolicy.redactedText.utf8), Data(expectedPolicy.redactedText.utf8))
        XCTAssertFalse(byPolicy.redactedText.contains("Jane"))
        XCTAssertEqual(byMethod.deidentifiedText, expectedMethod.deidentifiedText)
        XCTAssertEqual(byMethod.piiEntities, expectedMethod.piiEntities)
        XCTAssertEqual(analyzed.count, 4, "the 0.20 detection is below the threshold")
        await runtime.shutdown()
    }

    // MARK: Cancellation

    func testCancellationAtEveryStageReachesTerminalStateWithoutLeaks() async throws {
        let runtime = makeRuntime()
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        let policy = try Policy(named: Policy.defaultName)

        for stage in OpenMedInferenceStage.allCases {
            for operation in 0..<2 {
                probe.set(stage)
                await expectError(.cancelled) {
                    if operation == 0 {
                        return try await runtime.extractPII(syntheticNote).count
                    }
                    return try await runtime.deidentify(syntheticNote, policy: policy).actions.count
                }
                let snapshot = await runtime.snapshot()
                XCTAssertEqual(snapshot.lastCancellationStage, stage, "operation \(operation)")
                XCTAssertEqual(snapshot.state, .ready)
                XCTAssertEqual(snapshot.lastOutcome, "cancelled")
                XCTAssertNotNil(snapshot.lastCancellationLatencyMilliseconds)
                await assertNoLeaks(runtime)
            }
        }

        let recovered = try bytes(try await runtime.extractPII(syntheticNote))
        XCTAssertEqual(recovered, baseline)
        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.cancelled, OpenMedInferenceStage.allCases.count * 2)
        XCTAssertEqual(snapshot.loads, 1, "cancellation must not release or reload the model")
        await runtime.shutdown()
    }

    func testCallerCancellationInterruptsSuspendedModelWork() async throws {
        let runtime = makeRuntime()
        controls.pause(at: .modelPrediction, interruptible: true)
        let task = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        task.cancel()
        await expectError(.cancelled) { try await task.value }

        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.cancelled, 1)
        XCTAssertEqual(snapshot.completed, 0)
        XCTAssertEqual(snapshot.state, .ready)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testCancellationLatencyIsBoundedByTheNextCheckpoint() async throws {
        let runtime = makeRuntime()
        controls.pause(at: .windowing, interruptible: false)
        let task = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        task.cancel()
        clock.advance(by: 40)
        controls.stageGate.release()
        await expectError(.cancelled) { try await task.value }

        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.lastCancellationStage, .modelPrediction)
        XCTAssertEqual(snapshot.lastCancellationLatencyMilliseconds, 40)
        XCTAssertEqual(snapshot.maxCancellationLatencyMilliseconds, 40)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testCancellingAQueuedRequestNeverRunsTheModel() async throws {
        let runtime = makeRuntime(maxConcurrentInferences: 1, maxQueuedRequests: 2)
        _ = try await runtime.extractPII(syntheticNote)
        let callsBefore = controls.calls
        controls.pause(at: .tokenization)
        let running = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        let queuedA = Task { try await runtime.extractPII(syntheticNote) }
        let queuedB = Task { try await runtime.extractPII(syntheticNote) }
        try await waitFor(runtime) { $0.pendingAdmissions == 2 }

        await expectError(.capacityExceeded) { try await runtime.extractPII(syntheticNote) }
        queuedA.cancel()
        await expectError(.cancelled) { try await queuedA.value }
        controls.stageGate.release()

        let expected = try bytes(try expectedPII())
        let completed = try await guarded { [try await running.value, try await queuedB.value] }
        for result in completed { XCTAssertEqual(try bytes(result), expected) }
        XCTAssertEqual(controls.calls - callsBefore, 2, "the cancelled queued request never ran")
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testSynchronousCheckpointObservesOnlyTaskCancellation() async throws {
        XCTAssertNoThrow(try OpenMedInferenceCheckpoint().check(.tokenization))
        let task = Task { () -> OpenMedRuntimeError? in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                try OpenMedInferenceCheckpoint().check(.merging)
                return nil
            } catch let error as OpenMedRuntimeError {
                return error
            } catch {
                return nil
            }
        }
        let error = await task.value
        XCTAssertEqual(error, .cancelled)
    }

    // MARK: Memory pressure

    func testMemoryWarningDrainsInFlightRejectsNewWorkAndReleasesOnlyWhenIdle() async throws {
        let runtime = makeRuntime(memoryWarningResponse: .drainInFlight(timeoutMilliseconds: nil))
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        controls.pause(at: .modelPrediction)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)

        pressure.emit(.warning)
        let draining = try await waitFor(runtime) { $0.state == .draining }
        XCTAssertTrue(draining.hasResidentModel, "an active lease keeps the session resident")
        XCTAssertEqual(draining.releases, 0)
        XCTAssertEqual(draining.admissionHold, .memoryPressure)
        await expectError(.memoryPressure) { try await runtime.extractPII(syntheticNote) }

        controls.stageGate.release()
        let finished = try await guarded { try await inFlight.value }
        XCTAssertEqual(try bytes(finished), baseline, "a drained request completes whole")

        let idle = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertFalse(idle.hasResidentModel)
        XCTAssertEqual(idle.releases, 1)
        XCTAssertEqual(CachePurgeLedger.purges, [0], "caches purged after the last reference")
        XCTAssertEqual(controls.live, 0)
        await expectError(.memoryPressure) { try await runtime.extractPII(syntheticNote) }

        pressure.emit(.normal)
        try await waitFor(runtime) { $0.admissionHold == nil }
        let recovered = try bytes(try await runtime.extractPII(syntheticNote))
        XCTAssertEqual(recovered, baseline)
        XCTAssertEqual(controls.loads, 2, "a later request reacquires through the loader")
        XCTAssertEqual(controls.created, 2)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testCriticalPressureCancelsInFlightAndRecoversAfterNormal() async throws {
        let runtime = makeRuntime()
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        controls.pause(at: .modelPrediction, interruptible: false)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)

        pressure.emit(.critical)
        try await waitFor(runtime) { $0.state == .draining }
        await expectError(.memoryPressure) { try await runtime.extractPII(syntheticNote) }
        XCTAssertEqual(controls.live, 1, "the leased session survives until its lease ends")
        controls.stageGate.release()
        await expectError(.memoryPressure) { try await inFlight.value }

        let idle = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(idle.lastCancellationStage, .modelPrediction)
        XCTAssertEqual(controls.live, 0)
        pressure.emit(.normal)
        try await waitFor(runtime) { $0.admissionHold == nil }
        let recovered = try bytes(try await runtime.extractPII(syntheticNote))
        XCTAssertEqual(recovered, baseline)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testPressureObservedInsideMergingCancelsAtThatCheckpoint() async throws {
        let runtime = makeRuntime()
        try await runtime.load()
        probe.set(.merging, .criticalPressure)
        await expectError(.memoryPressure) { try await runtime.extractPII(syntheticNote) }
        let snapshot = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(snapshot.lastCancellationStage, .merging)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testWarningDrainTimeoutCancelsStragglersOnTheInjectedClock() async throws {
        let runtime = makeRuntime(memoryWarningResponse: .drainInFlight(timeoutMilliseconds: 500))
        try await runtime.load()
        controls.pause(at: .modelPrediction)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        pressure.emit(.warning)
        try await waitFor(runtime) { $0.state == .draining }
        try await waitForSleepers(1)
        clock.advance(by: 499)
        XCTAssertEqual(clock.pendingSleepers, 1)
        clock.advance(by: 1)
        await expectError(.memoryPressure) { try await inFlight.value }
        let snapshot = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(snapshot.drainTimeouts, 1)
        XCTAssertEqual(snapshot.lastCancellationLatencyMilliseconds, 0)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testPressureWhileReadyReleasesTheIdleSessionImmediately() async throws {
        let runtime = makeRuntime()
        try await runtime.load()
        XCTAssertEqual(controls.live, 1)
        pressure.emit(.warning)
        let snapshot = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(snapshot.releases, 1)
        XCTAssertEqual(controls.live, 0)
        await runtime.shutdown()
    }

    func testPressureDuringLoadDiscardsTheLoadedSession() async throws {
        let runtime = makeRuntime()
        controls.loadGate.arm(interruptible: false)
        let waiting = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.loadGate)
        pressure.emit(.critical)
        await expectError(.memoryPressure) { try await waiting.value }
        try await waitFor(runtime) { $0.state == .idle }

        controls.loadGate.release()
        try await waitFor(runtime) { _ in !CachePurgeLedger.purges.isEmpty }
        XCTAssertEqual(CachePurgeLedger.purges, [0], "the abandoned session was dropped, then purged")
        XCTAssertEqual(controls.live, 0)
        let snapshot = await runtime.snapshot()
        XCTAssertFalse(snapshot.hasResidentModel)

        pressure.emit(.normal)
        try await waitFor(runtime) { $0.admissionHold == nil }
        _ = try await runtime.extractPII(syntheticNote)
        XCTAssertEqual(controls.loads, 2)
        await runtime.shutdown()
    }

    // MARK: Background expiration

    func testBackgroundExpirationSuppressesOutputAndRequiresForeground() async throws {
        let runtime = makeRuntime()
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        XCTAssertEqual(background.beginCount, 1)
        XCTAssertEqual(background.activeAssertions, 0, "assertion ends when the runtime is ready")

        controls.pause(at: .modelPrediction, interruptible: false)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        XCTAssertEqual(background.activeAssertions, 1)
        background.expire()
        controls.stageGate.release()
        await expectError(.backgroundExpired) { try await inFlight.value }

        let idle = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(idle.admissionHold, .backgroundExpired)
        XCTAssertEqual(idle.backgroundExpirations, 1)
        XCTAssertEqual(controls.live, 0, "abandoned session state is not kept")
        await expectError(.backgroundExpired) { try await runtime.extractPII(syntheticNote) }

        background.enterForeground()
        try await waitFor(runtime) { $0.admissionHold == nil }
        let recovered = try bytes(try await runtime.extractPII(syntheticNote))
        XCTAssertEqual(recovered, baseline)
        XCTAssertEqual(controls.created, 2, "foreground work starts from a freshly loaded session")
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    func testBackgroundExpirationDuringSerializationIsSuppressed() async throws {
        let runtime = makeRuntime()
        try await runtime.load()
        probe.set(.serialization, .expireBackground)
        await expectError(.backgroundExpired) {
            try await runtime.deidentify(syntheticNote, method: .mask).deidentifiedText
        }
        let snapshot = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(snapshot.lastCancellationStage, .serialization)
        await runtime.resumeAfterForeground()
        _ = try await runtime.deidentify(syntheticNote, method: .mask)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
    }

    // MARK: Idle release, drain and shutdown

    func testIdleReleaseUsesTheInjectedClock() async throws {
        let runtime = makeRuntime(idleReleaseMilliseconds: 1_000)
        _ = try await runtime.extractPII(syntheticNote)
        try await waitForSleepers(1)
        clock.advance(by: 999)
        let ready = await runtime.snapshot()
        XCTAssertEqual(ready.state, .ready)
        clock.advance(by: 1)
        let idle = try await waitFor(runtime) { $0.state == .idle }
        XCTAssertEqual(idle.idleReleases, 1)
        XCTAssertEqual(controls.live, 0)

        _ = try await runtime.extractPII(syntheticNote)
        XCTAssertEqual(controls.loads, 2)
        await runtime.shutdown()
        XCTAssertEqual(clock.pendingSleepers, 0, "shutdown cancels the idle timer")
    }

    func testAdmissionCancelsAPendingIdleRelease() async throws {
        let runtime = makeRuntime(idleReleaseMilliseconds: 1_000)
        _ = try await runtime.extractPII(syntheticNote)
        try await waitForSleepers(1)
        controls.pause(at: .tokenization)
        let task = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        clock.advance(by: 5_000)
        let busy = await runtime.snapshot()
        XCTAssertEqual(busy.state, .busy(1))
        XCTAssertEqual(busy.idleReleases, 0)
        controls.stageGate.release()
        _ = try await guarded { try await task.value }
        await runtime.shutdown()
    }

    func testDrainFinishesInFlightThenReleasesTerminally() async throws {
        let runtime = makeRuntime()
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        controls.pause(at: .modelPrediction)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        let drain = Task { await runtime.drain() }
        try await waitFor(runtime) { $0.state == .draining }
        await expectError(.draining) { try await runtime.extractPII(syntheticNote) }

        controls.stageGate.release()
        let drained = try await guarded { try await inFlight.value }
        XCTAssertEqual(try bytes(drained), baseline)
        try await guarded { await drain.value }
        let snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.state, .released)
        XCTAssertEqual(controls.live, 0)
        XCTAssertEqual(pressure.observerCount, 0, "signal observations end at release")
        await expectError(.released) { try await runtime.extractPII(syntheticNote) }
        await assertNoLeaks(runtime)
    }

    func testShutdownCancelsInFlightAndClosesTheRuntime() async throws {
        let runtime = makeRuntime()
        controls.pause(at: .tokenization)
        let inFlight = Task { try await runtime.extractPII(syntheticNote) }
        try await arrival(controls.stageGate)
        try await guarded { await runtime.shutdown() }
        await expectError(.released) { try await inFlight.value }
        await expectError(.released) { try await runtime.load() }
        XCTAssertEqual(controls.live, 0)
        await assertNoLeaks(runtime)
    }

    // MARK: Failures and value-free evidence

    func testLoaderAndModelFailuresAreWithheldBehindValueFreeErrors() async throws {
        let runtime = makeRuntime()
        controls.failNextLoad(SentinelFailure())
        await expectError(.modelUnavailable) { try await runtime.extractPII(syntheticNote) }
        var snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.state, .idle)
        XCTAssertEqual(snapshot.loadFailures, 1)
        XCTAssertEqual(snapshot.lastFailureTypeName, "SentinelFailure")

        controls.failNextPrediction(SentinelFailure())
        await expectError(.inferenceFailed) { try await runtime.extractPII(syntheticNote) }
        snapshot = await runtime.snapshot()
        XCTAssertEqual(snapshot.failed, 1)
        XCTAssertEqual(snapshot.state, .ready)

        _ = try await runtime.extractPII(syntheticNote)
        await assertNoLeaks(runtime)
        await runtime.shutdown()
        snapshot = await runtime.snapshot()
        assertValueFree(String(describing: snapshot))
        assertValueFree(String(reflecting: snapshot))
    }

    func testLifecycleErrorsAndEvidencePassSentinelScans() async throws {
        for error in OpenMedRuntimeError.allCases {
            let rendered = [
                error.reasonCode,
                error.retryGuidance.rawValue,
                String(describing: error),
                String(reflecting: error),
                error.localizedDescription,
                (error as NSError).localizedDescription,
                error.errorDescription ?? "",
            ]
            for value in rendered { assertValueFree(value) }
            XCTAssertTrue(error.reasonCode.allSatisfy { $0.isLowercase || $0 == "_" })
        }
        for state in [OpenMedRuntimeState.idle, .loading, .ready, .busy(3), .draining, .released] {
            assertValueFree(String(describing: state))
        }

        let runtime = makeRuntime()
        _ = try await runtime.extractPII(syntheticNote)
        probe.set(.merging)
        await expectError(.cancelled) { try await runtime.extractPII(syntheticNote) }
        pressure.emit(.critical)
        let snapshot = try await waitFor(runtime) { $0.state == .idle }
        assertValueFree(String(describing: snapshot))
        assertValueFree(String(reflecting: snapshot))
        await runtime.shutdown()
    }

    private func assertValueFree(
        _ rendered: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        for forbidden in [phiSentinel, pathSentinel, "OMSENTINEL", "Jane", "Roe", "555-0100", "2024-01-02", "/private/"] {
            XCTAssertFalse(rendered.contains(forbidden), "leaked a protected value", file: file, line: line)
        }
    }

    // MARK: Cycles

    func testRepeatedLifecycleCyclesKeepResultsStableAndBounded() async throws {
        let runtime = makeRuntime(memoryWarningResponse: .drainInFlight(timeoutMilliseconds: nil))
        let policy = try Policy(named: Policy.defaultName)
        let baseline = try bytes(try await runtime.extractPII(syntheticNote))
        let baselinePolicy = try await runtime.deidentify(syntheticNote, policy: policy)

        for cycle in 0..<4 {
            probe.set(.modelPrediction)
            await expectError(.cancelled) { try await runtime.extractPII(syntheticNote) }

            controls.pause(at: .windowing)
            let backgrounded = Task { try await runtime.deidentify(syntheticNote, policy: policy) }
            try await arrival(controls.stageGate)
            background.expire()
            await expectError(.backgroundExpired) { try await backgrounded.value }
            try await waitFor(runtime) { $0.state == .idle }
            background.enterForeground()
            try await waitFor(runtime) { $0.admissionHold == nil }

            let recovered = try bytes(try await runtime.extractPII(syntheticNote))
            XCTAssertEqual(recovered, baseline, "cycle \(cycle)")
            pressure.emit(.warning)
            try await waitFor(runtime) { $0.state == .idle }
            pressure.emit(.normal)
            try await waitFor(runtime) { $0.admissionHold == nil }
            let policyResult = try await runtime.deidentify(syntheticNote, policy: policy)
            XCTAssertEqual(policyResult, baselinePolicy, "cycle \(cycle)")
            XCTAssertLessThanOrEqual(controls.live, 1)
            await assertNoLeaks(runtime)
        }

        XCTAssertEqual(controls.peakLive, 1, "never more than one resident session")
        await runtime.shutdown()
        XCTAssertEqual(controls.live, 0)
        XCTAssertTrue(CachePurgeLedger.purges.allSatisfy { $0 == 0 })
        await assertNoLeaks(runtime)
    }

    // MARK: Platform seams

    func testSystemSeamsAreInertWithoutSignals() async throws {
        let observation = OpenMedDispatchMemoryPressureSource().observe { _ in }
        observation.cancel()
        observation.cancel()
        let assertion = OpenMedNoBackgroundExecution().beginBackgroundExecution {}
        assertion.end()
        OpenMedNoBackgroundExecution().observeForeground {}.cancel()

        let systemClock = OpenMedSystemClock()
        let before = systemClock.nowMilliseconds()
        try await systemClock.sleep(milliseconds: 0)
        XCTAssertGreaterThanOrEqual(systemClock.nowMilliseconds(), before)
        XCTAssertEqual(OpenMedRuntimeConfiguration(maxConcurrentInferences: 0).maxConcurrentInferences, 1)
        XCTAssertEqual(OpenMedRuntimeConfiguration(maxQueuedRequests: -3).maxQueuedRequests, 0)
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testBackendRuntimeLoadsOfflineAndWithholdsArtifactErrors() async throws {
            let missing = FileManager.default.temporaryDirectory
                .appending(path: "OMSENTINEL-PATH-\(UUID().uuidString)", directoryHint: .isDirectory)
            let runtime = OpenMedRuntime(
                backend: .mlx(modelDirectoryURL: missing),
                memoryPressure: pressure,
                backgroundExecution: background,
                clock: clock
            )
            await expectError(.modelUnavailable) { try await runtime.load() }
            let snapshot = await runtime.snapshot()
            XCTAssertEqual(snapshot.loadFailures, 1)
            XCTAssertNotNil(snapshot.lastFailureTypeName)
            assertValueFree(String(describing: snapshot))
            XCTAssertFalse(String(describing: snapshot).contains(missing.path))
            await runtime.shutdown()
        }
    #endif
}
