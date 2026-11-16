import Foundation

// MARK: - Lifecycle vocabulary

/// Lifecycle state of an `OpenMedRuntime`.
///
/// The main path is `idle → loading → ready → busy(n) → draining → released`.
/// A drain caused by memory pressure or background expiration ends in `idle`
/// (the model session is released and reacquired on a later request); a drain
/// started by `drain()` or `shutdown()` ends in the terminal `released` state.
public enum OpenMedRuntimeState: Sendable, Equatable, CustomStringConvertible {
    /// No model session is resident. Admission loads one lazily.
    case idle
    /// A model session is loading from its verified local artifact.
    case loading
    /// A model session is resident and no inference holds a lease on it.
    case ready
    /// `n` admitted inferences hold leases on the resident session.
    case busy(Int)
    /// Admission is closed while in-flight leases finish or cancel.
    case draining
    /// Terminal: the runtime is shut down and holds no model session.
    case released

    /// Stable, value-free reason code.
    public var reasonCode: String {
        switch self {
        case .idle: return "idle"
        case .loading: return "loading"
        case .ready: return "ready"
        case .busy: return "busy"
        case .draining: return "draining"
        case .released: return "released"
        }
    }

    public var description: String {
        if case .busy(let count) = self { return "busy(\(count))" }
        return reasonCode
    }
}

/// What a caller may do after an `OpenMedRuntimeError`.
public enum OpenMedRuntimeRetryGuidance: String, Sendable, CaseIterable {
    /// The caller cancelled; resubmit only if the result is still wanted.
    case doNotRetry = "do_not_retry"
    /// Transient capacity limit; retry after in-flight work settles.
    case retryLater = "retry_later"
    /// Retry after the platform reports normal memory pressure.
    case retryAfterMemoryPressureClears = "retry_after_memory_pressure_clears"
    /// Retry after the application returns to the foreground.
    case retryAfterForeground = "retry_after_foreground"
    /// This runtime is closing or closed; create a new runtime.
    case retryWithNewRuntime = "retry_with_new_runtime"
    /// The model session could not be loaded; check the local artifact.
    case checkModelArtifact = "check_model_artifact"
}

/// Typed, value-free lifecycle outcome of an `OpenMedRuntime` operation.
///
/// Errors carry only a stable reason code and retry guidance. They never
/// contain input text, spans, labels, model paths or underlying error
/// messages, so they are safe to log.
public enum OpenMedRuntimeError: Error, Sendable, Hashable, CaseIterable, LocalizedError,
    CustomStringConvertible, CustomDebugStringConvertible
{
    /// The caller's task was cancelled; no output is delivered.
    case cancelled
    /// Refused or stopped because the platform reported memory pressure.
    case memoryPressure
    /// Stopped because background execution time expired.
    case backgroundExpired
    /// Refused because the runtime is draining for shutdown.
    case draining
    /// The runtime is shut down.
    case released
    /// Refused because the admission queue is full.
    case capacityExceeded
    /// The model session could not be loaded.
    case modelUnavailable
    /// The model failed while running; the underlying error is withheld.
    case inferenceFailed

    /// Stable, value-free reason code.
    public var reasonCode: String {
        switch self {
        case .cancelled: return "cancelled"
        case .memoryPressure: return "memory_pressure"
        case .backgroundExpired: return "background_expired"
        case .draining: return "draining"
        case .released: return "released"
        case .capacityExceeded: return "capacity_exceeded"
        case .modelUnavailable: return "model_unavailable"
        case .inferenceFailed: return "inference_failed"
        }
    }

    /// What the caller may do next.
    public var retryGuidance: OpenMedRuntimeRetryGuidance {
        switch self {
        case .cancelled, .inferenceFailed: return .doNotRetry
        case .memoryPressure: return .retryAfterMemoryPressureClears
        case .backgroundExpired: return .retryAfterForeground
        case .draining, .released: return .retryWithNewRuntime
        case .capacityExceeded: return .retryLater
        case .modelUnavailable: return .checkModelArtifact
        }
    }

    public var errorDescription: String? {
        "OpenMed runtime stopped the request (\(reasonCode); \(retryGuidance.rawValue))."
    }

    public var description: String {
        "OpenMedRuntimeError(reason=\(reasonCode), retry=\(retryGuidance.rawValue))"
    }

    public var debugDescription: String { description }
}

/// How the runtime treats in-flight inference when memory pressure arrives.
public enum OpenMedPressureResponse: Sendable, Equatable {
    /// Refuse new work and let in-flight inferences finish, cancelling any
    /// still running after `timeoutMilliseconds` (never, when `nil`). The
    /// model session is released once the last lease ends.
    case drainInFlight(timeoutMilliseconds: UInt64?)
    /// Refuse new work, cancel in-flight inferences at their next checkpoint
    /// and release the model session once their leases end.
    case cancelInFlight
}

/// Admission, release and pressure policy for an `OpenMedRuntime`.
public struct OpenMedRuntimeConfiguration: Sendable {
    /// Inferences that may hold a lease at once (1...64). The default of 1
    /// serializes model execution, which every bundled model requires.
    public var maxConcurrentInferences: Int
    /// Requests that may wait for a lease or a model load (0...1024).
    public var maxQueuedRequests: Int
    /// Release a resident model after it has been idle this long; `nil`
    /// keeps it resident until pressure, expiration or shutdown.
    public var idleReleaseMilliseconds: UInt64?
    /// Response to a memory-pressure warning.
    public var memoryWarningResponse: OpenMedPressureResponse
    /// Response to critical memory pressure.
    public var memoryCriticalResponse: OpenMedPressureResponse

    /// Test seam: invoked synchronously at every checkpoint, inside the
    /// inference task, with the stage name only.
    var stageProbe: (@Sendable (OpenMedInferenceStage) -> Void)?

    /// Create a configuration. Out-of-range limits are clamped.
    public init(
        maxConcurrentInferences: Int = 1,
        maxQueuedRequests: Int = 16,
        idleReleaseMilliseconds: UInt64? = 60_000,
        memoryWarningResponse: OpenMedPressureResponse = .drainInFlight(timeoutMilliseconds: 2_000),
        memoryCriticalResponse: OpenMedPressureResponse = .cancelInFlight
    ) {
        self.maxConcurrentInferences = min(max(1, maxConcurrentInferences), 64)
        self.maxQueuedRequests = min(max(0, maxQueuedRequests), 1_024)
        self.idleReleaseMilliseconds = idleReleaseMilliseconds
        self.memoryWarningResponse = memoryWarningResponse
        self.memoryCriticalResponse = memoryCriticalResponse
        self.stageProbe = nil
    }

    func response(for level: OpenMedMemoryPressureLevel) -> OpenMedPressureResponse? {
        switch level {
        case .normal: return nil
        case .warning: return memoryWarningResponse
        case .critical: return memoryCriticalResponse
        }
    }
}

/// Value-free lifecycle evidence: counts, reason codes, stage names and
/// monotonic latencies only.
public struct OpenMedRuntimeSnapshot: Sendable, Equatable, CustomStringConvertible {
    public let state: OpenMedRuntimeState
    /// Why admission is currently refused, if it is.
    public let admissionHold: OpenMedRuntimeError?
    public let memoryPressure: OpenMedMemoryPressureLevel
    public let hasResidentModel: Bool
    public let activeLeases: Int
    public let pendingAdmissions: Int
    public let loads: Int
    public let loadFailures: Int
    public let releases: Int
    public let idleReleases: Int
    public let admitted: Int
    public let completed: Int
    public let cancelled: Int
    public let rejected: Int
    public let failed: Int
    public let memoryPressureEvents: Int
    public let backgroundExpirations: Int
    public let drainTimeouts: Int
    public let transitions: Int
    /// Reason code of the most recent settled or refused request.
    public let lastOutcome: String?
    /// Stage at which the most recent cancellation was observed.
    public let lastCancellationStage: OpenMedInferenceStage?
    /// Milliseconds from the most recent cancellation request to its terminal state.
    public let lastCancellationLatencyMilliseconds: UInt64?
    public let maxCancellationLatencyMilliseconds: UInt64?
    /// Swift type name of the most recent withheld loader or model error.
    public let lastFailureTypeName: String?

    public var description: String {
        var fields = [
            "state=\(state)",
            "hold=\(admissionHold?.reasonCode ?? "none")",
            "pressure=\(memoryPressure.reasonCode)",
            "resident=\(hasResidentModel)",
            "active=\(activeLeases)",
            "pending=\(pendingAdmissions)",
            "loads=\(loads)",
            "releases=\(releases)",
            "admitted=\(admitted)",
            "completed=\(completed)",
            "cancelled=\(cancelled)",
            "rejected=\(rejected)",
            "failed=\(failed)",
        ]
        if let lastOutcome { fields.append("last=\(lastOutcome)") }
        if let lastCancellationStage { fields.append("cancel_stage=\(lastCancellationStage.rawValue)") }
        if let maxCancellationLatencyMilliseconds {
            fields.append("max_cancel_ms=\(maxCancellationLatencyMilliseconds)")
        }
        return "OpenMedRuntimeSnapshot(\(fields.joined(separator: ", ")))"
    }
}

// MARK: - Runtime

/// Lifecycle-managed, cancellable owner of one on-device text model session.
///
/// `OpenMedRuntime` admits requests, loads the model session lazily through a
/// local loader, leases it to in-flight inferences, and releases it when idle,
/// under memory pressure, on background expiration or at shutdown. Each
/// inference runs tokenization, windowing, model prediction, merging and
/// serialization behind cooperative checkpoints; output from cancelled work
/// is never delivered, and an active lease always outlives the session's use.
///
/// The runtime never performs network access itself and never selects a
/// remote backend. Recovery after a release calls the same local loader, so it
/// passes through the same artifact verification as the first load.
public actor OpenMedRuntime {
    private enum DrainIntent: Int, Comparable {
        case memoryPressure
        case backgroundExpired
        case drain
        case shutdown

        static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }

        var isTerminal: Bool { self >= .drain }

        var cancellationReason: OpenMedRuntimeError {
            switch self {
            case .memoryPressure: return .memoryPressure
            case .backgroundExpired: return .backgroundExpired
            case .drain: return .draining
            case .shutdown: return .released
            }
        }
    }

    private struct Waiter {
        let id: UInt64
        let continuation: CheckedContinuation<Void, Never>
    }

    private struct Counters {
        var loads = 0
        var loadFailures = 0
        var releases = 0
        var idleReleases = 0
        var admitted = 0
        var completed = 0
        var cancelled = 0
        var rejected = 0
        var failed = 0
        var memoryPressureEvents = 0
        var backgroundExpirations = 0
        var drainTimeouts = 0
        var transitions = 0
        var lastOutcome: String?
        var lastCancellationStage: OpenMedInferenceStage?
        var lastCancellationLatency: UInt64?
        var maxCancellationLatency: UInt64?
        var lastFailureTypeName: String?
    }

    /// The admission, release and pressure policy.
    public nonisolated let configuration: OpenMedRuntimeConfiguration

    private let loader: @Sendable () async throws -> any OpenMedRuntimeModel
    private let memoryPressureSource: any OpenMedMemoryPressureSource
    private let backgroundExecution: any OpenMedBackgroundExecutionProvider
    private let clock: any OpenMedRuntimeClock
    private nonisolated let leases = OpenMedLeaseRegistry()
    private nonisolated let signals = OpenMedSignalSequencer()

    /// Current lifecycle state.
    public private(set) var state: OpenMedRuntimeState = .idle

    private var model: (any OpenMedRuntimeModel)?
    private var loadTask: Task<Void, Never>?
    private var loadGeneration: UInt64 = 0
    private var failedLoadGeneration: UInt64?
    private var drainIntent: DrainIntent?
    private var pressureHold = false
    private var backgroundHold = false
    private var pressureLevel: OpenMedMemoryPressureLevel = .normal
    private var activeLeaseCount = 0
    private var pendingAdmissions = 0
    private var nextLeaseID: UInt64 = 0
    private var waiters: [Waiter] = []
    private var nextWaiterID: UInt64 = 0
    private var idleTimer: Task<Void, Never>?
    private var idleGeneration: UInt64 = 0
    private var drainTimer: Task<Void, Never>?
    private var drainGeneration: UInt64 = 0
    private var backgroundAssertion: (any OpenMedBackgroundExecutionAssertion)?
    private var observations: [any OpenMedRuntimeObservation] = []
    private var observingSignals = false
    private var lastPressureSignal: UInt64 = 0
    private var lastLifecycleSignal: UInt64 = 0
    private var counters = Counters()

    /// Create a runtime around a local model loader.
    ///
    /// - Parameters:
    ///   - configuration: Admission, idle-release and pressure policy.
    ///   - memoryPressure: Memory-pressure signal source.
    ///   - backgroundExecution: Background-time provider. The default never
    ///     requests background time; pass
    ///     `OpenMedUIApplicationBackgroundExecution` from an iOS application.
    ///   - clock: Monotonic clock for timers and latency evidence.
    ///   - loader: Loads and verifies a fresh model session from local assets.
    ///     It is called again after every release, so it must not reuse
    ///     abandoned mutable state and must not reach the network.
    public init(
        configuration: OpenMedRuntimeConfiguration = OpenMedRuntimeConfiguration(),
        memoryPressure: any OpenMedMemoryPressureSource = OpenMedDispatchMemoryPressureSource(),
        backgroundExecution: any OpenMedBackgroundExecutionProvider = OpenMedNoBackgroundExecution(),
        clock: any OpenMedRuntimeClock = OpenMedSystemClock(),
        loader: @escaping @Sendable () async throws -> any OpenMedRuntimeModel
    ) {
        self.configuration = configuration
        self.memoryPressureSource = memoryPressure
        self.backgroundExecution = backgroundExecution
        self.clock = clock
        self.loader = loader
    }

    deinit {
        loadTask?.cancel()
        idleTimer?.cancel()
        drainTimer?.cancel()
        backgroundAssertion?.end()
        for observation in observations { observation.cancel() }
        leases.cancelAll(.released)
    }

    // MARK: Public operations

    /// Load the model session now instead of on the first request.
    ///
    /// - Throws: `OpenMedRuntimeError` when admission is refused or loading fails.
    public func load() async throws {
        _ = try await acquire(lease: false)
    }

    /// Run named-entity recognition and return entities above the threshold.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - confidenceThreshold: Minimum confidence to include an entity.
    /// - Returns: Detected entities with Unicode scalar offsets into `text`.
    /// - Throws: `OpenMedRuntimeError`; never partial output.
    public func analyzeText(
        _ text: String,
        confidenceThreshold: Float = 0.5
    ) async throws -> [EntityPrediction] {
        try await perform { model, checkpoint in
            let raw = try await model.predictEntities(in: text, checkpoint: checkpoint)
            let entities = OpenMedInferenceCore.applyConfidenceThreshold(raw, confidenceThreshold)
            try checkpoint.check(.serialization)
            return entities
        }
    }

    /// Detect PII with OpenMed's span repair and semantic merging.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - confidenceThreshold: Minimum confidence to include an entity.
    ///   - useSmartMerging: Apply semantic-unit merging after span repair.
    /// - Returns: PII entities identical to `OpenMed.extractPII` for the same model.
    /// - Throws: `OpenMedRuntimeError`; never partial output.
    public func extractPII(
        _ text: String,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) async throws -> [EntityPrediction] {
        try await perform { model, checkpoint in
            let entities = try await Self.detectPII(
                model: model,
                text: text,
                confidenceThreshold: confidenceThreshold,
                useSmartMerging: useSmartMerging,
                checkpoint: checkpoint
            )
            try checkpoint.check(.serialization)
            return entities
        }
    }

    /// De-identify text under a bundled policy profile.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - policy: Bundled policy name or alias, resolved before admission.
    ///   - confidenceThreshold: Minimum confidence to include an entity.
    ///   - useSmartMerging: Apply semantic-unit merging after span repair.
    /// - Returns: A complete policy result; cancelled work returns nothing.
    /// - Throws: `PolicyError` for an unknown policy, else `OpenMedRuntimeError`.
    public func deidentify(
        _ text: String,
        policy: String,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) async throws -> PolicyDeidentificationResult {
        let resolvedPolicy = try Policy(named: policy)
        return try await deidentify(
            text,
            policy: resolvedPolicy,
            confidenceThreshold: confidenceThreshold,
            useSmartMerging: useSmartMerging
        )
    }

    /// De-identify text under an already loaded policy profile.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - policy: Loaded policy profile.
    ///   - confidenceThreshold: Minimum confidence to include an entity.
    ///   - useSmartMerging: Apply semantic-unit merging after span repair.
    /// - Returns: A complete policy result; cancelled work returns nothing.
    /// - Throws: `OpenMedRuntimeError`.
    public func deidentify(
        _ text: String,
        policy: Policy,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) async throws -> PolicyDeidentificationResult {
        try await perform { model, checkpoint in
            let entities = try await Self.detectPII(
                model: model,
                text: text,
                confidenceThreshold: confidenceThreshold,
                useSmartMerging: useSmartMerging,
                checkpoint: checkpoint
            )
            return try OpenMedInferenceCore.policyResult(
                text,
                entities: entities,
                policy: policy,
                checkpoint: checkpoint
            )
        }
    }

    /// Detect and de-identify PII, returning a Python-schema-compatible result.
    ///
    /// - Parameters:
    ///   - text: Input clinical text.
    ///   - method: Mask or remove detected spans.
    ///   - confidenceThreshold: Minimum confidence to include an entity.
    ///   - useSmartMerging: Apply semantic-unit merging after span repair.
    /// - Returns: A complete result; cancelled work returns nothing.
    /// - Throws: `OpenMedRuntimeError`.
    public func deidentify(
        _ text: String,
        method: DeidentificationMethod = .mask,
        confidenceThreshold: Float = 0.5,
        useSmartMerging: Bool = true
    ) async throws -> DeidentificationResult {
        try await perform { model, checkpoint in
            let entities = try await Self.detectPII(
                model: model,
                text: text,
                confidenceThreshold: confidenceThreshold,
                useSmartMerging: useSmartMerging,
                checkpoint: checkpoint
            )
            return try OpenMedInferenceCore.methodResult(
                text,
                entities: entities,
                method: method,
                checkpoint: checkpoint
            )
        }
    }

    /// Release the resident model session if no inference holds a lease.
    ///
    /// - Returns: `true` when a session was released.
    @discardableResult
    public func releaseIdleResources() -> Bool {
        guard state == .ready else { return false }
        cancelIdleTimer()
        releaseModel()
        transition(to: .idle)
        return true
    }

    /// Apply a memory-pressure level, for applications that observe their own
    /// signals (for example `UIApplication.didReceiveMemoryWarningNotification`).
    ///
    /// Warning and critical levels refuse new admissions, respond to in-flight
    /// work per `configuration`, and release the model session once no lease
    /// holds it. `normal` lifts the admission hold.
    public func handleMemoryPressure(_ level: OpenMedMemoryPressureLevel) {
        guard state != .released else { return }
        pressureLevel = level
        guard let response = configuration.response(for: level) else {
            pressureHold = false
            noteChange()
            return
        }
        counters.memoryPressureEvents += 1
        pressureHold = true
        switch response {
        case .cancelInFlight:
            beginDrain(.memoryPressure, cancelInFlight: true, timeoutMilliseconds: nil)
        case .drainInFlight(let timeoutMilliseconds):
            beginDrain(
                .memoryPressure,
                cancelInFlight: false,
                timeoutMilliseconds: timeoutMilliseconds
            )
        }
    }

    /// Apply background-time expiration: cancel in-flight work, suppress its
    /// output, release the model session and refuse admission until
    /// `resumeAfterForeground()`.
    public func handleBackgroundExpiration() {
        guard state != .released else { return }
        counters.backgroundExpirations += 1
        backgroundHold = true
        endBackgroundAssertion()
        beginDrain(.backgroundExpired, cancelInFlight: true, timeoutMilliseconds: nil)
    }

    /// Lift the background-expiration admission hold. A later request reloads
    /// the model session through the loader; no abandoned state is reused.
    public func resumeAfterForeground() {
        guard backgroundHold else { return }
        backgroundHold = false
        noteChange()
    }

    /// Stop admission, let in-flight inferences finish, release the model
    /// session and enter the terminal `released` state.
    public func drain() async {
        beginDrain(.drain, cancelInFlight: false, timeoutMilliseconds: nil)
        await waitUntilReleased()
    }

    /// Stop admission, cancel in-flight inferences, release the model session
    /// and enter the terminal `released` state.
    public func shutdown() async {
        beginDrain(.shutdown, cancelInFlight: true, timeoutMilliseconds: nil)
        await waitUntilReleased()
    }

    /// Value-free lifecycle evidence.
    public func snapshot() -> OpenMedRuntimeSnapshot {
        OpenMedRuntimeSnapshot(
            state: state,
            admissionHold: admissionError(),
            memoryPressure: pressureLevel,
            hasResidentModel: model != nil,
            activeLeases: activeLeaseCount,
            pendingAdmissions: pendingAdmissions,
            loads: counters.loads,
            loadFailures: counters.loadFailures,
            releases: counters.releases,
            idleReleases: counters.idleReleases,
            admitted: counters.admitted,
            completed: counters.completed,
            cancelled: counters.cancelled,
            rejected: counters.rejected,
            failed: counters.failed,
            memoryPressureEvents: counters.memoryPressureEvents,
            backgroundExpirations: counters.backgroundExpirations,
            drainTimeouts: counters.drainTimeouts,
            transitions: counters.transitions,
            lastOutcome: counters.lastOutcome,
            lastCancellationStage: counters.lastCancellationStage,
            lastCancellationLatencyMilliseconds: counters.lastCancellationLatency,
            maxCancellationLatencyMilliseconds: counters.maxCancellationLatency,
            lastFailureTypeName: counters.lastFailureTypeName
        )
    }

    /// Suspend until `predicate` holds for the current snapshot or the caller
    /// is cancelled. Used by tests to observe transitions without sleeping.
    func awaitSnapshot(
        where predicate: @Sendable (OpenMedRuntimeSnapshot) -> Bool
    ) async -> OpenMedRuntimeSnapshot {
        while true {
            let current = snapshot()
            if predicate(current) || Task.isCancelled { return current }
            await waitForChange()
        }
    }

    static func isValidTransition(
        from current: OpenMedRuntimeState,
        to next: OpenMedRuntimeState
    ) -> Bool {
        switch (current, next) {
        case (.released, _): return false
        case (_, .released): return true
        case (.idle, .loading): return true
        case (.loading, .ready), (.loading, .idle): return true
        case (.ready, .busy(1)), (.ready, .idle): return true
        case (.busy(let lhs), .busy(let rhs)): return rhs == lhs + 1 || rhs == lhs - 1
        case (.busy(1), .ready): return true
        case (.busy, .draining): return true
        case (.draining, .idle): return true
        default: return false
        }
    }

    // MARK: Execution

    private static func detectPII(
        model: any OpenMedRuntimeModel,
        text: String,
        confidenceThreshold: Float,
        useSmartMerging: Bool,
        checkpoint: OpenMedInferenceCheckpoint
    ) async throws -> [EntityPrediction] {
        let raw = try await model.predictEntities(in: text, checkpoint: checkpoint)
        let analyzed = OpenMedInferenceCore.applyConfidenceThreshold(raw, confidenceThreshold)
        return try OpenMedInferenceCore.mergePII(
            analyzed,
            text: text,
            useSmartMerging: useSmartMerging,
            profile: model.mergeProfile,
            checkpoint: checkpoint
        )
    }

    private func perform<Output: Sendable>(
        _ body: @escaping @Sendable (any OpenMedRuntimeModel, OpenMedInferenceCheckpoint) async throws -> Output
    ) async throws -> Output {
        let lease: OpenMedInferenceLease
        let result: Result<Output, Error>
        do {
            guard let admitted = try await acquire(lease: true) else {
                throw OpenMedRuntimeError.released
            }
            lease = admitted.lease
            let box = OpenMedModelBox(admitted.model)
            result = await Self.execute(box: box, lease: lease, body: body)
        }
        return try settle(lease, result: result)
    }

    /// Run `body` in a child unstructured task that runtime cancellation can
    /// interrupt, and wait for it to finish so the lease outlives every use of
    /// the model session.
    private nonisolated static func execute<Output: Sendable>(
        box: OpenMedModelBox,
        lease: OpenMedInferenceLease,
        body: @escaping @Sendable (any OpenMedRuntimeModel, OpenMedInferenceCheckpoint) async throws -> Output
    ) async -> Result<Output, Error> {
        let checkpoint = OpenMedInferenceCheckpoint(lease: lease)
        let work = Task<Output, Error> {
            guard let model = box.take() else { throw OpenMedRuntimeError.released }
            return try await body(model, checkpoint)
        }
        lease.attachInterrupt { work.cancel() }
        return await withTaskCancellationHandler {
            await work.result
        } onCancel: {
            lease.cancel(.cancelled)
        }
    }

    private func settle<Output>(
        _ lease: OpenMedInferenceLease,
        result: Result<Output, Error>
    ) throws -> Output {
        leases.remove(lease)
        activeLeaseCount -= 1
        if Task.isCancelled { lease.cancel(.cancelled) }

        let outcome: Result<Output, OpenMedRuntimeError>
        if let cancellation = lease.cancellation {
            recordCancellation(cancellation)
            outcome = .failure(cancellation.reason)
        } else {
            switch result {
            case .success(let value):
                counters.completed += 1
                outcome = .success(value)
            case .failure(let error):
                outcome = .failure(classify(error))
            }
        }
        switch outcome {
        case .success: counters.lastOutcome = "completed"
        case .failure(let error): counters.lastOutcome = error.reasonCode
        }
        leaseEnded()
        return try outcome.get()
    }

    private func classify(_ error: Error) -> OpenMedRuntimeError {
        if let runtimeError = error as? OpenMedRuntimeError {
            if runtimeError == .cancelled { counters.cancelled += 1 } else { counters.failed += 1 }
            return runtimeError
        }
        if error is CancellationError {
            counters.cancelled += 1
            return .cancelled
        }
        counters.failed += 1
        counters.lastFailureTypeName = String(describing: type(of: error))
        return .inferenceFailed
    }

    private func recordCancellation(_ cancellation: OpenMedInferenceLease.Cancellation) {
        counters.cancelled += 1
        counters.lastCancellationStage = cancellation.observedStage
        let now = clock.nowMilliseconds()
        let latency =
            now >= cancellation.requestedAtMilliseconds
            ? now - cancellation.requestedAtMilliseconds : 0
        counters.lastCancellationLatency = latency
        counters.maxCancellationLatency = max(counters.maxCancellationLatency ?? 0, latency)
    }

    // MARK: Admission

    /// Wait until a model session is resident and, when `lease` is true,
    /// return a lease on it.
    private func acquire(
        lease wantsLease: Bool
    ) async throws -> (lease: OpenMedInferenceLease, model: any OpenMedRuntimeModel)? {
        activateSignals()
        let capacity = configuration.maxConcurrentInferences + configuration.maxQueuedRequests
        guard !wantsLease || pendingAdmissions + activeLeaseCount < capacity else {
            throw reject(.capacityExceeded)
        }
        pendingAdmissions += 1
        noteChange()
        defer {
            pendingAdmissions -= 1
            noteChange()
        }

        var awaitedLoad: UInt64?
        while true {
            if Task.isCancelled { throw reject(.cancelled) }
            if let error = admissionError() { throw reject(error) }
            if let awaitedLoad, failedLoadGeneration == awaitedLoad {
                throw reject(.modelUnavailable)
            }
            switch state {
            case .idle:
                startLoad()
                awaitedLoad = loadGeneration
            case .loading:
                awaitedLoad = loadGeneration
            case .ready, .busy:
                guard let model else { break }
                guard wantsLease else { return nil }
                if activeLeaseCount < configuration.maxConcurrentInferences {
                    return (makeLease(), model)
                }
            case .draining:
                // A pressure drain whose hold already lifted: wait for the
                // release to finish, then reload.
                break
            case .released:
                throw reject(.released)
            }
            await waitForChange()
        }
    }

    private func admissionError() -> OpenMedRuntimeError? {
        if state == .released { return .released }
        if let drainIntent, drainIntent.isTerminal { return .draining }
        if backgroundHold { return .backgroundExpired }
        if pressureHold { return .memoryPressure }
        return nil
    }

    private func reject(_ error: OpenMedRuntimeError) -> OpenMedRuntimeError {
        counters.rejected += 1
        counters.lastOutcome = error.reasonCode
        noteChange()
        return error
    }

    private func makeLease() -> OpenMedInferenceLease {
        cancelIdleTimer()
        nextLeaseID &+= 1
        let lease = OpenMedInferenceLease(
            id: nextLeaseID,
            clock: clock,
            probe: configuration.stageProbe
        )
        leases.insert(lease)
        activeLeaseCount += 1
        counters.admitted += 1
        beginBackgroundAssertionIfNeeded()
        transition(to: .busy(activeLeaseCount))
        return lease
    }

    private func leaseEnded() {
        if activeLeaseCount == 0 { endBackgroundAssertion() }
        if drainIntent != nil {
            if activeLeaseCount == 0 { completeDrain() }
        } else if activeLeaseCount > 0 {
            transition(to: .busy(activeLeaseCount))
        } else {
            transition(to: .ready)
            scheduleIdleRelease()
        }
        noteChange()
    }

    // MARK: Loading and release

    private func startLoad() {
        loadGeneration &+= 1
        let generation = loadGeneration
        counters.loads += 1
        let loader = self.loader
        loadTask = Task { [weak self] in
            let box = OpenMedModelBox(nil)
            var failureTypeName: String?
            do {
                box.store(try await loader())
            } catch {
                failureTypeName = String(describing: type(of: error))
            }
            guard let self else {
                box.discard()
                return
            }
            await self.finishLoad(
                generation: generation,
                box: box,
                failureTypeName: failureTypeName
            )
        }
        transition(to: .loading)
    }

    private func finishLoad(
        generation: UInt64,
        box: OpenMedModelBox,
        failureTypeName: String?
    ) {
        guard generation == loadGeneration, state == .loading else {
            box.discard()
            noteChange()
            return
        }
        loadTask = nil
        if let loaded = box.take() {
            model = loaded
            transition(to: .ready)
            scheduleIdleRelease()
        } else {
            failedLoadGeneration = generation
            counters.loadFailures += 1
            counters.lastFailureTypeName = failureTypeName
            transition(to: .idle)
        }
    }

    private func abandonLoad() {
        guard let loadTask else { return }
        loadGeneration &+= 1
        loadTask.cancel()
        self.loadTask = nil
    }

    private func releaseModel() {
        guard let modelType = model.map({ type(of: $0) }) else { return }
        model = nil
        counters.releases += 1
        modelType.releaseUnreferencedCaches()
    }

    private func scheduleIdleRelease() {
        cancelIdleTimer()
        guard let milliseconds = configuration.idleReleaseMilliseconds, state == .ready else {
            return
        }
        let generation = idleGeneration
        let clock = self.clock
        idleTimer = Task { [weak self] in
            do {
                try await clock.sleep(milliseconds: milliseconds)
            } catch {
                return
            }
            await self?.idleReleaseFired(generation: generation)
        }
    }

    private func cancelIdleTimer() {
        idleGeneration &+= 1
        idleTimer?.cancel()
        idleTimer = nil
    }

    private func idleReleaseFired(generation: UInt64) {
        guard generation == idleGeneration, state == .ready else { return }
        idleTimer = nil
        counters.idleReleases += 1
        releaseModel()
        transition(to: .idle)
    }

    // MARK: Draining

    private func beginDrain(
        _ intent: DrainIntent,
        cancelInFlight: Bool,
        timeoutMilliseconds: UInt64?
    ) {
        switch state {
        case .released:
            return
        case .idle:
            if intent.isTerminal { terminate() }
        case .loading:
            abandonLoad()
            if intent.isTerminal { terminate() } else { transition(to: .idle) }
        case .ready:
            cancelIdleTimer()
            releaseModel()
            if intent.isTerminal { terminate() } else { transition(to: .idle) }
        case .busy, .draining:
            drainIntent = max(drainIntent ?? intent, intent)
            cancelIdleTimer()
            transition(to: .draining)
            if cancelInFlight {
                cancelDrainTimer()
                leases.cancelAll(intent.cancellationReason)
            } else if let timeoutMilliseconds {
                scheduleDrainTimeout(timeoutMilliseconds, reason: intent.cancellationReason)
            }
        }
        noteChange()
    }

    private func completeDrain() {
        guard let intent = drainIntent else { return }
        drainIntent = nil
        cancelDrainTimer()
        releaseModel()
        if intent.isTerminal { terminate() } else { transition(to: .idle) }
    }

    private func scheduleDrainTimeout(_ milliseconds: UInt64, reason: OpenMedRuntimeError) {
        guard drainTimer == nil else { return }
        drainGeneration &+= 1
        let generation = drainGeneration
        let clock = self.clock
        drainTimer = Task { [weak self] in
            do {
                try await clock.sleep(milliseconds: milliseconds)
            } catch {
                return
            }
            await self?.drainTimeoutFired(generation: generation, reason: reason)
        }
    }

    private func cancelDrainTimer() {
        drainGeneration &+= 1
        drainTimer?.cancel()
        drainTimer = nil
    }

    private func drainTimeoutFired(generation: UInt64, reason: OpenMedRuntimeError) {
        guard generation == drainGeneration, state == .draining else { return }
        drainTimer = nil
        counters.drainTimeouts += 1
        leases.cancelAll(reason)
        noteChange()
    }

    private func terminate() {
        guard state != .released else { return }
        drainIntent = nil
        abandonLoad()
        cancelIdleTimer()
        cancelDrainTimer()
        releaseModel()
        endBackgroundAssertion()
        for observation in observations { observation.cancel() }
        observations.removeAll()
        transition(to: .released)
    }

    private func waitUntilReleased() async {
        while state != .released, !Task.isCancelled {
            await waitForChange()
        }
    }

    // MARK: Signals

    private func activateSignals() {
        guard !observingSignals, state != .released else { return }
        observingSignals = true
        let leases = self.leases
        let signals = self.signals
        let configuration = self.configuration
        observations.append(
            memoryPressureSource.observe { [weak self] level in
                let sequence = signals.next()
                if configuration.response(for: level) == .cancelInFlight {
                    leases.cancelAll(.memoryPressure)
                }
                Task { await self?.applyMemoryPressure(level, sequence: sequence) }
            }
        )
        observations.append(
            backgroundExecution.observeForeground { [weak self] in
                let sequence = signals.next()
                Task { await self?.applyForeground(sequence: sequence) }
            }
        )
    }

    private func applyMemoryPressure(_ level: OpenMedMemoryPressureLevel, sequence: UInt64) {
        guard sequence > lastPressureSignal else { return }
        lastPressureSignal = sequence
        handleMemoryPressure(level)
    }

    private func applyBackgroundExpiration(sequence: UInt64) {
        guard sequence > lastLifecycleSignal else { return }
        lastLifecycleSignal = sequence
        handleBackgroundExpiration()
    }

    private func applyForeground(sequence: UInt64) {
        guard sequence > lastLifecycleSignal else { return }
        lastLifecycleSignal = sequence
        resumeAfterForeground()
    }

    private func beginBackgroundAssertionIfNeeded() {
        guard backgroundAssertion == nil else { return }
        let leases = self.leases
        let signals = self.signals
        backgroundAssertion = backgroundExecution.beginBackgroundExecution { [weak self] in
            let sequence = signals.next()
            leases.cancelAll(.backgroundExpired)
            Task { await self?.applyBackgroundExpiration(sequence: sequence) }
        }
    }

    private func endBackgroundAssertion() {
        backgroundAssertion?.end()
        backgroundAssertion = nil
    }

    // MARK: State and waiting

    private func transition(to next: OpenMedRuntimeState) {
        guard next != state else { return }
        assert(
            Self.isValidTransition(from: state, to: next),
            "Invalid OpenMedRuntime lifecycle transition"
        )
        state = next
        counters.transitions += 1
        noteChange()
    }

    private func waitForChange() async {
        let id = nextWaiterID
        nextWaiterID &+= 1
        await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                if Task.isCancelled {
                    continuation.resume()
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        } onCancel: {
            Task { await self.resumeWaiter(id) }
        }
    }

    private func resumeWaiter(_ id: UInt64) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume()
    }

    private func noteChange() {
        guard !waiters.isEmpty else { return }
        let current = waiters
        waiters.removeAll()
        for waiter in current { waiter.continuation.resume() }
    }
}

// MARK: - Support

/// Hands a model session to exactly one consumer so no stray reference keeps
/// a released session alive.
final class OpenMedModelBox: @unchecked Sendable {
    private let lock = NSLock()
    private var model: (any OpenMedRuntimeModel)?

    init(_ model: (any OpenMedRuntimeModel)?) { self.model = model }

    func store(_ model: any OpenMedRuntimeModel) {
        lock.lock()
        self.model = model
        lock.unlock()
    }

    func take() -> (any OpenMedRuntimeModel)? {
        lock.lock()
        defer { lock.unlock() }
        let taken = model
        model = nil
        return taken
    }

    /// Drop a session that will never be used and purge its type's caches.
    func discard() {
        let modelType: (any OpenMedRuntimeModel.Type)?
        lock.lock()
        modelType = model.map { type(of: $0) }
        model = nil
        lock.unlock()
        modelType?.releaseUnreferencedCaches()
    }
}

/// Orders asynchronously delivered platform signals.
final class OpenMedSignalSequencer: @unchecked Sendable {
    private let lock = NSLock()
    private var value: UInt64 = 0

    func next() -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        value &+= 1
        return value
    }
}
