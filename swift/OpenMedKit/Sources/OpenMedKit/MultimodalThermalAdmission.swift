import Foundation

/// The four thermal states reported by Apple platforms. Unknown readings are nil.
public enum MultimodalThermalState: Int, Sendable { case nominal, fair, serious, critical }

public protocol ThermalObservationSubscription: Sendable { func cancel() }

/// A current read and change notifications. Reads must be synchronous and side-effect free;
/// invoke callbacks after releasing any source lock. The session re-reads on admission and return.
public protocol MultimodalThermalSource: Sendable {
    func current() -> MultimodalThermalState?
    func observe(_ change: @escaping @Sendable () -> Void) -> any ThermalObservationSubscription
}

private final class ThermalNotificationSubscription: ThermalObservationSubscription, @unchecked Sendable {
    private let lock = NSLock()
    private var token: NSObjectProtocol?
    init(_ token: NSObjectProtocol) { self.token = token }
    func cancel() {
        lock.lock()
        let previous = token
        token = nil
        lock.unlock()
        if let previous { NotificationCenter.default.removeObserver(previous) }
    }
    deinit { cancel() }
}

/// Native source: no device identifiers, polling service or network access.
public struct AppleSystemThermalSource: MultimodalThermalSource {
    public init() {}
    public func current() -> MultimodalThermalState? {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return .nominal
        case .fair: return .fair
        case .serious: return .serious
        case .critical: return .critical
        @unknown default: return nil
        }
    }
    public func observe(_ change: @escaping @Sendable () -> Void) -> any ThermalObservationSubscription {
        ThermalNotificationSubscription(NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification, object: nil, queue: nil) { _ in change() })
    }
}

public enum MultimodalAdmissionMode: Int, Sendable { case normal, reducedCadence, reducedWindow, paused }

/// Same reason vocabulary as the Python thermal/latency gate.
public enum MultimodalAdmissionReason: String, Sendable {
    case ready, latency, resources, closed
    case queueExpired = "queue_expired"
    case outputExpired = "output_expired"
    case observationUnavailable = "observation_unavailable"
    case clockInvalid = "clock_invalid"
    case providerRejected = "provider_rejected"
    case operationLimit = "operation_limit"
}

/// Explicit, workload-specific budgets. No model or device qualification is inferred.
public struct MultimodalThermalBudget: Sendable {
    public let maxQueueMs: UInt64
    public let maxLatencyMs: UInt64
    public let maxOutputAgeMs: UInt64
    public let recoveryMs: UInt64
    public let historyAgeMs: UInt64
    public let historySize: Int
    public let reducedCadenceMs: UInt64
    public let normalWindowMs: UInt64
    public let reducedWindowMs: UInt64
    public let maxOperations: Int

    public init(maxQueueMs: UInt64, maxLatencyMs: UInt64, maxOutputAgeMs: UInt64, recoveryMs: UInt64, historyAgeMs: UInt64, historySize: Int = 20, reducedCadenceMs: UInt64, normalWindowMs: UInt64, reducedWindowMs: UInt64, maxOperations: Int = 100_000) throws {
        guard [maxQueueMs, maxLatencyMs, maxOutputAgeMs, recoveryMs, historyAgeMs, reducedCadenceMs, normalWindowMs, reducedWindowMs].allSatisfy({ (1...3_600_000).contains($0) }),
            maxOutputAgeMs >= maxLatencyMs, reducedWindowMs < normalWindowMs,
            (1...1024).contains(historySize), (1...1_000_000).contains(maxOperations)
        else { throw MultimodalThermalError.invalidBudget }
        self.maxQueueMs = maxQueueMs
        self.maxLatencyMs = maxLatencyMs
        self.maxOutputAgeMs = maxOutputAgeMs
        self.recoveryMs = recoveryMs
        self.historyAgeMs = historyAgeMs
        self.historySize = historySize
        self.reducedCadenceMs = reducedCadenceMs
        self.normalWindowMs = normalWindowMs
        self.reducedWindowMs = reducedWindowMs
        self.maxOperations = maxOperations
    }
}

public enum MultimodalThermalError: Error { case invalidBudget }

public struct MultimodalWorkProfile: Sendable {
    public let mode: MultimodalAdmissionMode
    public let minimumStartIntervalMs: UInt64
    public let maximumWindowMs: UInt64
}

/// A trusted local operation. An operation declaring network use is never called.
/// The gate never shortens input, substitutes a model or retries protected work.
public protocol LocalMultimodalWork: Sendable {
    associatedtype Output: Sendable
    var requiresNetwork: Bool { get }
    func run(profile: MultimodalWorkProfile) async throws -> Output
}

public struct MultimodalAdmissionOutcome<Value: Sendable>: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let value: Value?
    public let reason: MultimodalAdmissionReason
    public let mode: MultimodalAdmissionMode
    public var description: String { "MultimodalAdmissionOutcome(reason=\(reason.rawValue), value=<private>)" }
    public var debugDescription: String { description }
}

/// Aggregate evidence only. No patient content, source references, device names or wall clock.
public struct MultimodalThermalEvidence: Sendable {
    public let thermal: MultimodalThermalState?
    public let mode: MultimodalAdmissionMode
    public let reason: MultimodalAdmissionReason
    public let attempts: Int
    public let completed: Int
    public let withheld: Int
    public let transitions: UInt64
    public let queueAgeMs: UInt64?
    public let latencyP95Ms: UInt64?
    public let latencySamples: Int
}

/// Local admission and result-release gate. Hotter readings take effect immediately;
/// a lower profile must remain stable for the declared recovery interval.
public final class MultimodalThermalSession: @unchecked Sendable {
    private typealias Sample = (time: UInt64, duration: UInt64)
    private struct Ticket {
        let start: UInt64
        let epoch: UInt64
        let profile: MultimodalWorkProfile
    }
    private enum Admission {
        case run(Ticket)
        case hold(MultimodalAdmissionReason)
        case terminal(MultimodalAdmissionReason)
    }
    private let lock = NSLock()
    private let source: any MultimodalThermalSource
    private let budget: MultimodalThermalBudget
    private let clock: @Sendable () -> UInt64
    private var subscription: (any ThermalObservationSubscription)?
    private var mode: MultimodalAdmissionMode = .paused
    private var thermal: MultimodalThermalState?
    private var observed = false
    private var cooling: (mode: MultimodalAdmissionMode, since: UInt64)?
    private var history = [[Sample]](repeating: [], count: 3)
    private var lastTime: UInt64?
    private var lastStart: UInt64?
    private var lastQueueAge: UInt64?
    private var lastMeasuredMode: MultimodalAdmissionMode?
    private var lastReason: MultimodalAdmissionReason = .observationUnavailable
    private var pressureReason: MultimodalAdmissionReason = .resources
    private var terminal: MultimodalAdmissionReason?
    private var active = false
    private var epoch: UInt64 = 0
    private var transitions: UInt64 = 0
    private var attempts = 0, completed = 0, withheld = 0

    public init(budget: MultimodalThermalBudget, source: any MultimodalThermalSource = AppleSystemThermalSource(), monotonicMilliseconds: @escaping @Sendable () -> UInt64 = { DispatchTime.now().uptimeNanoseconds / 1_000_000 }) {
        self.budget = budget
        self.source = source
        clock = monotonicMilliseconds
        subscription = source.observe { [weak self] in self?.observeChange() }
        observeChange()
    }
    deinit { subscription?.cancel() }
    private func locked<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
    private func observeChange() { locked { _ = refresh() } }

    private func p95(_ index: Int) -> UInt64? {
        let values = history[index].map(\.duration).sorted()
        guard !values.isEmpty else { return nil }
        return values[(95 * values.count + 99) / 100 - 1]
    }

    /// Called only under lock. Unknown readings immediately stop work.
    private func refresh() -> UInt64 {
        let now = clock()
        guard terminal == nil else { return now }
        if let lastTime, now < lastTime {
            terminal = .clockInvalid
            mode = .paused
            return now
        }
        lastTime = now
        for index in history.indices { history[index].removeAll { now - $0.time > budget.historyAgeMs } }
        thermal = source.current()
        var wanted: MultimodalAdmissionMode
        switch thermal {
        case .nominal: wanted = .normal
        case .fair: wanted = .reducedCadence
        case .serious: wanted = .reducedWindow
        case .critical, nil: wanted = .paused
        }
        pressureReason = thermal == nil ? .observationUnavailable : .resources
        while wanted != .paused, let measured = p95(wanted.rawValue), measured > budget.maxLatencyMs {
            wanted = MultimodalAdmissionMode(rawValue: wanted.rawValue + 1)!
            pressureReason = .latency
        }
        if !observed || wanted.rawValue >= mode.rawValue {
            cooling = nil
            if wanted != mode {
                transitions &+= 1
                if wanted.rawValue > mode.rawValue { epoch &+= 1 }
            }
            mode = wanted
        } else {
            if cooling?.mode != wanted { cooling = (wanted, now) }
            if let cooling, now - cooling.since >= budget.recoveryMs {
                mode = wanted
                self.cooling = nil
                transitions &+= 1
            }
        }
        observed = true
        return now
    }

    public func evidence() -> MultimodalThermalEvidence {
        locked {
            _ = refresh()
            let index = lastMeasuredMode?.rawValue
            return MultimodalThermalEvidence(thermal: thermal, mode: mode, reason: terminal ?? (mode == .paused ? pressureReason : lastReason), attempts: attempts, completed: completed, withheld: withheld, transitions: transitions, queueAgeMs: lastQueueAge, latencyP95Ms: index.flatMap(p95), latencySamples: history.reduce(0) { $0 + $1.count })
        }
    }

    public func close() {
        let old = locked { () -> (any ThermalObservationSubscription)? in
            if terminal == nil { terminal = .closed }
            mode = .paused
            let old = subscription
            subscription = nil
            return old
        }
        old?.cancel()
    }

    private func begin(queuedAt: UInt64, requestedWindowMs: UInt64?, requiresNetwork: Bool) -> Admission {
        let now = refresh()
        if let terminal { return .terminal(terminal) }
        guard attempts < budget.maxOperations else {
            if !active {
                terminal = .operationLimit
                mode = .paused
            }
            return .terminal(.operationLimit)
        }
        attempts += 1
        guard !Task.isCancelled else { return .hold(.closed) }
        guard !requiresNetwork else { return .hold(.providerRejected) }
        guard now >= queuedAt else {
            lastQueueAge = nil
            return .hold(.queueExpired)
        }
        let age = now - queuedAt
        lastQueueAge = age
        guard age <= min(budget.maxQueueMs, budget.maxOutputAgeMs) else { return .hold(.queueExpired) }
        guard mode != .paused else { return .hold(pressureReason) }
        guard !active else { return .hold(.resources) }
        let profile = MultimodalWorkProfile(mode: mode, minimumStartIntervalMs: mode == .normal ? 0 : budget.reducedCadenceMs, maximumWindowMs: mode == .reducedWindow ? budget.reducedWindowMs : budget.normalWindowMs)
        if let requestedWindowMs, requestedWindowMs == 0 || requestedWindowMs > profile.maximumWindowMs { return .hold(.resources) }
        if let lastStart, now - lastStart < profile.minimumStartIntervalMs { return .hold(.resources) }
        active = true
        lastStart = now
        return .run(Ticket(start: now, epoch: epoch, profile: profile))
    }

    private func outcome<T: Sendable>(_ value: T?, _ reason: MultimodalAdmissionReason, mode: MultimodalAdmissionMode) -> MultimodalAdmissionOutcome<T> {
        lastReason = reason
        if reason == .ready { completed += 1 } else if withheld < budget.maxOperations { withheld += 1 }
        return MultimodalAdmissionOutcome(value: value, reason: reason, mode: mode)
    }

    /// Checks queue age and policy before dispatch and rechecks clock, latency,
    /// thermal changes and output age before releasing a result. No implicit fallback.
    public func run<Work: LocalMultimodalWork>(_ work: Work, queuedAtMilliseconds: UInt64, requestedWindowMs: UInt64? = nil) async -> MultimodalAdmissionOutcome<Work.Output> {
        let admission = locked { begin(queuedAt: queuedAtMilliseconds, requestedWindowMs: requestedWindowMs, requiresNetwork: work.requiresNetwork) }
        switch admission {
        case .terminal(let reason): return locked { MultimodalAdmissionOutcome(value: nil, reason: reason, mode: mode) }
        case .hold(let reason): return locked { outcome(nil, reason, mode: mode) }
        case .run(let ticket):
            let result: Result<Work.Output, Error>
            do { result = .success(try await work.run(profile: ticket.profile)) } catch { result = .failure(error) }
            return locked {
                defer { active = false }
                let now = refresh()
                if let terminal { return outcome(nil, terminal, mode: mode) }
                let duration = now - ticket.start
                let index = ticket.profile.mode.rawValue
                if history[index].count >= budget.historySize { history[index].removeFirst() }
                history[index].append((now, duration))
                lastMeasuredMode = ticket.profile.mode
                let releasedAt = refresh()
                if let terminal { return outcome(nil, terminal, mode: mode) }
                guard !Task.isCancelled else { return outcome(nil, .closed, mode: mode) }
                guard case .success(let value) = result else { return outcome(nil, .providerRejected, mode: mode) }
                guard releasedAt - queuedAtMilliseconds <= budget.maxOutputAgeMs else { return outcome(nil, .outputExpired, mode: mode) }
                guard duration <= budget.maxLatencyMs, (p95(index) ?? 0) <= budget.maxLatencyMs else { return outcome(nil, .latency, mode: mode) }
                guard thermal != nil else { return outcome(nil, .observationUnavailable, mode: mode) }
                guard mode != .paused, epoch == ticket.epoch else { return outcome(nil, .resources, mode: mode) }
                return outcome(value, .ready, mode: ticket.profile.mode)
            }
        }
    }
}
