import Dispatch
import Foundation

#if canImport(UIKit) && !os(watchOS)
    import UIKit
#endif

// MARK: - Observation

/// A cancellable registration with a platform signal source.
///
/// `cancel()` must be idempotent and safe to call from any thread.
public protocol OpenMedRuntimeObservation: Sendable {
    /// Stop delivering events to the registered handler.
    func cancel()
}

/// An observation that has nothing to cancel.
public struct OpenMedInactiveObservation: OpenMedRuntimeObservation {
    /// Create an inactive observation.
    public init() {}

    /// Does nothing.
    public func cancel() {}
}

// MARK: - Clock

/// Monotonic time used by `OpenMedRuntime` for idle release, drain timeouts and
/// value-free cancellation-latency evidence.
///
/// Implementations must be monotonic and must never consult wall-clock time.
public protocol OpenMedRuntimeClock: Sendable {
    /// Milliseconds since an arbitrary, fixed, monotonic origin.
    func nowMilliseconds() -> UInt64

    /// Suspend for at least `milliseconds`.
    ///
    /// - Throws: `CancellationError` when the calling task is cancelled.
    func sleep(milliseconds: UInt64) async throws
}

/// The system monotonic clock.
public struct OpenMedSystemClock: OpenMedRuntimeClock {
    /// Create the system clock.
    public init() {}

    /// Monotonic uptime in milliseconds.
    public func nowMilliseconds() -> UInt64 {
        DispatchTime.now().uptimeNanoseconds / 1_000_000
    }

    /// Suspend the calling task with `Task.sleep`.
    public func sleep(milliseconds: UInt64) async throws {
        let capped = min(milliseconds, UInt64.max / 1_000_000)
        try await Task.sleep(nanoseconds: capped * 1_000_000)
    }
}

// MARK: - Memory pressure

/// Platform memory-pressure levels, ordered from least to most severe.
public enum OpenMedMemoryPressureLevel: Int, Sendable, Comparable, CaseIterable {
    case normal
    case warning
    case critical

    /// Stable, value-free reason code.
    public var reasonCode: String {
        switch self {
        case .normal: return "normal"
        case .warning: return "warning"
        case .critical: return "critical"
        }
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

/// A source of memory-pressure transitions.
///
/// Handlers may be invoked on any thread and must not block.
public protocol OpenMedMemoryPressureSource: Sendable {
    /// Start delivering memory-pressure levels to `handler`.
    func observe(
        _ handler: @escaping @Sendable (OpenMedMemoryPressureLevel) -> Void
    ) -> any OpenMedRuntimeObservation
}

/// Memory-pressure source backed by `DispatchSource.makeMemoryPressureSource`.
///
/// Available on every Apple platform OpenMedKit supports. It reads only the
/// pressure level, never process or device identifiers.
public struct OpenMedDispatchMemoryPressureSource: OpenMedMemoryPressureSource {
    /// Create a dispatch-backed memory-pressure source.
    public init() {}

    /// Begin observing normal, warning and critical transitions.
    public func observe(
        _ handler: @escaping @Sendable (OpenMedMemoryPressureLevel) -> Void
    ) -> any OpenMedRuntimeObservation {
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical],
            queue: DispatchQueue.global(qos: .utility)
        )
        source.setEventHandler { [weak source] in
            guard let event = source?.data else { return }
            if event.contains(.critical) {
                handler(.critical)
            } else if event.contains(.warning) {
                handler(.warning)
            } else {
                handler(.normal)
            }
        }
        source.activate()
        return OpenMedDispatchSourceObservation(source)
    }
}

private final class OpenMedDispatchSourceObservation: OpenMedRuntimeObservation, @unchecked Sendable {
    private let lock = NSLock()
    private var source: (any DispatchSourceProtocol)?

    init(_ source: any DispatchSourceProtocol) { self.source = source }

    func cancel() {
        lock.lock()
        let current = source
        source = nil
        lock.unlock()
        current?.cancel()
    }

    deinit { cancel() }
}

// MARK: - Background execution

/// One request for extra background execution time.
public protocol OpenMedBackgroundExecutionAssertion: Sendable {
    /// Release the background-time request. Must be idempotent and thread-safe.
    func end()
}

/// Requests background execution time while inference is in flight.
///
/// The runtime begins one assertion when it becomes busy and ends it when the
/// last in-flight inference settles. When the platform reports expiration, the
/// runtime cancels in-flight work, suppresses its output, releases the model
/// session and refuses admission until the application returns to the
/// foreground.
public protocol OpenMedBackgroundExecutionProvider: Sendable {
    /// Begin a background-time request.
    ///
    /// - Parameter onExpiration: Called at most once, on any thread, when the
    ///   platform is about to suspend the application. It must return quickly.
    func beginBackgroundExecution(
        onExpiration: @escaping @Sendable () -> Void
    ) -> any OpenMedBackgroundExecutionAssertion

    /// Observe the application returning to the foreground.
    func observeForeground(
        _ handler: @escaping @Sendable () -> Void
    ) -> any OpenMedRuntimeObservation
}

extension OpenMedBackgroundExecutionProvider {
    /// Default: no foreground notifications. Call
    /// `OpenMedRuntime.resumeAfterForeground()` from the application instead.
    public func observeForeground(
        _ handler: @escaping @Sendable () -> Void
    ) -> any OpenMedRuntimeObservation {
        OpenMedInactiveObservation()
    }
}

/// A provider that never requests background time and never expires.
///
/// This is the default on every platform, including macOS and watchOS, and in
/// application extensions where `UIApplication` is unavailable.
public struct OpenMedNoBackgroundExecution: OpenMedBackgroundExecutionProvider {
    /// Create the no-op provider.
    public init() {}

    /// Return an assertion that does nothing.
    public func beginBackgroundExecution(
        onExpiration: @escaping @Sendable () -> Void
    ) -> any OpenMedBackgroundExecutionAssertion {
        OpenMedInactiveBackgroundAssertion()
    }
}

private struct OpenMedInactiveBackgroundAssertion: OpenMedBackgroundExecutionAssertion {
    func end() {}
}

#if canImport(UIKit) && !os(watchOS)
    /// Background execution backed by `UIApplication` background tasks.
    ///
    /// Pass the host application explicitly, for example
    /// `OpenMedUIApplicationBackgroundExecution(application: .shared)`, so that
    /// OpenMedKit never touches `UIApplication.shared` from an app extension.
    /// The task name is a fixed, value-free constant.
    public struct OpenMedUIApplicationBackgroundExecution: OpenMedBackgroundExecutionProvider {
        private let application: UIApplication

        /// Create a provider for the given application.
        public init(application: UIApplication) {
            self.application = application
        }

        /// Begin a `UIApplication` background task on the main queue.
        public func beginBackgroundExecution(
            onExpiration: @escaping @Sendable () -> Void
        ) -> any OpenMedBackgroundExecutionAssertion {
            let assertion = OpenMedUIApplicationBackgroundAssertion(application: application)
            assertion.begin(onExpiration: onExpiration)
            return assertion
        }

        /// Observe `UIApplication.willEnterForegroundNotification`.
        public func observeForeground(
            _ handler: @escaping @Sendable () -> Void
        ) -> any OpenMedRuntimeObservation {
            let token = NotificationCenter.default.addObserver(
                forName: UIApplication.willEnterForegroundNotification,
                object: nil,
                queue: nil
            ) { _ in handler() }
            return OpenMedNotificationObservation(token)
        }
    }

    private final class OpenMedUIApplicationBackgroundAssertion: OpenMedBackgroundExecutionAssertion, @unchecked Sendable {
        private static let taskName = "OpenMedKit.inference"
        private let application: UIApplication
        private let lock = NSLock()
        private var identifier: UIBackgroundTaskIdentifier = .invalid
        private var ended = false

        init(application: UIApplication) {
            self.application = application
        }

        func begin(onExpiration: @escaping @Sendable () -> Void) {
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self.beginOnMain(onExpiration: onExpiration) }
            }
        }

        func end() {
            lock.lock()
            ended = true
            lock.unlock()
            DispatchQueue.main.async {
                MainActor.assumeIsolated { self.finishOnMain() }
            }
        }

        @MainActor
        private func beginOnMain(onExpiration: @escaping @Sendable () -> Void) {
            lock.lock()
            let alreadyEnded = ended
            lock.unlock()
            guard !alreadyEnded else { return }
            let identifier = application.beginBackgroundTask(withName: Self.taskName) {
                onExpiration()
                MainActor.assumeIsolated { self.finishOnMain() }
            }
            lock.lock()
            self.identifier = identifier
            lock.unlock()
        }

        @MainActor
        private func finishOnMain() {
            lock.lock()
            let current = identifier
            identifier = .invalid
            ended = true
            lock.unlock()
            if current != .invalid {
                application.endBackgroundTask(current)
            }
        }
    }

    private final class OpenMedNotificationObservation: OpenMedRuntimeObservation, @unchecked Sendable {
        private let lock = NSLock()
        private var token: NSObjectProtocol?

        init(_ token: NSObjectProtocol) { self.token = token }

        func cancel() {
            lock.lock()
            let current = token
            token = nil
            lock.unlock()
            if let current { NotificationCenter.default.removeObserver(current) }
        }

        deinit { cancel() }
    }
#endif
