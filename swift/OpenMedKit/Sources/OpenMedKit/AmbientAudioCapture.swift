import Foundation

public enum CapturePermission: String, Sendable {
    case undetermined, denied, granted, unavailable
}

public enum CaptureInterruption: String, Sendable {
    case interruption, routeChanged, background, sourceFailure
}

public enum AmbientAudioCaptureError: String, Error, Sendable {
    case invalidBuffer, permissionRequired, invalidTransition, sourceFailed
}

/// One bounded mono Float32 allocation. Disposal overwrites owned memory only.
/// Sources transfer this object to capture; neither side may retain sample pointers.
public final class CapturePCMBuffer: @unchecked Sendable, CustomStringConvertible {
    public static let maximumFrames = 16_384
    public let sampleRateHz: Int
    public let startSample: Int64
    public let frameCount: Int
    private let lock = NSRecursiveLock()
    private let storage: UnsafeMutablePointer<Float>
    private var cleared = false

    public var description: String { "CapturePCMBuffer(samples=<private>)" }

    public init(samples: UnsafeBufferPointer<Float>, sampleRateHz: Int, startSample: Int64) throws {
        guard (1...Self.maximumFrames).contains(samples.count), (8000...192_000).contains(sampleRateHz),
            startSample >= 0, startSample <= Int64.max - Int64(samples.count), samples.allSatisfy(\.isFinite)
        else { throw AmbientAudioCaptureError.invalidBuffer }
        self.sampleRateHz = sampleRateHz
        self.startSample = startSample
        frameCount = samples.count
        storage = .allocate(capacity: samples.count)
        storage.initialize(from: samples.baseAddress!, count: samples.count)
    }

    deinit {
        clear()
        storage.deinitialize(count: frameCount)
        storage.deallocate()
    }

    /// Idempotent overwrite; no claim about source, OS, or consumer-owned copies.
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        storage.update(repeating: 0, count: frameCount)
        cleared = true
    }

    public var isCleared: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cleared && UnsafeBufferPointer(start: storage, count: frameCount).allSatisfy { $0 == 0 }
    }

    fileprivate func borrow(_ body: (UnsafeBufferPointer<Float>) throws -> Void) throws {
        lock.lock()
        defer { lock.unlock() }
        guard !cleared else { throw AmbientAudioCaptureError.invalidBuffer }
        try body(UnsafeBufferPointer(start: storage, count: frameCount))
    }
}

/// Start must not request permission. Stop fences callbacks and clears queued owned buffers.
/// A late callback is allowed only with an owned buffer that the receiver can dispose.
public protocol AmbientAudioSource: AnyObject, Sendable {
    var permission: CapturePermission { get }
    func requestPermission(_ completion: @escaping @Sendable (CapturePermission) -> Void)
    func start(
        receive: @escaping @Sendable (CapturePCMBuffer) -> Void,
        interrupted: @escaping @Sendable (CaptureInterruption) -> Void
    ) throws
    /// Return whether all source-owned pending buffers were cleared and capture stopped.
    func stop() -> Bool
}

/// Synchronous local borrowing only. Do not retain pointers, copy/export samples, or
/// call capture lifecycle methods from this callback. Model state is separately owned.
public protocol AmbientPCMConsumer: Sendable {
    func consume(
        _ samples: UnsafeBufferPointer<Float>, sampleRateHz: Int,
        startSample: Int64, captureEpoch: UInt64
    ) throws
}

/// Content-free accounting. Session deletion and producer cleanup are distinct claims.
public struct AmbientAudioCaptureSnapshot: Sendable {
    public let session: AmbientSessionSnapshot
    public let permission: CapturePermission
    public let running: Bool
    public let captureEpoch: UInt64
    public let receivedBuffers: UInt64
    public let clearedBuffers: UInt64
    public let admittedBuffers: UInt64
    public let sourceCleanupComplete: Bool
    public let reason: String
    public var ownedBuffers: UInt64 { receivedBuffers - clearedBuffers }
    public var cleanupComplete: Bool { !running && ownedBuffers == 0 && sourceCleanupComplete }
}

/// Owns its consent session and serializes consent boundaries with local consumption.
/// Capture has no raw-audio retention/export API; only the current bounded buffer exists.
public final class AmbientAudioCapture: @unchecked Sendable, CustomStringConvertible {
    private let lock = NSRecursiveLock()
    private let accounting = NSLock()
    private let session: AmbientSession
    private let source: any AmbientAudioSource
    private let consumer: any AmbientPCMConsumer
    private let maximumBytes: Int
    private var running = false
    private var epoch: UInt64 = 0
    private var received: UInt64 = 0
    private var cleared: UInt64 = 0
    private var admitted: UInt64 = 0
    private var sourceCleanup = true
    private var closedHandled = false
    private var reason = "none"
    private var timer: DispatchWorkItem?

    public var description: String { "AmbientAudioCapture(samples=<private>)" }

    /// Disabling the timer requires a reliable host scheduler calling `expire()`.
    public init(
        sessionRef: String, participants: [String], policy: AmbientSessionPolicy,
        source: any AmbientAudioSource, consumer: any AmbientPCMConsumer,
        clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime },
        automaticExpiry: Bool = true
    ) throws {
        session = try AmbientSession(
            sessionRef: sessionRef, participants: participants, policy: policy,
            clock: clock, automaticExpiry: false)
        self.source = source
        self.consumer = consumer
        maximumBytes = policy.maxBytes
        if automaticExpiry { scheduleExpiry() }
    }

    deinit { delete() }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func scheduleExpiry() {
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.locked {
                if !self.expire() { self.scheduleExpiry() }
            }
        }
        timer = work
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 0.05, execute: work)
    }

    private func halt(_ code: String) {
        running = false
        epoch += 1
        sourceCleanup = source.stop() && sourceCleanup
        reason = code
    }

    /// Permission grants never record consent and never start capture.
    public func requestPermission(_ completion: @escaping @Sendable (CapturePermission) -> Void) {
        source.requestPermission(completion)
    }

    public func consent(_ participantRef: String) throws {
        try locked {
            defer { _ = expire() }
            try session.consent(participantRef)
        }
    }

    /// Explicit start from ready, or explicit resume from paused/interrupted.
    public func start(resuming: Bool = false) throws {
        try locked {
            guard !running, sourceCleanup else { throw AmbientAudioCaptureError.invalidTransition }
            guard source.permission == .granted else {
                reason = "permission_required"
                throw AmbientAudioCaptureError.permissionRequired
            }
            try session.transition(resuming ? "resume" : "start")
            epoch += 1
            let admittedEpoch = epoch
            do {
                try source.start(
                    receive: { [weak self] buffer in
                        guard let self else {
                            buffer.clear()
                            return
                        }
                        self.receive(buffer, epoch: admittedEpoch)
                    },
                    interrupted: { [weak self] event in self?.interrupt(event, epoch: admittedEpoch) })
                // Synchronous startup events may already have revoked this attempt.
                guard epoch == admittedEpoch, !session.expire(), source.permission == .granted else {
                    throw AmbientAudioCaptureError.sourceFailed
                }
                running = true
                reason = "none"
            } catch {
                halt("source_failed")
                try? session.transition("interrupt")
                throw AmbientAudioCaptureError.sourceFailed
            }
        }
    }

    private func receive(_ buffer: CapturePCMBuffer, epoch expected: UInt64) {
        accounting.lock()
        received += 1
        accounting.unlock()
        var ownsGate = false
        defer {
            buffer.clear()
            accounting.lock()
            cleared += 1
            accounting.unlock()
            if ownsGate { lock.unlock() }
        }
        // Never block an audio producer behind stop(), which may wait for its tap.
        guard lock.try() else { return }
        ownsGate = true
        guard running, epoch == expected else { return }
        guard !expire() else { return }
        guard source.permission == .granted else {
            interrupt(.interruption, epoch: expected)
            reason = "permission_required"
            return
        }
        guard buffer.frameCount * MemoryLayout<Float>.size <= maximumBytes else {
            halt("buffer_limit")
            session.withdraw()
            closedHandled = true
            return
        }
        do {
            try buffer.borrow {
                try consumer.consume(
                    $0, sampleRateHz: buffer.sampleRateHz,
                    startSample: buffer.startSample, captureEpoch: expected)
            }
            admitted += 1
            _ = expire()
        } catch {
            halt("consumer_failed")
            session.withdraw()
            closedHandled = true
        }
    }

    public func pause() throws {
        try locked {
            defer { _ = expire() }
            try session.transition("pause")
            halt("paused")
        }
    }

    /// Host lifecycle adapters must forward suspension before leaving the foreground.
    public func interrupt(_ event: CaptureInterruption) {
        locked { interrupt(event, epoch: epoch) }
    }

    private func interrupt(_ event: CaptureInterruption, epoch expected: UInt64) {
        locked {
            guard epoch == expected else { return }
            halt(event.rawValue)
            try? session.transition("interrupt")
        }
    }

    public func changeParticipants(_ participants: [String]) throws {
        try locked {
            halt("participants_changed")
            do { try session.changeParticipants(participants) } catch {
                try? session.transition("interrupt")
                throw error
            }
        }
    }

    public func stop() throws {
        try locked {
            defer { _ = expire() }
            try session.transition("stop")
            halt("stopped")
        }
    }

    public func withdraw() {
        locked {
            session.withdraw()
            halt("consent_withdrawn")
            closedHandled = true
            timer?.cancel()
            timer = nil
        }
    }

    public func delete() {
        locked {
            session.delete()
            halt("deleted")
            closedHandled = true
            timer?.cancel()
            timer = nil
        }
    }

    /// Enforces the original session deadline before admitting any new buffer.
    @discardableResult
    public func expire() -> Bool {
        locked {
            guard session.expire() else { return false }
            if !closedHandled {
                halt("session_closed")
                closedHandled = true
            }
            timer?.cancel()
            timer = nil
            return true
        }
    }

    public func snapshot() -> AmbientAudioCaptureSnapshot {
        locked {
            _ = expire()
            accounting.lock()
            defer { accounting.unlock() }
            return AmbientAudioCaptureSnapshot(
                session: session.snapshot(), permission: source.permission, running: running,
                captureEpoch: epoch, receivedBuffers: received, clearedBuffers: cleared,
                admittedBuffers: admitted, sourceCleanupComplete: sourceCleanup, reason: reason)
        }
    }
}
