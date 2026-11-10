import Foundation

/// Controlled refusals that never contain participant names or retained content.
public enum AmbientSessionError: String, Error {
    case invalidSessionContract = "invalid_session_contract"
    case sessionClosed = "session_closed"
    case operationLimit = "operation_limit"
    case invalidTransition = "invalid_transition"
    case unknownParticipant = "unknown_participant"
    case consentRequired = "consent_required"
    case captureNotAdmitted = "capture_not_admitted"
    case duplicateReference = "duplicate_reference"
    case retentionLimit = "retention_limit"
    case staleReference = "stale_reference"
    case readerFailed = "reader_failed"
    case externalLimit = "external_limit"
    case exportNotPermitted = "export_not_permitted"
    case exportLimit = "export_limit"
}

/// Visible state; an interruption never resumes capture automatically.
public enum AmbientCaptureState: String, Codable, Sendable {
    case awaitingConsent = "awaiting_consent"
    case ready, capturing, paused, interrupted, stopped, withdrawn, expired, deleted
    case deletionIncomplete = "deletion_incomplete"

    fileprivate var terminal: Bool {
        [.withdrawn, .expired, .deleted, .deletionIncomplete].contains(self)
    }

    fileprivate var readable: Bool { [.capturing, .paused, .stopped].contains(self) }
}

/// Owned content categories share the same consent and destruction boundary.
public enum AmbientArtifactKind: String, Codable, Sendable {
    case audio, transcript, cache, draft
}

private func ambientHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count
        && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

private func ambientParticipants(_ values: [String]) throws {
    guard (1...16).contains(values.count), Set(values).count == values.count,
        values.allSatisfy({ ambientHex($0, count: 32) })
    else { throw AmbientSessionError.invalidSessionContract }
}

/// Approved policy identity and non-renewable local retention limits.
public struct AmbientSessionPolicy: Sendable {
    public let revision: String
    public let maxLifetimeMs: Int
    public let maxArtifacts: Int
    public let maxBytes: Int
    public let maxOperations: Int
    public let allowReviewExport: Bool

    /// `revision` is the digest of the host-approved policy, not a legal determination.
    public init(
        revision: String, maxLifetimeMs: Int = 300_000, maxArtifacts: Int = 64,
        maxBytes: Int = 4 * 1024 * 1024, maxOperations: Int = 4096,
        allowReviewExport: Bool = false
    ) throws {
        guard ambientHex(revision, count: 64), (1...3_600_000).contains(maxLifetimeMs),
            (1...256).contains(maxArtifacts), (1...64 * 1024 * 1024).contains(maxBytes),
            (1...16_384).contains(maxOperations)
        else { throw AmbientSessionError.invalidSessionContract }
        self.revision = revision
        self.maxLifetimeMs = maxLifetimeMs
        self.maxArtifacts = maxArtifacts
        self.maxBytes = maxBytes
        self.maxOperations = maxOperations
        self.allowReviewExport = allowReviewExport
    }
}

/// An original-object capability, invalidated when its cohort or lifetime ends.
public final class AmbientReference: Sendable {
    public let artifactRef: String
    public let kind: AmbientArtifactKind
    public let generation: Int

    fileprivate init(_ artifactRef: String, _ kind: AmbientArtifactKind, _ generation: Int) {
        self.artifactRef = artifactRef
        self.kind = kind
        self.generation = generation
    }
}

/// Explicit review copy outside session ownership; never a finalized clinical note.
public struct AmbientReviewCopy: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public let reference: AmbientReference
    public let content: Data
    public let draftOnly = true
    public let requiresReview = true
    public var description: String { "AmbientReviewCopy(content=<private>)" }
    public var debugDescription: String { description }
}

/// Bounded state without content or participant names. It cannot restore consent.
public struct AmbientSessionSnapshot: Codable, Equatable, Sendable {
    public let schema: String
    public let sessionRef: String
    public let policyRevision: String
    public let state: AmbientCaptureState
    public let generation: Int
    public let participants: Int
    public let consents: Int
    public let ownedArtifacts: Int
    public let ownedBytes: Int
    public let unaccountedExternalCopies: Int
    public let deletionVerified: Bool
    public let reason: String
    public let draftOnly: Bool
    public let requiresReview: Bool

    enum CodingKeys: String, CodingKey {
        case schema, state, generation, participants, consents, reason
        case sessionRef = "session_ref"
        case policyRevision = "policy_revision"
        case ownedArtifacts = "owned_artifacts"
        case ownedBytes = "owned_bytes"
        case unaccountedExternalCopies = "unaccounted_external_copies"
        case deletionVerified = "deletion_verified"
        case draftOnly = "draft_only"
        case requiresReview = "requires_review"
    }
}

/// Consent-gated owned memory. This class never opens a microphone or a network connection.
///
/// The host must stop producers when capture state changes, clear its input copies,
/// and declare external copies. Use `delete()` for deterministic cleanup. The lock
/// protects all state and content; callbacks must remain short, synchronous and local.
public final class AmbientSession: @unchecked Sendable, CustomStringConvertible {
    private struct Owned {
        let reference: AmbientReference
        let content: NSMutableData
    }

    private let lock = NSRecursiveLock()
    private let policy: AmbientSessionPolicy
    private let sessionRef: String
    private let clock: () -> Double
    private let started: Double
    private var lastTime: Double
    private var participants: Set<String>
    private var consents: Set<String> = []
    private var state: AmbientCaptureState = .awaitingConsent
    private var generation = 0
    private var owned: [String: Owned] = [:]
    private var seen: Set<String> = []
    private var external = 0
    private var exportBytes = 0
    private var operations = 0
    private var cleanupFailed = false
    private var reason = "none"
    private var timer: DispatchWorkItem?

    public var description: String { "AmbientSession(content=<private>)" }

    /// Disable automatic expiry only when the host supplies a reliable idle scheduler.
    public init(
        sessionRef: String, participants: [String], policy: AmbientSessionPolicy,
        clock: @escaping () -> Double = { ProcessInfo.processInfo.systemUptime },
        automaticExpiry: Bool = true
    ) throws {
        guard ambientHex(sessionRef, count: 32) else {
            throw AmbientSessionError.invalidSessionContract
        }
        try ambientParticipants(participants)
        let now = clock()
        guard now.isFinite, now >= 0 else { throw AmbientSessionError.invalidSessionContract }
        self.sessionRef = sessionRef
        self.participants = Set(participants)
        self.policy = policy
        self.clock = clock
        self.started = now
        self.lastTime = now
        if automaticExpiry { schedule(Double(policy.maxLifetimeMs) / 1000) }
    }

    deinit { delete() }

    private func locked<T>(_ body: () throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body()
    }

    private func schedule(_ delay: Double) {
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.locked {
                _ = self.expire()
                if !self.state.terminal {
                    self.schedule(
                        max(
                            0.001,
                            Double(self.policy.maxLifetimeMs) / 1000
                                - (self.lastTime - self.started)))
                }
            }
        }
        timer = work
        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + delay, execute: work)
    }

    private func invalidate(_ target: AmbientCaptureState, _ reason: String) {
        generation += 1
        for value in owned.values {
            let bytes = value.content.mutableBytes.assumingMemoryBound(to: UInt8.self)
            bytes.update(repeating: 0, count: value.content.length)
            if UnsafeBufferPointer(start: bytes, count: value.content.length).contains(where: { $0 != 0 }) {
                cleanupFailed = true
            }
        }
        owned.removeAll()
        consents.removeAll()
        self.reason = reason
        state = cleanupFailed ? .deletionIncomplete : target
        if state.terminal {
            timer?.cancel()
            timer = nil
        }
    }

    /// Apply the original deadline, including when called by host lifecycle callbacks.
    @discardableResult
    public func expire() -> Bool {
        locked {
            if state.terminal { return true }
            let now = clock()
            guard now.isFinite, now >= lastTime else {
                invalidate(.expired, "clock_invalid")
                return true
            }
            lastTime = now
            if (now - started) * 1000 >= Double(policy.maxLifetimeMs) {
                invalidate(.expired, "expired")
                return true
            }
            return false
        }
    }

    private func active() throws {
        guard !expire() else { throw AmbientSessionError.sessionClosed }
        operations += 1
        if operations > policy.maxOperations {
            invalidate(.expired, "operation_limit")
            throw AmbientSessionError.operationLimit
        }
    }

    /// Record host-acquired consent for the current policy, cohort and generation.
    public func consent(_ participantRef: String) throws {
        try locked {
            try active()
            guard ambientHex(participantRef, count: 32) else {
                throw AmbientSessionError.invalidSessionContract
            }
            guard state == .awaitingConsent else { throw AmbientSessionError.invalidTransition }
            guard participants.contains(participantRef) else { throw AmbientSessionError.unknownParticipant }
            consents.insert(participantRef)
            if consents == participants { state = .ready }
        }
    }

    /// Apply an explicit start, pause, resume, interrupt or stop action.
    public func transition(_ event: String) throws {
        try locked {
            try active()
            let allowed: Set<AmbientCaptureState>
            let target: AmbientCaptureState
            switch event {
            case "start": (allowed, target) = ([.ready], .capturing)
            case "pause": (allowed, target) = ([.capturing], .paused)
            case "resume": (allowed, target) = ([.paused, .interrupted], .capturing)
            case "interrupt": (allowed, target) = ([.ready, .capturing, .paused], .interrupted)
            case "stop": (allowed, target) = ([.ready, .capturing, .paused, .interrupted], .stopped)
            default: throw AmbientSessionError.invalidTransition
            }
            guard allowed.contains(state) else { throw AmbientSessionError.invalidTransition }
            guard consents == participants else { throw AmbientSessionError.consentRequired }
            state = target
        }
    }

    /// Destroy the previous cohort's data and require new unanimous consent.
    public func changeParticipants(_ values: [String]) throws {
        try locked {
            try active()
            try ambientParticipants(values)
            guard state != .stopped else { throw AmbientSessionError.invalidTransition }
            invalidate(external > 0 ? .deletionIncomplete : .awaitingConsent, "participants_changed")
            participants = Set(values)
        }
    }

    /// Revoke all references and overwrite owned content immediately.
    public func withdraw() {
        locked {
            if !state.terminal { invalidate(.withdrawn, "consent_withdrawn") }
        }
    }

    /// Destroy owned content. Undeletable external copies prevent verification.
    public func delete() {
        locked {
            if state == .deleted || state == .deletionIncomplete { return }
            invalidate(external > 0 ? .deletionIncomplete : .deleted, "deleted")
        }
    }

    /// Copy into owned mutable memory. The caller remains responsible for its input copy.
    public func retain(
        artifactRef: String, kind: AmbientArtifactKind, content: Data
    ) throws -> AmbientReference {
        try locked {
            try active()
            guard ambientHex(artifactRef, count: 32), !content.isEmpty else {
                throw AmbientSessionError.invalidSessionContract
            }
            guard kind == .audio ? state == .capturing : state.readable else {
                throw AmbientSessionError.captureNotAdmitted
            }
            guard !seen.contains(artifactRef) else { throw AmbientSessionError.duplicateReference }
            let size = owned.values.reduce(0) { $0 + $1.content.length }
            guard seen.count < policy.maxArtifacts, content.count <= policy.maxBytes - size else {
                invalidate(.expired, "retention_limit")
                throw AmbientSessionError.retentionLimit
            }
            let reference = AmbientReference(artifactRef, kind, generation)
            owned[artifactRef] = Owned(reference: reference, content: NSMutableData(data: content))
            seen.insert(artifactRef)
            guard !expire() else { throw AmbientSessionError.sessionClosed }
            return reference
        }
    }

    private func resolve(_ reference: AmbientReference) throws -> Owned {
        guard let value = owned[reference.artifactRef], value.reference === reference else {
            throw AmbientSessionError.staleReference
        }
        guard state.readable else { throw AmbientSessionError.captureNotAdmitted }
        return value
    }

    /// Borrow for a short local callback. Never retain the pointer or copy, log or send content.
    public func use<T>(
        _ reference: AmbientReference, reader: (UnsafeRawBufferPointer) throws -> T
    ) throws -> T {
        try locked {
            try active()
            let value = try resolve(reference)
            let result: T
            do {
                result = try reader(UnsafeRawBufferPointer(start: value.content.bytes, count: value.content.length))
            } catch {
                invalidate(.expired, "reader_failed")
                throw AmbientSessionError.readerFailed
            }
            guard !expire() else { throw AmbientSessionError.sessionClosed }
            _ = try resolve(reference)
            return result
        }
    }

    /// Register a copy outside ownership; this session cannot attest to its deletion.
    public func declareExternalCopy() throws {
        try locked {
            try active()
            guard external < policy.maxArtifacts else {
                invalidate(.deletionIncomplete, "external_limit")
                throw AmbientSessionError.externalLimit
            }
            external += 1
        }
    }

    /// Make a draft-only local review copy after capture stops and policy permits export.
    public func exportForReview(_ reference: AmbientReference) throws -> AmbientReviewCopy {
        try locked {
            try active()
            guard policy.allowReviewExport else { throw AmbientSessionError.exportNotPermitted }
            guard state == .stopped else { throw AmbientSessionError.invalidTransition }
            let value = try resolve(reference)
            guard reference.kind != .audio else { throw AmbientSessionError.exportNotPermitted }
            guard value.content.length <= policy.maxBytes - exportBytes else {
                throw AmbientSessionError.exportLimit
            }
            try declareExternalCopy()
            exportBytes += value.content.length
            let copy = AmbientReviewCopy(reference: reference, content: Data(bytes: value.content.bytes, count: value.content.length))
            guard !expire() else { throw AmbientSessionError.sessionClosed }
            return copy
        }
    }

    /// Content-free current state. Deletion means owned-buffer overwrite, not physical erasure.
    public func snapshot() -> AmbientSessionSnapshot {
        locked {
            _ = expire()
            return AmbientSessionSnapshot(
                schema: "openmed.ambient.session.v1", sessionRef: sessionRef,
                policyRevision: policy.revision, state: state, generation: generation,
                participants: participants.count, consents: consents.count,
                ownedArtifacts: owned.count, ownedBytes: owned.values.reduce(0) { $0 + $1.content.length },
                unaccountedExternalCopies: external,
                deletionVerified: state.terminal && owned.isEmpty && external == 0 && !cleanupFailed,
                reason: reason, draftOnly: true, requiresReview: true)
        }
    }

    /// Serialize only the state snapshot; retained content is never encoded.
    public func toJSON() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(snapshot())
    }
}
