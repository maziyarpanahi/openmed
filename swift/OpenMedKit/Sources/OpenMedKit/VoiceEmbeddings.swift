import CryptoKit
import Foundation

/// Controlled, value-free refusals. No case carries a vector or caller payload.
public enum VoiceEmbeddingError: String, Error {
    case serializationRefused = "embedding_serialization_refused"
    case crossSessionRefused = "embedding_cross_session_refused"
    case destroyed = "embedding_destroyed"
    case persistenceRefused = "embedding_persistence_refused"
    case sessionClosed = "embedding_session_closed"
    case capacityExceeded = "embedding_capacity_exceeded"
    case vectorInvalid = "embedding_vector_invalid"
    case limitsInvalid = "embedding_limits_invalid"
    case allocatorFailed = "embedding_allocator_failed"
    case registrationFailed = "embedding_registration_failed"
    case dimensionsMismatch = "embedding_dimensions_mismatch"
}

/// Only these lifecycle boundaries can appear in destruction evidence.
public enum VoiceEmbeddingBoundary: String {
    case pause, withdrawal, cancellation, finalization
}

/// Value-free evidence. Digests identify random handles, never vector content.
public struct EmbeddingDestructionReceipt: Codable, Equatable {
    public let reasonCode: String
    public let destroyedCount: Int
    public let handleDigests: [String]
}

/// Non-diagnostic similarity; consequential use requires explicit human review.
public struct VoiceSimilarity {
    public let score: Double
    public let notice = "non_diagnostic_voice_similarity"
    public let reviewerConfirmationRequired = true
}

/// Diagnostics contain counts and opaque handle digests only.
public struct VoiceEmbeddingDiagnostics: Codable, Equatable {
    public let handleCount: Int
    public let handleDigests: [String]
}

/// Owned mutable allocation. There is no public read or export operation.
/// An injected allocator transfers a fresh allocation to the session.
public final class VoiceEmbeddingBuffer: CustomStringConvertible, CustomReflectable {
    fileprivate var storage: UnsafeMutablePointer<Double>?
    public private(set) var count: Int

    /// Allocate zero-filled storage; the session enforces its dimension bound.
    public init(count: Int) throws {
        guard (1...65536).contains(count) else { throw VoiceEmbeddingError.limitsInvalid }
        self.count = count
        storage = .allocate(capacity: count)
        storage?.initialize(repeating: 0, count: count)
    }

    fileprivate func erase() {
        guard let storage else { return }
        for index in 0..<count { storage[index] = 0 }
        storage.deinitialize(count: count)
        storage.deallocate()
        self.storage = nil
        count = 0
    }

    public var description: String { "VoiceEmbeddingBuffer(opaque)" }
    public var customMirror: Mirror { Mirror(self, unlabeledChildren: [Any]()) }
    deinit { erase() }
}

/// Non-Codable opaque handle with a weak owner, safe reflection and no vector.
public final class VoiceEmbeddingHandle: CustomStringConvertible, CustomReflectable {
    fileprivate weak var owner: VoiceEmbeddingSession?
    private let sessionToken: UUID
    public let handleDigest: String

    fileprivate init(owner: VoiceEmbeddingSession, digest: String) {
        self.owner = owner
        sessionToken = owner.sessionToken
        handleDigest = digest
    }

    public var description: String { "VoiceEmbeddingHandle(\(handleDigest))" }
    public var customMirror: Mirror { Mirror(self, children: ["handleDigest": handleDigest]) }

    /// Export an opaque reference only, never the vector or session identifier.
    public func evidence() -> [String: String] { ["handle_digest": handleDigest] }

    /// Refuse serialization even when explicitly requested.
    public func serialize() throws -> Data { throw VoiceEmbeddingError.serializationRefused }

    /// Persistence requires a separate reviewed policy outside this API.
    public func persist() throws { throw VoiceEmbeddingError.persistenceRefused }

    /// Compare within the same live owner; this never assigns identity or role.
    public func similarity(to other: VoiceEmbeddingHandle) throws -> VoiceSimilarity {
        guard other.sessionToken == sessionToken else { throw VoiceEmbeddingError.crossSessionRefused }
        guard let owner else { throw VoiceEmbeddingError.destroyed }
        return try owner.similarity(self, other)
    }
}

/// Embedding-only lifecycle owner. This is not a consent or audio retention policy.
/// The host invokes `destroy` on every boundary; callbacks erase individual buffers.
/// Any boundary permanently closes this owner; resumption requires a fresh one.
public final class VoiceEmbeddingSession: CustomStringConvertible, CustomReflectable {
    public typealias Allocator = (Int) throws -> VoiceEmbeddingBuffer
    public typealias Registrar = (String, @escaping () -> EmbeddingDestructionReceipt) throws -> Void

    fileprivate let sessionToken = UUID()
    private var buffers: [String: VoiceEmbeddingBuffer] = [:]
    private let lock = NSRecursiveLock()
    private var closed = false
    private let allocator: Allocator
    private let register: Registrar?
    private let maxHandles: Int
    private let maxDimensions: Int

    /// Inject a trusted allocator and optional embedding-owned retention hook.
    /// Hooks receive only an opaque digest and idempotent erasure callback.
    public init(
        maxHandles: Int = 256, maxDimensions: Int = 4096,
        allocator: @escaping Allocator = { try VoiceEmbeddingBuffer(count: $0) },
        register: Registrar? = nil
    ) throws {
        guard (1...65536).contains(maxHandles), (1...65536).contains(maxDimensions) else {
            throw VoiceEmbeddingError.limitsInvalid
        }
        self.maxHandles = maxHandles
        self.maxDimensions = maxDimensions
        self.allocator = allocator
        self.register = register
    }

    public var description: String { "VoiceEmbeddingSession(handle_count=\(diagnostics().handleCount))" }
    public var customMirror: Mirror { Mirror(self, unlabeledChildren: [Any]()) }

    /// Copy finite nonzero caller values into owned storage. The caller/provider
    /// remains responsible for erasing its source allocations; no model is used.
    public func add(_ values: [Double]) throws -> VoiceEmbeddingHandle {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { throw VoiceEmbeddingError.sessionClosed }
        guard buffers.count < maxHandles else { throw VoiceEmbeddingError.capacityExceeded }
        guard !values.isEmpty, values.count <= maxDimensions,
            values.allSatisfy({ $0.isFinite }), values.contains(where: { $0 != 0 })
        else { throw VoiceEmbeddingError.vectorInvalid }
        let buffer: VoiceEmbeddingBuffer
        do { buffer = try allocator(values.count) } catch { throw VoiceEmbeddingError.allocatorFailed }
        guard !buffers.values.contains(where: { $0 === buffer }) else {
            throw VoiceEmbeddingError.allocatorFailed
        }
        guard buffer.count == values.count, let storage = buffer.storage,
            (0..<buffer.count).allSatisfy({ storage[$0] == 0 })
        else {
            buffer.erase()
            throw VoiceEmbeddingError.allocatorFailed
        }
        for index in values.indices { storage[index] = values[index] }
        // UUID bytes are unrelated to patient/session identifiers or vectors.
        let digest = SHA256.hash(data: Data(UUID().uuidString.utf8)).map {
            String(format: "%02x", $0)
        }.joined()
        buffers[digest] = buffer
        do {
            try register?(
                digest,
                { [weak self] in
                    self?.eraseHandle(digest)
                        ?? EmbeddingDestructionReceipt(
                            reasonCode: "embedding_destroyed", destroyedCount: 0, handleDigests: [])
                })
        } catch {
            _ = eraseHandle(digest)
            throw VoiceEmbeddingError.registrationFailed
        }
        guard !closed, buffers[digest] != nil else {
            _ = eraseHandle(digest)
            throw VoiceEmbeddingError.destroyed
        }
        return VoiceEmbeddingHandle(owner: self, digest: digest)
    }

    /// Return counts and random reference digests, without dimensions or scores.
    public func diagnostics() -> VoiceEmbeddingDiagnostics {
        lock.lock()
        defer { lock.unlock() }
        return VoiceEmbeddingDiagnostics(handleCount: buffers.count, handleDigests: buffers.keys.sorted())
    }

    /// Overwrite, deallocate and release all owned buffers, closing the owner.
    @discardableResult
    public func destroy(_ boundary: VoiceEmbeddingBoundary = .finalization) -> EmbeddingDestructionReceipt {
        lock.lock()
        defer { lock.unlock() }
        closed = true
        let digests = buffers.keys.sorted()
        for buffer in buffers.values { buffer.erase() }
        buffers.removeAll()
        return EmbeddingDestructionReceipt(
            reasonCode: boundary.rawValue, destroyedCount: digests.count, handleDigests: digests)
    }

    private func eraseHandle(_ digest: String) -> EmbeddingDestructionReceipt {
        lock.lock()
        defer { lock.unlock() }
        guard let buffer = buffers.removeValue(forKey: digest) else {
            return EmbeddingDestructionReceipt(
                reasonCode: "embedding_destroyed", destroyedCount: 0, handleDigests: [])
        }
        buffer.erase()
        return EmbeddingDestructionReceipt(
            reasonCode: "embedding_destroyed", destroyedCount: 1, handleDigests: [digest])
    }

    fileprivate func similarity(_ left: VoiceEmbeddingHandle, _ right: VoiceEmbeddingHandle) throws -> VoiceSimilarity {
        lock.lock()
        defer { lock.unlock() }
        guard let a = buffers[left.handleDigest], let b = buffers[right.handleDigest],
            let x = a.storage, let y = b.storage
        else { throw VoiceEmbeddingError.destroyed }
        guard a.count == b.count else { throw VoiceEmbeddingError.dimensionsMismatch }
        let scaleX = (0..<a.count).map { abs(x[$0]) }.max()!
        let scaleY = (0..<b.count).map { abs(y[$0]) }.max()!
        var normX = 0.0
        var normY = 0.0
        for i in 0..<a.count {
            normX += (x[i] / scaleX) * (x[i] / scaleX)
            normY += (y[i] / scaleY) * (y[i] / scaleY)
        }
        normX = sqrt(normX)
        normY = sqrt(normY)
        var score = 0.0
        for i in 0..<a.count { score += (x[i] / scaleX / normX) * (y[i] / scaleY / normY) }
        return VoiceSimilarity(score: max(-1, min(1, score)))
    }

    deinit { destroy() }
}
