import Foundation

#if canImport(CryptoKit)
    import CryptoKit
#endif

#if canImport(Darwin)
    import Darwin
#endif

/// Lifecycle states of one bundle generation (`docs/models/bundle-cache.md`).
public enum OpenMedBundleGenerationState: String, CaseIterable, Sendable {
    case absent, staging, resumable, verifying, ready, active, superseded, evictable, quarantined, evicted
}

/// Why a generation is protected from eviction, in snapshot order.
public enum OpenMedBundleProtection: String, CaseIterable, Sendable {
    case active
    case lastGreen = "last_green"
    case rollback, pinned, leased
}

/// Stable, value-free failure reasons of the generation cache.
public enum OpenMedBundleCacheReason: String, CaseIterable, Sendable {
    case bundleRejected = "bundle_rejected"
    case lockTimeout = "lock_timeout"
    case quotaExceeded = "quota_exceeded"
    case storagePressure = "storage_pressure"
    case transferInterrupted = "transfer_interrupted"
    case cancelled
    case rangeMismatch = "range_mismatch"
    case verificationFailed = "verification_failed"
    case storageIO = "storage_io"
    case generationNotReady = "generation_not_ready"
    case generationCorrupt = "generation_corrupt"
    case noActiveGeneration = "no_active_generation"
    case rollbackUnavailable = "rollback_unavailable"
    case leaseUnknown = "lease_unknown"
}

/// Why staged bytes were discarded or partly dropped on resume.
public enum OpenMedBundleResumeReason: String, CaseIterable, Sendable {
    case sourceChanged = "source_changed"
    case trustPolicyChanged = "trust_policy_changed"
    case chunkSizeChanged = "chunk_size_changed"
    case partialDigestMismatch = "partial_digest_mismatch"
    case rangeNotHonored = "range_not_honored"
}

/// One crash-recovery action, in report order.
public enum OpenMedBundleRecoveryAction: String, CaseIterable, Sendable {
    case staleLockCleared = "stale_lock_cleared"
    case trashPurged = "trash_purged"
    case quarantinePurged = "quarantine_purged"
    case stagingOrphaned = "staging_orphaned"
    case stagingExpired = "staging_expired"
    case stagingResumable = "staging_resumable"
    case generationQuarantined = "generation_quarantined"
    case pointerRepaired = "pointer_repaired"
    case leaseExpired = "lease_expired"
}

/// How thoroughly `open` checks the active generation.
public enum OpenMedBundleOpenVerification: Sendable {
    /// Manifest identity, declared file set, and sizes.
    case metadata
    /// Every file is also re-hashed against its declared SHA-256.
    case full
}

/// Error carrying only controlled reason codes; it never contains a path or URL.
public struct OpenMedBundleCacheError: Error, Equatable, Sendable, CustomStringConvertible {
    public let reason: OpenMedBundleCacheReason
    public let bundleReason: PortableBundleReason?

    public init(_ reason: OpenMedBundleCacheReason, bundleReason: PortableBundleReason? = nil) {
        self.reason = reason
        self.bundleReason = bundleReason
    }

    public var description: String { reason.rawValue }
}

/// Reads one byte range; an empty result means end of file.
public protocol OpenMedBundleRangeReader: AnyObject {
    func read(upToCount count: Int) throws -> Data
}

/// A byte range returned by a range source.
public struct OpenMedBundleRangeResponse {
    /// Offset of the first byte the reader returns; `0` when the source ignored the range.
    public let offset: Int64
    public let reader: any OpenMedBundleRangeReader

    public init(offset: Int64, reader: any OpenMedBundleRangeReader) {
        self.offset = offset
        self.reader = reader
    }
}

/// Supplies bundle files by byte range. Implementations perform any network
/// access; the cache itself never does.
public protocol OpenMedBundleRangeSource {
    /// Opaque identity of the source (for example a URL plus an ETag). Only its digest is stored.
    var identity: String { get }
    /// Opens a declared bundle path starting at `offset`.
    func open(path: String, offset: Int64) throws -> OpenMedBundleRangeResponse
}

/// Range source over a local prepared bundle directory. Performs no network access.
public struct OpenMedDirectoryBundleSource: OpenMedBundleRangeSource, CustomStringConvertible {
    private let directory: URL
    public let identity: String

    public init(directory: URL, identity: String = "local-directory") {
        self.directory = directory
        self.identity = identity
    }

    private final class HandleReader: OpenMedBundleRangeReader {
        let handle: FileHandle
        init(handle: FileHandle) { self.handle = handle }
        deinit { try? handle.close() }
        func read(upToCount count: Int) throws -> Data { try handle.read(upToCount: count) ?? Data() }
    }

    public func open(path: String, offset: Int64) throws -> OpenMedBundleRangeResponse {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL.path
        let file = directory.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard file.path.hasPrefix(root + "/"), let handle = try? FileHandle(forReadingFrom: file) else {
            throw OpenMedBundleCacheError(.transferInterrupted)
        }
        try handle.seek(toOffset: UInt64(max(0, offset)))
        return OpenMedBundleRangeResponse(offset: offset, reader: HandleReader(handle: handle))
    }

    public var description: String { "OpenMedDirectoryBundleSource" }
}

/// Tunables of an `OpenMedBundleCache`.
public struct OpenMedBundleCacheOptions: @unchecked Sendable {
    /// Byte budget over the ready generations of every bundle.
    public var quotaBytes: Int64
    /// Resume checkpoint size; each committed chunk records its SHA-256.
    public var chunkBytes: Int
    /// Bounded wait for a bundle lock, in milliseconds.
    public var lockTimeoutMilliseconds: Int64
    /// How long a lease protects its generation once its owner is gone.
    public var leaseTTLMilliseconds: Int64
    /// Staging older than this since its last checkpoint is removed by recovery.
    public var stagingMaxAgeMilliseconds: Int64
    /// Clock in epoch milliseconds.
    public var now: () -> Int64
    /// Free-storage probe; defaults to the volume's available capacity.
    public var availableBytes: ((URL) -> Int64)?
    /// Opaque id of this cache instance in lock and lease records.
    public var ownerID: String

    public init(
        quotaBytes: Int64 = 2 << 30,
        chunkBytes: Int = 4 << 20,
        lockTimeoutMilliseconds: Int64 = 30_000,
        leaseTTLMilliseconds: Int64 = 600_000,
        stagingMaxAgeMilliseconds: Int64 = 604_800_000,
        now: @escaping () -> Int64 = { Int64(Date().timeIntervalSince1970 * 1000) },
        availableBytes: ((URL) -> Int64)? = nil,
        ownerID: String = UUID().uuidString
    ) {
        self.quotaBytes = max(0, quotaBytes)
        self.chunkBytes = max(1, chunkBytes)
        self.lockTimeoutMilliseconds = max(0, lockTimeoutMilliseconds)
        self.leaseTTLMilliseconds = max(1, leaseTTLMilliseconds)
        self.stagingMaxAgeMilliseconds = max(1, stagingMaxAgeMilliseconds)
        self.now = now
        self.availableBytes = availableBytes
        self.ownerID = ownerID
    }
}

/// One acquisition: a signed manifest and envelope plus the source of its files.
public struct OpenMedBundleAcquireRequest {
    public var manifest: Data
    public var envelope: Data
    public var source: any OpenMedBundleRangeSource
    public var trustPolicy: PortableBundleTrustPolicy
    public var context: PortableBundleContext
    public var activate: Bool
    public var confirm: Bool
    /// Checked after every committed chunk; staged bytes stay resumable.
    public var isCancelled: () -> Bool

    public init(
        manifest: Data, envelope: Data, source: any OpenMedBundleRangeSource,
        trustPolicy: PortableBundleTrustPolicy, context: PortableBundleContext,
        activate: Bool = true, confirm: Bool = true, isCancelled: @escaping () -> Bool = { false }
    ) {
        self.manifest = manifest
        self.envelope = envelope
        self.source = source
        self.trustPolicy = trustPolicy
        self.context = context
        self.activate = activate
        self.confirm = confirm
        self.isCancelled = isCancelled
    }
}

/// Value-free result of `OpenMedBundleCache.acquire`.
public struct OpenMedBundleAcquireResult: Equatable, Sendable {
    public let ok: Bool
    public let reason: OpenMedBundleCacheReason?
    public let bundleReason: PortableBundleReason?
    public let bundleKey: String?
    public let generation: String?
    public let state: OpenMedBundleGenerationState?
    public let fromCache: Bool
    public let resumed: Bool
    public let resumeReason: OpenMedBundleResumeReason?
    public let bytesTransferred: Int64
    public let verifiedBytes: Int64
    public let filesVerified: Int
    public let evicted: [String]
}

/// One generation in a snapshot.
public struct OpenMedBundleGenerationSnapshot: Equatable, Sendable {
    public let generation: String
    public let state: OpenMedBundleGenerationState
    public let totalBytes: Int64
    public let verifiedBytes: Int64
    public let protections: [OpenMedBundleProtection]
}

/// One quarantined generation and its reason code.
public struct OpenMedBundleQuarantineEntry: Equatable, Sendable {
    public let generation: String
    public let reason: String
}

/// Versioned, value-free view of one bundle (`openmed.bundle_cache.snapshot` v1).
public struct OpenMedBundleCacheSnapshot: Equatable, Sendable {
    public static let schema = "openmed.bundle_cache.snapshot"
    public static let schemaVersion = 1
    public let bundleKey: String
    public let active: String?
    public let lastGreen: String?
    public let rollback: String?
    public let generations: [OpenMedBundleGenerationSnapshot]
    public let quarantined: [OpenMedBundleQuarantineEntry]
    public let usedBytes: Int64
}

/// Deterministic quota decision.
public struct OpenMedBundleQuotaDecision: Equatable, Sendable {
    public let quotaBytes: Int64
    public let usedBytes: Int64
    public let additionalBytes: Int64
    public let evicted: [String]
    public let satisfied: Bool
}

/// One recovery action.
public struct OpenMedBundleRecoveryStep: Equatable, Sendable {
    public let action: OpenMedBundleRecoveryAction
    public let bundleKey: String
    public let generation: String?
}

/// Ordered report of `OpenMedBundleCache.recover`.
public struct OpenMedBundleRecoveryReport: Equatable, Sendable {
    public let actions: [OpenMedBundleRecoveryStep]
}

/// A lease on a verified generation. While held it is never evicted. Its
/// description omits `payloadDirectory` so the private path does not reach logs.
public final class OpenMedBundleLease: CustomStringConvertible, @unchecked Sendable {
    public let id: String
    public let bundleKey: String
    public let generation: String
    public let payloadDirectory: URL
    public let filesChecked: Int
    private weak var cache: OpenMedBundleCache?

    init(id: String, bundleKey: String, generation: String, payloadDirectory: URL, filesChecked: Int, cache: OpenMedBundleCache) {
        self.id = id
        self.bundleKey = bundleKey
        self.generation = generation
        self.payloadDirectory = payloadDirectory
        self.filesChecked = filesChecked
        self.cache = cache
    }

    /// Releases the lease. Safe to call more than once.
    public func release() { try? cache?.release(self) }

    public var description: String {
        "OpenMedBundleLease(bundleKey: \(bundleKey), generation: \(generation), filesChecked: \(filesChecked))"
    }
}

/// Thrown only by test checkpoints to simulate a process dying at a named point.
struct OpenMedBundleCacheCrash: Error {
    let checkpoint: String
}

/// Crash-safe, resumable, integrity-checked cache of signed portable bundles.
///
/// Every bundle has isolated staging, verified generations, an atomic
/// activation pointer, a `flock(2)` lock file, and lease records. Acquisition
/// resumes only from verified chunks, verifies the staged directory with
/// `PortableBundle.verifyDirectory`, and commits with one directory rename.
/// Quota eviction never removes active, last-green, rollback, pinned, or leased
/// generations. The cache never opens a network connection and records only
/// opaque ids, digests, sizes, and reason codes. See `docs/models/bundle-cache.md`.
public final class OpenMedBundleCache: @unchecked Sendable, CustomStringConvertible {
    /// Contract version of the cache records and results.
    public static let contractVersion = 1

    private let root: URL
    private let options: OpenMedBundleCacheOptions
    private let fileManager = FileManager.default
    private let leaseLock = NSLock()
    private var heldLeases: Set<String> = []

    /// Test seam: called at each named crash checkpoint.
    var checkpointHook: ((String) throws -> Void)?

    public init(rootDirectory: URL, options: OpenMedBundleCacheOptions = OpenMedBundleCacheOptions()) throws {
        try FileManager.default.createDirectory(at: rootDirectory, withIntermediateDirectories: true)
        root = rootDirectory.resolvingSymlinksInPath().standardizedFileURL
        self.options = options
    }

    public var description: String { "OpenMedBundleCache(quotaBytes: \(options.quotaBytes))" }

    // MARK: - Public API

    /// Verifies, stages, commits, and optionally activates one signed bundle.
    public func acquire(_ request: OpenMedBundleAcquireRequest) -> OpenMedBundleAcquireResult {
        (try? acquireChecked(request)) ?? failure(.storageIO, nil, nil, nil)
    }

    /// Same as `acquire` but lets a test crash checkpoint escape.
    func acquireChecked(_ request: OpenMedBundleAcquireRequest) throws -> OpenMedBundleAcquireResult {
        let verification = PortableBundle.verify(
            manifest: request.manifest, envelope: request.envelope, trustPolicy: request.trustPolicy,
            context: request.context)
        guard verification.ok, let contentID = verification.contentID else {
            return failure(.bundleRejected, verification.reason, nil, nil)
        }
        guard let header = ManifestHeader(request.manifest) else {
            return failure(.bundleRejected, .manifestMalformed, nil, nil)
        }
        let key = Self.bundleKey(bundleID: header.bundleID)
        let generation = Self.generationID(contentID: contentID)
        do {
            return try locked(key) {
                try acquireLocked(request, header: header, key: key, generation: generation, contentID: contentID)
            }
        } catch let crash as OpenMedBundleCacheCrash {
            throw crash
        } catch let error as OpenMedBundleCacheError {
            return failure(error.reason, error.bundleReason, key, generation)
        } catch {
            return failure(.storageIO, nil, key, generation)
        }
    }

    /// Activates a ready generation; with `confirm` it also becomes last green.
    public func activate(bundleID: String, generation: String, confirm: Bool = true) throws {
        let key = Self.bundleKey(bundleID: bundleID)
        try locked(key) { try activateLocked(key, generation, confirm: confirm) }
    }

    /// Marks the active generation as last green after a probation activation.
    public func confirm(bundleID: String) throws {
        let key = Self.bundleKey(bundleID: bundleID)
        try locked(key) {
            var pointer = readPointer(key)
            guard let active = pointer.active else { throw OpenMedBundleCacheError(.noActiveGeneration) }
            pointer.lastGreen = active
            try writePointer(key, pointer)
        }
    }

    /// Reactivates the retained rollback generation without any transfer.
    public func rollback(bundleID: String, quarantineAbandoned: Bool = false) throws {
        let key = Self.bundleKey(bundleID: bundleID)
        try locked(key) {
            let pointer = readPointer(key)
            guard let target = pointer.rollback, metadataCheck(key, target) == nil else {
                throw OpenMedBundleCacheError(.rollbackUnavailable)
            }
            let abandoned = pointer.active
            try touch(key, target, activated: true)
            let lastGreen = pointer.lastGreen == abandoned ? target : pointer.lastGreen
            try writePointer(key, Pointer(active: target, lastGreen: lastGreen, rollback: nil))
            if quarantineAbandoned, let abandoned, abandoned != target {
                try quarantine(key, generationDirectory(key, abandoned), abandoned, reason: Self.rolledBack)
            }
        }
    }

    /// Pins a ready generation so eviction never removes it.
    public func pin(bundleID: String, generation: String) throws { try setPinned(bundleID, generation, true) }

    /// Removes a pin.
    public func unpin(bundleID: String, generation: String) throws { try setPinned(bundleID, generation, false) }

    /// Leases a ready generation (the active one by default) after a metadata check.
    public func lease(bundleID: String, generation: String? = nil) throws -> OpenMedBundleLease {
        let key = Self.bundleKey(bundleID: bundleID)
        return try locked(key) {
            guard let target = generation ?? readPointer(key).active else {
                throw OpenMedBundleCacheError(.noActiveGeneration)
            }
            return try createLease(key, target, verification: .metadata)
        }
    }

    /// Opens the active generation for loading and returns a lease on it. A
    /// mismatch quarantines the generation, falls the pointer back, and fails
    /// with `generationCorrupt`.
    public func open(bundleID: String, verification: OpenMedBundleOpenVerification = .metadata) throws -> OpenMedBundleLease {
        let key = Self.bundleKey(bundleID: bundleID)
        return try locked(key) {
            guard let active = readPointer(key).active else { throw OpenMedBundleCacheError(.noActiveGeneration) }
            return try createLease(key, active, verification: verification)
        }
    }

    /// Releases a lease.
    public func release(_ lease: OpenMedBundleLease) throws {
        try locked(lease.bundleKey) {
            leaseLock.lock()
            let known = heldLeases.remove(lease.id) != nil
            leaseLock.unlock()
            let file = leaseFile(lease.bundleKey, lease.generation, lease.id)
            let removed = (try? fileManager.removeItem(at: file)) != nil
            if !removed && !known { throw OpenMedBundleCacheError(.leaseUnknown) }
        }
    }

    /// Evicts least recently used unprotected generations until `additionalBytes` fit.
    public func evict(additionalBytes: Int64 = 0) -> OpenMedBundleQuotaDecision {
        (try? evictChecked(additionalBytes: additionalBytes))
            ?? OpenMedBundleQuotaDecision(
                quotaBytes: options.quotaBytes, usedBytes: usedBytes(), additionalBytes: additionalBytes, evicted: [],
                satisfied: false)
    }

    func evictChecked(additionalBytes: Int64) throws -> OpenMedBundleQuotaDecision {
        try applyQuota(additionalBytes: max(0, additionalBytes), exclude: nil, heldKey: nil)
    }

    /// Idempotent, bounded crash recovery over every bundle; call at startup.
    public func recover() -> OpenMedBundleRecoveryReport {
        (try? recoverChecked()) ?? OpenMedBundleRecoveryReport(actions: [])
    }

    func recoverChecked() throws -> OpenMedBundleRecoveryReport {
        var actions: [OpenMedBundleRecoveryStep] = []
        for key in bundleKeys() {
            guard let held = try lockBundle(key, timeoutMilliseconds: options.lockTimeoutMilliseconds) else { continue }
            try held.guarded { try recoverLocked(key, stale: held.stale, into: &actions) }
        }
        return OpenMedBundleRecoveryReport(actions: actions)
    }

    /// Quarantines `generation` with reason `rolled_back` and falls the pointer
    /// back to last green, then rollback, then none. The managed importer calls
    /// it when the activation check of a probation generation fails.
    func withdraw(bundleID: String, generation: String) throws {
        let key = Self.bundleKey(bundleID: bundleID)
        try locked(key) { try quarantineAndRepair(key, generation, reason: Self.rolledBack) }
    }

    /// Signed anti-rollback sequence of a ready generation.
    func sequence(bundleID: String, generation: String) -> Int64? {
        let key = Self.bundleKey(bundleID: bundleID)
        return try? locked(key) { readGenerationRecord(key, generation)?.sequence }
    }

    /// Value-free snapshot of one bundle.
    public func snapshot(bundleID: String) throws -> OpenMedBundleCacheSnapshot {
        let key = Self.bundleKey(bundleID: bundleID)
        return try locked(key) { snapshotLocked(key) }
    }

    // MARK: - Identities

    /// Opaque directory key of a bundle id.
    public static func bundleKey(bundleID: String) -> String {
        "b-" + String(PortableBundle.sha256Hex(Data(bundleID.utf8)).prefix(32))
    }

    /// Opaque generation id of a manifest content identity (`sha256:<hex>`).
    public static func generationID(contentID: String) -> String {
        let hex = contentID.hasPrefix("sha256:") ? String(contentID.dropFirst(7)) : contentID
        return "g-" + String(hex.prefix(32))
    }

    /// Binding digest of a trust policy, recorded with resumable bytes.
    public static func trustDigest(_ policy: PortableBundleTrustPolicy) -> String {
        let keys = policy.keys.sorted { $0.kid.utf8.lexicographicallyPrecedes($1.kid.utf8) }.map { key in
            PortableBundleJSON.object([
                ("kid", .string(key.kid)), ("not_after", .string(key.notAfter)),
                ("not_before", .string(key.notBefore)), ("revoked", .bool(key.revoked)),
                ("x", .string(PortableBundle.base64URLEncode(key.publicKey))),
            ])
        }
        let revoked = policy.revokedContentIDs.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
        let value = PortableBundleJSON.object([
            ("keys", .array(keys)), ("revoked_content_ids", .array(revoked.map { .string($0) })),
            ("threshold", .int(Int64(policy.threshold))),
        ])
        return sha256Text(Data((try? PortableBundleJSON.encode(value)) ?? []))
    }

    // MARK: - Acquisition

    private func acquireLocked(
        _ request: OpenMedBundleAcquireRequest, header: ManifestHeader, key: String, generation: String, contentID: String
    ) throws -> OpenMedBundleAcquireResult {
        let files = header.files
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        if metadataCheck(key, generation) == nil {
            try touch(key, generation, activated: nil)
            if request.activate { try activateLocked(key, generation, confirm: request.confirm) }
            return success(key, generation, totalBytes, files.count, fromCache: true)
        }
        if readGenerationRecord(key, generation) != nil {
            // A ready record whose files changed is unusable: quarantine it first.
            try quarantineAndRepair(key, generation, reason: Self.payloadMismatch)
        }

        for other in children(stagingRoot(key)) where other.lastPathComponent != generation {
            try discard(key, other)
        }
        let staleQuarantine = quarantineDirectory(key, generation)
        if exists(staleQuarantine) { try discard(key, staleQuarantine) }

        let stage = stagingDirectory(key, generation)
        let payload = stage.appendingPathComponent(Self.payload)
        let sourceDigest = Self.sha256Text(Data(request.source.identity.utf8))
        let trust = Self.trustDigest(request.trustPolicy)
        let chunk = options.chunkBytes
        var resumeReason: OpenMedBundleResumeReason?
        var chunks: [[String]] = Array(repeating: [], count: files.count)
        let record = readStagingRecord(stage)
        var fresh = true
        if let record, record.contentID == contentID, record.files.count == files.count {
            if record.sourceDigest != sourceDigest {
                resumeReason = .sourceChanged
            } else if record.trustDigest != trust {
                resumeReason = .trustPolicyChanged
            } else if record.chunkBytes != chunk {
                resumeReason = .chunkSizeChanged
            }
            if resumeReason == nil {
                fresh = false
                for (index, file) in files.enumerated() {
                    let kept = try keptChunks(payload.appendingPathComponent(file.path), size: file.size, recorded: record.files[index])
                    if kept.mismatch { resumeReason = .partialDigestMismatch }
                    chunks[index] = kept.digests
                }
            }
        }
        if fresh, exists(stage) { try discard(key, stage) }
        let keptBytes = files.indices.reduce(Int64(0)) { $0 + verifiedBytes(chunks[$1].count, files[$1].size) }
        let resumed = keptBytes > 0
        let created = (fresh ? nil : record?.createdAt) ?? now()
        func checkpointRecord(_ phase: String) throws {
            try writeStagingRecord(
                stage,
                StagingRecord(
                    contentID: contentID, sourceDigest: sourceDigest, trustDigest: trust, chunkBytes: chunk, phase: phase,
                    createdAt: created, updatedAt: now(), files: chunks))
        }

        let available = options.availableBytes?(root) ?? availableCapacity()
        if available < totalBytes - keptBytes {
            if !fresh { try checkpointRecord(Self.phaseStaging) }
            throw OpenMedBundleCacheError(.storagePressure)
        }
        let plan = planQuota(additionalBytes: totalBytes, exclude: nil)
        guard plan.satisfied else {
            if !fresh { try checkpointRecord(Self.phaseStaging) }
            throw OpenMedBundleCacheError(.quotaExceeded)
        }
        var evicted = try applyPlan(plan.candidates, heldKey: key)

        if fresh {
            try fileManager.createDirectory(at: payload, withIntermediateDirectories: true)
            try request.manifest.write(to: payload.appendingPathComponent(PortableBundle.manifestFilename), options: .atomic)
            try request.envelope.write(to: payload.appendingPathComponent(PortableBundle.envelopeFilename), options: .atomic)
        }
        try checkpointRecord(Self.phaseStaging)
        if fresh { try checkpoint("staging_record_written") }

        var transferred: Int64 = 0
        for (index, file) in files.enumerated() {
            if request.isCancelled() { throw OpenMedBundleCacheError(.cancelled) }
            var have = verifiedBytes(chunks[index].count, file.size)
            if have == file.size { continue }
            let target = payload.appendingPathComponent(file.path)
            try fileManager.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            let response: OpenMedBundleRangeResponse
            do {
                response = try request.source.open(path: file.path, offset: have)
            } catch {
                throw OpenMedBundleCacheError(.transferInterrupted)
            }
            if response.offset != have {
                guard response.offset == 0 else { throw OpenMedBundleCacheError(.rangeMismatch) }
                chunks[index] = []
                have = 0
                if resumeReason == nil { resumeReason = .rangeNotHonored }
            }
            if !exists(target) { fileManager.createFile(atPath: target.path, contents: nil) }
            let output = try FileHandle(forWritingTo: target)
            defer { try? output.close() }
            try output.truncate(atOffset: UInt64(have))
            try output.seek(toOffset: UInt64(have))
            while have < file.size {
                let want = Int(min(Int64(chunk), file.size - have))
                let data = readFully(response.reader, want)
                transferred += Int64(data.count)
                guard data.count == want else { throw OpenMedBundleCacheError(.transferInterrupted) }
                try output.write(contentsOf: data)
                chunks[index].append(Self.sha256Text(data))
                have += Int64(want)
                try checkpointRecord(Self.phaseStaging)
                try checkpoint("chunk_committed")
                if request.isCancelled() { throw OpenMedBundleCacheError(.cancelled) }
            }
            let extra = (try? response.reader.read(upToCount: 1)) ?? Data()
            if !extra.isEmpty {
                try? output.close()
                try quarantine(key, stage, generation, reason: PortableBundleReason.fileSizeMismatch.rawValue)
                throw OpenMedBundleCacheError(.verificationFailed, bundleReason: .fileSizeMismatch)
            }
        }

        try checkpointRecord(Self.phaseVerifying)
        try checkpoint("verifying_record_written")
        let verified = PortableBundle.verifyDirectory(
            at: payload, trustPolicy: request.trustPolicy, context: request.context)
        guard verified.eligibleForActivation, verified.contentID == contentID else {
            let reason = verified.reason ?? .fileMissing
            try quarantine(key, stage, generation, reason: reason.rawValue)
            throw OpenMedBundleCacheError(.verificationFailed, bundleReason: reason)
        }

        let stamp = now()
        try writeGenerationRecord(
            stage.appendingPathComponent(Self.generationRecord),
            GenerationRecord(
                contentID: contentID, sequence: header.sequence, totalBytes: totalBytes, fileCount: files.count,
                activated: false, pinned: false, verifiedAt: stamp, lastAccessed: stamp))
        let destination = generationDirectory(key, generation)
        if exists(destination) { try discard(key, destination) }
        try move(stage, destination)
        try checkpoint("generation_committed")
        try? fileManager.removeItem(at: destination.appendingPathComponent(Self.stagingRecord))

        if request.activate { try activateLocked(key, generation, confirm: request.confirm) }
        evicted += try applyQuota(additionalBytes: 0, exclude: generation, heldKey: key).evicted
        return success(
            key, generation, totalBytes, files.count, fromCache: false, resumed: resumed, resumeReason: resumeReason,
            transferred: transferred, evicted: evicted)
    }

    private func keptChunks(_ file: URL, size: Int64, recorded: [String]) throws -> (digests: [String], mismatch: Bool) {
        guard isRegularFile(file), let handle = try? FileHandle(forUpdating: file) else { return ([], false) }
        defer { try? handle.close() }
        let length = Int64(try handle.seekToEnd())
        var kept: [String] = []
        var mismatch = false
        for (index, digest) in recorded.enumerated() {
            let start = Int64(index) * Int64(options.chunkBytes)
            let want = Int(min(Int64(options.chunkBytes), size - start))
            if want <= 0 || start + Int64(want) > length { break }
            try handle.seek(toOffset: UInt64(start))
            let data = try handle.read(upToCount: want) ?? Data()
            guard data.count == want, Self.sha256Text(data) == digest else {
                mismatch = true
                break
            }
            kept.append(digest)
        }
        try handle.truncate(atOffset: UInt64(verifiedBytes(kept.count, size)))
        return (kept, mismatch)
    }

    private func verifiedBytes(_ chunkCount: Int, _ size: Int64) -> Int64 {
        min(Int64(chunkCount) * Int64(options.chunkBytes), size)
    }

    private func readFully(_ reader: any OpenMedBundleRangeReader, _ want: Int) -> Data {
        var data = Data()
        while data.count < want {
            guard let next = try? reader.read(upToCount: want - data.count), !next.isEmpty else { break }
            data.append(next)
        }
        return data
    }

    /// First verifier reason the payload fails, or `nil`.
    private func verifyPayload(_ payload: URL, _ files: [DeclaredFile], full: Bool) -> PortableBundleReason? {
        for file in files {
            let target = payload.appendingPathComponent(file.path)
            guard isRegularFile(target) else { return .fileMissing }
            guard fileSize(target) == file.size else { return .fileSizeMismatch }
            if full && Self.sha256File(target) != file.sha256 { return .fileDigestMismatch }
        }
        if full {
            var declared = Set(files.map(\.path))
            declared.insert(PortableBundle.manifestFilename)
            declared.insert(PortableBundle.envelopeFilename)
            let subpaths = (try? fileManager.subpathsOfDirectory(atPath: payload.path)) ?? []
            for relative in subpaths where isRegularFile(payload.appendingPathComponent(relative)) {
                if !declared.contains(relative) { return .fileUndeclared }
            }
        }
        return nil
    }

    // MARK: - Activation, leases, pins

    private func activateLocked(_ key: String, _ generation: String, confirm: Bool) throws {
        guard readGenerationRecord(key, generation) != nil else { throw OpenMedBundleCacheError(.generationNotReady) }
        if let reason = metadataCheck(key, generation) {
            try quarantineAndRepair(key, generation, reason: reason.rawValue)
            throw OpenMedBundleCacheError(.generationCorrupt, bundleReason: reason)
        }
        let pointer = readPointer(key)
        let next: Pointer
        if pointer.active == generation {
            next = Pointer(active: generation, lastGreen: confirm ? generation : pointer.lastGreen, rollback: pointer.rollback)
        } else {
            let previous = pointer.active ?? pointer.rollback
            next = Pointer(
                active: generation, lastGreen: confirm ? generation : pointer.lastGreen,
                rollback: previous == generation ? nil : previous)
        }
        try touch(key, generation, activated: true)
        try writePointer(key, next)
        try checkpoint("pointer_committed")
    }

    private func setPinned(_ bundleID: String, _ generation: String, _ pinned: Bool) throws {
        let key = Self.bundleKey(bundleID: bundleID)
        try locked(key) {
            guard var record = readGenerationRecord(key, generation) else {
                throw OpenMedBundleCacheError(.generationNotReady)
            }
            record.pinned = pinned
            try writeGenerationRecord(generationRecordFile(key, generation), record)
        }
    }

    private func createLease(_ key: String, _ generation: String, verification: OpenMedBundleOpenVerification) throws
        -> OpenMedBundleLease
    {
        guard let record = readGenerationRecord(key, generation) else {
            throw OpenMedBundleCacheError(.generationNotReady)
        }
        let payload = generationDirectory(key, generation).appendingPathComponent(Self.payload)
        var reason = metadataCheck(key, generation)
        if reason == nil, verification == .full {
            reason = verifyPayload(payload, declaredFiles(payload) ?? [], full: true)
        }
        if let reason {
            try quarantineAndRepair(key, generation, reason: reason.rawValue)
            throw OpenMedBundleCacheError(.generationCorrupt, bundleReason: reason)
        }
        let id = UUID().uuidString.lowercased()
        let file = leaseFile(key, generation, id)
        try fileManager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try writeJSON(
            file,
            .object([
                ("schema", .string(Self.leaseSchema)), ("schema_version", .int(1)), ("lease", .string(id)),
                ("generation", .string(generation)), ("owner", .string(options.ownerID)),
                ("expires_at_ms", .int(now() + options.leaseTTLMilliseconds)),
            ]))
        leaseLock.lock()
        heldLeases.insert(id)
        leaseLock.unlock()
        try touch(key, generation, activated: nil)
        return OpenMedBundleLease(
            id: id, bundleKey: key, generation: generation, payloadDirectory: payload, filesChecked: record.fileCount,
            cache: self)
    }

    // MARK: - Quota and eviction

    private struct Candidate {
        let key: String
        let generation: String
        let lastAccessed: Int64
        let size: Int64
    }

    private func planQuota(additionalBytes: Int64, exclude: String?) -> (candidates: [Candidate], satisfied: Bool) {
        var used: Int64 = 0
        var candidates: [Candidate] = []
        for key in bundleKeys() {
            let pointer = readPointer(key)
            for generation in readyGenerations(key) {
                guard let record = readGenerationRecord(key, generation) else { continue }
                used += record.totalBytes
                if generation != exclude, protections(key, generation, pointer, record).isEmpty {
                    candidates.append(
                        Candidate(key: key, generation: generation, lastAccessed: record.lastAccessed, size: record.totalBytes))
                }
            }
        }
        candidates.sort {
            $0.lastAccessed != $1.lastAccessed
                ? $0.lastAccessed < $1.lastAccessed : $0.generation.utf8.lexicographicallyPrecedes($1.generation.utf8)
        }
        var chosen: [Candidate] = []
        for candidate in candidates {
            if used + additionalBytes <= options.quotaBytes { break }
            chosen.append(candidate)
            used -= candidate.size
        }
        return (chosen, used + additionalBytes <= options.quotaBytes)
    }

    private func applyQuota(additionalBytes: Int64, exclude: String?, heldKey: String?) throws -> OpenMedBundleQuotaDecision {
        let plan = planQuota(additionalBytes: additionalBytes, exclude: exclude)
        let evicted = try applyPlan(plan.candidates, heldKey: heldKey)
        let used = usedBytes()
        return OpenMedBundleQuotaDecision(
            quotaBytes: options.quotaBytes, usedBytes: used, additionalBytes: additionalBytes, evicted: evicted,
            satisfied: used + additionalBytes <= options.quotaBytes)
    }

    private func applyPlan(_ candidates: [Candidate], heldKey: String?) throws -> [String] {
        var evicted: [String] = []
        for candidate in candidates {
            let evict: () throws -> Void = { [self] in
                guard let record = readGenerationRecord(candidate.key, candidate.generation),
                    protections(candidate.key, candidate.generation, readPointer(candidate.key), record).isEmpty
                else { return }
                let trash = trashEntry(candidate.key, candidate.generation)
                try move(generationDirectory(candidate.key, candidate.generation), trash)
                try checkpoint("eviction_committed")
                try? fileManager.removeItem(at: trash)
                try? fileManager.removeItem(at: leasesRoot(candidate.key).appendingPathComponent(candidate.generation))
                evicted.append(candidate.generation)
            }
            if candidate.key == heldKey {
                try evict()
            } else {
                // Wait only when no bundle lock is held, so two acquisitions
                // evicting each other's generations cannot deadlock.
                let wait = heldKey == nil ? options.lockTimeoutMilliseconds : 0
                if let held = try lockBundle(candidate.key, timeoutMilliseconds: wait) {
                    try held.guarded(evict)
                }
            }
        }
        return evicted
    }

    private func usedBytes() -> Int64 {
        bundleKeys().reduce(Int64(0)) { total, key in
            total + readyGenerations(key).reduce(Int64(0)) { $0 + (readGenerationRecord(key, $1)?.totalBytes ?? 0) }
        }
    }

    private func protections(_ key: String, _ generation: String, _ pointer: Pointer, _ record: GenerationRecord)
        -> [OpenMedBundleProtection]
    {
        var result: [OpenMedBundleProtection] = []
        if pointer.active == generation { result.append(.active) }
        if pointer.lastGreen == generation { result.append(.lastGreen) }
        if pointer.rollback == generation { result.append(.rollback) }
        if record.pinned { result.append(.pinned) }
        if isLeased(key, generation) { result.append(.leased) }
        return result
    }

    private func isLeased(_ key: String, _ generation: String) -> Bool {
        let now = now()
        return children(leasesRoot(key).appendingPathComponent(generation)).contains { file in
            guard let lease = readLease(file) else { return false }
            return (lease.owner == options.ownerID && holds(lease.id)) || lease.expiresAt > now
        }
    }

    private func holds(_ id: String) -> Bool {
        leaseLock.lock()
        defer { leaseLock.unlock() }
        return heldLeases.contains(id)
    }

    private func state(_ key: String, _ generation: String, _ pointer: Pointer, _ record: GenerationRecord)
        -> OpenMedBundleGenerationState
    {
        let protections = protections(key, generation, pointer, record)
        if protections.contains(.active) { return .active }
        if protections.isEmpty { return .evictable }
        return record.activated ? .superseded : .ready
    }

    // MARK: - Recovery and snapshots

    private func recoverLocked(_ key: String, stale: Bool, into actions: inout [OpenMedBundleRecoveryStep]) throws {
        func report(_ action: OpenMedBundleRecoveryAction, _ generation: String?) {
            actions.append(OpenMedBundleRecoveryStep(action: action, bundleKey: key, generation: generation))
        }
        if stale { report(.staleLockCleared, nil) }
        for entry in children(trashRoot(key)) {
            try? fileManager.removeItem(at: entry)
            let name = String(entry.lastPathComponent.split(separator: ".").first ?? "")
            report(.trashPurged, Self.isGeneration(name) ? name : nil)
        }
        for entry in children(quarantineRoot(key)) {
            try discard(key, entry)
            report(.quarantinePurged, Self.isGeneration(entry.lastPathComponent) ? entry.lastPathComponent : nil)
        }
        let ready = Set(readyGenerations(key))
        for entry in children(stagingRoot(key)) {
            let name = entry.lastPathComponent
            let record = readStagingRecord(entry)
            if !Self.isGeneration(name) || record == nil {
                try discard(key, entry)
                report(.stagingOrphaned, Self.isGeneration(name) ? name : nil)
            } else if ready.contains(name) {
                try discard(key, entry)
            } else if let record, now() - record.updatedAt > options.stagingMaxAgeMilliseconds {
                try discard(key, entry)
                report(.stagingExpired, name)
            } else if var record {
                if record.phase != Self.phaseStaging {
                    record.phase = Self.phaseStaging
                    try writeStagingRecord(entry, record)
                }
                report(.stagingResumable, name)
            }
        }
        for entry in children(generationsRoot(key)) {
            let name = entry.lastPathComponent
            try? fileManager.removeItem(at: entry.appendingPathComponent(Self.stagingRecord))
            if !Self.isGeneration(name) {
                try discard(key, entry)
            } else if metadataCheck(key, name) != nil {
                try quarantine(key, entry, name, reason: Self.payloadMismatch)
                report(.generationQuarantined, name)
            }
        }
        if try repairPointer(key) { report(.pointerRepaired, nil) }
        let now = now()
        for directory in children(leasesRoot(key)) {
            for file in children(directory) {
                let lease = readLease(file)
                let mine = lease.map { $0.owner == options.ownerID && holds($0.id) } ?? false
                if lease == nil || (!mine && (lease?.expiresAt ?? 0) <= now) {
                    try? fileManager.removeItem(at: file)
                    let name = directory.lastPathComponent
                    report(.leaseExpired, Self.isGeneration(name) ? name : nil)
                }
            }
            if children(directory).isEmpty { try? fileManager.removeItem(at: directory) }
        }
    }

    private func snapshotLocked(_ key: String) -> OpenMedBundleCacheSnapshot {
        let pointer = readPointer(key)
        var generations: [OpenMedBundleGenerationSnapshot] = []
        var used: Int64 = 0
        for generation in readyGenerations(key) {
            guard let record = readGenerationRecord(key, generation) else { continue }
            used += record.totalBytes
            generations.append(
                OpenMedBundleGenerationSnapshot(
                    generation: generation, state: state(key, generation, pointer, record), totalBytes: record.totalBytes,
                    verifiedBytes: record.totalBytes, protections: protections(key, generation, pointer, record)))
        }
        for entry in children(stagingRoot(key)) {
            let name = entry.lastPathComponent
            guard Self.isGeneration(name), !generations.contains(where: { $0.generation == name }),
                let record = readStagingRecord(entry),
                let files = declaredFiles(entry.appendingPathComponent(Self.payload))
            else { continue }
            let verified = files.indices.reduce(Int64(0)) {
                $0 + verifiedBytes($1 < record.files.count ? record.files[$1].count : 0, files[$1].size)
            }
            generations.append(
                OpenMedBundleGenerationSnapshot(
                    generation: name, state: .resumable, totalBytes: files.reduce(Int64(0)) { $0 + $1.size },
                    verifiedBytes: verified, protections: []))
        }
        let quarantined = children(quarantineRoot(key)).filter { Self.isGeneration($0.lastPathComponent) }.map { entry in
            var reason = "unknown"
            if case .object(let members)? = readJSON(entry.appendingPathComponent(Self.quarantineRecord)),
                case .string(let value)? = PortableBundleJSON.member(members, "reason")
            {
                reason = value
            }
            return OpenMedBundleQuarantineEntry(generation: entry.lastPathComponent, reason: reason)
        }
        return OpenMedBundleCacheSnapshot(
            bundleKey: key, active: pointer.active, lastGreen: pointer.lastGreen, rollback: pointer.rollback,
            generations: generations.sorted { $0.generation.utf8.lexicographicallyPrecedes($1.generation.utf8) },
            quarantined: quarantined, usedBytes: used)
    }

    private func quarantineAndRepair(_ key: String, _ generation: String, reason: String) throws {
        try quarantine(key, generationDirectory(key, generation), generation, reason: reason)
        _ = try repairPointer(key)
    }

    /// Falls an invalid active generation back to last green, then rollback.
    private func repairPointer(_ key: String) throws -> Bool {
        let pointer = readPointer(key)
        let valid = { (generation: String?) -> Bool in
            generation.map { self.readGenerationRecord(key, $0) != nil } ?? false
        }
        let lastGreen = valid(pointer.lastGreen) ? pointer.lastGreen : nil
        var rollback = valid(pointer.rollback) ? pointer.rollback : nil
        let active = valid(pointer.active) ? pointer.active : (lastGreen ?? rollback)
        if rollback == active { rollback = nil }
        let next = Pointer(active: active, lastGreen: lastGreen, rollback: rollback)
        guard next != pointer else { return false }
        try writePointer(key, next)
        return true
    }

    // MARK: - Records

    private struct Pointer: Equatable {
        var active: String?
        var lastGreen: String?
        var rollback: String?
    }

    private struct GenerationRecord {
        let contentID: String
        let sequence: Int64
        let totalBytes: Int64
        let fileCount: Int
        var activated: Bool
        var pinned: Bool
        let verifiedAt: Int64
        var lastAccessed: Int64
    }

    private struct StagingRecord {
        let contentID: String
        let sourceDigest: String
        let trustDigest: String
        let chunkBytes: Int
        var phase: String
        let createdAt: Int64
        let updatedAt: Int64
        let files: [[String]]
    }

    private struct LeaseRecord {
        let id: String
        let owner: String
        let expiresAt: Int64
    }

    struct DeclaredFile {
        let path: String
        let size: Int64
        let sha256: String
    }

    struct ManifestHeader {
        let bundleID: String
        let sequence: Int64
        let files: [DeclaredFile]

        init?(_ manifest: Data) {
            guard case .object(let root)? = try? PortableBundleJSON.parse(manifest),
                case .object(let bundle)? = PortableBundleJSON.member(root, "bundle"),
                case .string(let bundleID)? = PortableBundleJSON.member(bundle, "id"),
                case .int(let sequence)? = PortableBundleJSON.member(bundle, "sequence"),
                case .array(let items)? = PortableBundleJSON.member(root, "files")
            else { return nil }
            var files: [DeclaredFile] = []
            for item in items {
                guard case .object(let entry) = item,
                    case .string(let path)? = PortableBundleJSON.member(entry, "path"),
                    case .int(let size)? = PortableBundleJSON.member(entry, "size_bytes"),
                    case .string(let digest)? = PortableBundleJSON.member(entry, "sha256")
                else { return nil }
                files.append(DeclaredFile(path: path, size: size, sha256: digest))
            }
            self.bundleID = bundleID
            self.sequence = sequence
            self.files = files
        }
    }

    private static func string(_ members: [(String, PortableBundleJSON)], _ key: String) -> String? {
        if case .string(let value)? = PortableBundleJSON.member(members, key) { return value }
        return nil
    }

    private static func int(_ members: [(String, PortableBundleJSON)], _ key: String) -> Int64? {
        if case .int(let value)? = PortableBundleJSON.member(members, key) { return value }
        return nil
    }

    private static func bool(_ members: [(String, PortableBundleJSON)], _ key: String) -> Bool? {
        if case .bool(let value)? = PortableBundleJSON.member(members, key) { return value }
        return nil
    }

    private func readPointer(_ key: String) -> Pointer {
        guard case .object(let members)? = readJSON(bundleDirectory(key).appendingPathComponent(Self.pointer)) else {
            return Pointer()
        }
        func field(_ name: String) -> String? { Self.string(members, name).flatMap { Self.isGeneration($0) ? $0 : nil } }
        return Pointer(active: field("active"), lastGreen: field("last_green"), rollback: field("rollback"))
    }

    private func writePointer(_ key: String, _ pointer: Pointer) throws {
        func ref(_ value: String?) -> PortableBundleJSON { value.map { .string($0) } ?? .null }
        try writeJSON(
            bundleDirectory(key).appendingPathComponent(Self.pointer),
            .object([
                ("schema", .string(Self.pointerSchema)), ("schema_version", .int(1)), ("active", ref(pointer.active)),
                ("last_green", ref(pointer.lastGreen)), ("rollback", ref(pointer.rollback)), ("updated_at_ms", .int(now())),
            ]))
    }

    private func readGenerationRecord(_ key: String, _ generation: String) -> GenerationRecord? {
        guard Self.isGeneration(generation),
            case .object(let members)? = readJSON(generationRecordFile(key, generation)),
            Self.string(members, "schema") == Self.generationSchema, Self.string(members, "state") == "ready",
            let contentID = Self.string(members, "content_id"), let sequence = Self.int(members, "sequence"),
            let total = Self.int(members, "total_bytes"), let count = Self.int(members, "file_count"),
            let activated = Self.bool(members, "activated"), let pinned = Self.bool(members, "pinned"),
            let verifiedAt = Self.int(members, "verified_at_ms"), let lastAccessed = Self.int(members, "last_accessed_ms"),
            Self.generationID(contentID: contentID) == generation
        else { return nil }
        return GenerationRecord(
            contentID: contentID, sequence: sequence, totalBytes: total, fileCount: Int(count), activated: activated,
            pinned: pinned, verifiedAt: verifiedAt, lastAccessed: lastAccessed)
    }

    private func writeGenerationRecord(_ file: URL, _ record: GenerationRecord) throws {
        try writeJSON(
            file,
            .object([
                ("schema", .string(Self.generationSchema)), ("schema_version", .int(1)), ("state", .string("ready")),
                ("generation", .string(Self.generationID(contentID: record.contentID))),
                ("content_id", .string(record.contentID)), ("sequence", .int(record.sequence)),
                ("total_bytes", .int(record.totalBytes)), ("file_count", .int(Int64(record.fileCount))),
                ("activated", .bool(record.activated)), ("pinned", .bool(record.pinned)),
                ("verified_at_ms", .int(record.verifiedAt)), ("last_accessed_ms", .int(record.lastAccessed)),
            ]))
    }

    private func touch(_ key: String, _ generation: String, activated: Bool?) throws {
        guard var record = readGenerationRecord(key, generation) else { return }
        if let activated { record.activated = activated }
        record.lastAccessed = now()
        try writeGenerationRecord(generationRecordFile(key, generation), record)
    }

    private func readStagingRecord(_ stage: URL) -> StagingRecord? {
        guard case .object(let members)? = readJSON(stage.appendingPathComponent(Self.stagingRecord)),
            Self.string(members, "schema") == Self.stagingSchema,
            let contentID = Self.string(members, "content_id"), let source = Self.string(members, "source_digest"),
            let trust = Self.string(members, "trust_digest"), let chunk = Self.int(members, "chunk_bytes"),
            let phase = Self.string(members, "phase"), let created = Self.int(members, "created_at_ms"),
            let updated = Self.int(members, "updated_at_ms"),
            case .array(let rawFiles)? = PortableBundleJSON.member(members, "files")
        else { return nil }
        var files: [[String]] = []
        for raw in rawFiles {
            guard case .array(let items) = raw else { return nil }
            var digests: [String] = []
            for item in items {
                guard case .string(let digest) = item else { return nil }
                digests.append(digest)
            }
            files.append(digests)
        }
        return StagingRecord(
            contentID: contentID, sourceDigest: source, trustDigest: trust, chunkBytes: Int(chunk), phase: phase,
            createdAt: created, updatedAt: updated, files: files)
    }

    private func writeStagingRecord(_ stage: URL, _ record: StagingRecord) throws {
        try fileManager.createDirectory(at: stage, withIntermediateDirectories: true)
        try writeJSON(
            stage.appendingPathComponent(Self.stagingRecord),
            .object([
                ("schema", .string(Self.stagingSchema)), ("schema_version", .int(1)),
                ("content_id", .string(record.contentID)), ("source_digest", .string(record.sourceDigest)),
                ("trust_digest", .string(record.trustDigest)), ("chunk_bytes", .int(Int64(record.chunkBytes))),
                ("phase", .string(record.phase)), ("created_at_ms", .int(record.createdAt)),
                ("updated_at_ms", .int(record.updatedAt)),
                ("files", .array(record.files.map { file in .array(file.map { .string($0) }) })),
            ]))
    }

    private func readLease(_ file: URL) -> LeaseRecord? {
        guard case .object(let members)? = readJSON(file), let id = Self.string(members, "lease"),
            let owner = Self.string(members, "owner"), let expires = Self.int(members, "expires_at_ms")
        else { return nil }
        return LeaseRecord(id: id, owner: owner, expiresAt: expires)
    }

    private func declaredFiles(_ payload: URL) -> [DeclaredFile]? {
        guard let data = try? Data(contentsOf: payload.appendingPathComponent(PortableBundle.manifestFilename)) else {
            return nil
        }
        return ManifestHeader(data)?.files
    }

    /// Cheap check: the record, the manifest identity, the declared files, and their sizes.
    private func metadataCheck(_ key: String, _ generation: String) -> PortableBundleReason? {
        guard let record = readGenerationRecord(key, generation) else { return .fileMissing }
        let payload = generationDirectory(key, generation).appendingPathComponent(Self.payload)
        guard let manifest = try? Data(contentsOf: payload.appendingPathComponent(PortableBundle.manifestFilename)) else {
            return .fileMissing
        }
        guard PortableBundle.contentID(of: manifest) == record.contentID else { return .fileDigestMismatch }
        guard let files = ManifestHeader(manifest)?.files else { return .manifestMalformed }
        guard files.count == record.fileCount, files.reduce(Int64(0), { $0 + $1.size }) == record.totalBytes else {
            return .fileSizeMismatch
        }
        return verifyPayload(payload, files, full: false)
    }

    private func readyGenerations(_ key: String) -> [String] {
        children(generationsRoot(key)).map(\.lastPathComponent).filter { readGenerationRecord(key, $0) != nil }
    }

    // MARK: - Filesystem

    private func bundleKeys() -> [String] {
        children(root).map(\.lastPathComponent).filter { Self.isBundle($0) }
    }

    private func bundleDirectory(_ key: String) -> URL { root.appendingPathComponent(key, isDirectory: true) }
    private func stagingRoot(_ key: String) -> URL { bundleDirectory(key).appendingPathComponent("staging") }
    private func generationsRoot(_ key: String) -> URL { bundleDirectory(key).appendingPathComponent("generations") }
    private func quarantineRoot(_ key: String) -> URL { bundleDirectory(key).appendingPathComponent("quarantine") }
    private func trashRoot(_ key: String) -> URL { bundleDirectory(key).appendingPathComponent("trash") }
    private func leasesRoot(_ key: String) -> URL { bundleDirectory(key).appendingPathComponent("leases") }
    private func stagingDirectory(_ key: String, _ generation: String) -> URL {
        stagingRoot(key).appendingPathComponent(generation)
    }
    private func generationDirectory(_ key: String, _ generation: String) -> URL {
        generationsRoot(key).appendingPathComponent(generation)
    }
    private func quarantineDirectory(_ key: String, _ generation: String) -> URL {
        quarantineRoot(key).appendingPathComponent(generation)
    }
    private func generationRecordFile(_ key: String, _ generation: String) -> URL {
        generationDirectory(key, generation).appendingPathComponent(Self.generationRecord)
    }
    private func leaseFile(_ key: String, _ generation: String, _ id: String) -> URL {
        leasesRoot(key).appendingPathComponent(generation).appendingPathComponent(id + ".json")
    }

    private func trashEntry(_ key: String, _ generation: String) -> URL {
        try? fileManager.createDirectory(at: trashRoot(key), withIntermediateDirectories: true)
        return trashRoot(key).appendingPathComponent(generation + "." + UUID().uuidString.lowercased())
    }

    private func children(_ directory: URL) -> [URL] {
        let names = (try? fileManager.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { !$0.hasPrefix(".") }
            .sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
            .map { directory.appendingPathComponent($0) }
    }

    private func exists(_ url: URL) -> Bool { fileManager.fileExists(atPath: url.path) }

    private func isRegularFile(_ url: URL) -> Bool {
        (try? fileManager.attributesOfItem(atPath: url.path)[.type]) as? FileAttributeType == .typeRegular
    }

    private func fileSize(_ url: URL) -> Int64 {
        ((try? fileManager.attributesOfItem(atPath: url.path)[.size]) as? NSNumber)?.int64Value ?? -1
    }

    private func availableCapacity() -> Int64 {
        let values = try? root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage ?? Int64.max
    }

    /// Moves `entry` out of reach with one rename, then deletes it.
    private func discard(_ key: String, _ entry: URL) throws {
        guard exists(entry) else { return }
        let name = String(entry.lastPathComponent.split(separator: ".").first ?? "entry")
        let trash = trashEntry(key, name)
        try move(entry, trash)
        try? fileManager.removeItem(at: trash)
    }

    private func quarantine(_ key: String, _ entry: URL, _ generation: String, reason: String) throws {
        guard exists(entry) else { return }
        try writeJSON(
            entry.appendingPathComponent(Self.quarantineRecord),
            .object([
                ("schema", .string(Self.quarantineSchema)), ("schema_version", .int(1)),
                ("generation", .string(generation)), ("reason", .string(reason)), ("quarantined_at_ms", .int(now())),
            ]))
        let destination = quarantineDirectory(key, generation)
        if exists(destination) { try discard(key, destination) }
        try move(entry, destination)
    }

    private func move(_ source: URL, _ destination: URL) throws {
        try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        try fileManager.moveItem(at: source, to: destination)
    }

    private func writeJSON(_ file: URL, _ value: PortableBundleJSON) throws {
        try fileManager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(PortableBundleJSON.encode(value)).write(to: file, options: .atomic)
    }

    private func readJSON(_ file: URL) -> PortableBundleJSON? {
        guard let data = try? Data(contentsOf: file) else { return nil }
        return try? PortableBundleJSON.parse(data)
    }

    private func now() -> Int64 { options.now() }

    private func checkpoint(_ name: String) throws { try checkpointHook?(name) }

    // MARK: - Locks

    /// Process-wide mutex per lock file, so threads never contend on `flock`.
    private final class MutexRegistry: @unchecked Sendable {
        private let lock = NSLock()
        private var mutexes: [String: NSLock] = [:]

        func mutex(for path: String) -> NSLock {
            lock.lock()
            defer { lock.unlock() }
            if let existing = mutexes[path] { return existing }
            let created = NSLock()
            mutexes[path] = created
            return created
        }
    }

    private static let registry = MutexRegistry()

    private static func mutex(for path: String) -> NSLock { registry.mutex(for: path) }

    private final class HeldLock {
        private let mutex: NSLock
        private let descriptor: Int32
        private let record: URL
        let stale: Bool
        private var released = false

        init(mutex: NSLock, descriptor: Int32, record: URL, stale: Bool) {
            self.mutex = mutex
            self.descriptor = descriptor
            self.record = record
            self.stale = stale
        }

        func release(keepRecord: Bool) {
            guard !released else { return }
            released = true
            if !keepRecord { try? FileManager.default.removeItem(at: record) }
            _ = flock(descriptor, LOCK_UN)
            _ = Darwin.close(descriptor)
            mutex.unlock()
        }

        func guarded<T>(_ block: () throws -> T) throws -> T {
            do {
                let result = try block()
                release(keepRecord: false)
                return result
            } catch let crash as OpenMedBundleCacheCrash {
                release(keepRecord: true)
                throw crash
            } catch {
                release(keepRecord: false)
                throw error
            }
        }
    }

    private func locked<T>(_ key: String, _ block: () throws -> T) throws -> T {
        guard let held = try lockBundle(key, timeoutMilliseconds: options.lockTimeoutMilliseconds) else {
            throw OpenMedBundleCacheError(.lockTimeout)
        }
        return try held.guarded(block)
    }

    private func lockBundle(_ key: String, timeoutMilliseconds: Int64) throws -> HeldLock? {
        let directory = bundleDirectory(key)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let path = directory.appendingPathComponent("lock").path
        let mutex = Self.mutex(for: path)
        let deadline = Date(timeIntervalSinceNow: Double(timeoutMilliseconds) / 1000)
        guard mutex.lock(before: deadline) else { return nil }
        let descriptor = Darwin.open(path, O_RDWR | O_CREAT, 0o600)
        guard descriptor >= 0 else {
            mutex.unlock()
            throw OpenMedBundleCacheError(.storageIO)
        }
        while flock(descriptor, LOCK_EX | LOCK_NB) != 0 {
            guard Date() < deadline else {
                _ = Darwin.close(descriptor)
                mutex.unlock()
                return nil
            }
            usleep(10_000)
        }
        let record = directory.appendingPathComponent("lock.json")
        let stale = exists(record)
        do {
            try writeJSON(
                record,
                .object([
                    ("schema", .string(Self.lockSchema)), ("schema_version", .int(1)),
                    ("owner", .string(options.ownerID)), ("acquired_at_ms", .int(now())),
                ]))
        } catch {
            _ = flock(descriptor, LOCK_UN)
            _ = Darwin.close(descriptor)
            mutex.unlock()
            throw OpenMedBundleCacheError(.storageIO)
        }
        return HeldLock(mutex: mutex, descriptor: descriptor, record: record, stale: stale)
    }

    // MARK: - Results

    private func success(
        _ key: String, _ generation: String, _ totalBytes: Int64, _ fileCount: Int, fromCache: Bool, resumed: Bool = false,
        resumeReason: OpenMedBundleResumeReason? = nil, transferred: Int64 = 0, evicted: [String] = []
    ) -> OpenMedBundleAcquireResult {
        let pointer = readPointer(key)
        return OpenMedBundleAcquireResult(
            ok: true, reason: nil, bundleReason: nil, bundleKey: key, generation: generation,
            state: readGenerationRecord(key, generation).map { state(key, generation, pointer, $0) },
            fromCache: fromCache, resumed: resumed, resumeReason: resumeReason, bytesTransferred: transferred,
            verifiedBytes: totalBytes, filesVerified: fileCount, evicted: evicted)
    }

    private func failure(
        _ reason: OpenMedBundleCacheReason, _ bundleReason: PortableBundleReason?, _ key: String?, _ generation: String?
    ) -> OpenMedBundleAcquireResult {
        OpenMedBundleAcquireResult(
            ok: false, reason: reason, bundleReason: bundleReason, bundleKey: key, generation: generation, state: nil,
            fromCache: false, resumed: false, resumeReason: nil, bytesTransferred: 0, verifiedBytes: 0, filesVerified: 0,
            evicted: [])
    }

    // MARK: - Constants and hashing

    private static let payload = "payload"
    private static let pointer = "pointer.json"
    private static let stagingRecord = "staging.json"
    private static let generationRecord = "generation.json"
    private static let quarantineRecord = "quarantine.json"
    private static let phaseStaging = "staging"
    private static let phaseVerifying = "verifying"
    private static let payloadMismatch = "payload_mismatch"
    private static let rolledBack = "rolled_back"
    private static let pointerSchema = "openmed.bundle_cache.pointer"
    private static let generationSchema = "openmed.bundle_cache.generation"
    private static let stagingSchema = "openmed.bundle_cache.staging"
    private static let quarantineSchema = "openmed.bundle_cache.quarantine"
    private static let leaseSchema = "openmed.bundle_cache.lease"
    private static let lockSchema = "openmed.bundle_cache.lock"

    private static func isGeneration(_ name: String) -> Bool { isOpaque(name, prefix: "g-") }
    private static func isBundle(_ name: String) -> Bool { isOpaque(name, prefix: "b-") }

    private static func isOpaque(_ name: String, prefix: String) -> Bool {
        guard name.hasPrefix(prefix), name.utf8.count == 34 else { return false }
        return name.utf8.dropFirst(2).allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
    }

    static func sha256Text(_ data: Data) -> String { "sha256:" + PortableBundle.sha256Hex(data) }

    static func sha256File(_ url: URL) -> String? {
        #if canImport(CryptoKit)
            guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? handle.close() }
            var hasher = SHA256()
            while let chunk = try? handle.read(upToCount: 1 << 20), !chunk.isEmpty {
                hasher.update(data: chunk)
            }
            return "sha256:" + hasher.finalize().map { String(format: "%02x", $0) }.joined()
        #else
            return nil
        #endif
    }
}
