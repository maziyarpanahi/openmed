import Foundation

// Managed, offline import of signed portable bundles into an
// `OpenMedBundleCache` (`docs/mobile/managed-model-bundles.md`).

/// Outcome of one managed import.
public enum OpenMedManagedImportOutcome: String, CaseIterable, Sendable {
    /// Verified and committed as a new generation; not activated (stage policy).
    case imported
    /// The generation was already committed (and active, when activation was requested).
    case alreadyPresent = "already_present"
    /// The generation is now active.
    case activated
    /// Nothing to import yet: no managed configuration, or the delivery is pending.
    case deferred
    /// The activation check failed; the prior last-green generation is active again.
    case rolledBack = "rolled_back"
    /// The configuration version, handoff kind, or archive format is not supported.
    case unsupported
    /// Refused before activation; the cache is unchanged except for quarantine.
    case rejected
}

/// Stable, value-free reason codes of a managed import.
public enum OpenMedManagedImportReason: String, CaseIterable, Sendable {
    case managedConfigAbsent = "managed_config_absent"
    case handoffPending = "handoff_pending"
    case configVersionUnsupported = "config_version_unsupported"
    case handoffUnsupported = "handoff_unsupported"
    case archiveFormatUnsupported = "archive_format_unsupported"
    case configInvalid = "config_invalid"
    case trustPolicyMissing = "trust_policy_missing"
    case trustPolicyInvalid = "trust_policy_invalid"
    case trustAnchorUnpinned = "trust_anchor_unpinned"
    case locationUnapproved = "location_unapproved"
    case handoffUnreadable = "handoff_unreadable"
    case budgetArchiveBytes = "budget_archive_bytes"
    case budgetFileCount = "budget_file_count"
    case budgetTotalBytes = "budget_total_bytes"
    case budgetCompressionRatio = "budget_compression_ratio"
    case archiveMalformed = "archive_malformed"
    case entryUnsafe = "entry_unsafe"
    case entryDuplicate = "entry_duplicate"
    case bundleRejected = "bundle_rejected"
    case bundleNotExpected = "bundle_not_expected"
    case verificationFailed = "verification_failed"
    case storagePressure = "storage_pressure"
    case quotaExceeded = "quota_exceeded"
    case transferInterrupted = "transfer_interrupted"
    case cancelled
    case lockTimeout = "lock_timeout"
    case storageIO = "storage_io"
    case activationCheckFailed = "activation_check_failed"

    /// Outcome a refusal with this reason reports.
    var refusalOutcome: OpenMedManagedImportOutcome {
        switch self {
        case .managedConfigAbsent, .handoffPending: return .deferred
        case .configVersionUnsupported, .handoffUnsupported, .archiveFormatUnsupported: return .unsupported
        default: return .rejected
        }
    }
}

/// How an imported generation is activated.
public enum OpenMedManagedActivationPolicy: String, CaseIterable, Sendable {
    /// Activate and, after an optional activation check, mark last green.
    case activate
    /// Activate on probation; the host calls `confirm(bundleID:)` later.
    case probation
    /// Commit only; the active generation does not change.
    case stage
}

/// Kind of local platform handoff that delivered the bundle.
public enum OpenMedManagedHandoffKind: String, Sendable {
    case file, directory, stream, bookmark
}

/// An approved local handle to a bundle. Only `file:` URLs are accepted; the
/// importer never opens a network connection.
public enum OpenMedManagedBundleHandoff {
    /// A ZIP archive file or an extracted bundle directory. Security-scoped
    /// access is started and stopped around the import.
    case file(URL)
    /// Bookmark data (for example from a document picker) resolving to a file URL.
    case bookmark(Data)
    /// Archive bytes; spooled into the importer's private area first.
    case stream(InputStream)
}

/// Resource budgets enforced before any byte reaches the cache.
public struct OpenMedManagedImportLimits: Sendable, Equatable {
    /// Largest archive file or stream, in bytes.
    public var maxArchiveBytes: Int64
    /// Largest number of archive entries (files and directories).
    public var maxEntries: Int
    /// Largest sum of uncompressed entry sizes.
    public var maxTotalBytes: Int64
    /// Largest uncompressed-to-compressed ratio of one DEFLATE entry.
    public var maxCompressionRatio: Int64

    public init(
        maxArchiveBytes: Int64 = 4_294_967_295, maxEntries: Int = 4_096, maxTotalBytes: Int64 = 4_294_967_295,
        maxCompressionRatio: Int64 = 200
    ) {
        self.maxArchiveBytes = max(0, maxArchiveBytes)
        self.maxEntries = max(1, maxEntries)
        self.maxTotalBytes = max(0, maxTotalBytes)
        self.maxCompressionRatio = max(1, maxCompressionRatio)
    }
}

/// Error carrying only a reason code and the outcome it maps to.
public struct OpenMedManagedImportError: Error, Equatable, Sendable, CustomStringConvertible {
    public let reason: OpenMedManagedImportReason
    /// Portable-bundle verifier reason for `bundle_rejected` and `verification_failed`.
    public let bundleReason: PortableBundleReason?

    public init(_ reason: OpenMedManagedImportReason, bundleReason: PortableBundleReason? = nil) {
        self.reason = reason
        self.bundleReason = bundleReason
    }

    public var outcome: OpenMedManagedImportOutcome { reason.refusalOutcome }
    public var description: String { reason.rawValue }
}

/// A parsed managed configuration (Apple managed app configuration or Android
/// app restrictions). Keys are shared by both platforms.
public struct OpenMedManagedBundleConfiguration: Sendable {
    /// Supported configuration version.
    public static let version = 1
    public static let versionKey = "openmed_bundle_config_version"
    public static let trustPolicyKey = "openmed_bundle_trust_policy"
    public static let keyPinsKey = "openmed_bundle_key_pins"
    public static let locationKey = "openmed_bundle_location"
    public static let expectedIDKey = "openmed_bundle_expected_id"
    public static let expectedContentIDKey = "openmed_bundle_expected_content_id"
    public static let minimumSequenceKey = "openmed_bundle_min_sequence"
    public static let activationKey = "openmed_bundle_activation"
    /// Every key the importer reads; other `openmed_bundle_` keys are refused.
    public static let keys: [String] = [
        versionKey, trustPolicyKey, keyPinsKey, locationKey, expectedIDKey, expectedContentIDKey, minimumSequenceKey,
        activationKey,
    ]
    static let prefix = "openmed_bundle_"

    /// Trust roots delivered by the administrator. Public keys only.
    public var trustPolicy: PortableBundleTrustPolicy
    /// `sha256:` fingerprints of raw Ed25519 public keys; empty means no pinning.
    public var keyPins: Set<String>
    /// Where the delivered bundle is found (a `file:` URL or a relative path).
    public var location: String?
    public var expectedBundleID: String?
    public var expectedContentID: String?
    public var minimumSequence: Int64?
    public var activation: OpenMedManagedActivationPolicy?

    public init(
        trustPolicy: PortableBundleTrustPolicy, keyPins: Set<String> = [], location: String? = nil,
        expectedBundleID: String? = nil, expectedContentID: String? = nil, minimumSequence: Int64? = nil,
        activation: OpenMedManagedActivationPolicy? = nil
    ) {
        self.trustPolicy = trustPolicy
        self.keyPins = keyPins
        self.location = location
        self.expectedBundleID = expectedBundleID
        self.expectedContentID = expectedContentID
        self.minimumSequence = minimumSequence
        self.activation = activation
    }

    /// Parses managed values. Keys without the `openmed_bundle_` prefix are
    /// ignored; the checks run in the documented order.
    ///
    /// - Throws: `OpenMedManagedImportError` with `config_invalid`,
    ///   `config_version_unsupported`, `trust_policy_missing`, or
    ///   `trust_policy_invalid`.
    public init(managedValues values: [String: Any]) throws {
        let invalid = OpenMedManagedImportError(.configInvalid)
        guard let rawVersion = values[Self.versionKey] else { throw invalid }
        guard let version = Self.integer(rawVersion) else { throw invalid }
        guard version == Int64(Self.version) else { throw OpenMedManagedImportError(.configVersionUnsupported) }
        let known = Set(Self.keys)
        guard values.keys.allSatisfy({ !$0.hasPrefix(Self.prefix) || known.contains($0) }) else { throw invalid }
        guard let rawPolicy = values[Self.trustPolicyKey] else { throw OpenMedManagedImportError(.trustPolicyMissing) }
        guard let policyText = rawPolicy as? String, policyText.utf8.count <= 64 << 10,
            let policy = try? PortableBundleTrustPolicy(json: Data(policyText.utf8))
        else { throw OpenMedManagedImportError(.trustPolicyInvalid) }
        var pins = Set<String>()
        if let raw = values[Self.keyPinsKey] {
            guard let text = raw as? String else { throw invalid }
            let separators = CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: ","))
            for token in text.components(separatedBy: separators) where !token.isEmpty {
                guard PortableBundlePattern.sha256.matches(token) else { throw invalid }
                pins.insert(token)
            }
        }
        func optionalString(_ key: String, _ valid: (String) -> Bool) throws -> String? {
            guard let raw = values[key] else { return nil }
            guard let text = raw as? String else { throw invalid }
            if text.isEmpty { return nil }
            guard valid(text) else { throw invalid }
            return text
        }
        let location = try optionalString(Self.locationKey) { $0.utf8.count <= 1024 }
        let expectedID = try optionalString(Self.expectedIDKey) { PortableBundlePattern.bundleID.matches($0) }
        let expectedContent = try optionalString(Self.expectedContentIDKey) { PortableBundlePattern.sha256.matches($0) }
        var minimum: Int64?
        if let raw = values[Self.minimumSequenceKey] {
            guard let value = Self.integer(raw), value >= 0, value <= PortableBundle.maxSafeInteger else { throw invalid }
            minimum = value
        }
        var activation: OpenMedManagedActivationPolicy?
        if let raw = try optionalString(Self.activationKey, { OpenMedManagedActivationPolicy(rawValue: $0) != nil }) {
            activation = OpenMedManagedActivationPolicy(rawValue: raw)
        }
        self.init(
            trustPolicy: policy, keyPins: pins, location: location, expectedBundleID: expectedID,
            expectedContentID: expectedContent, minimumSequence: minimum, activation: activation)
    }

    /// The pin of a trust-policy key: `sha256:` of its raw 32-byte public key.
    public static func pin(of key: PortableBundleTrustedKey) -> String {
        "sha256:" + PortableBundle.sha256Hex(key.publicKey)
    }

    private static func integer(_ value: Any) -> Int64? {
        if let number = value as? NSNumber {
            #if canImport(Darwin)
                if CFGetTypeID(number) == CFBooleanGetTypeID() { return nil }
            #endif
            let double = number.doubleValue
            guard double.rounded() == double, abs(double) <= Double(PortableBundle.maxSafeInteger) else { return nil }
            return number.int64Value
        }
        if let value = value as? Int { return Int64(value) }
        if let value = value as? Int64 { return value }
        return nil
    }
}

/// Supplies the managed configuration dictionary, or `nil` when none is installed.
public protocol OpenMedManagedConfigurationSource {
    func managedConfiguration() -> [String: Any]?
}

/// Apple managed app configuration: the dictionary an MDM server installs under
/// `com.apple.configuration.managed` in the app's user defaults.
public struct OpenMedManagedAppConfiguration: OpenMedManagedConfigurationSource, @unchecked Sendable {
    /// User-defaults key of the managed configuration dictionary.
    public static let configurationKey = "com.apple.configuration.managed"
    private let defaults: UserDefaults

    /// - Parameter defaults: Defaults to read; `.standard` in an app, a private
    ///   suite in tests.
    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    public func managedConfiguration() -> [String: Any]? {
        defaults.dictionary(forKey: Self.configurationKey)
    }
}

/// One managed import.
public struct OpenMedManagedImportRequest {
    public var handoff: OpenMedManagedBundleHandoff
    public var configuration: OpenMedManagedBundleConfiguration
    /// Verification instant, device target, caller floors, and manifest limits.
    public var context: PortableBundleContext
    /// Pins compiled into the app; intersected with the configuration's pins.
    public var keyPins: Set<String>
    public var limits: OpenMedManagedImportLimits
    /// Used when the configuration does not set an activation policy.
    public var activation: OpenMedManagedActivationPolicy
    /// Optional check on the newly active generation (for example loading the
    /// model). Returning `false` rolls back to last green.
    public var activationCheck: ((OpenMedBundleLease) -> Bool)?
    /// Checked while spooling and after every committed chunk.
    public var isCancelled: () -> Bool

    public init(
        handoff: OpenMedManagedBundleHandoff, configuration: OpenMedManagedBundleConfiguration,
        context: PortableBundleContext, keyPins: Set<String> = [], limits: OpenMedManagedImportLimits = .init(),
        activation: OpenMedManagedActivationPolicy = .activate,
        activationCheck: ((OpenMedBundleLease) -> Bool)? = nil, isCancelled: @escaping () -> Bool = { false }
    ) {
        self.handoff = handoff
        self.configuration = configuration
        self.context = context
        self.keyPins = keyPins
        self.limits = limits
        self.activation = activation
        self.activationCheck = activationCheck
        self.isCancelled = isCancelled
    }
}

/// Bounded, value-free receipt of one managed import
/// (`openmed.managed_import.receipt` v1). It holds opaque identities, a
/// sequence number, counts, a digest, and reason codes only.
public struct OpenMedManagedImportReceipt: Equatable, Sendable {
    public static let schema = "openmed.managed_import.receipt"
    public static let schemaVersion = 1

    public let outcome: OpenMedManagedImportOutcome
    public let reason: OpenMedManagedImportReason?
    public let bundleReason: PortableBundleReason?
    public let handoff: OpenMedManagedHandoffKind?
    /// Opaque bundle key; present once the manifest authenticated.
    public let bundleKey: String?
    public let generation: String?
    public let sequence: Int64?
    public let previousActive: String?
    public let active: String?
    public let lastGreen: String?
    public let filesVerified: Int
    public let bytesStaged: Int64
    public let resumed: Bool
    /// Binding digest of the trust policy used (`OpenMedBundleCache.trustDigest`).
    public let trustDigest: String?
    public let evicted: [String]

    /// Canonical JSON of the receipt, safe to persist or report as managed feedback.
    public func jsonData() -> Data {
        Data((try? PortableBundleJSON.encode(json)) ?? [])
    }

    var json: PortableBundleJSON {
        func text(_ value: String?) -> PortableBundleJSON { value.map { .string($0) } ?? .null }
        return .object([
            ("schema", .string(Self.schema)), ("schema_version", .int(Int64(Self.schemaVersion))),
            ("outcome", .string(outcome.rawValue)), ("reason", text(reason?.rawValue)),
            ("bundle_reason", text(bundleReason?.rawValue)), ("handoff", text(handoff?.rawValue)),
            ("bundle_key", text(bundleKey)), ("generation", text(generation)),
            ("sequence", sequence.map { .int($0) } ?? .null), ("previous_active", text(previousActive)),
            ("active", text(active)), ("last_green", text(lastGreen)), ("files_verified", .int(Int64(filesVerified))),
            ("bytes_staged", .int(bytesStaged)), ("resumed", .bool(resumed)), ("trust_digest", text(trustDigest)),
            ("evicted", .array(evicted.map { .string($0) })),
        ])
    }

    init?(json value: PortableBundleJSON) {
        guard case .object(let members) = value else { return nil }
        func text(_ key: String) -> String? {
            if case .string(let item)? = PortableBundleJSON.member(members, key) { return item }
            return nil
        }
        func number(_ key: String) -> Int64? {
            if case .int(let item)? = PortableBundleJSON.member(members, key) { return item }
            return nil
        }
        guard text("schema") == Self.schema, let outcome = text("outcome").flatMap(OpenMedManagedImportOutcome.init)
        else { return nil }
        var evicted: [String] = []
        if case .array(let items)? = PortableBundleJSON.member(members, "evicted") {
            for case .string(let item) in items { evicted.append(item) }
        }
        var resumed = false
        if case .bool(let flag)? = PortableBundleJSON.member(members, "resumed") { resumed = flag }
        self.outcome = outcome
        reason = text("reason").flatMap(OpenMedManagedImportReason.init)
        bundleReason = text("bundle_reason").flatMap(PortableBundleReason.init)
        handoff = text("handoff").flatMap(OpenMedManagedHandoffKind.init)
        bundleKey = text("bundle_key")
        generation = text("generation")
        sequence = number("sequence")
        previousActive = text("previous_active")
        active = text("active")
        lastGreen = text("last_green")
        filesVerified = Int(number("files_verified") ?? 0)
        bytesStaged = number("bytes_staged") ?? 0
        self.resumed = resumed
        trustDigest = text("trust_digest")
        self.evicted = evicted
    }

    fileprivate init(_ draft: Draft) {
        outcome = draft.outcome
        reason = draft.reason
        bundleReason = draft.bundleReason
        handoff = draft.handoff
        bundleKey = draft.bundleKey
        generation = draft.generation
        sequence = draft.sequence
        previousActive = draft.previousActive
        active = draft.active
        lastGreen = draft.lastGreen
        filesVerified = draft.filesVerified
        bytesStaged = draft.bytesStaged
        resumed = draft.resumed
        trustDigest = draft.trustDigest
        evicted = draft.evicted
    }

    fileprivate struct Draft {
        var outcome: OpenMedManagedImportOutcome = .rejected
        var reason: OpenMedManagedImportReason?
        var bundleReason: PortableBundleReason?
        var handoff: OpenMedManagedHandoffKind?
        var bundleKey: String?
        var generation: String?
        var sequence: Int64?
        var previousActive: String?
        var active: String?
        var lastGreen: String?
        var filesVerified = 0
        var bytesStaged: Int64 = 0
        var resumed = false
        var trustDigest: String?
        var evicted: [String] = []
    }
}

/// Result of `OpenMedManagedBundleImporter.recover()`.
public struct OpenMedManagedImportRecovery: Equatable, Sendable {
    public let cache: OpenMedBundleRecoveryReport
    /// Spooled stream copies left by an interrupted import and now deleted.
    public let spoolPurged: Int
}

/// Imports signed portable bundles delivered through managed, local platform
/// handoffs, verifies them fully offline against administrator trust roots,
/// and activates or rolls them back atomically through `OpenMedBundleCache`.
///
/// The importer accepts only `file:` URLs, bookmarks that resolve to them, and
/// streams. It never opens a network connection, never executes bundled
/// content, and never extracts an archive by its own entry names. It keeps a
/// small state file with per-bundle sequence floors and the last receipts, all
/// value-free. Imports through one state directory are serialized.
public final class OpenMedManagedBundleImporter: @unchecked Sendable, CustomStringConvertible {
    /// Number of receipts kept in the state file.
    public static let receiptCapacity = 16
    static let stateSchema = "openmed.managed_import.state"

    private let cache: OpenMedBundleCache
    private let stateDirectory: URL
    private let fileManager = FileManager.default
    private let mutex: NSLock

    /// Test seam: called at the importer's named crash checkpoints.
    var checkpointHook: ((String) throws -> Void)?

    /// - Parameters:
    ///   - cache: The bundle generation cache that stages and activates.
    ///   - stateDirectory: App-private directory for the state file and the
    ///     stream spool. Never shared with other apps.
    public init(cache: OpenMedBundleCache, stateDirectory: URL) throws {
        try FileManager.default.createDirectory(at: stateDirectory, withIntermediateDirectories: true)
        self.cache = cache
        self.stateDirectory = stateDirectory.resolvingSymlinksInPath().standardizedFileURL
        mutex = Self.registry.mutex(for: self.stateDirectory.path)
    }

    public var description: String { "OpenMedManagedBundleImporter" }

    // MARK: - Public API

    /// Reads the managed configuration and imports the bundle it names, or the
    /// explicit `handoff` (for example a stream from a managed document provider).
    ///
    /// A configured `location` that is a relative path resolves against the
    /// `approvedRoots` in order; a `file:` URL must lie inside one of them.
    /// Any other scheme is `unsupported` (`handoff_unsupported`).
    public func importManaged(
        from source: any OpenMedManagedConfigurationSource, approvedRoots: [URL] = [],
        handoff explicit: OpenMedManagedBundleHandoff? = nil, context: PortableBundleContext,
        keyPins: Set<String> = [], limits: OpenMedManagedImportLimits = .init(),
        activation: OpenMedManagedActivationPolicy = .activate,
        activationCheck: ((OpenMedBundleLease) -> Bool)? = nil, isCancelled: @escaping () -> Bool = { false }
    ) -> OpenMedManagedImportReceipt {
        (try? importManagedChecked(
            from: source, approvedRoots: approvedRoots, handoff: explicit, context: context, keyPins: keyPins,
            limits: limits, activation: activation, activationCheck: activationCheck, isCancelled: isCancelled))
            ?? record(refusal(.storageIO, handoff: nil))
    }

    /// Imports one bundle delivered through an approved local handoff.
    public func importBundle(_ request: OpenMedManagedImportRequest) -> OpenMedManagedImportReceipt {
        (try? importChecked(request)) ?? record(refusal(.storageIO, handoff: nil))
    }

    /// Confirms a probation activation: the active generation becomes last green
    /// and its sequence becomes the bundle's floor.
    public func confirm(bundleID: String) throws {
        mutex.lock()
        defer { mutex.unlock() }
        try cache.confirm(bundleID: bundleID)
        let key = OpenMedBundleCache.bundleKey(bundleID: bundleID)
        if let active = try cache.snapshot(bundleID: bundleID).active,
            let sequence = cache.sequence(bundleID: bundleID, generation: active)
        {
            var state = readState()
            state.floors[key] = max(state.floors[key] ?? 0, sequence)
            try writeState(state)
        }
    }

    /// Crash recovery: runs `OpenMedBundleCache.recover()` and deletes stream
    /// spools left by an interrupted import. Idempotent; call at startup.
    public func recover() -> OpenMedManagedImportRecovery {
        mutex.lock()
        defer { mutex.unlock() }
        let report = cache.recover()
        var purged = 0
        let names = (try? fileManager.contentsOfDirectory(atPath: spoolDirectory.path)) ?? []
        for name in names.sorted() {
            if (try? fileManager.removeItem(at: spoolDirectory.appendingPathComponent(name))) != nil { purged += 1 }
        }
        return OpenMedManagedImportRecovery(cache: report, spoolPurged: purged)
    }

    /// Recorded receipts, oldest first (at most `receiptCapacity`).
    public func receipts() -> [OpenMedManagedImportReceipt] {
        mutex.lock()
        defer { mutex.unlock() }
        return readState().receipts
    }

    /// Highest sequence accepted as last green for a bundle, if any.
    public func sequenceFloor(bundleID: String) -> Int64? {
        mutex.lock()
        defer { mutex.unlock() }
        return readState().floors[OpenMedBundleCache.bundleKey(bundleID: bundleID)]
    }

    // MARK: - Managed configuration

    func importManagedChecked(
        from source: any OpenMedManagedConfigurationSource, approvedRoots: [URL],
        handoff explicit: OpenMedManagedBundleHandoff?, context: PortableBundleContext, keyPins: Set<String>,
        limits: OpenMedManagedImportLimits, activation: OpenMedManagedActivationPolicy,
        activationCheck: ((OpenMedBundleLease) -> Bool)?, isCancelled: @escaping () -> Bool
    ) throws -> OpenMedManagedImportReceipt {
        guard let values = source.managedConfiguration() else {
            return record(refusal(.managedConfigAbsent, handoff: nil))
        }
        let configuration: OpenMedManagedBundleConfiguration
        do {
            configuration = try OpenMedManagedBundleConfiguration(managedValues: values)
        } catch let error as OpenMedManagedImportError {
            return record(refusal(error.reason, handoff: nil))
        }
        let handoff: OpenMedManagedBundleHandoff
        if let explicit {
            handoff = explicit
        } else {
            switch resolve(configuration.location, approvedRoots: approvedRoots) {
            case .success(let url): handoff = .file(url)
            case .failure(let error): return record(refusal(error.reason, handoff: .file))
            }
        }
        return try importChecked(
            OpenMedManagedImportRequest(
                handoff: handoff, configuration: configuration, context: context, keyPins: keyPins, limits: limits,
                activation: activation, activationCheck: activationCheck, isCancelled: isCancelled))
    }

    private func resolve(_ location: String?, approvedRoots: [URL]) -> Result<URL, OpenMedManagedImportError> {
        guard let location else { return .failure(OpenMedManagedImportError(.handoffPending)) }
        let roots = approvedRoots.map { $0.resolvingSymlinksInPath().standardizedFileURL }
        if let colon = location.firstIndex(of: ":"),
            location[..<colon].allSatisfy({ $0.isLetter || $0.isNumber || "+.-".contains($0) })
        {
            guard location[..<colon].lowercased() == "file", let url = URL(string: location), url.isFileURL else {
                return .failure(OpenMedManagedImportError(.handoffUnsupported))
            }
            let resolved = url.resolvingSymlinksInPath().standardizedFileURL
            guard roots.contains(where: { resolved.path.hasPrefix($0.path + "/") }) else {
                return .failure(OpenMedManagedImportError(.locationUnapproved))
            }
            guard fileManager.fileExists(atPath: resolved.path) else {
                return .failure(OpenMedManagedImportError(.handoffPending))
            }
            return .success(resolved)
        }
        guard OpenMedBundleArchive.isSafePath(location) else {
            return .failure(OpenMedManagedImportError(.locationUnapproved))
        }
        for root in roots {
            let candidate = root.appendingPathComponent(location)
            guard fileManager.fileExists(atPath: candidate.path) else { continue }
            // A symbolic link inside an approved directory must not lead out of it.
            let resolved = candidate.resolvingSymlinksInPath().standardizedFileURL
            guard resolved.path.hasPrefix(root.path + "/") else {
                return .failure(OpenMedManagedImportError(.locationUnapproved))
            }
            return .success(resolved)
        }
        return .failure(OpenMedManagedImportError(.handoffPending))
    }

    // MARK: - Import

    func importChecked(_ request: OpenMedManagedImportRequest) throws -> OpenMedManagedImportReceipt {
        mutex.lock()
        defer { mutex.unlock() }
        var draft = OpenMedManagedImportReceipt.Draft()
        draft.handoff = kind(request.handoff)
        var spooled: URL?
        do {
            try importLocked(request, &draft, spooled: &spooled)
        } catch let crash as OpenMedBundleCacheCrash {
            // A simulated process death leaves the spool for `recover()`.
            throw crash
        } catch let error as OpenMedManagedImportError {
            draft.outcome = error.outcome
            draft.reason = error.reason
            draft.bundleReason = error.bundleReason
        } catch {
            draft.outcome = .rejected
            draft.reason = .storageIO
        }
        if let spooled { try? fileManager.removeItem(at: spooled) }
        return recordLocked(OpenMedManagedImportReceipt(draft))
    }

    private func importLocked(
        _ request: OpenMedManagedImportRequest, _ draft: inout OpenMedManagedImportReceipt.Draft, spooled: inout URL?
    ) throws {
        let configuration = request.configuration
        let policy = configuration.trustPolicy
        // Every delivered trust root must appear in every non-empty pin set.
        for pins in [configuration.keyPins, request.keyPins] where !pins.isEmpty {
            guard policy.keys.allSatisfy({ pins.contains(OpenMedManagedBundleConfiguration.pin(of: $0)) }) else {
                throw OpenMedManagedImportError(.trustAnchorUnpinned)
            }
        }
        draft.trustDigest = OpenMedBundleCache.trustDigest(policy)
        if request.isCancelled() { throw OpenMedManagedImportError(.cancelled) }

        let url: URL
        switch request.handoff {
        case .file(let file):
            guard file.isFileURL else { throw OpenMedManagedImportError(.handoffUnsupported) }
            url = file
        case .bookmark(let data):
            var stale = false
            guard let resolved = try? URL(resolvingBookmarkData: data, options: [], bookmarkDataIsStale: &stale) else {
                throw OpenMedManagedImportError(.handoffUnreadable)
            }
            guard resolved.isFileURL else { throw OpenMedManagedImportError(.handoffUnsupported) }
            url = resolved
        case .stream(let stream):
            url = try spool(stream, limits: request.limits, isCancelled: request.isCancelled)
            spooled = url
            try checkpoint("archive_spooled")
        }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }

        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw OpenMedManagedImportError(.handoffPending)
        }
        if isDirectory.boolValue, case .file = request.handoff { draft.handoff = .directory }
        let delivery: Delivery =
            isDirectory.boolValue
            ? try DirectoryDelivery(url.resolvingSymlinksInPath(), context: request.context)
            : try ArchiveDelivery(url, limits: request.limits, context: request.context)

        // Authenticate the manifest before trusting anything it declares.
        let header = OpenMedBundleCache.ManifestHeader(delivery.manifest)
        var context = request.context
        if let header {
            let key = OpenMedBundleCache.bundleKey(bundleID: header.bundleID)
            let floor =
                [
                    request.context.sequenceFloors[header.bundleID] ?? 0, configuration.minimumSequence ?? 0,
                    readState().floors[key] ?? 0,
                ].max() ?? 0
            var floors = request.context.sequenceFloors
            if floor > 0 { floors[header.bundleID] = floor }
            context = try PortableBundleContext(
                now: context.now, target: context.target, sequenceFloors: floors,
                understoodExtensions: context.understoodExtensions, maxManifestBytes: context.maxManifestBytes,
                maxEnvelopeBytes: context.maxEnvelopeBytes, maxTotalBytes: context.maxTotalBytes,
                maxFiles: context.maxFiles)
        }
        let verification = PortableBundle.verify(
            manifest: delivery.manifest, envelope: delivery.envelope, trustPolicy: policy, context: context)
        guard verification.ok, let contentID = verification.contentID, let header else {
            throw OpenMedManagedImportError(
                .bundleRejected, bundleReason: verification.ok ? .manifestMalformed : verification.reason)
        }
        let generation = OpenMedBundleCache.generationID(contentID: contentID)
        draft.bundleKey = OpenMedBundleCache.bundleKey(bundleID: header.bundleID)
        draft.generation = generation
        draft.sequence = header.sequence
        let before = try pointer(header.bundleID)
        draft.previousActive = before.active
        draft.active = before.active
        draft.lastGreen = before.lastGreen

        if let expected = configuration.expectedBundleID, expected != header.bundleID {
            throw OpenMedManagedImportError(.bundleNotExpected)
        }
        if let expected = configuration.expectedContentID, expected != contentID {
            throw OpenMedManagedImportError(.bundleNotExpected)
        }
        if let reason = delivery.inventory(header.files) {
            throw OpenMedManagedImportError(.verificationFailed, bundleReason: reason)
        }

        let activation = configuration.activation ?? request.activation
        let source = delivery.source(identity: "openmed.managed_import:" + contentID)
        let result = try cache.acquireChecked(
            OpenMedBundleAcquireRequest(
                manifest: delivery.manifest, envelope: delivery.envelope, source: source, trustPolicy: policy,
                context: context, activate: activation != .stage,
                confirm: activation == .activate && request.activationCheck == nil, isCancelled: request.isCancelled))
        guard result.ok else {
            let streamFailed = (source as? OpenMedArchiveBundleSource)?.streamFailed ?? false
            if streamFailed { throw OpenMedManagedImportError(.archiveMalformed) }
            throw OpenMedManagedImportError(Self.reason(result.reason), bundleReason: result.bundleReason)
        }
        draft.filesVerified = result.filesVerified
        draft.bytesStaged = result.bytesTransferred
        draft.resumed = result.resumed
        draft.evicted = result.evicted

        switch activation {
        case .stage:
            draft.outcome = result.fromCache ? .alreadyPresent : .imported
        case .activate, .probation:
            let unchanged =
                result.fromCache && before.active == generation
                && (before.lastGreen == generation || activation == .probation)
            if unchanged {
                draft.outcome = .alreadyPresent
            } else if let check = request.activationCheck {
                let lease = try cache.lease(bundleID: header.bundleID, generation: generation)
                let healthy = check(lease)
                lease.release()
                if !healthy {
                    try cache.withdraw(bundleID: header.bundleID, generation: generation)
                    let after = try pointer(header.bundleID)
                    draft.active = after.active
                    draft.lastGreen = after.lastGreen
                    draft.outcome = after.active == nil ? .rejected : .rolledBack
                    draft.reason = .activationCheckFailed
                    return
                }
                if activation == .activate { try cache.confirm(bundleID: header.bundleID) }
                draft.outcome = .activated
            } else {
                draft.outcome = .activated
            }
        }
        let after = try pointer(header.bundleID)
        draft.active = after.active
        draft.lastGreen = after.lastGreen
        if after.lastGreen == generation {
            var state = readState()
            let key = OpenMedBundleCache.bundleKey(bundleID: header.bundleID)
            state.floors[key] = max(state.floors[key] ?? 0, header.sequence)
            try writeState(state)
        }
    }

    private func pointer(_ bundleID: String) throws -> (active: String?, lastGreen: String?) {
        do {
            let snapshot = try cache.snapshot(bundleID: bundleID)
            return (snapshot.active, snapshot.lastGreen)
        } catch let error as OpenMedBundleCacheError {
            throw OpenMedManagedImportError(Self.reason(error.reason))
        }
    }

    private static func reason(_ reason: OpenMedBundleCacheReason?) -> OpenMedManagedImportReason {
        switch reason {
        case .bundleRejected?: return .bundleRejected
        case .lockTimeout?: return .lockTimeout
        case .quotaExceeded?: return .quotaExceeded
        case .storagePressure?: return .storagePressure
        case .transferInterrupted?, .rangeMismatch?: return .transferInterrupted
        case .cancelled?: return .cancelled
        case .verificationFailed?, .generationCorrupt?: return .verificationFailed
        default: return .storageIO
        }
    }

    private func kind(_ handoff: OpenMedManagedBundleHandoff) -> OpenMedManagedHandoffKind {
        switch handoff {
        case .file: return .file
        case .bookmark: return .bookmark
        case .stream: return .stream
        }
    }

    private func refusal(_ reason: OpenMedManagedImportReason, handoff: OpenMedManagedHandoffKind?)
        -> OpenMedManagedImportReceipt
    {
        var draft = OpenMedManagedImportReceipt.Draft()
        draft.outcome = reason.refusalOutcome
        draft.reason = reason
        draft.handoff = handoff
        return OpenMedManagedImportReceipt(draft)
    }

    // MARK: - Spooling

    private var spoolDirectory: URL { stateDirectory.appendingPathComponent("spool", isDirectory: true) }

    private func spool(_ stream: InputStream, limits: OpenMedManagedImportLimits, isCancelled: () -> Bool) throws -> URL {
        try fileManager.createDirectory(at: spoolDirectory, withIntermediateDirectories: true)
        let file = spoolDirectory.appendingPathComponent(UUID().uuidString.lowercased() + ".zip")
        guard fileManager.createFile(atPath: file.path, contents: nil), let output = try? FileHandle(forWritingTo: file)
        else { throw OpenMedManagedImportError(.storageIO) }
        var complete = false
        defer {
            try? output.close()
            if !complete { try? fileManager.removeItem(at: file) }
        }
        stream.open()
        defer { stream.close() }
        let capacity = 1 << 20
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: capacity)
        defer { buffer.deallocate() }
        var total: Int64 = 0
        while true {
            if isCancelled() { throw OpenMedManagedImportError(.cancelled) }
            let count = stream.read(buffer, maxLength: capacity)
            if count < 0 { throw OpenMedManagedImportError(.transferInterrupted) }
            if count == 0 { break }
            total += Int64(count)
            guard total <= limits.maxArchiveBytes else { throw OpenMedManagedImportError(.budgetArchiveBytes) }
            do {
                try output.write(contentsOf: Data(bytes: buffer, count: count))
            } catch {
                throw OpenMedManagedImportError(.storageIO)
            }
        }
        complete = true
        return file
    }

    // MARK: - State

    private struct State {
        var floors: [String: Int64] = [:]
        var receipts: [OpenMedManagedImportReceipt] = []
    }

    private var stateFile: URL { stateDirectory.appendingPathComponent("managed-import.json") }

    private func readState() -> State {
        guard let data = try? Data(contentsOf: stateFile), case .object(let members)? = try? PortableBundleJSON.parse(data),
            case .string(Self.stateSchema)? = PortableBundleJSON.member(members, "schema")
        else { return State() }
        var state = State()
        if case .object(let floors)? = PortableBundleJSON.member(members, "sequence_floors") {
            for (key, value) in floors {
                if case .int(let floor) = value { state.floors[key] = floor }
            }
        }
        if case .array(let items)? = PortableBundleJSON.member(members, "receipts") {
            state.receipts = items.compactMap(OpenMedManagedImportReceipt.init(json:))
        }
        return state
    }

    private func writeState(_ state: State) throws {
        let floors = state.floors.keys.sorted().map { ($0, PortableBundleJSON.int(state.floors[$0] ?? 0)) }
        let value = PortableBundleJSON.object([
            ("schema", .string(Self.stateSchema)), ("schema_version", .int(1)), ("sequence_floors", .object(floors)),
            ("receipts", .array(state.receipts.suffix(Self.receiptCapacity).map(\.json))),
        ])
        try Data(PortableBundleJSON.encode(value)).write(to: stateFile, options: .atomic)
    }

    private func record(_ receipt: OpenMedManagedImportReceipt) -> OpenMedManagedImportReceipt {
        mutex.lock()
        defer { mutex.unlock() }
        return recordLocked(receipt)
    }

    private func recordLocked(_ receipt: OpenMedManagedImportReceipt) -> OpenMedManagedImportReceipt {
        var state = readState()
        state.receipts.append(receipt)
        state.receipts = Array(state.receipts.suffix(Self.receiptCapacity))
        try? writeState(state)
        return receipt
    }

    private func checkpoint(_ name: String) throws { try checkpointHook?(name) }

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
}

// MARK: - Deliveries

/// A located bundle: its manifest, envelope, inventory, and a range source.
private protocol Delivery {
    var manifest: Data { get }
    var envelope: Data { get }
    /// First inventory refusal against the declared files, in manifest order.
    func inventory(_ declared: [OpenMedBundleCache.DeclaredFile]) -> PortableBundleReason?
    func source(identity: String) -> any OpenMedBundleRangeSource
}

private func inventoryReason(
    declared: [OpenMedBundleCache.DeclaredFile], sizes: [String: Int64], order: [String]
) -> PortableBundleReason? {
    for file in declared {
        guard let size = sizes[file.path] else { return .fileMissing }
        guard size == file.size else { return .fileSizeMismatch }
    }
    var known = Set(declared.map(\.path))
    known.insert(PortableBundle.manifestFilename)
    known.insert(PortableBundle.envelopeFilename)
    return order.contains(where: { !known.contains($0) }) ? .fileUndeclared : nil
}

private struct ArchiveDelivery: Delivery {
    let archive: OpenMedBundleArchive
    let manifest: Data
    let envelope: Data

    init(_ url: URL, limits: OpenMedManagedImportLimits, context: PortableBundleContext) throws {
        do {
            archive = try OpenMedBundleArchive(url: url, limits: limits)
        } catch let failure as OpenMedBundleArchive.Failure {
            throw OpenMedManagedImportError(failure.reason)
        } catch {
            throw OpenMedManagedImportError(.archiveMalformed)
        }
        manifest = try Self.document(archive, PortableBundle.manifestFilename, context.maxManifestBytes, .manifestTooLarge)
        envelope = try Self.document(archive, PortableBundle.envelopeFilename, context.maxEnvelopeBytes, .envelopeTooLarge)
    }

    private static func document(
        _ archive: OpenMedBundleArchive, _ name: String, _ limit: Int, _ tooLarge: PortableBundleReason
    ) throws -> Data {
        guard let entry = archive.files[name] else { throw OpenMedManagedImportError(.bundleRejected, bundleReason: .fileMissing) }
        guard entry.size <= Int64(limit) else { throw OpenMedManagedImportError(.bundleRejected, bundleReason: tooLarge) }
        do {
            return try archive.read(entry)
        } catch {
            throw OpenMedManagedImportError(.archiveMalformed)
        }
    }

    func inventory(_ declared: [OpenMedBundleCache.DeclaredFile]) -> PortableBundleReason? {
        let order = archive.entries.filter { !$0.isDirectory }.map { String($0.name.dropFirst(archive.root.count)) }
        return inventoryReason(declared: declared, sizes: archive.files.mapValues(\.size), order: order)
    }

    func source(identity: String) -> any OpenMedBundleRangeSource {
        OpenMedArchiveBundleSource(archive: archive, identity: identity)
    }
}

private struct DirectoryDelivery: Delivery {
    let root: URL
    let manifest: Data
    let envelope: Data
    private let sizes: [String: Int64]
    private let order: [String]

    init(_ root: URL, context: PortableBundleContext) throws {
        let manager = FileManager.default
        guard let subpaths = try? manager.subpathsOfDirectory(atPath: root.path) else {
            throw OpenMedManagedImportError(.handoffUnreadable)
        }
        var sizes: [String: Int64] = [:]
        var order: [String] = []
        for relative in subpaths.sorted(by: { $0.utf8.lexicographicallyPrecedes($1.utf8) }) {
            guard let attributes = try? manager.attributesOfItem(atPath: root.appendingPathComponent(relative).path)
            else { throw OpenMedManagedImportError(.handoffUnreadable) }
            switch attributes[.type] as? FileAttributeType {
            case .typeDirectory?: continue
            case .typeRegular?:
                sizes[relative] = (attributes[.size] as? NSNumber)?.int64Value ?? -1
                order.append(relative)
            default: throw OpenMedManagedImportError(.entryUnsafe)
            }
        }
        func document(_ name: String, _ limit: Int, _ tooLarge: PortableBundleReason) throws -> Data {
            guard let size = sizes[name] else { throw OpenMedManagedImportError(.bundleRejected, bundleReason: .fileMissing) }
            guard size <= Int64(limit) else { throw OpenMedManagedImportError(.bundleRejected, bundleReason: tooLarge) }
            guard let data = try? Data(contentsOf: root.appendingPathComponent(name)) else {
                throw OpenMedManagedImportError(.handoffUnreadable)
            }
            return data
        }
        self.root = root
        self.sizes = sizes
        self.order = order
        manifest = try document(PortableBundle.manifestFilename, context.maxManifestBytes, .manifestTooLarge)
        envelope = try document(PortableBundle.envelopeFilename, context.maxEnvelopeBytes, .envelopeTooLarge)
    }

    func inventory(_ declared: [OpenMedBundleCache.DeclaredFile]) -> PortableBundleReason? {
        inventoryReason(declared: declared, sizes: sizes, order: order)
    }

    func source(identity: String) -> any OpenMedBundleRangeSource {
        OpenMedDirectoryBundleSource(directory: root, identity: identity)
    }
}
