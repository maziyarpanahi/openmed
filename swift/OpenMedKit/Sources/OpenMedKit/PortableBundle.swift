import Foundation

#if canImport(CryptoKit)
    import CryptoKit
#endif

/// Stable, value-free reason codes shared by every portable-bundle reader.
///
/// Cases are declared in evaluation order. Readers in Python, Swift, Kotlin,
/// and TypeScript report the first failing check in this order.
public enum PortableBundleReason: String, CaseIterable, Sendable {
    case manifestTooLarge = "manifest_too_large"
    case envelopeTooLarge = "envelope_too_large"
    case envelopeMalformed = "envelope_malformed"
    case envelopeVersionUnsupported = "envelope_version_unsupported"
    case contentIDMismatch = "content_id_mismatch"
    case bundleRevoked = "bundle_revoked"
    case signatureMissing = "signature_missing"
    case signatureAlgorithmUnsupported = "signature_algorithm_unsupported"
    case untrustedKey = "untrusted_key"
    case keyRevoked = "key_revoked"
    case keyNotYetValid = "key_not_yet_valid"
    case keyExpired = "key_expired"
    case signatureInvalid = "signature_invalid"
    case signatureVerifierUnavailable = "signature_verifier_unavailable"
    case signatureThresholdNotMet = "signature_threshold_not_met"
    case manifestNotCanonical = "manifest_not_canonical"
    case schemaUnsupported = "schema_unsupported"
    case schemaVersionUnsupported = "schema_version_unsupported"
    case unknownCriticalField = "unknown_critical_field"
    case manifestMalformed = "manifest_malformed"
    case pathUnsafe = "path_unsafe"
    case executableFileForbidden = "executable_file_forbidden"
    case runtimeCombinationUnsupported = "runtime_combination_unsupported"
    case duplicatePath = "duplicate_path"
    case pathConflict = "path_conflict"
    case fileOrderInvalid = "file_order_invalid"
    case fileReferenceUndeclared = "file_reference_undeclared"
    case fallbackInvalid = "fallback_invalid"
    case unknownCriticalExtension = "unknown_critical_extension"
    case sizeLimitExceeded = "size_limit_exceeded"
    case bundleNotYetValid = "bundle_not_yet_valid"
    case bundleExpired = "bundle_expired"
    case rollbackRejected = "rollback_rejected"
    case sdkIncompatible = "sdk_incompatible"
    case policyIncompatible = "policy_incompatible"
    case platformIncompatible = "platform_incompatible"
    case fileMissing = "file_missing"
    case fileSizeMismatch = "file_size_mismatch"
    case fileDigestMismatch = "file_digest_mismatch"
    case fileUndeclared = "file_undeclared"
    case trustPolicyInvalid = "trust_policy_invalid"
    case contextInvalid = "context_invalid"
    case signingKeyInvalid = "signing_key_invalid"
}

/// Error that carries only a controlled reason code.
public struct PortableBundleError: Error, Equatable, Sendable, CustomStringConvertible {
    public let reason: PortableBundleReason

    public init(_ reason: PortableBundleReason) {
        self.reason = reason
    }

    public var description: String { reason.rawValue }
}

/// One caller-controlled Ed25519 trust anchor.
public struct PortableBundleTrustedKey: Sendable, CustomStringConvertible {
    public let kid: String
    public let publicKey: Data
    public let notBefore: String
    public let notAfter: String
    public let revoked: Bool

    public init(kid: String, publicKey: Data, notBefore: String, notAfter: String, revoked: Bool = false) throws {
        guard PortableBundlePattern.kid.matches(kid), publicKey.count == 32,
            PortableBundlePattern.timestamp.matches(notBefore),
            PortableBundlePattern.timestamp.matches(notAfter),
            PortableBundle.utf8Less(notBefore, notAfter)
        else { throw PortableBundleError(.trustPolicyInvalid) }
        self.kid = kid
        self.publicKey = publicKey
        self.notBefore = notBefore
        self.notAfter = notAfter
        self.revoked = revoked
    }

    public var description: String { "PortableBundleTrustedKey(kid: \(kid), revoked: \(revoked))" }
}

/// Offline trust roots, rotation windows, and revocations.
public struct PortableBundleTrustPolicy: Sendable {
    public let keys: [PortableBundleTrustedKey]
    public let threshold: Int
    public let revokedContentIDs: Set<String>

    public init(keys: [PortableBundleTrustedKey], threshold: Int = 1, revokedContentIDs: Set<String> = []) throws {
        guard !keys.isEmpty, keys.count <= 64, (1...PortableBundle.maxSignatures).contains(threshold),
            Set(keys.map(\.kid)).count == keys.count,
            revokedContentIDs.allSatisfy({ PortableBundlePattern.sha256.matches($0) })
        else { throw PortableBundleError(.trustPolicyInvalid) }
        self.keys = keys
        self.threshold = threshold
        self.revokedContentIDs = revokedContentIDs
    }

    /// Parses a JSON trust-policy document (`portable_bundle_trust_policy`).
    public init(json: Data) throws {
        let invalid = PortableBundleError(.trustPolicyInvalid)
        guard let value = try? PortableBundleJSON.parse(json), case .object(let members) = value,
            PortableBundleJSON.hasExactKeys(members, ["schema", "schema_version", "threshold", "keys", "revoked_content_ids"]),
            case .string(let schema)? = PortableBundleJSON.member(members, "schema"),
            schema == "openmed.portable_bundle.trust_policy",
            case .int(1)? = PortableBundleJSON.member(members, "schema_version"),
            case .int(let threshold)? = PortableBundleJSON.member(members, "threshold"),
            case .array(let rawKeys)? = PortableBundleJSON.member(members, "keys"),
            case .array(let rawRevoked)? = PortableBundleJSON.member(members, "revoked_content_ids")
        else { throw invalid }
        var keys: [PortableBundleTrustedKey] = []
        for raw in rawKeys {
            guard case .object(let entry) = raw,
                PortableBundleJSON.hasExactKeys(entry, ["kid", "kty", "crv", "x", "not_before", "not_after", "status"]),
                case .string(let kid)? = PortableBundleJSON.member(entry, "kid"),
                case .string("OKP")? = PortableBundleJSON.member(entry, "kty"),
                case .string("Ed25519")? = PortableBundleJSON.member(entry, "crv"),
                case .string(let x)? = PortableBundleJSON.member(entry, "x"),
                let publicKey = PortableBundle.base64URLDecode(x, length: 32),
                case .string(let notBefore)? = PortableBundleJSON.member(entry, "not_before"),
                case .string(let notAfter)? = PortableBundleJSON.member(entry, "not_after"),
                case .string(let status)? = PortableBundleJSON.member(entry, "status"),
                status == "active" || status == "revoked"
            else { throw invalid }
            keys.append(try PortableBundleTrustedKey(kid: kid, publicKey: publicKey, notBefore: notBefore, notAfter: notAfter, revoked: status == "revoked"))
        }
        var revoked = Set<String>()
        for raw in rawRevoked {
            guard case .string(let item) = raw else { throw invalid }
            revoked.insert(item)
        }
        try self.init(keys: keys, threshold: Int(threshold), revokedContentIDs: revoked)
    }

    func key(_ kid: String) -> PortableBundleTrustedKey? {
        keys.first { PortableBundle.same($0.kid, kid) }
    }
}

/// Runtime environment a bundle must be compatible with.
public struct PortableBundleTarget: Sendable {
    public let platform: String
    public let sdkName: String
    public let sdkVersion: String
    public let runtimes: [String]
    public let osVersion: String?
    public let policyVersion: Int64?
    public let memoryBytes: Int64?
    public let storageBytes: Int64?

    public init(
        platform: String, sdkName: String, sdkVersion: String, runtimes: [String], osVersion: String? = nil,
        policyVersion: Int64? = nil, memoryBytes: Int64? = nil, storageBytes: Int64? = nil
    ) throws {
        let optionalUInt: (Int64?) -> Bool = { $0.map { $0 >= 0 && $0 <= PortableBundle.maxSafeInteger } ?? true }
        guard PortableBundle.platforms.contains(platform), PortableBundlePattern.token.matches(sdkName),
            PortableBundlePattern.version.matches(sdkVersion),
            runtimes.allSatisfy({ PortableBundle.formatPlatforms[$0] != nil }),
            osVersion.map({ PortableBundlePattern.version.matches($0) }) ?? true,
            optionalUInt(policyVersion), optionalUInt(memoryBytes), optionalUInt(storageBytes)
        else { throw PortableBundleError(.contextInvalid) }
        self.platform = platform
        self.sdkName = sdkName
        self.sdkVersion = sdkVersion
        self.runtimes = runtimes
        self.osVersion = osVersion
        self.policyVersion = policyVersion
        self.memoryBytes = memoryBytes
        self.storageBytes = storageBytes
    }
}

/// Caller-controlled verification inputs.
public struct PortableBundleContext: Sendable {
    public let now: String
    public let target: PortableBundleTarget?
    public let sequenceFloors: [String: Int64]
    public let understoodExtensions: Set<String>
    public let maxManifestBytes: Int
    public let maxEnvelopeBytes: Int
    public let maxTotalBytes: Int64
    public let maxFiles: Int

    public init(
        now: String, target: PortableBundleTarget? = nil, sequenceFloors: [String: Int64] = [:],
        understoodExtensions: Set<String> = [], maxManifestBytes: Int = 1 << 20, maxEnvelopeBytes: Int = 64 << 10,
        maxTotalBytes: Int64 = 64 << 30, maxFiles: Int = 4096
    ) throws {
        guard PortableBundlePattern.timestamp.matches(now), sequenceFloors.values.allSatisfy({ $0 >= 0 }),
            understoodExtensions.allSatisfy({ PortableBundlePattern.extensionName.matches($0) }),
            maxManifestBytes > 0, maxEnvelopeBytes > 0, maxTotalBytes > 0, maxFiles > 0
        else { throw PortableBundleError(.contextInvalid) }
        self.now = now
        self.target = target
        self.sequenceFloors = sequenceFloors
        self.understoodExtensions = understoodExtensions
        self.maxManifestBytes = maxManifestBytes
        self.maxEnvelopeBytes = maxEnvelopeBytes
        self.maxTotalBytes = maxTotalBytes
        self.maxFiles = maxFiles
    }
}

/// Bounded, value-free verification result.
public struct PortableBundleVerification: Equatable, Sendable {
    public let ok: Bool
    public let reason: PortableBundleReason?
    public let contentID: String?
    public let validSignatures: Int
    public let eligibleRuntimes: [Int]
    public let filesChecked: Int
    public let filesVerified: Bool

    /// Whether a separate activation step may consider the bundle.
    public var eligibleForActivation: Bool { ok && filesVerified }
}

/// Offline reader for signed OpenMed portable model bundles.
///
/// Mirrors `openmed/core/portable_bundle.py`: the same checks run in the same
/// order and produce the same result for `tests/fixtures/portable_bundle`.
/// Verification uses CryptoKit only, never touches the network, never executes
/// bundled content, and reports only reason codes, digests, counts, and indexes.
public enum PortableBundle {
    public static let schema = "openmed.portable_bundle"
    public static let schemaVersion: Int64 = 1
    public static let envelopeSchema = "openmed.portable_bundle.signatures"
    public static let manifestFilename = "openmed-bundle.json"
    public static let envelopeFilename = "openmed-bundle.sig.json"
    public static let payloadType = "application/vnd.openmed.portable-bundle.v1+json"
    public static let maxSafeInteger: Int64 = 9_007_199_254_740_991
    static let maxSignatures = 16

    static let formatPlatforms: [String: Set<String>] = [
        "coreml": ["ios", "macos", "watchos", "visionos"],
        "mlx": ["ios", "macos"],
        "onnx": ["android", "ios", "macos", "linux", "windows", "web"],
        "ort": ["android", "ios"],
        "transformersjs": ["web"],
        "pytorch": ["linux", "macos", "windows"],
    ]
    static let formatPrecisions: [String: Set<String>] = [
        "coreml": ["fp32", "fp16", "int8", "int4"],
        "mlx": ["fp32", "fp16", "bf16", "int8", "int4", "int2"],
        "onnx": ["fp32", "fp16", "int8", "int4"],
        "ort": ["fp32", "fp16", "int8"],
        "transformersjs": ["fp32", "fp16", "int8", "int4"],
        "pytorch": ["fp32", "fp16", "bf16"],
    ]
    static let platforms: Set<String> = ["ios", "macos", "watchos", "visionos", "android", "web", "linux", "windows"]
    static let precisions: Set<String> = ["fp32", "fp16", "bf16", "int8", "int4", "int2"]
    static let fileRoles: Set<String> = [
        "calibration", "config", "decoder", "evidence", "graph", "labels", "license", "metadata", "policy",
        "provenance", "segmenter", "tokenizer", "weights",
    ]
    static let offsetUnits: Set<String> = ["unicode_scalar", "utf16_code_unit", "utf8_byte"]
    static let topKeys = [
        "schema", "schema_version", "bundle", "model", "tokenizer", "labels", "offsets", "decoder", "segmenter",
        "policy", "calibration", "compatibility", "runtimes", "licenses", "provenance", "evidence", "files",
        "extensions", "critical_extensions",
    ]

    /// Returns the `sha256:<hex>` content identity of canonical manifest bytes.
    public static func contentID(of manifest: Data) -> String {
        "sha256:" + sha256Hex(manifest)
    }

    /// Returns the DSSE v1 pre-authentication encoding that signatures cover.
    public static func signingInput(for manifest: Data) -> Data {
        var output = Data("DSSEv1 \(payloadType.utf8.count) \(payloadType) \(manifest.count) ".utf8)
        output.append(manifest)
        return output
    }

    /// Returns RFC 8785 canonical bytes for any JSON text in the portable subset.
    public static func canonicalJSON(_ data: Data) throws -> Data {
        do {
            return Data(try PortableBundleJSON.encode(PortableBundleJSON.parse(data)))
        } catch {
            throw PortableBundleError(.manifestNotCanonical)
        }
    }

    /// Validates canonical manifest bytes without checking signatures.
    public static func validateManifest(_ manifest: Data, understoodExtensions: Set<String> = []) throws {
        do {
            _ = try validate(parseCanonical(manifest), understood: understoodExtensions)
        } catch let refusal as Refusal {
            throw PortableBundleError(refusal.reason)
        }
    }

    /// Verifies a portable bundle offline and returns a value-free result.
    ///
    /// - Parameters:
    ///   - manifest: Exact bytes of `openmed-bundle.json`.
    ///   - envelope: Exact bytes of `openmed-bundle.sig.json`.
    ///   - trustPolicy: Caller-controlled trust roots.
    ///   - context: Verification instant, target, rollback floors, and limits.
    ///   - files: Bundle files by relative path; `nil` checks only the
    ///     manifest and signatures.
    public static func verify(
        manifest: Data, envelope: Data, trustPolicy: PortableBundleTrustPolicy, context: PortableBundleContext,
        files: [String: Data]? = nil
    ) -> PortableBundleVerification {
        verify(
            manifest: manifest, envelope: envelope, trustPolicy: trustPolicy, context: context,
            tree: files.map { MemoryTree(files: $0) })
    }

    /// Verifies a bundle directory that contains its manifest and envelope.
    ///
    /// Files are hashed in streaming chunks. Symbolic links and special files
    /// are refused, and nothing outside the directory is read.
    public static func verifyDirectory(
        at root: URL, trustPolicy: PortableBundleTrustPolicy, context: PortableBundleContext
    ) -> PortableBundleVerification {
        let failed = { (reason: PortableBundleReason) in
            PortableBundleVerification(
                ok: false, reason: reason, contentID: nil, validSignatures: 0, eligibleRuntimes: [], filesChecked: 0,
                filesVerified: false)
        }
        let tree: DirectoryTree
        do {
            tree = try DirectoryTree(root: root)
        } catch let refusal as Refusal {
            return failed(refusal.reason)
        } catch {
            return failed(.fileMissing)
        }
        guard let manifest = tree.read(manifestFilename, limit: context.maxManifestBytes),
            let envelope = tree.read(envelopeFilename, limit: context.maxEnvelopeBytes)
        else { return failed(.fileMissing) }
        guard manifest.count <= context.maxManifestBytes else { return failed(.manifestTooLarge) }
        guard envelope.count <= context.maxEnvelopeBytes else { return failed(.envelopeTooLarge) }
        return verify(manifest: manifest, envelope: envelope, trustPolicy: trustPolicy, context: context, tree: tree)
    }

    static func verify(
        manifest: Data, envelope: Data, trustPolicy: PortableBundleTrustPolicy, context: PortableBundleContext,
        tree: (any PortableBundleTree)?
    ) -> PortableBundleVerification {
        var contentID: String?
        var validSignatures = 0
        var eligible: [Int] = []
        var checked = 0
        do {
            guard manifest.count <= context.maxManifestBytes else { throw Refusal(.manifestTooLarge) }
            let identity = Self.contentID(of: manifest)
            contentID = identity
            guard envelope.count <= context.maxEnvelopeBytes else { throw Refusal(.envelopeTooLarge) }
            let (declared, signatures) = try parseEnvelope(envelope)
            guard same(declared, identity) else { throw Refusal(.contentIDMismatch) }
            guard !trustPolicy.revokedContentIDs.contains(identity) else { throw Refusal(.bundleRevoked) }
            validSignatures = try checkSignatures(signatures, manifest: manifest, policy: trustPolicy, now: context.now)
            let validated = try validate(parseCanonical(manifest), understood: context.understoodExtensions)
            let declaredBytes = validated.files.reduce(Int64(0)) { $0 + $1.size }
            guard validated.files.count <= context.maxFiles, declaredBytes <= context.maxTotalBytes else {
                throw Refusal(.sizeLimitExceeded)
            }
            guard !utf8Less(context.now, validated.issuedAt) else { throw Refusal(.bundleNotYetValid) }
            guard utf8Less(context.now, validated.expiresAt) else { throw Refusal(.bundleExpired) }
            if let floor = context.sequenceFloors[validated.bundleID], validated.sequence < floor {
                throw Refusal(.rollbackRejected)
            }
            eligible = try eligibleRuntimes(validated, target: context.target)
            if let tree {
                for item in validated.files {
                    guard let size = tree.size(item.path) else { throw Refusal(.fileMissing) }
                    guard size == item.size else { throw Refusal(.fileSizeMismatch) }
                    guard let digest = tree.digest(item.path), same(digest, item.sha256) else {
                        throw Refusal(.fileDigestMismatch)
                    }
                    checked += 1
                }
                let declaredPaths = Set(validated.files.map { Array($0.path.utf8) })
                let reserved = [Array(manifestFilename.utf8), Array(envelopeFilename.utf8)]
                for path in tree.paths where !declaredPaths.contains(path) && !reserved.contains(path) {
                    throw Refusal(.fileUndeclared)
                }
            }
        } catch let refusal as Refusal {
            return PortableBundleVerification(
                ok: false, reason: refusal.reason, contentID: contentID, validSignatures: validSignatures,
                eligibleRuntimes: eligible, filesChecked: checked, filesVerified: false)
        } catch {
            return PortableBundleVerification(
                ok: false, reason: .manifestMalformed, contentID: contentID, validSignatures: validSignatures,
                eligibleRuntimes: eligible, filesChecked: checked, filesVerified: false)
        }
        return PortableBundleVerification(
            ok: true, reason: nil, contentID: contentID, validSignatures: validSignatures, eligibleRuntimes: eligible,
            filesChecked: checked, filesVerified: tree != nil)
    }

    // MARK: - Helpers

    struct Refusal: Error {
        let reason: PortableBundleReason
        init(_ reason: PortableBundleReason) { self.reason = reason }
    }

    struct ValidatedFile {
        let path: String
        let size: Int64
        let sha256: String
    }

    struct ValidatedRuntime {
        let format: String
        let platforms: [(name: String, minOS: String?, maxOS: String?)]
        let minMemory: Int64
        let minStorage: Int64
    }

    struct ValidatedManifest {
        let bundleID: String
        let sequence: Int64
        let issuedAt: String
        let expiresAt: String
        let sdks: [(name: String, min: String, max: String?)]
        let policyRange: (Int64, Int64)
        let runtimes: [ValidatedRuntime]
        let files: [ValidatedFile]
    }

    static func same(_ left: String, _ right: String) -> Bool {
        left.utf8.elementsEqual(right.utf8)
    }

    static func utf8Less(_ left: String, _ right: String) -> Bool {
        left.utf8.lexicographicallyPrecedes(right.utf8)
    }

    static func sha256Hex(_ data: Data) -> String {
        #if canImport(CryptoKit)
            return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        #else
            return ""
        #endif
    }

    static func ed25519Verify(publicKey: Data, signature: Data, message: Data) throws -> Bool {
        #if canImport(CryptoKit)
            guard let key = try? Curve25519.Signing.PublicKey(rawRepresentation: publicKey) else { return false }
            return key.isValidSignature(signature, for: message)
        #else
            throw Refusal(.signatureVerifierUnavailable)
        #endif
    }

    private static let base64Alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".utf8)

    static func base64URLEncode(_ data: Data) -> String {
        var output: [UInt8] = []
        let bytes = [UInt8](data)
        var index = 0
        while index < bytes.count {
            let remaining = bytes.count - index
            let a = UInt32(bytes[index])
            let b = remaining > 1 ? UInt32(bytes[index + 1]) : 0
            let c = remaining > 2 ? UInt32(bytes[index + 2]) : 0
            let triple = (a << 16) | (b << 8) | c
            output.append(base64Alphabet[Int((triple >> 18) & 63)])
            output.append(base64Alphabet[Int((triple >> 12) & 63)])
            if remaining > 1 { output.append(base64Alphabet[Int((triple >> 6) & 63)]) }
            if remaining > 2 { output.append(base64Alphabet[Int(triple & 63)]) }
            index += 3
        }
        return String(decoding: output, as: UTF8.self)
    }

    static func base64URLDecode(_ text: String, length: Int) -> Data? {
        let scalars = Array(text.utf8)
        guard !scalars.isEmpty, scalars.count % 4 != 1 else { return nil }
        var bytes: [UInt8] = []
        var accumulator: UInt32 = 0
        var bits = 0
        for scalar in scalars {
            guard let value = base64Alphabet.firstIndex(of: scalar) else { return nil }
            accumulator = (accumulator << 6) | UInt32(value)
            bits += 6
            if bits >= 8 {
                bits -= 8
                bytes.append(UInt8((accumulator >> UInt32(bits)) & 0xff))
            }
        }
        let data = Data(bytes)
        guard data.count == length, same(base64URLEncode(data), text) else { return nil }
        return data
    }

    static func parseCanonical(_ data: Data) throws -> PortableBundleJSON {
        guard let value = try? PortableBundleJSON.parse(data),
            let encoded = try? PortableBundleJSON.encode(value),
            encoded.elementsEqual(data)
        else { throw Refusal(.manifestNotCanonical) }
        return value
    }

    static func parseEnvelope(_ data: Data) throws -> (String, [[(String, PortableBundleJSON)]]) {
        guard let value = try? parseCanonical(data), case .object(let members) = value,
            case .string(let schema)? = PortableBundleJSON.member(members, "schema"), same(schema, envelopeSchema)
        else { throw Refusal(.envelopeMalformed) }
        guard case .int(let version)? = PortableBundleJSON.member(members, "schema_version") else {
            throw Refusal(.envelopeMalformed)
        }
        guard version == 1 else { throw Refusal(.envelopeVersionUnsupported) }
        guard PortableBundleJSON.hasExactKeys(members, ["schema", "schema_version", "content_id", "signatures"]),
            case .string(let contentID)? = PortableBundleJSON.member(members, "content_id"),
            PortableBundlePattern.sha256.matches(contentID),
            case .array(let rawSignatures)? = PortableBundleJSON.member(members, "signatures"),
            rawSignatures.count <= maxSignatures
        else { throw Refusal(.envelopeMalformed) }
        var signatures: [[(String, PortableBundleJSON)]] = []
        for raw in rawSignatures {
            guard case .object(let entry) = raw, PortableBundleJSON.hasExactKeys(entry, ["kid", "alg", "sig"]),
                case .string(let kid)? = PortableBundleJSON.member(entry, "kid"), PortableBundlePattern.kid.matches(kid),
                case .string? = PortableBundleJSON.member(entry, "alg"),
                case .string? = PortableBundleJSON.member(entry, "sig")
            else { throw Refusal(.envelopeMalformed) }
            signatures.append(entry)
        }
        return (contentID, signatures)
    }

    static func checkSignatures(
        _ signatures: [[(String, PortableBundleJSON)]], manifest: Data, policy: PortableBundleTrustPolicy, now: String
    ) throws -> Int {
        guard !signatures.isEmpty else { throw Refusal(.signatureMissing) }
        let message = signingInput(for: manifest)
        var valid: [String] = []
        var firstFailure: PortableBundleReason?
        for entry in signatures {
            guard case .string(let kid)? = PortableBundleJSON.member(entry, "kid"),
                case .string(let algorithm)? = PortableBundleJSON.member(entry, "alg"),
                case .string(let encoded)? = PortableBundleJSON.member(entry, "sig")
            else { throw Refusal(.envelopeMalformed) }
            if valid.contains(where: { same($0, kid) }) { continue }
            let failure = try signatureFailure(kid: kid, algorithm: algorithm, encoded: encoded, message: message, policy: policy, now: now)
            if let failure {
                if firstFailure == nil { firstFailure = failure }
            } else {
                valid.append(kid)
            }
        }
        if valid.count >= policy.threshold { return valid.count }
        throw Refusal(firstFailure ?? .signatureThresholdNotMet)
    }

    static func signatureFailure(
        kid: String, algorithm: String, encoded: String, message: Data, policy: PortableBundleTrustPolicy, now: String
    ) throws -> PortableBundleReason? {
        guard same(algorithm, "Ed25519") else { return .signatureAlgorithmUnsupported }
        guard let key = policy.key(kid) else { return .untrustedKey }
        if key.revoked { return .keyRevoked }
        if utf8Less(now, key.notBefore) { return .keyNotYetValid }
        if !utf8Less(now, key.notAfter) { return .keyExpired }
        guard let signature = base64URLDecode(encoded, length: 64),
            try ed25519Verify(publicKey: key.publicKey, signature: signature, message: message)
        else { return .signatureInvalid }
        return nil
    }

    static func versionKey(_ value: String) -> [Int64] {
        var parts = value.split(separator: ".").map { Int64($0) ?? 0 }
        while parts.count < 4 { parts.append(0) }
        return parts
    }

    static func compareVersions(_ left: String, _ right: String) -> Int {
        for (a, b) in zip(versionKey(left), versionKey(right)) where a != b {
            return a < b ? -1 : 1
        }
        return 0
    }

    static func eligibleRuntimes(_ manifest: ValidatedManifest, target: PortableBundleTarget?) throws -> [Int] {
        guard let target else { return Array(manifest.runtimes.indices) }
        guard let sdk = manifest.sdks.first(where: { same($0.name, target.sdkName) }),
            compareVersions(target.sdkVersion, sdk.min) >= 0,
            sdk.max.map({ compareVersions(target.sdkVersion, $0) < 0 }) ?? true
        else { throw Refusal(.sdkIncompatible) }
        if let policyVersion = target.policyVersion,
            !(manifest.policyRange.0 <= policyVersion && policyVersion < manifest.policyRange.1)
        {
            throw Refusal(.policyIncompatible)
        }
        let indexes = manifest.runtimes.indices.filter { runtimeFits(manifest.runtimes[$0], target: target) }
        guard !indexes.isEmpty else { throw Refusal(.platformIncompatible) }
        return indexes
    }

    static func runtimeFits(_ runtime: ValidatedRuntime, target: PortableBundleTarget) -> Bool {
        guard target.runtimes.contains(where: { same($0, runtime.format) }) else { return false }
        if let memory = target.memoryBytes, runtime.minMemory > memory { return false }
        if let storage = target.storageBytes, runtime.minStorage > storage { return false }
        guard let platform = runtime.platforms.first(where: { same($0.name, target.platform) }) else { return false }
        if platform.minOS == nil && platform.maxOS == nil { return true }
        guard let osVersion = target.osVersion else { return false }
        return (platform.minOS.map { compareVersions(osVersion, $0) >= 0 } ?? true)
            && (platform.maxOS.map { compareVersions(osVersion, $0) < 0 } ?? true)
    }
}

// MARK: - Manifest structure

extension PortableBundle {
    fileprivate typealias Members = [(String, PortableBundleJSON)]

    private static func object(_ value: PortableBundleJSON?, _ keys: [String]) throws -> Members {
        guard case .object(let members)? = value else { throw Refusal(.manifestMalformed) }
        for (key, _) in members where !keys.contains(where: { same($0, key) }) {
            throw Refusal(.unknownCriticalField)
        }
        guard members.count == keys.count else { throw Refusal(.manifestMalformed) }
        return members
    }

    private static func string(_ value: PortableBundleJSON?, _ pattern: PortableBundlePattern) throws -> String {
        guard case .string(let text)? = value, pattern.matches(text) else { throw Refusal(.manifestMalformed) }
        return text
    }

    private static func choice(_ value: PortableBundleJSON?, _ choices: Set<String>) throws -> String {
        guard case .string(let text)? = value, text.utf8.allSatisfy({ $0 < 0x80 }), choices.contains(text) else {
            throw Refusal(.manifestMalformed)
        }
        return text
    }

    private static func uint(_ value: PortableBundleJSON?, minimum: Int64 = 0) throws -> Int64 {
        guard case .int(let number)? = value, number >= minimum, number <= maxSafeInteger else {
            throw Refusal(.manifestMalformed)
        }
        return number
    }

    private static func array(_ value: PortableBundleJSON?, _ minimum: Int, _ maximum: Int) throws -> [PortableBundleJSON] {
        guard case .array(let items)? = value, items.count >= minimum, items.count <= maximum else {
            throw Refusal(.manifestMalformed)
        }
        return items
    }

    private static func unique(_ items: [String]) throws {
        guard Set(items.map { Array($0.utf8) }).count == items.count else { throw Refusal(.manifestMalformed) }
    }

    private static func path(_ value: PortableBundleJSON?) throws -> String {
        guard case .string(let text)? = value else { throw Refusal(.manifestMalformed) }
        let segments = text.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard text.utf8.count > 0, text.utf8.count <= 512, segments.count <= 16,
            segments.allSatisfy({ PortableBundlePattern.pathSegment.matches($0) }),
            !same(text.lowercased(), manifestFilename), !same(text.lowercased(), envelopeFilename)
        else { throw Refusal(.pathUnsafe) }
        return text
    }

    private static func pathList(_ value: PortableBundleJSON?, _ minimum: Int, _ maximum: Int) throws -> [String] {
        let items = try array(value, minimum, maximum).map { try path($0) }
        try unique(items)
        return items
    }

    private static func version(_ value: PortableBundleJSON?) throws -> String {
        try string(value, .version)
    }

    private static func isNull(_ value: PortableBundleJSON?) -> Bool {
        if case .null? = value { return true }
        return false
    }

    static func validate(_ value: PortableBundleJSON, understood: Set<String>) throws -> ValidatedManifest {
        guard case .object(let root) = value else { throw Refusal(.manifestMalformed) }
        let member = { (members: Members, key: String) in PortableBundleJSON.member(members, key) }
        guard case .string(let schemaName)? = member(root, "schema"), same(schemaName, schema) else {
            throw Refusal(.schemaUnsupported)
        }
        guard case .int(let declaredVersion)? = member(root, "schema_version") else { throw Refusal(.manifestMalformed) }
        guard declaredVersion == schemaVersion else { throw Refusal(.schemaVersionUnsupported) }
        let top = try object(value, topKeys)
        var references: [String] = []

        let bundle = try object(member(top, "bundle"), ["id", "version", "sequence", "issued_at", "expires_at"])
        let bundleID = try string(member(bundle, "id"), .bundleID)
        _ = try string(member(bundle, "version"), .semver)
        let sequence = try uint(member(bundle, "sequence"))
        let issuedAt = try string(member(bundle, "issued_at"), .timestamp)
        let expiresAt = try string(member(bundle, "expires_at"), .timestamp)
        guard utf8Less(issuedAt, expiresAt) else { throw Refusal(.manifestMalformed) }

        let model = try object(member(top, "model"), ["id", "revision", "family", "architecture", "task", "languages"])
        _ = try string(member(model, "id"), .modelID)
        if !isNull(member(model, "revision")) { _ = try string(member(model, "revision"), .revision) }
        _ = try string(member(model, "family"), .token)
        _ = try string(member(model, "architecture"), .token)
        _ = try string(member(model, "task"), .token)
        try unique(try array(member(model, "languages"), 1, 32).map { try string($0, .language) })

        if !isNull(member(top, "tokenizer")) {
            let tokenizer = try object(member(top, "tokenizer"), ["kind", "contract_version", "files"])
            _ = try string(member(tokenizer, "kind"), .token)
            _ = try uint(member(tokenizer, "contract_version"), minimum: 1)
            references += try pathList(member(tokenizer, "files"), 1, 64)
        }
        if !isNull(member(top, "labels")) {
            let labels = try object(member(top, "labels"), ["path", "count", "scheme"])
            references.append(try path(member(labels, "path")))
            _ = try uint(member(labels, "count"), minimum: 1)
            _ = try string(member(labels, "scheme"), .token)
        }
        let offsets = try object(member(top, "offsets"), ["unit", "contract_version"])
        _ = try choice(member(offsets, "unit"), offsetUnits)
        _ = try uint(member(offsets, "contract_version"), minimum: 1)
        let decoder = try object(member(top, "decoder"), ["kind", "contract_version"])
        _ = try string(member(decoder, "kind"), .token)
        _ = try uint(member(decoder, "contract_version"), minimum: 1)
        if !isNull(member(top, "segmenter")) {
            let segmenter = try object(member(top, "segmenter"), ["id", "files"])
            _ = try string(member(segmenter, "id"), .token)
            references += try pathList(member(segmenter, "files"), 1, 64)
        }
        let policy = try object(member(top, "policy"), ["min_version", "max_version_exclusive", "path"])
        let policyMin = try uint(member(policy, "min_version"), minimum: 1)
        let policyMax = try uint(member(policy, "max_version_exclusive"), minimum: 1)
        guard policyMin < policyMax else { throw Refusal(.manifestMalformed) }
        if !isNull(member(policy, "path")) { references.append(try path(member(policy, "path"))) }
        if !isNull(member(top, "calibration")) {
            let calibration = try object(member(top, "calibration"), ["profile_version", "path"])
            _ = try uint(member(calibration, "profile_version"), minimum: 1)
            references.append(try path(member(calibration, "path")))
        }

        let compatibility = try object(member(top, "compatibility"), ["sdks"])
        var sdks: [(name: String, min: String, max: String?)] = []
        for entry in try array(member(compatibility, "sdks"), 1, 16) {
            let sdk = try object(entry, ["name", "min", "max_exclusive"])
            let name = try string(member(sdk, "name"), .token)
            let minimum = try version(member(sdk, "min"))
            let maximum = isNull(member(sdk, "max_exclusive")) ? nil : try version(member(sdk, "max_exclusive"))
            if let maximum, compareVersions(minimum, maximum) >= 0 { throw Refusal(.manifestMalformed) }
            sdks.append((name, minimum, maximum))
        }
        try unique(sdks.map(\.name))

        var runtimes: [ValidatedRuntime] = []
        var runtimeIDs: [String] = []
        var fallbacks: [String?] = []
        let runtimeKeys = [
            "id", "format", "precision", "entrypoint", "files", "platforms", "min_memory_bytes", "min_storage_bytes",
            "fallback",
        ]
        for entry in try array(member(top, "runtimes"), 1, 32) {
            let runtime = try object(entry, runtimeKeys)
            let runtimeID = try string(member(runtime, "id"), .token)
            let format = try choice(member(runtime, "format"), Set(formatPlatforms.keys))
            let precision = try choice(member(runtime, "precision"), precisions)
            let entrypoint = try path(member(runtime, "entrypoint"))
            let files = try pathList(member(runtime, "files"), 1, 1024)
            guard files.contains(where: { same($0, entrypoint) }) else { throw Refusal(.manifestMalformed) }
            references += files
            var platforms: [(name: String, minOS: String?, maxOS: String?)] = []
            for item in try array(member(runtime, "platforms"), 1, 16) {
                let platform = try object(item, ["platform", "min_os", "max_os_exclusive"])
                let name = try choice(member(platform, "platform"), self.platforms)
                let minOS = isNull(member(platform, "min_os")) ? nil : try version(member(platform, "min_os"))
                let maxOS =
                    isNull(member(platform, "max_os_exclusive")) ? nil : try version(member(platform, "max_os_exclusive"))
                if let minOS, let maxOS, compareVersions(minOS, maxOS) >= 0 { throw Refusal(.manifestMalformed) }
                platforms.append((name, minOS, maxOS))
            }
            try unique(platforms.map(\.name))
            let minMemory = try uint(member(runtime, "min_memory_bytes"))
            let minStorage = try uint(member(runtime, "min_storage_bytes"))
            let fallback = isNull(member(runtime, "fallback")) ? nil : try string(member(runtime, "fallback"), .token)
            guard formatPrecisions[format]?.contains(precision) == true,
                platforms.allSatisfy({ formatPlatforms[format]?.contains($0.name) == true })
            else { throw Refusal(.runtimeCombinationUnsupported) }
            runtimes.append(ValidatedRuntime(format: format, platforms: platforms, minMemory: minMemory, minStorage: minStorage))
            runtimeIDs.append(runtimeID)
            fallbacks.append(fallback)
        }

        for entry in try array(member(top, "licenses"), 1, 16) {
            let license = try object(entry, ["id", "files"])
            _ = try string(member(license, "id"), .license)
            references += try pathList(member(license, "files"), 0, 64)
        }
        let provenance = try object(member(top, "provenance"), ["source_digest", "export_record"])
        if !isNull(member(provenance, "source_digest")) { _ = try string(member(provenance, "source_digest"), .sha256) }
        if !isNull(member(provenance, "export_record")) { references.append(try path(member(provenance, "export_record"))) }
        for entry in try array(member(top, "evidence"), 0, 64) {
            let evidence = try object(entry, ["kind", "path"])
            _ = try string(member(evidence, "kind"), .token)
            references.append(try path(member(evidence, "path")))
        }

        var files: [ValidatedFile] = []
        for entry in try array(member(top, "files"), 1, 65536) {
            let item = try object(entry, ["path", "media_type", "size_bytes", "sha256", "role", "executable"])
            let filePath = try path(member(item, "path"))
            _ = try string(member(item, "media_type"), .mediaType)
            let size = try uint(member(item, "size_bytes"))
            let digest = try string(member(item, "sha256"), .sha256)
            _ = try choice(member(item, "role"), fileRoles)
            guard case .bool(let executable)? = member(item, "executable") else { throw Refusal(.manifestMalformed) }
            if executable { throw Refusal(.executableFileForbidden) }
            files.append(ValidatedFile(path: filePath, size: size, sha256: digest))
        }

        guard case .object(let extensions)? = member(top, "extensions"), extensions.count <= 64 else {
            throw Refusal(.manifestMalformed)
        }
        for (name, _) in extensions { _ = try string(.string(name), .extensionName) }
        let critical = try array(member(top, "critical_extensions"), 0, 64).map { try string($0, .extensionName) }
        try unique(critical)
        guard critical.allSatisfy({ name in extensions.contains(where: { same($0.0, name) }) }) else {
            throw Refusal(.manifestMalformed)
        }

        let paths = files.map(\.path)
        let folded = paths.map { $0.lowercased() }
        guard Set(folded).count == folded.count else { throw Refusal(.duplicatePath) }
        let foldedSet = Set(folded)
        for item in folded {
            let segments = item.split(separator: "/").map(String.init)
            for index in segments.indices.dropFirst() where foldedSet.contains(segments[..<index].joined(separator: "/")) {
                throw Refusal(.pathConflict)
            }
        }
        for index in paths.indices.dropLast() where !utf8Less(paths[index], paths[index + 1]) {
            throw Refusal(.fileOrderInvalid)
        }
        let declared = Set(paths)
        guard references.allSatisfy({ declared.contains($0) }) else { throw Refusal(.fileReferenceUndeclared) }

        try unique(runtimeIDs)
        for (index, fallback) in fallbacks.enumerated() {
            guard var current = fallback else { continue }
            guard runtimeIDs.contains(current), current != runtimeIDs[index] else { throw Refusal(.fallbackInvalid) }
            var seen: Set<String> = [runtimeIDs[index]]
            while true {
                guard !seen.contains(current), let position = runtimeIDs.firstIndex(of: current) else {
                    throw Refusal(.fallbackInvalid)
                }
                seen.insert(current)
                guard let next = fallbacks[position] else { break }
                current = next
            }
        }

        guard critical.allSatisfy({ understood.contains($0) }) else { throw Refusal(.unknownCriticalExtension) }

        return ValidatedManifest(
            bundleID: bundleID, sequence: sequence, issuedAt: issuedAt, expiresAt: expiresAt, sdks: sdks,
            policyRange: (policyMin, policyMax), runtimes: runtimes, files: files)
    }
}

// MARK: - File trees

/// Read-only file tree; paths compare by exact UTF-8 bytes.
protocol PortableBundleTree {
    var paths: [[UInt8]] { get }
    func size(_ path: String) -> Int64?
    func digest(_ path: String) -> String?
}

struct MemoryTree: PortableBundleTree {
    private let files: [[UInt8]: Data]

    init(files: [String: Data]) {
        self.files = Dictionary(files.map { (Array($0.key.utf8), $0.value) }, uniquingKeysWith: { first, _ in first })
    }

    var paths: [[UInt8]] { Array(files.keys) }

    func size(_ path: String) -> Int64? {
        files[Array(path.utf8)].map { Int64($0.count) }
    }

    func digest(_ path: String) -> String? {
        files[Array(path.utf8)].map { "sha256:" + PortableBundle.sha256Hex($0) }
    }
}

struct DirectoryTree: PortableBundleTree {
    private let root: URL
    private let entries: [[UInt8]: (relative: String, size: Int64)]

    init(root: URL) throws {
        let manager = FileManager.default
        var isDirectory: ObjCBool = false
        guard manager.fileExists(atPath: root.path, isDirectory: &isDirectory), isDirectory.boolValue,
            let rootAttributes = try? manager.attributesOfItem(atPath: root.path),
            rootAttributes[.type] as? FileAttributeType != .typeSymbolicLink,
            let subpaths = try? manager.subpathsOfDirectory(atPath: root.path)
        else { throw PortableBundle.Refusal(.fileMissing) }
        var entries: [[UInt8]: (String, Int64)] = [:]
        for relative in subpaths {
            let full = root.appendingPathComponent(relative).path
            guard let attributes = try? manager.attributesOfItem(atPath: full) else {
                throw PortableBundle.Refusal(.fileMissing)
            }
            switch attributes[.type] as? FileAttributeType {
            case .typeDirectory?: continue
            case .typeRegular?:
                let size = (attributes[.size] as? NSNumber)?.int64Value ?? -1
                entries[Array(relative.utf8)] = (relative, size)
            default:
                throw PortableBundle.Refusal(.pathUnsafe)
            }
        }
        self.root = root
        self.entries = entries
    }

    var paths: [[UInt8]] { Array(entries.keys) }

    func size(_ path: String) -> Int64? {
        entries[Array(path.utf8)]?.size
    }

    func read(_ path: String, limit: Int) -> Data? {
        guard let entry = entries[Array(path.utf8)],
            let handle = try? FileHandle(forReadingFrom: root.appendingPathComponent(entry.relative))
        else { return nil }
        defer { try? handle.close() }
        return try? handle.read(upToCount: limit + 1) ?? Data()
    }

    func digest(_ path: String) -> String? {
        #if canImport(CryptoKit)
            guard let entry = entries[Array(path.utf8)],
                let handle = try? FileHandle(forReadingFrom: root.appendingPathComponent(entry.relative))
            else { return nil }
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

// MARK: - Patterns

/// ASCII-only validators; non-ASCII input never matches.
struct PortableBundlePattern: @unchecked Sendable {
    // NSRegularExpression is immutable and documented as thread-safe.
    private let expression: NSRegularExpression?

    init(_ pattern: String) {
        expression = try? NSRegularExpression(pattern: "^(?:" + pattern + ")\\z")
    }

    func matches(_ text: String) -> Bool {
        guard text.utf8.allSatisfy({ $0 < 0x80 }), let expression else { return false }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return expression.firstMatch(in: text, options: [], range: range) != nil
    }

    static let token = Self("[a-z0-9][a-z0-9._-]{0,63}")
    static let bundleID = Self("[a-z0-9][a-z0-9._-]{0,127}")
    static let modelID = Self("[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
    static let revision = Self("[0-9a-f]{40}")
    static let language = Self("[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}")
    static let license = Self("[a-z0-9][a-z0-9.+-]{0,63}")
    static let sha256 = Self("sha256:[0-9a-f]{64}")
    static let mediaType = Self("[a-z]+/[a-z0-9][a-z0-9.+-]{0,126}")
    static let extensionName = Self("[a-z0-9][a-z0-9-]{0,31}(?:\\.[a-z0-9][a-z0-9_-]{0,31}){1,7}")
    static let semver = Self("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){2}")
    static let version = Self("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){0,3}")
    static let timestamp = Self(
        "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z")
    static let kid = Self("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    static let pathSegment = Self("[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}")
}

// MARK: - Strict JSON

/// Minimal strict JSON value used for canonical checks.
///
/// Object members keep their exact keys and order, so canonically equivalent
/// but distinct Unicode keys are never merged the way Swift `String` keys in a
/// `Dictionary` would be.
enum PortableBundleJSON: Sendable {
    case null
    case bool(Bool)
    case int(Int64)
    case string(String)
    case array([PortableBundleJSON])
    case object([(String, PortableBundleJSON)])

    struct Invalid: Error {}

    static func member(_ members: [(String, PortableBundleJSON)], _ key: String) -> PortableBundleJSON? {
        members.first(where: { PortableBundle.same($0.0, key) })?.1
    }

    static func hasExactKeys(_ members: [(String, PortableBundleJSON)], _ keys: [String]) -> Bool {
        members.count == keys.count && keys.allSatisfy { key in members.contains(where: { PortableBundle.same($0.0, key) }) }
    }

    static func parse(_ data: Data) throws -> PortableBundleJSON {
        var parser = Parser(bytes: [UInt8](data))
        parser.skipWhitespace()
        let value = try parser.value(depth: 0)
        parser.skipWhitespace()
        guard parser.index == parser.bytes.count else { throw Invalid() }
        return value
    }

    static func encode(_ value: PortableBundleJSON) throws -> [UInt8] {
        var output: [UInt8] = []
        try encode(value, into: &output, depth: 0)
        return output
    }

    private static func encode(_ value: PortableBundleJSON, into output: inout [UInt8], depth: Int) throws {
        guard depth <= 32 else { throw Invalid() }
        switch value {
        case .null: output += Array("null".utf8)
        case .bool(let flag): output += Array((flag ? "true" : "false").utf8)
        case .int(let number):
            guard number >= -PortableBundle.maxSafeInteger, number <= PortableBundle.maxSafeInteger else { throw Invalid() }
            output += Array(String(number).utf8)
        case .string(let text): encodeString(text, into: &output)
        case .array(let items):
            output.append(UInt8(ascii: "["))
            for (index, item) in items.enumerated() {
                if index > 0 { output.append(UInt8(ascii: ",")) }
                try encode(item, into: &output, depth: depth + 1)
            }
            output.append(UInt8(ascii: "]"))
        case .object(let members):
            let sorted = members.sorted { $0.0.utf16.lexicographicallyPrecedes($1.0.utf16) }
            for index in sorted.indices.dropFirst() where sorted[index - 1].0.utf16.elementsEqual(sorted[index].0.utf16) {
                throw Invalid()
            }
            output.append(UInt8(ascii: "{"))
            for (index, member) in sorted.enumerated() {
                if index > 0 { output.append(UInt8(ascii: ",")) }
                encodeString(member.0, into: &output)
                output.append(UInt8(ascii: ":"))
                try encode(member.1, into: &output, depth: depth + 1)
            }
            output.append(UInt8(ascii: "}"))
        }
    }

    private static func encodeString(_ text: String, into output: inout [UInt8]) {
        output.append(UInt8(ascii: "\""))
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x22: output += Array("\\\"".utf8)
            case 0x5C: output += Array("\\\\".utf8)
            case 0x08: output += Array("\\b".utf8)
            case 0x09: output += Array("\\t".utf8)
            case 0x0A: output += Array("\\n".utf8)
            case 0x0C: output += Array("\\f".utf8)
            case 0x0D: output += Array("\\r".utf8)
            case 0..<0x20: output += Array(String(format: "\\u%04x", scalar.value).utf8)
            default: output += Array(String(scalar).utf8)
            }
        }
        output.append(UInt8(ascii: "\""))
    }

    private struct Parser {
        let bytes: [UInt8]
        var index = 0

        mutating func skipWhitespace() {
            while index < bytes.count, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[index]) { index += 1 }
        }

        mutating func expect(_ literal: String) throws {
            for byte in literal.utf8 {
                guard index < bytes.count, bytes[index] == byte else { throw Invalid() }
                index += 1
            }
        }

        mutating func value(depth: Int) throws -> PortableBundleJSON {
            guard depth <= 32, index < bytes.count else { throw Invalid() }
            switch bytes[index] {
            case UInt8(ascii: "n"):
                try expect("null")
                return .null
            case UInt8(ascii: "t"):
                try expect("true")
                return .bool(true)
            case UInt8(ascii: "f"):
                try expect("false")
                return .bool(false)
            case UInt8(ascii: "\""):
                return .string(try string())
            case UInt8(ascii: "["):
                index += 1
                var items: [PortableBundleJSON] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == UInt8(ascii: "]") {
                    index += 1
                    return .array(items)
                }
                while true {
                    skipWhitespace()
                    items.append(try value(depth: depth + 1))
                    skipWhitespace()
                    guard index < bytes.count else { throw Invalid() }
                    if bytes[index] == UInt8(ascii: "]") {
                        index += 1
                        return .array(items)
                    }
                    try expect(",")
                }
            case UInt8(ascii: "{"):
                index += 1
                var members: [(String, PortableBundleJSON)] = []
                skipWhitespace()
                if index < bytes.count, bytes[index] == UInt8(ascii: "}") {
                    index += 1
                    return .object(members)
                }
                while true {
                    skipWhitespace()
                    guard index < bytes.count, bytes[index] == UInt8(ascii: "\"") else { throw Invalid() }
                    let key = try string()
                    skipWhitespace()
                    try expect(":")
                    skipWhitespace()
                    members.append((key, try value(depth: depth + 1)))
                    skipWhitespace()
                    guard index < bytes.count else { throw Invalid() }
                    if bytes[index] == UInt8(ascii: "}") {
                        index += 1
                        return .object(members)
                    }
                    try expect(",")
                }
            default:
                return .int(try integer())
            }
        }

        mutating func integer() throws -> Int64 {
            let start = index
            if index < bytes.count, bytes[index] == UInt8(ascii: "-") { index += 1 }
            let digitsStart = index
            while index < bytes.count, (0x30...0x39).contains(bytes[index]) { index += 1 }
            let digits = index - digitsStart
            guard digits > 0, digits <= 16, !(digits > 1 && bytes[digitsStart] == 0x30) else { throw Invalid() }
            if index < bytes.count, [UInt8(ascii: "."), UInt8(ascii: "e"), UInt8(ascii: "E")].contains(bytes[index]) {
                throw Invalid()
            }
            guard let text = String(bytes: bytes[start..<index], encoding: .ascii), let number = Int64(text),
                abs(number) <= PortableBundle.maxSafeInteger
            else { throw Invalid() }
            return number
        }

        mutating func hexQuad() throws -> UInt32 {
            guard index + 4 <= bytes.count,
                let text = String(bytes: bytes[index..<index + 4], encoding: .ascii),
                let value = UInt32(text, radix: 16)
            else { throw Invalid() }
            index += 4
            return value
        }

        mutating func string() throws -> String {
            try expect("\"")
            var output: [UInt8] = []
            while true {
                guard index < bytes.count else { throw Invalid() }
                let byte = bytes[index]
                index += 1
                switch byte {
                case UInt8(ascii: "\""):
                    guard let text = String(bytes: output, encoding: .utf8) else { throw Invalid() }
                    return text
                case UInt8(ascii: "\\"):
                    guard index < bytes.count else { throw Invalid() }
                    let escape = bytes[index]
                    index += 1
                    switch escape {
                    case UInt8(ascii: "\""): output.append(0x22)
                    case UInt8(ascii: "\\"): output.append(0x5C)
                    case UInt8(ascii: "/"): output.append(0x2F)
                    case UInt8(ascii: "b"): output.append(0x08)
                    case UInt8(ascii: "f"): output.append(0x0C)
                    case UInt8(ascii: "n"): output.append(0x0A)
                    case UInt8(ascii: "r"): output.append(0x0D)
                    case UInt8(ascii: "t"): output.append(0x09)
                    case UInt8(ascii: "u"):
                        var code = try hexQuad()
                        if (0xD800...0xDBFF).contains(code) {
                            try expect("\\u")
                            let low = try hexQuad()
                            guard (0xDC00...0xDFFF).contains(low) else { throw Invalid() }
                            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
                        } else if (0xDC00...0xDFFF).contains(code) {
                            throw Invalid()
                        }
                        guard let scalar = Unicode.Scalar(code) else { throw Invalid() }
                        output += Array(String(scalar).utf8)
                    default:
                        throw Invalid()
                    }
                case 0..<0x20:
                    throw Invalid()
                default:
                    output.append(byte)
                }
            }
        }
    }
}
