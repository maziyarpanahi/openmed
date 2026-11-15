import CryptoKit
import Foundation

/// What a provider-held secret is used for.
///
/// The purpose is folded into the opaque reference, so one caller label under
/// two purposes names two separate items.
public enum OpenMedSecretPurpose: String, CaseIterable, Codable, Sendable {
    /// Key material a de-identification policy uses, such as a surrogate or hash key.
    case policySecret = "policy_secret"
    /// Verification material for portable bundles, such as a public key or a pin set.
    case bundleTrustAnchor = "bundle_trust_anchor"
}

/// Value-free failures from a secret provider.
///
/// Each case is a fixed code. No case carries, and no description repeats,
/// secret bytes, caller labels, Keychain service or account strings, access
/// groups, paths or raw platform messages.
public enum OpenMedSecretProviderError: String, Error, LocalizedError, CaseIterable, Sendable {
    /// The caller label or opaque digest is not a valid reference.
    case invalidReference = "secret_invalid_reference"
    /// The secret is empty or larger than the provider accepts.
    case invalidSecret = "secret_invalid_value"
    /// An expected version is not a positive integer, or the next version would overflow.
    case invalidVersion = "secret_invalid_version"
    /// The provider configuration is incomplete or malformed.
    case invalidConfiguration = "secret_invalid_configuration"
    /// No item exists for the reference.
    case missing = "secret_missing"
    /// An item already exists; rotate it instead of storing it again.
    case duplicate = "secret_duplicate"
    /// The device is locked, or the system did not allow user interaction.
    case locked = "secret_locked"
    /// The user dismissed the authentication prompt.
    case userCanceled = "secret_user_canceled"
    /// Authentication failed or is locked out.
    case authenticationFailed = "secret_authentication_failed"
    /// The item exists but can no longer be read, or its stored metadata is not trustworthy.
    case invalidated = "secret_invalidated"
    /// The item is at a different version than the caller expected.
    case rotated = "secret_rotated"
    /// The stored item's protection does not match the configured protection.
    case protectionMismatch = "secret_protection_mismatch"
    /// The configured protection is not available on this device or OS.
    case unsupported = "secret_unsupported"
    /// The process lacks the Keychain entitlement the configuration needs.
    case missingEntitlement = "secret_missing_entitlement"
    /// No Keychain is available to the process.
    case unavailable = "secret_unavailable"
    /// The platform returned a status this adapter does not classify.
    case unexpectedStatus = "secret_unexpected_status"

    public var errorDescription: String? { rawValue }
}

/// An opaque, caller-managed handle for one provider-held secret.
///
/// The caller label is hashed with the purpose and is not retained, so the
/// reference, its storage name and every rendering of it carry only a
/// SHA-256 digest. Use labels that are application configuration names, never
/// patient or user data.
public struct OpenMedSecretReference: Hashable, Sendable, Codable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    /// The largest caller label accepted, in UTF-8 bytes.
    public static let maximumLabelBytes = 256

    private static let domain = "openmed.secret-reference.v1"

    /// What the referenced secret is used for.
    public let purpose: OpenMedSecretPurpose
    /// Lowercase hexadecimal SHA-256 digest of the purpose and caller label.
    public let digest: String

    /// Create a reference from a caller label.
    ///
    /// - Parameters:
    ///   - purpose: What the secret is used for.
    ///   - label: A non-blank application label without control characters,
    ///     at most ``maximumLabelBytes`` UTF-8 bytes. It is not retained.
    /// - Throws: ``OpenMedSecretProviderError/invalidReference`` for an invalid label.
    public init(purpose: OpenMedSecretPurpose, label: String) throws {
        guard Self.isValidLabel(label) else {
            throw OpenMedSecretProviderError.invalidReference
        }
        var hash = SHA256()
        hash.update(data: Data(Self.domain.utf8))
        for part in [purpose.rawValue, label] {
            hash.update(data: Data([0]))
            hash.update(data: Data(part.utf8))
        }
        self.purpose = purpose
        self.digest = hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Recreate a reference from a digest a provider or policy file recorded.
    ///
    /// - Throws: ``OpenMedSecretProviderError/invalidReference`` unless the
    ///   digest is 64 lowercase hexadecimal characters.
    public init(purpose: OpenMedSecretPurpose, digest: String) throws {
        guard digest.utf8.count == 64, digest.utf8.allSatisfy(Self.isLowercaseHex) else {
            throw OpenMedSecretProviderError.invalidReference
        }
        self.purpose = purpose
        self.digest = digest
    }

    private enum CodingKeys: String, CodingKey {
        case purpose
        case digest
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            purpose: container.decode(OpenMedSecretPurpose.self, forKey: .purpose),
            digest: container.decode(String.self, forKey: .digest)
        )
    }

    /// A short, non-reversible handle for logs, receipts and diagnostics.
    public var fingerprint: String { "ref:" + digest.prefix(16) }

    public var description: String {
        "OpenMedSecretReference(\(purpose.rawValue), \(fingerprint))"
    }

    public var debugDescription: String { description }

    public var customMirror: Mirror {
        Mirror(self, children: ["purpose": purpose, "fingerprint": fingerprint], displayStyle: .struct)
    }

    private static func isValidLabel(_ label: String) -> Bool {
        let byteCount = label.utf8.count
        guard byteCount > 0, byteCount <= maximumLabelBytes else { return false }
        guard !label.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        return !label.unicodeScalars.contains { scalar in
            scalar.properties.generalCategory == .control || CharacterSet.newlines.contains(scalar)
        }
    }

    private static func isLowercaseHex(_ byte: UInt8) -> Bool {
        (UInt8(ascii: "0")...UInt8(ascii: "9")).contains(byte) || (UInt8(ascii: "a")...UInt8(ascii: "f")).contains(byte)
    }
}

/// Non-secret state of one stored secret.
public struct OpenMedSecretMetadata: Equatable, Sendable, Codable, CustomStringConvertible {
    /// The opaque reference the item is stored under.
    public let reference: OpenMedSecretReference
    /// The current version; stores start at 1 and each rotation adds 1.
    public let version: Int

    /// Create metadata for a provider result.
    public init(reference: OpenMedSecretReference, version: Int) {
        self.reference = reference
        self.version = version
    }

    public var description: String {
        "OpenMedSecretMetadata(\(reference.purpose.rawValue), \(reference.fingerprint), version: \(version))"
    }
}

/// A secret read from a provider.
///
/// Descriptions, debug descriptions and mirrors report the reference
/// fingerprint, version and byte count only. The type is not `Codable` so it
/// cannot be exported by accident.
public struct OpenMedSecretValue: Sendable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    /// The opaque reference the secret was read from.
    public let reference: OpenMedSecretReference
    /// The version that was read.
    public let version: Int
    /// The secret bytes. Pass them straight to the local operation that needs
    /// them; never log, cache, persist or export them.
    public let bytes: Data

    /// Create a value for a provider result.
    public init(reference: OpenMedSecretReference, version: Int, bytes: Data) {
        self.reference = reference
        self.version = version
        self.bytes = bytes
    }

    /// The number of secret bytes.
    public var byteCount: Int { bytes.count }

    /// The value's non-secret metadata.
    public var metadata: OpenMedSecretMetadata {
        OpenMedSecretMetadata(reference: reference, version: version)
    }

    public var description: String {
        "OpenMedSecretValue(\(reference.purpose.rawValue), \(reference.fingerprint), version: \(version), \(bytes.count) bytes)"
    }

    public var debugDescription: String { description }

    public var customMirror: Mirror {
        Mirror(self, children: ["reference": reference, "version": version, "byteCount": bytes.count], displayStyle: .struct)
    }
}

/// A value-free record of a store or rotation, suitable for export.
public struct OpenMedSecretReceipt: Equatable, Sendable, Codable, CustomStringConvertible {
    /// The operation a receipt records.
    public enum Operation: String, Codable, Sendable {
        case stored
        case rotated
    }

    /// The operation performed.
    public let operation: Operation
    /// The opaque reference written.
    public let reference: OpenMedSecretReference
    /// The version now stored.
    public let version: Int
    /// The number of secret bytes written.
    public let byteCount: Int

    /// Create a receipt for a provider result.
    public init(operation: Operation, reference: OpenMedSecretReference, version: Int, byteCount: Int) {
        self.operation = operation
        self.reference = reference
        self.version = version
        self.byteCount = byteCount
    }

    public var description: String {
        "OpenMedSecretReceipt(\(operation.rawValue), \(reference.purpose.rawValue), \(reference.fingerprint), version: \(version), \(byteCount) bytes)"
    }
}

/// The outcome of deleting a reference.
public enum OpenMedSecretDeletion: String, Codable, Sendable {
    case deleted
    case alreadyAbsent = "already_absent"
}

/// Whether this device can currently satisfy a provider's configured
/// passcode and user-authentication requirements.
public enum OpenMedProtectionAvailability: String, CaseIterable, Codable, Sendable {
    /// The configuration requires no passcode or user authentication.
    case notRequested = "not_requested"
    /// Every configured requirement can be evaluated now.
    case available
    /// A required device passcode is not set.
    case passcodeNotSet = "passcode_not_set"
    /// Biometry is required but the device has none, or it is disabled.
    case biometryNotAvailable = "biometry_not_available"
    /// Biometry is required but nothing is enrolled.
    case biometryNotEnrolled = "biometry_not_enrolled"
    /// Biometry is required but locked out after failed attempts.
    case biometryLockout = "biometry_lockout"
    /// The platform cannot provide the required protection.
    case unsupportedPlatform = "unsupported_platform"
    /// The platform did not say why the requirement cannot be evaluated.
    case unknown
}

/// Value-free description of the protection a provider requests and can get.
///
/// The report states configuration and probe results only. It makes no claim
/// that a secret is held in, or never leaves, secure hardware.
public struct OpenMedSecretProtectionReport: Equatable, Sendable, Codable, CustomStringConvertible {
    /// The provider kind, such as `apple_keychain`.
    public let provider: String
    /// The configured accessibility class code.
    public let accessibility: String
    /// The configured user-authentication code.
    public let userAuthentication: String
    /// `application_default` or `named`; the group itself is never reported.
    public let accessGroup: String
    /// Whether items stay on this device and are not restored to another device from a backup.
    public let deviceOnly: Bool
    /// Whether items sync through a cloud keychain.
    public let synchronizable: Bool
    /// Whether this device can currently satisfy the configured requirements.
    public let availability: OpenMedProtectionAvailability
    /// Whether the provider claims secure-hardware custody of the secret bytes.
    public let secureHardwareClaimed: Bool

    /// Create a report for a provider.
    public init(
        provider: String,
        accessibility: String,
        userAuthentication: String,
        accessGroup: String,
        deviceOnly: Bool,
        synchronizable: Bool,
        availability: OpenMedProtectionAvailability,
        secureHardwareClaimed: Bool
    ) {
        self.provider = provider
        self.accessibility = accessibility
        self.userAuthentication = userAuthentication
        self.accessGroup = accessGroup
        self.deviceOnly = deviceOnly
        self.synchronizable = synchronizable
        self.availability = availability
        self.secureHardwareClaimed = secureHardwareClaimed
    }

    public var description: String {
        "OpenMedSecretProtectionReport(\(provider), \(accessibility), user_authentication: \(userAuthentication), access_group: \(accessGroup), device_only: \(deviceOnly), synchronizable: \(synchronizable), availability: \(availability.rawValue), secure_hardware_claimed: \(secureHardwareClaimed))"
    }
}

/// A small, injectable custody boundary for policy secrets and bundle trust
/// material, addressed by opaque reference.
///
/// Providers never create secrets for the caller, never fall back to weaker
/// or plaintext storage, and report every failure as an
/// ``OpenMedSecretProviderError``. Calls may block or show a system
/// authentication prompt, so call them off the main thread.
public protocol OpenMedSecretProvider: Sendable {
    /// Read the current secret.
    ///
    /// - Parameters:
    ///   - reference: The opaque reference to read.
    ///   - expectedVersion: When set, fail with
    ///     ``OpenMedSecretProviderError/rotated`` unless the stored version
    ///     matches, instead of returning a different version.
    func get(_ reference: OpenMedSecretReference, expectedVersion: Int?) throws -> OpenMedSecretValue

    /// Read the current version without reading or authenticating the secret bytes.
    func metadata(for reference: OpenMedSecretReference) throws -> OpenMedSecretMetadata

    /// Store a caller-supplied secret as version 1. Never overwrites an existing item.
    func put(_ secret: Data, for reference: OpenMedSecretReference) throws -> OpenMedSecretReceipt

    /// Replace the secret, only if it is still at `expectedVersion`.
    func rotate(_ reference: OpenMedSecretReference, to secret: Data, expectedVersion: Int) throws -> OpenMedSecretReceipt

    /// Delete the item. Deleting an absent item is not an error.
    func delete(_ reference: OpenMedSecretReference) throws -> OpenMedSecretDeletion

    /// Describe the configured protection and whether this device can provide it.
    func protectionReport() -> OpenMedSecretProtectionReport
}

extension OpenMedSecretProvider {
    /// Read the current secret at whatever version is stored.
    public func get(_ reference: OpenMedSecretReference) throws -> OpenMedSecretValue {
        try get(reference, expectedVersion: nil)
    }
}
