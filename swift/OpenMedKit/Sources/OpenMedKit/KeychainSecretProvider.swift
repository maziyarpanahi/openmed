import Foundation
import Security

#if canImport(LocalAuthentication)
    import LocalAuthentication
#endif

/// The four `SecItem` calls the Keychain provider makes, so tests and hosts
/// can substitute a fake or an instrumented store.
///
/// Query and attribute dictionaries use the `Security` framework keys. Add
/// and update dictionaries carry secret bytes under `kSecValueData`, so an
/// implementation must never log or persist them.
public protocol OpenMedKeychainItemClient: Sendable {
    /// Mirror `SecItemAdd` without returning a result.
    func add(_ attributes: [String: Any]) -> OSStatus
    /// Mirror `SecItemCopyMatching`.
    func copyMatching(_ query: [String: Any]) -> (status: OSStatus, result: AnyObject?)
    /// Mirror `SecItemUpdate`.
    func update(_ query: [String: Any], attributes: [String: Any]) -> OSStatus
    /// Mirror `SecItemDelete`.
    func delete(_ query: [String: Any]) -> OSStatus
}

/// The system Keychain through `SecItemAdd`, `SecItemCopyMatching`,
/// `SecItemUpdate` and `SecItemDelete`.
public struct OpenMedSystemKeychainItemClient: OpenMedKeychainItemClient {
    public init() {}

    public func add(_ attributes: [String: Any]) -> OSStatus {
        SecItemAdd(attributes as CFDictionary, nil)
    }

    public func copyMatching(_ query: [String: Any]) -> (status: OSStatus, result: AnyObject?) {
        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        return (status, result)
    }

    public func update(_ query: [String: Any], attributes: [String: Any]) -> OSStatus {
        SecItemUpdate(query as CFDictionary, attributes as CFDictionary)
    }

    public func delete(_ query: [String: Any]) -> OSStatus {
        SecItemDelete(query as CFDictionary)
    }
}

/// A local-authentication capability a Keychain configuration depends on.
public enum OpenMedLocalAuthenticationRequirement: String, CaseIterable, Sendable {
    case devicePasscode = "device_passcode"
    case biometry
}

/// Reports whether this device can currently evaluate a local-authentication
/// requirement. Probing never shows a prompt.
public protocol OpenMedLocalAuthenticationProbe: Sendable {
    func availability(of requirement: OpenMedLocalAuthenticationRequirement) -> OpenMedProtectionAvailability
}

/// Probes `LocalAuthentication` with `LAContext.canEvaluatePolicy(_:error:)`.
public struct OpenMedSystemLocalAuthenticationProbe: OpenMedLocalAuthenticationProbe {
    public init() {}

    public func availability(of requirement: OpenMedLocalAuthenticationRequirement) -> OpenMedProtectionAvailability {
        #if canImport(LocalAuthentication)
            let policy: LAPolicy
            switch requirement {
            case .devicePasscode:
                policy = .deviceOwnerAuthentication
            case .biometry:
                #if os(watchOS)
                    return .unsupportedPlatform
                #else
                    policy = .deviceOwnerAuthenticationWithBiometrics
                #endif
            }
            var error: NSError?
            if LAContext().canEvaluatePolicy(policy, error: &error) {
                return .available
            }
            return Self.availability(forErrorCode: error?.code)
        #else
            return .unsupportedPlatform
        #endif
    }

    /// Map a `LocalAuthentication` error code to an availability code.
    static func availability(forErrorCode code: Int?) -> OpenMedProtectionAvailability {
        #if canImport(LocalAuthentication)
            switch code {
            case Int(kLAErrorPasscodeNotSet):
                return .passcodeNotSet
            case Int(kLAErrorBiometryNotAvailable):
                return .biometryNotAvailable
            case Int(kLAErrorBiometryNotEnrolled):
                return .biometryNotEnrolled
            case Int(kLAErrorBiometryLockout):
                return .biometryLockout
            default:
                return .unknown
            }
        #else
            return .unsupportedPlatform
        #endif
    }
}

/// Explicit Keychain placement and protection for ``OpenMedKeychainSecretProvider``.
///
/// Every item is a generic password in the data-protection Keychain, bound
/// to this device (`ThisDeviceOnly`) and never synchronizable. Descriptions
/// and mirrors omit the service and access-group strings.
public struct OpenMedKeychainConfiguration: Equatable, Sendable, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    /// When the system may decrypt an item. Only device-bound classes are offered.
    public enum Accessibility: String, CaseIterable, Codable, Sendable {
        /// `kSecAttrAccessibleWhenUnlockedThisDeviceOnly`; the default.
        case whenUnlockedThisDeviceOnly = "when_unlocked_this_device_only"
        /// `kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly`; items are removed if the passcode is removed.
        case whenPasscodeSetThisDeviceOnly = "when_passcode_set_this_device_only"
        /// `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`; readable while locked after the first unlock.
        case afterFirstUnlockThisDeviceOnly = "after_first_unlock_this_device_only"

        var constant: CFString {
            switch self {
            case .whenUnlockedThisDeviceOnly:
                return kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            case .whenPasscodeSetThisDeviceOnly:
                return kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly
            case .afterFirstUnlockThisDeviceOnly:
                return kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
            }
        }
    }

    /// Which Keychain access group holds the items.
    public enum AccessGroup: Equatable, Sendable {
        /// The application's default group; no `kSecAttrAccessGroup` is sent.
        case applicationDefault
        /// A named group the app is entitled to, such as one shared with an extension.
        case named(String)

        var code: String {
            switch self {
            case .applicationDefault:
                return "application_default"
            case .named:
                return "named"
            }
        }
    }

    /// The user authentication the system must see before releasing the bytes.
    public enum UserAuthentication: String, CaseIterable, Codable, Sendable {
        /// Only the accessibility class gates reads.
        case notRequired = "not_required"
        /// `SecAccessControlCreateFlags.userPresence`: biometry or the device passcode.
        case userPresence = "user_presence"
        /// `SecAccessControlCreateFlags.devicePasscode`.
        case devicePasscode = "device_passcode"
        /// `SecAccessControlCreateFlags.biometryAny`; survives enrollment changes.
        case biometryAny = "biometry_any"
        /// `SecAccessControlCreateFlags.biometryCurrentSet`; invalidated when enrollment changes.
        case biometryCurrentSet = "biometry_current_set"

        var flags: SecAccessControlCreateFlags? {
            switch self {
            case .notRequired:
                return nil
            case .userPresence:
                return .userPresence
            case .devicePasscode:
                return .devicePasscode
            case .biometryAny:
                return .biometryAny
            case .biometryCurrentSet:
                return .biometryCurrentSet
            }
        }
    }

    /// The largest service or access-group string accepted, in UTF-8 bytes.
    public static let maximumAttributeBytes = 256

    /// The Keychain service namespace for this provider's items.
    public let service: String
    /// The access group holding the items.
    public let accessGroup: AccessGroup
    /// When the system may decrypt the items.
    public let accessibility: Accessibility
    /// The user authentication required to read the bytes.
    public let userAuthentication: UserAuthentication

    /// Always `true`: only `ThisDeviceOnly` accessibility classes are offered.
    public var deviceOnly: Bool { true }
    /// Always `false`: items never sync through iCloud Keychain.
    public var synchronizable: Bool { false }

    /// Create a configuration.
    ///
    /// - Parameters:
    ///   - service: A non-blank namespace without control characters, at most
    ///     ``maximumAttributeBytes`` UTF-8 bytes.
    ///   - accessGroup: Where the items live; there is no implicit default.
    ///   - accessibility: When the system may decrypt the items.
    ///   - userAuthentication: The authentication required to read the bytes.
    /// - Throws: ``OpenMedSecretProviderError/invalidConfiguration`` for an
    ///   invalid service or access group.
    public init(
        service: String,
        accessGroup: AccessGroup,
        accessibility: Accessibility = .whenUnlockedThisDeviceOnly,
        userAuthentication: UserAuthentication = .notRequired
    ) throws {
        guard Self.isValidAttribute(service) else {
            throw OpenMedSecretProviderError.invalidConfiguration
        }
        if case .named(let group) = accessGroup, !Self.isValidAttribute(group) {
            throw OpenMedSecretProviderError.invalidConfiguration
        }
        self.service = service
        self.accessGroup = accessGroup
        self.accessibility = accessibility
        self.userAuthentication = userAuthentication
    }

    /// The local-authentication capabilities this configuration depends on.
    public var requirements: [OpenMedLocalAuthenticationRequirement] {
        var requirements: [OpenMedLocalAuthenticationRequirement] = []
        switch (accessibility, userAuthentication) {
        case (.whenPasscodeSetThisDeviceOnly, _), (_, .userPresence), (_, .devicePasscode):
            requirements.append(.devicePasscode)
        default:
            break
        }
        if userAuthentication == .biometryAny || userAuthentication == .biometryCurrentSet {
            requirements.append(.biometry)
        }
        return requirements
    }

    public var description: String {
        "OpenMedKeychainConfiguration(\(accessibility.rawValue), access_group: \(accessGroup.code), user_authentication: \(userAuthentication.rawValue), synchronizable: false)"
    }

    public var debugDescription: String { description }

    public var customMirror: Mirror {
        Mirror(
            self,
            children: [
                "accessibility": accessibility,
                "accessGroup": accessGroup.code,
                "userAuthentication": userAuthentication,
            ],
            displayStyle: .struct
        )
    }

    private static func isValidAttribute(_ value: String) -> Bool {
        let byteCount = value.utf8.count
        guard byteCount > 0, byteCount <= maximumAttributeBytes else { return false }
        guard !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        return !value.unicodeScalars.contains { scalar in
            scalar.properties.generalCategory == .control || CharacterSet.newlines.contains(scalar)
        }
    }
}

/// An ``OpenMedSecretProvider`` backed by the Apple data-protection Keychain.
///
/// Each reference maps to one generic-password item whose account is derived
/// from the reference digest. The item's non-secret metadata (format, purpose,
/// user-authentication code and version) lives in `kSecAttrGeneric`; the
/// secret bytes live only in `kSecValueData`. The provider never generates a
/// secret, never retains or caches bytes, never syncs, and never downgrades
/// to weaker or plaintext storage when the Keychain refuses a call.
public struct OpenMedKeychainSecretProvider: OpenMedSecretProvider, CustomStringConvertible, CustomDebugStringConvertible, CustomReflectable {
    /// The largest secret accepted, in bytes.
    public static let maximumSecretBytes = 16_384
    /// The provider kind reported in ``OpenMedSecretProtectionReport``.
    public static let providerName = "apple_keychain"

    static let accountPrefix = "omk.secret.v1."

    /// The placement and protection every item uses.
    public let configuration: OpenMedKeychainConfiguration
    private let client: any OpenMedKeychainItemClient
    private let probe: any OpenMedLocalAuthenticationProbe

    /// Create a provider.
    ///
    /// - Parameters:
    ///   - configuration: Placement and protection for every item.
    ///   - client: The `SecItem` calls; the system Keychain by default.
    ///   - authenticationProbe: Checks passcode and biometry availability
    ///     before a write; `LocalAuthentication` by default.
    public init(
        configuration: OpenMedKeychainConfiguration,
        client: any OpenMedKeychainItemClient = OpenMedSystemKeychainItemClient(),
        authenticationProbe: any OpenMedLocalAuthenticationProbe = OpenMedSystemLocalAuthenticationProbe()
    ) {
        self.configuration = configuration
        self.client = client
        self.probe = authenticationProbe
    }

    public func get(_ reference: OpenMedSecretReference, expectedVersion: Int?) throws -> OpenMedSecretValue {
        if let expectedVersion, expectedVersion < 1 {
            throw OpenMedSecretProviderError.invalidVersion
        }
        var query = itemQuery(for: reference)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnAttributes as String] = true
        query[kSecReturnData as String] = true
        let (status, result) = client.copyMatching(query)
        switch status {
        case errSecSuccess:
            break
        case errSecItemNotFound:
            // The data read found nothing. If the attributes are still
            // readable, the item exists but its access control no longer
            // releases the bytes, for example after a biometry change.
            _ = try attributes(for: reference)
            throw OpenMedSecretProviderError.invalidated
        default:
            throw Self.error(for: status)
        }
        guard let attributes = result as? [String: Any],
            let bytes = attributes[kSecValueData as String] as? Data
        else {
            throw OpenMedSecretProviderError.invalidated
        }
        let metadata = try validatedMetadata(attributes, for: reference)
        if let expectedVersion, metadata.version != expectedVersion {
            throw OpenMedSecretProviderError.rotated
        }
        guard Self.isAcceptedLength(bytes.count) else {
            throw OpenMedSecretProviderError.invalidated
        }
        return OpenMedSecretValue(reference: reference, version: metadata.version, bytes: bytes)
    }

    public func metadata(for reference: OpenMedSecretReference) throws -> OpenMedSecretMetadata {
        let metadata = try validatedMetadata(attributes(for: reference), for: reference)
        return OpenMedSecretMetadata(reference: reference, version: metadata.version)
    }

    public func put(_ secret: Data, for reference: OpenMedSecretReference) throws -> OpenMedSecretReceipt {
        guard Self.isAcceptedLength(secret.count) else {
            throw OpenMedSecretProviderError.invalidSecret
        }
        try requireProtectionAvailable()
        var attributes = itemQuery(for: reference)
        attributes[kSecAttrGeneric as String] = KeychainSecretItemMetadata(purpose: reference.purpose, userAuthentication: configuration.userAuthentication, version: 1).encoded
        attributes[kSecValueData as String] = secret
        try applyProtection(to: &attributes)
        let status = client.add(attributes)
        guard status == errSecSuccess else {
            throw Self.error(for: status)
        }
        return OpenMedSecretReceipt(operation: .stored, reference: reference, version: 1, byteCount: secret.count)
    }

    public func rotate(_ reference: OpenMedSecretReference, to secret: Data, expectedVersion: Int) throws -> OpenMedSecretReceipt {
        guard Self.isAcceptedLength(secret.count) else {
            throw OpenMedSecretProviderError.invalidSecret
        }
        guard expectedVersion >= 1, expectedVersion < Int.max else {
            throw OpenMedSecretProviderError.invalidVersion
        }
        try requireProtectionAvailable()
        let current = try validatedMetadata(attributes(for: reference), for: reference)
        guard current.version == expectedVersion else {
            throw OpenMedSecretProviderError.rotated
        }
        let next = KeychainSecretItemMetadata(purpose: reference.purpose, userAuthentication: configuration.userAuthentication, version: expectedVersion + 1)
        // Matching on the stored metadata makes the update a compare-and-swap:
        // a concurrent rotation changes kSecAttrGeneric and the update misses.
        var query = itemQuery(for: reference)
        query[kSecAttrGeneric as String] = current.encoded
        let changes: [String: Any] = [
            kSecValueData as String: secret,
            kSecAttrGeneric as String: next.encoded,
        ]
        let status = client.update(query, attributes: changes)
        switch status {
        case errSecSuccess:
            return OpenMedSecretReceipt(operation: .rotated, reference: reference, version: next.version, byteCount: secret.count)
        case errSecItemNotFound:
            _ = try attributes(for: reference)
            throw OpenMedSecretProviderError.rotated
        default:
            throw Self.error(for: status)
        }
    }

    public func delete(_ reference: OpenMedSecretReference) throws -> OpenMedSecretDeletion {
        let status = client.delete(itemQuery(for: reference))
        switch status {
        case errSecSuccess:
            return .deleted
        case errSecItemNotFound:
            return .alreadyAbsent
        default:
            throw Self.error(for: status)
        }
    }

    public func protectionReport() -> OpenMedSecretProtectionReport {
        OpenMedSecretProtectionReport(
            provider: Self.providerName,
            accessibility: configuration.accessibility.rawValue,
            userAuthentication: configuration.userAuthentication.rawValue,
            accessGroup: configuration.accessGroup.code,
            deviceOnly: configuration.deviceOnly,
            synchronizable: configuration.synchronizable,
            availability: availability(),
            secureHardwareClaimed: false
        )
    }

    public var description: String {
        "OpenMedKeychainSecretProvider(\(configuration.description))"
    }

    public var debugDescription: String { description }

    public var customMirror: Mirror {
        Mirror(self, children: ["configuration": configuration], displayStyle: .struct)
    }

    /// Map a `Security` status to a value-free error code.
    static func error(for status: OSStatus) -> OpenMedSecretProviderError {
        switch status {
        case errSecItemNotFound:
            return .missing
        case errSecDuplicateItem:
            return .duplicate
        case errSecInteractionNotAllowed:
            return .locked
        case errSecUserCanceled:
            return .userCanceled
        case errSecAuthFailed:
            return .authenticationFailed
        case errSecMissingEntitlement:
            return .missingEntitlement
        case errSecNotAvailable:
            return .unavailable
        case errSecUnimplemented, errSecParam:
            return .unsupported
        default:
            return .unexpectedStatus
        }
    }

    /// The account an item is stored under; derived from the digest only.
    static func account(for reference: OpenMedSecretReference) -> String {
        accountPrefix + reference.digest
    }

    func itemQuery(for reference: OpenMedSecretReference) -> [String: Any] {
        var query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: configuration.service,
            kSecAttrAccount as String: Self.account(for: reference),
            kSecAttrSynchronizable as String: false,
            kSecUseDataProtectionKeychain as String: true,
        ]
        if case .named(let group) = configuration.accessGroup {
            query[kSecAttrAccessGroup as String] = group
        }
        return query
    }

    func availability() -> OpenMedProtectionAvailability {
        let requirements = configuration.requirements
        guard !requirements.isEmpty else { return .notRequested }
        for requirement in requirements {
            let availability = probe.availability(of: requirement)
            if availability != .available {
                return availability
            }
        }
        return .available
    }

    private func requireProtectionAvailable() throws {
        switch availability() {
        case .notRequested, .available:
            return
        case .biometryLockout:
            throw OpenMedSecretProviderError.authenticationFailed
        case .passcodeNotSet, .biometryNotAvailable, .biometryNotEnrolled, .unsupportedPlatform, .unknown:
            throw OpenMedSecretProviderError.unsupported
        }
    }

    private func applyProtection(to attributes: inout [String: Any]) throws {
        guard let flags = configuration.userAuthentication.flags else {
            attributes[kSecAttrAccessible as String] = configuration.accessibility.constant
            return
        }
        var error: Unmanaged<CFError>?
        guard let accessControl = SecAccessControlCreateWithFlags(nil, configuration.accessibility.constant, flags, &error) else {
            error?.release()
            throw OpenMedSecretProviderError.unsupported
        }
        attributes[kSecAttrAccessControl as String] = accessControl
    }

    private func attributes(for reference: OpenMedSecretReference) throws -> [String: Any] {
        var query = itemQuery(for: reference)
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        query[kSecReturnAttributes as String] = true
        let (status, result) = client.copyMatching(query)
        guard status == errSecSuccess else {
            throw Self.error(for: status)
        }
        guard let attributes = result as? [String: Any] else {
            throw OpenMedSecretProviderError.invalidated
        }
        return attributes
    }

    private func validatedMetadata(_ attributes: [String: Any], for reference: OpenMedSecretReference) throws -> KeychainSecretItemMetadata {
        if let synchronizable = attributes[kSecAttrSynchronizable as String] as? Bool, synchronizable {
            throw OpenMedSecretProviderError.protectionMismatch
        }
        if let accessible = attributes[kSecAttrAccessible as String] as? String,
            accessible != configuration.accessibility.constant as String
        {
            throw OpenMedSecretProviderError.protectionMismatch
        }
        guard let encoded = attributes[kSecAttrGeneric as String] as? Data,
            let metadata = KeychainSecretItemMetadata(encoded),
            metadata.purpose == reference.purpose
        else {
            throw OpenMedSecretProviderError.invalidated
        }
        guard metadata.userAuthentication == configuration.userAuthentication else {
            throw OpenMedSecretProviderError.protectionMismatch
        }
        return metadata
    }

    private static func isAcceptedLength(_ count: Int) -> Bool {
        count > 0 && count <= maximumSecretBytes
    }
}

/// The non-secret record stored in `kSecAttrGeneric`.
struct KeychainSecretItemMetadata: Equatable {
    static let format = "openmed.secret.v1"

    let purpose: OpenMedSecretPurpose
    let userAuthentication: OpenMedKeychainConfiguration.UserAuthentication
    let version: Int

    init(purpose: OpenMedSecretPurpose, userAuthentication: OpenMedKeychainConfiguration.UserAuthentication, version: Int) {
        self.purpose = purpose
        self.userAuthentication = userAuthentication
        self.version = version
    }

    /// Parse the canonical encoding only; anything else is rejected.
    init?(_ data: Data) {
        guard data.count <= 256, let text = String(data: data, encoding: .utf8) else { return nil }
        let fields = text.split(separator: ";", omittingEmptySubsequences: false)
        guard fields.count == 4, fields[0] == Self.format,
            fields[1].hasPrefix("purpose="), fields[2].hasPrefix("auth="), fields[3].hasPrefix("version="),
            let purpose = OpenMedSecretPurpose(rawValue: String(fields[1].dropFirst("purpose=".count))),
            let userAuthentication = OpenMedKeychainConfiguration.UserAuthentication(rawValue: String(fields[2].dropFirst("auth=".count))),
            let version = Int(fields[3].dropFirst("version=".count)), version >= 1
        else {
            return nil
        }
        self.init(purpose: purpose, userAuthentication: userAuthentication, version: version)
        guard encoded == data else { return nil }
    }

    var encoded: Data {
        Data("\(Self.format);purpose=\(purpose.rawValue);auth=\(userAuthentication.rawValue);version=\(version)".utf8)
    }
}
