import Foundation
import Security
import XCTest

@testable import OpenMedKit

/// In-memory stand-in for the four `SecItem` calls.
///
/// It matches every query attribute the way the data-protection Keychain
/// does for generic passwords, records each call, and can simulate a locked
/// device, an access control that no longer releases the bytes, forced
/// statuses and a concurrent writer.
private final class FakeKeychain: OpenMedKeychainItemClient, @unchecked Sendable {
    struct Call {
        let operation: String
        let query: [String: Any]
        let attributes: [String: Any]?
    }

    private static let defaultGroup = "fake.default-group"
    private static let controlKeys: Set<String> = [
        kSecClass as String,
        kSecMatchLimit as String,
        kSecReturnData as String,
        kSecReturnAttributes as String,
        kSecUseDataProtectionKeychain as String,
        kSecValueData as String,
    ]

    private let lock = NSLock()
    private var items: [(attributes: [String: Any], data: Data)] = []
    private var recorded: [Call] = []
    private var lockedDevice = false
    private var dataUnreadable = false
    private var forced: [String: OSStatus] = [:]
    private var beforeUpdate: (() -> Void)?

    var calls: [Call] {
        lock.lock()
        defer { lock.unlock() }
        return recorded
    }

    var itemCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return items.count
    }

    func setLocked(_ value: Bool) {
        lock.lock()
        lockedDevice = value
        lock.unlock()
    }

    func setDataUnreadable(_ value: Bool) {
        lock.lock()
        dataUnreadable = value
        lock.unlock()
    }

    func force(_ operation: String, _ status: OSStatus?) {
        lock.lock()
        forced[operation] = status
        lock.unlock()
    }

    func onNextUpdate(_ action: @escaping () -> Void) {
        lock.lock()
        beforeUpdate = action
        lock.unlock()
    }

    /// Insert an item directly, bypassing the provider.
    func plant(_ attributes: [String: Any], data: Data) {
        lock.lock()
        items.append((normalized(attributes), data))
        lock.unlock()
    }

    /// Change one attribute of every stored item, bypassing the provider.
    func rewrite(_ key: String, to value: Any, data: Data? = nil) {
        lock.lock()
        for index in items.indices {
            items[index].attributes[key] = value
            if let data { items[index].data = data }
        }
        lock.unlock()
    }

    func add(_ attributes: [String: Any]) -> OSStatus {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(Call(operation: "add", query: attributes, attributes: nil))
        if let status = forced["add"] { return status }
        if lockedDevice { return errSecInteractionNotAllowed }
        guard let data = attributes[kSecValueData as String] as? Data else { return errSecParam }
        let candidate = normalized(attributes)
        let identity = [kSecAttrService, kSecAttrAccount, kSecAttrAccessGroup, kSecAttrSynchronizable].map { $0 as String }
        if items.contains(where: { item in identity.allSatisfy { Self.equal(item.attributes[$0], candidate[$0]) } }) {
            return errSecDuplicateItem
        }
        items.append((candidate, data))
        return errSecSuccess
    }

    func copyMatching(_ query: [String: Any]) -> (status: OSStatus, result: AnyObject?) {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(Call(operation: "copy", query: query, attributes: nil))
        if let status = forced["copy"] { return (status, nil) }
        if lockedDevice { return (errSecInteractionNotAllowed, nil) }
        guard let item = items.first(where: { matches($0.attributes, query) }) else {
            return (errSecItemNotFound, nil)
        }
        let wantsData = query[kSecReturnData as String] as? Bool == true
        if wantsData, dataUnreadable { return (errSecItemNotFound, nil) }
        var result = item.attributes
        if wantsData { result[kSecValueData as String] = item.data }
        if query[kSecReturnAttributes as String] as? Bool != true { return (errSecSuccess, item.data as NSData) }
        return (errSecSuccess, result as NSDictionary)
    }

    func update(_ query: [String: Any], attributes: [String: Any]) -> OSStatus {
        lock.lock()
        let hook = beforeUpdate
        beforeUpdate = nil
        lock.unlock()
        hook?()

        lock.lock()
        defer { lock.unlock() }
        recorded.append(Call(operation: "update", query: query, attributes: attributes))
        if let status = forced["update"] { return status }
        if lockedDevice { return errSecInteractionNotAllowed }
        guard let index = items.firstIndex(where: { matches($0.attributes, query) }) else { return errSecItemNotFound }
        for (key, value) in attributes where key != kSecValueData as String {
            items[index].attributes[key] = value
        }
        if let data = attributes[kSecValueData as String] as? Data { items[index].data = data }
        return errSecSuccess
    }

    func delete(_ query: [String: Any]) -> OSStatus {
        lock.lock()
        defer { lock.unlock() }
        recorded.append(Call(operation: "delete", query: query, attributes: nil))
        if let status = forced["delete"] { return status }
        let before = items.count
        items.removeAll { matches($0.attributes, query) }
        return items.count == before ? errSecItemNotFound : errSecSuccess
    }

    private func normalized(_ attributes: [String: Any]) -> [String: Any] {
        var stored = attributes
        for key in Self.controlKeys { stored.removeValue(forKey: key) }
        if stored[kSecAttrAccessGroup as String] == nil { stored[kSecAttrAccessGroup as String] = Self.defaultGroup }
        if stored[kSecAttrSynchronizable as String] == nil { stored[kSecAttrSynchronizable as String] = false }
        return stored
    }

    private func matches(_ attributes: [String: Any], _ query: [String: Any]) -> Bool {
        query.allSatisfy { key, value in
            Self.controlKeys.contains(key) || Self.equal(attributes[key], value)
        }
    }

    private static func equal(_ lhs: Any?, _ rhs: Any?) -> Bool {
        switch (lhs, rhs) {
        case (nil, nil):
            return true
        case (let lhs?, let rhs?):
            return (lhs as AnyObject).isEqual(rhs)
        default:
            return false
        }
    }
}

private struct FakeProbe: OpenMedLocalAuthenticationProbe {
    var passcode: OpenMedProtectionAvailability = .available
    var biometry: OpenMedProtectionAvailability = .available

    func availability(of requirement: OpenMedLocalAuthenticationRequirement) -> OpenMedProtectionAvailability {
        requirement == .devicePasscode ? passcode : biometry
    }
}

final class KeychainSecretProviderTests: XCTestCase {
    private static let sentinel = "OMKSENTINEL7Q4Z"
    private static let service = "org.example.\(sentinel).service"
    private static let group = "TEAMID.org.example.\(sentinel).shared"
    private static let secret = Data("synthetic-\(sentinel)-key-material".utf8)
    private static let rotatedSecret = Data("synthetic-\(sentinel)-rotated-material".utf8)

    private var keychain: FakeKeychain!

    override func setUp() {
        super.setUp()
        keychain = FakeKeychain()
    }

    private func makeProvider(
        accessGroup: OpenMedKeychainConfiguration.AccessGroup = .applicationDefault,
        accessibility: OpenMedKeychainConfiguration.Accessibility = .whenUnlockedThisDeviceOnly,
        userAuthentication: OpenMedKeychainConfiguration.UserAuthentication = .notRequired,
        probe: FakeProbe = FakeProbe()
    ) throws -> OpenMedKeychainSecretProvider {
        let configuration = try OpenMedKeychainConfiguration(
            service: Self.service,
            accessGroup: accessGroup,
            accessibility: accessibility,
            userAuthentication: userAuthentication
        )
        return OpenMedKeychainSecretProvider(configuration: configuration, client: keychain, authenticationProbe: probe)
    }

    private func makeReference(_ purpose: OpenMedSecretPurpose = .policySecret) throws -> OpenMedSecretReference {
        try OpenMedSecretReference(purpose: purpose, label: "study-\(Self.sentinel).hash-key")
    }

    private func assertThrows(
        _ expected: OpenMedSecretProviderError,
        file: StaticString = #filePath,
        line: UInt = #line,
        _ body: () throws -> Void
    ) {
        XCTAssertThrowsError(try body(), file: file, line: line) { error in
            XCTAssertEqual(error as? OpenMedSecretProviderError, expected, file: file, line: line)
            assertValueFree(error, file: file, line: line)
        }
    }

    private func assertValueFree(_ subject: Any, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertFalse(leaks(subject), "rendering leaks a synthetic value", file: file, line: line)
    }

    /// Whether any rendering of `subject` contains the sentinel label, service,
    /// group or secret bytes (as text, base64 or hex).
    private func leaks(_ subject: Any) -> Bool {
        var renderings = [String(describing: subject), String(reflecting: subject)]
        var dumped = ""
        dump(subject, to: &dumped)
        renderings.append(dumped)
        if let error = subject as? Error {
            let bridged = error as NSError
            renderings += [error.localizedDescription, String(describing: bridged), bridged.userInfo.description]
            if let localized = error as? LocalizedError {
                renderings += [localized.errorDescription, localized.failureReason, localized.recoverySuggestion].compactMap { $0 }
            }
        }
        let forbidden = [
            Self.sentinel,
            Self.secret.base64EncodedString(),
            Self.secret.map { String(format: "%02x", $0) }.joined(),
        ]
        return renderings.contains { rendering in forbidden.contains { rendering.contains($0) } }
    }

    private func storedValue(of key: CFString, in call: FakeKeychain.Call) -> Any? {
        call.query[key as String]
    }

    // MARK: - Round trip and declared protection

    func testPutThenGetRoundTripsWithValueFreeReceipts() throws {
        let provider = try makeProvider()
        let reference = try makeReference()

        let receipt = try provider.put(Self.secret, for: reference)
        XCTAssertEqual(receipt.operation, .stored)
        XCTAssertEqual(receipt.version, 1)
        XCTAssertEqual(receipt.byteCount, Self.secret.count)

        let value = try provider.get(reference)
        XCTAssertEqual(value.bytes, Self.secret)
        XCTAssertEqual(value.version, 1)
        XCTAssertEqual(value.metadata, try provider.metadata(for: reference))
        XCTAssertEqual(try provider.get(reference, expectedVersion: 1).bytes, Self.secret)

        let exported = try String(decoding: JSONEncoder().encode(receipt), as: UTF8.self)
        for subject in [receipt, value, value.metadata, reference, provider, provider.configuration, exported] as [Any] {
            assertValueFree(subject)
        }
    }

    func testAddDeclaresDeviceOnlyNonSynchronizingDataProtectionItem() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)

        let add = try XCTUnwrap(keychain.calls.first { $0.operation == "add" })
        XCTAssertEqual(storedValue(of: kSecClass, in: add) as? String, kSecClassGenericPassword as String)
        XCTAssertEqual(storedValue(of: kSecAttrService, in: add) as? String, Self.service)
        XCTAssertEqual(storedValue(of: kSecAttrAccessible, in: add) as? String, kSecAttrAccessibleWhenUnlockedThisDeviceOnly as String)
        XCTAssertEqual(storedValue(of: kSecAttrSynchronizable, in: add) as? Bool, false)
        XCTAssertEqual(storedValue(of: kSecUseDataProtectionKeychain, in: add) as? Bool, true)
        XCTAssertNil(storedValue(of: kSecAttrAccessGroup, in: add))
        XCTAssertNil(storedValue(of: kSecAttrAccessControl, in: add))

        let account = try XCTUnwrap(storedValue(of: kSecAttrAccount, in: add) as? String)
        XCTAssertEqual(account, "omk.secret.v1." + reference.digest)
        XCTAssertFalse(account.contains(Self.sentinel))
        let metadata = try XCTUnwrap(storedValue(of: kSecAttrGeneric, in: add) as? Data)
        XCTAssertEqual(String(decoding: metadata, as: UTF8.self), "openmed.secret.v1;purpose=policy_secret;auth=not_required;version=1")

        // Every later call addresses the same item and never widens to synchronizable items.
        _ = try provider.get(reference)
        _ = try provider.delete(reference)
        for call in keychain.calls {
            XCTAssertEqual(storedValue(of: kSecAttrSynchronizable, in: call) as? Bool, false)
            XCTAssertEqual(storedValue(of: kSecUseDataProtectionKeychain, in: call) as? Bool, true)
            XCTAssertEqual(storedValue(of: kSecAttrAccount, in: call) as? String, account)
        }
    }

    func testNamedAccessGroupAndAccessibilityClassAreSentOnEveryCall() throws {
        let provider = try makeProvider(accessGroup: .named(Self.group), accessibility: .afterFirstUnlockThisDeviceOnly)
        let reference = try makeReference(.bundleTrustAnchor)
        _ = try provider.put(Self.secret, for: reference)
        _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1)
        XCTAssertEqual(try provider.get(reference).bytes, Self.rotatedSecret)
        XCTAssertEqual(try provider.delete(reference), .deleted)

        let add = try XCTUnwrap(keychain.calls.first { $0.operation == "add" })
        XCTAssertEqual(storedValue(of: kSecAttrAccessible, in: add) as? String, kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String)
        for call in keychain.calls {
            XCTAssertEqual(storedValue(of: kSecAttrAccessGroup, in: call) as? String, Self.group)
        }
        XCTAssertEqual(provider.protectionReport().accessGroup, "named")
    }

    func testUserAuthenticationUsesAccessControlInsteadOfAccessibility() throws {
        let provider = try makeProvider(accessibility: .whenPasscodeSetThisDeviceOnly, userAuthentication: .biometryCurrentSet)
        _ = try provider.put(Self.secret, for: makeReference())

        let add = try XCTUnwrap(keychain.calls.first { $0.operation == "add" })
        XCTAssertNil(storedValue(of: kSecAttrAccessible, in: add))
        let accessControl = try XCTUnwrap(storedValue(of: kSecAttrAccessControl, in: add))
        XCTAssertEqual(CFGetTypeID(accessControl as CFTypeRef), SecAccessControlGetTypeID())
        let metadata = try XCTUnwrap(storedValue(of: kSecAttrGeneric, in: add) as? Data)
        XCTAssertTrue(String(decoding: metadata, as: UTF8.self).contains("auth=biometry_current_set"))

        typealias Auth = OpenMedKeychainConfiguration.UserAuthentication
        let expected: [Auth: SecAccessControlCreateFlags?] = [
            .notRequired: nil,
            .userPresence: .userPresence,
            .devicePasscode: .devicePasscode,
            .biometryAny: .biometryAny,
            .biometryCurrentSet: .biometryCurrentSet,
        ]
        XCTAssertEqual(Set(expected.keys), Set(Auth.allCases))
        for (mode, flags) in expected {
            XCTAssertEqual(mode.flags, flags, mode.rawValue)
        }
    }

    // MARK: - Missing, locked, invalidated, rotated

    func testMissingItemsAreTyped() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        assertThrows(.missing) { _ = try provider.get(reference) }
        assertThrows(.missing) { _ = try provider.metadata(for: reference) }
        assertThrows(.missing) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }
        XCTAssertEqual(try provider.delete(reference), .alreadyAbsent)
        XCTAssertEqual(keychain.itemCount, 0)
    }

    func testLockedKeychainFailsClosedWithoutFallback() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)
        let other = try OpenMedSecretReference(purpose: .policySecret, label: "second-\(Self.sentinel)")

        keychain.setLocked(true)
        assertThrows(.locked) { _ = try provider.get(reference) }
        assertThrows(.locked) { _ = try provider.get(reference, expectedVersion: 1) }
        assertThrows(.locked) { _ = try provider.metadata(for: reference) }
        assertThrows(.locked) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }
        assertThrows(.locked) { _ = try provider.put(Self.secret, for: other) }
        XCTAssertEqual(keychain.itemCount, 1, "a locked keychain must not cause a second write")
        XCTAssertFalse(keychain.calls.contains { $0.operation == "update" })

        keychain.setLocked(false)
        let value = try provider.get(reference)
        XCTAssertEqual(value.bytes, Self.secret)
        XCTAssertEqual(value.version, 1)
    }

    func testInvalidatedAccessControlIsDistinguishedFromMissing() throws {
        let provider = try makeProvider(userAuthentication: .biometryCurrentSet)
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)

        // The item is still listed, but its access control no longer releases the bytes.
        keychain.setDataUnreadable(true)
        assertThrows(.invalidated) { _ = try provider.get(reference) }
        XCTAssertEqual(try provider.metadata(for: reference).version, 1)

        XCTAssertEqual(try provider.delete(reference), .deleted)
        assertThrows(.missing) { _ = try provider.get(reference) }
    }

    func testUntrustworthyMetadataIsInvalidated() throws {
        let reference = try makeReference()
        let base = try makeProvider().itemQuery(for: reference)
        let cases: [Data?] = [
            nil,
            Data("not-openmed".utf8),
            Data("openmed.secret.v1;purpose=policy_secret;auth=not_required;version=0".utf8),
            Data("openmed.secret.v1;purpose=policy_secret;auth=not_required;version=+2".utf8),
            Data("openmed.secret.v1;purpose=bundle_trust_anchor;auth=not_required;version=1".utf8),
            Data("openmed.secret.v1;purpose=policy_secret;auth=not_required;version=1;extra".utf8),
        ]
        for metadata in cases {
            keychain = FakeKeychain()
            let provider = try makeProvider()
            var attributes = base
            attributes[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            if let metadata { attributes[kSecAttrGeneric as String] = metadata }
            keychain.plant(attributes, data: Self.secret)
            assertThrows(.invalidated) { _ = try provider.get(reference) }
            assertThrows(.invalidated) { _ = try provider.metadata(for: reference) }
            assertThrows(.invalidated) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }
        }
    }

    func testStoredProtectionMismatchFailsClosed() throws {
        let reference = try makeReference()
        let validMetadata = KeychainSecretItemMetadata(purpose: .policySecret, userAuthentication: .notRequired, version: 1).encoded

        // A weaker, migratable accessibility class planted under the same account.
        var provider = try makeProvider()
        var attributes = provider.itemQuery(for: reference)
        attributes[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        attributes[kSecAttrGeneric as String] = validMetadata
        keychain.plant(attributes, data: Self.secret)
        assertThrows(.protectionMismatch) { _ = try provider.get(reference) }
        assertThrows(.protectionMismatch) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }

        // An item written without user authentication, read by a configuration that requires it.
        keychain = FakeKeychain()
        provider = try makeProvider(userAuthentication: .userPresence)
        attributes = provider.itemQuery(for: reference)
        attributes[kSecAttrGeneric as String] = validMetadata
        keychain.plant(attributes, data: Self.secret)
        assertThrows(.protectionMismatch) { _ = try provider.get(reference) }

        // A synchronizable item with the same account is never matched or read.
        keychain = FakeKeychain()
        provider = try makeProvider()
        attributes = provider.itemQuery(for: reference)
        attributes[kSecAttrSynchronizable as String] = true
        attributes[kSecAttrGeneric as String] = validMetadata
        keychain.plant(attributes, data: Self.secret)
        assertThrows(.missing) { _ = try provider.get(reference) }
    }

    func testRotationIsCompareAndSwapAndPinnedReadsReportRotation() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)

        let receipt = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1)
        XCTAssertEqual(receipt.operation, .rotated)
        XCTAssertEqual(receipt.version, 2)
        assertValueFree(receipt)

        let update = try XCTUnwrap(keychain.calls.last { $0.operation == "update" })
        XCTAssertEqual(
            (storedValue(of: kSecAttrGeneric, in: update) as? Data).map { String(decoding: $0, as: UTF8.self) },
            "openmed.secret.v1;purpose=policy_secret;auth=not_required;version=1",
            "the update must match the version it read"
        )

        assertThrows(.rotated) { _ = try provider.get(reference, expectedVersion: 1) }
        let current = try provider.get(reference)
        XCTAssertEqual(current.bytes, Self.rotatedSecret)
        XCTAssertEqual(current.version, 2)

        assertThrows(.rotated) { _ = try provider.rotate(reference, to: Self.secret, expectedVersion: 1) }
        XCTAssertEqual(try provider.get(reference).bytes, Self.rotatedSecret)
        assertThrows(.invalidVersion) { _ = try provider.rotate(reference, to: Self.secret, expectedVersion: 0) }
        assertThrows(.invalidVersion) { _ = try provider.get(reference, expectedVersion: 0) }
    }

    func testConcurrentRotationBetweenReadAndUpdateIsNotOverwritten() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)
        let concurrent = Data("synthetic-concurrent-writer".utf8)
        let concurrentMetadata = KeychainSecretItemMetadata(purpose: .policySecret, userAuthentication: .notRequired, version: 2).encoded

        keychain.onNextUpdate { [keychain] in
            keychain?.rewrite(kSecAttrGeneric as String, to: concurrentMetadata, data: concurrent)
        }
        assertThrows(.rotated) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }
        let value = try provider.get(reference)
        XCTAssertEqual(value.bytes, concurrent)
        XCTAssertEqual(value.version, 2)

        // A concurrent delete between the read and the update reports the item as missing.
        keychain.onNextUpdate { _ = try? provider.delete(reference) }
        assertThrows(.missing) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 2) }
    }

    func testDuplicatePutDoesNotOverwrite() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        _ = try provider.put(Self.secret, for: reference)
        assertThrows(.duplicate) { _ = try provider.put(Self.rotatedSecret, for: reference) }
        XCTAssertEqual(try provider.get(reference).bytes, Self.secret)
        XCTAssertEqual(keychain.itemCount, 1)
    }

    // MARK: - Available versus unsupported protection

    func testUnavailableProtectionRefusesToWriteInsteadOfDowngrading() throws {
        let reference = try makeReference()
        let cases: [(OpenMedKeychainConfiguration.Accessibility, OpenMedKeychainConfiguration.UserAuthentication, FakeProbe, OpenMedSecretProviderError)] = [
            (.whenUnlockedThisDeviceOnly, .biometryCurrentSet, FakeProbe(biometry: .biometryNotEnrolled), .unsupported),
            (.whenUnlockedThisDeviceOnly, .biometryAny, FakeProbe(biometry: .biometryNotAvailable), .unsupported),
            (.whenUnlockedThisDeviceOnly, .biometryAny, FakeProbe(biometry: .unsupportedPlatform), .unsupported),
            (.whenUnlockedThisDeviceOnly, .biometryAny, FakeProbe(biometry: .unknown), .unsupported),
            (.whenPasscodeSetThisDeviceOnly, .notRequired, FakeProbe(passcode: .passcodeNotSet), .unsupported),
            (.whenUnlockedThisDeviceOnly, .devicePasscode, FakeProbe(passcode: .passcodeNotSet), .unsupported),
            (.whenUnlockedThisDeviceOnly, .userPresence, FakeProbe(passcode: .passcodeNotSet), .unsupported),
            (.whenUnlockedThisDeviceOnly, .biometryCurrentSet, FakeProbe(biometry: .biometryLockout), .authenticationFailed),
        ]
        for (accessibility, userAuthentication, probe, expected) in cases {
            keychain = FakeKeychain()
            let provider = try makeProvider(accessibility: accessibility, userAuthentication: userAuthentication, probe: probe)
            assertThrows(expected) { _ = try provider.put(Self.secret, for: reference) }
            XCTAssertTrue(keychain.calls.isEmpty, "no SecItem call may run when protection is unavailable")
            let report = provider.protectionReport()
            XCTAssertNotEqual(report.availability, .available)
            XCTAssertNotEqual(report.availability, .notRequested)
            XCTAssertFalse(report.secureHardwareClaimed)
        }

        // A configuration with no requirements never consults the probe.
        keychain = FakeKeychain()
        let open = try makeProvider(probe: FakeProbe(passcode: .passcodeNotSet, biometry: .biometryNotAvailable))
        XCTAssertEqual(open.protectionReport().availability, .notRequested)
        XCTAssertEqual(try open.put(Self.secret, for: reference).version, 1)
    }

    func testProtectionReportStatesConfigurationWithoutHardwareClaim() throws {
        let provider = try makeProvider(accessGroup: .named(Self.group), userAuthentication: .biometryAny)
        let report = provider.protectionReport()
        XCTAssertEqual(report.provider, "apple_keychain")
        XCTAssertEqual(report.accessibility, "when_unlocked_this_device_only")
        XCTAssertEqual(report.userAuthentication, "biometry_any")
        XCTAssertEqual(report.accessGroup, "named")
        XCTAssertTrue(report.deviceOnly)
        XCTAssertFalse(report.synchronizable)
        XCTAssertEqual(report.availability, .available)
        XCTAssertFalse(report.secureHardwareClaimed)
        XCTAssertEqual(provider.configuration.requirements, [.biometry])

        let json = try String(decoding: JSONEncoder().encode(report), as: UTF8.self)
        XCTAssertTrue(json.contains("\"secureHardwareClaimed\":false"))
        assertValueFree(json)
        assertValueFree(report)
    }

    func testSystemProbeReportsAvailabilityWithoutPrompting() {
        // The host decides what is available; the result must be one typed code.
        let probe = OpenMedSystemLocalAuthenticationProbe()
        for requirement in OpenMedLocalAuthenticationRequirement.allCases {
            XCTAssertTrue(OpenMedProtectionAvailability.allCases.contains(probe.availability(of: requirement)))
        }
        #if os(watchOS)
            XCTAssertEqual(probe.availability(of: .biometry), .unsupportedPlatform)
        #endif
        #if canImport(LocalAuthentication)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: -5), .passcodeNotSet)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: -6), .biometryNotAvailable)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: -7), .biometryNotEnrolled)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: -8), .biometryLockout)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: -1004), .unknown)
            XCTAssertEqual(OpenMedSystemLocalAuthenticationProbe.availability(forErrorCode: nil), .unknown)
        #endif
    }

    // MARK: - Status mapping and validation

    func testSecurityStatusesMapToValueFreeCodes() throws {
        let expected: [(OSStatus, OpenMedSecretProviderError)] = [
            (errSecItemNotFound, .missing),
            (errSecDuplicateItem, .duplicate),
            (errSecInteractionNotAllowed, .locked),
            (errSecUserCanceled, .userCanceled),
            (errSecAuthFailed, .authenticationFailed),
            (errSecMissingEntitlement, .missingEntitlement),
            (errSecNotAvailable, .unavailable),
            (errSecUnimplemented, .unsupported),
            (errSecParam, .unsupported),
            (errSecDecode, .unexpectedStatus),
            (-1, .unexpectedStatus),
        ]
        for (status, error) in expected {
            XCTAssertEqual(OpenMedKeychainSecretProvider.error(for: status), error, "\(status)")
        }

        // Forced statuses surface through each operation without a fallback write.
        let provider = try makeProvider()
        let reference = try makeReference()
        keychain.force("add", errSecMissingEntitlement)
        assertThrows(.missingEntitlement) { _ = try provider.put(Self.secret, for: reference) }
        keychain.force("add", nil)
        _ = try provider.put(Self.secret, for: reference)
        keychain.force("copy", errSecUserCanceled)
        assertThrows(.userCanceled) { _ = try provider.get(reference) }
        keychain.force("copy", errSecAuthFailed)
        assertThrows(.authenticationFailed) { _ = try provider.get(reference) }
        keychain.force("copy", nil)
        keychain.force("update", errSecAuthFailed)
        assertThrows(.authenticationFailed) { _ = try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1) }
        keychain.force("delete", errSecInteractionNotAllowed)
        assertThrows(.locked) { _ = try provider.delete(reference) }
        XCTAssertEqual(keychain.itemCount, 1)
        XCTAssertEqual(try provider.get(reference).bytes, Self.secret)
    }

    func testSecretAndConfigurationValidation() throws {
        let provider = try makeProvider()
        let reference = try makeReference()
        assertThrows(.invalidSecret) { _ = try provider.put(Data(), for: reference) }
        assertThrows(.invalidSecret) {
            _ = try provider.put(Data(count: OpenMedKeychainSecretProvider.maximumSecretBytes + 1), for: reference)
        }
        assertThrows(.invalidSecret) { _ = try provider.rotate(reference, to: Data(), expectedVersion: 1) }
        XCTAssertTrue(keychain.calls.isEmpty)

        for service in ["", "   ", "line\nbreak", "tab\u{0009}", String(repeating: "s", count: 257)] {
            assertThrows(.invalidConfiguration) {
                _ = try OpenMedKeychainConfiguration(service: service, accessGroup: .applicationDefault)
            }
        }
        assertThrows(.invalidConfiguration) {
            _ = try OpenMedKeychainConfiguration(service: "org.example", accessGroup: .named(" "))
        }
        let configuration = try OpenMedKeychainConfiguration(service: Self.service, accessGroup: .named(Self.group))
        XCTAssertEqual(configuration.accessibility, .whenUnlockedThisDeviceOnly)
        XCTAssertEqual(configuration.userAuthentication, .notRequired)
        XCTAssertTrue(configuration.deviceOnly)
        XCTAssertFalse(configuration.synchronizable)
        XCTAssertEqual(configuration.requirements, [])
        assertValueFree(configuration)
    }

    func testReferencesAreOpaqueDomainSeparatedAndPortable() throws {
        let policy = try OpenMedSecretReference(purpose: .policySecret, label: "shared-\(Self.sentinel)")
        let again = try OpenMedSecretReference(purpose: .policySecret, label: "shared-\(Self.sentinel)")
        let trust = try OpenMedSecretReference(purpose: .bundleTrustAnchor, label: "shared-\(Self.sentinel)")
        XCTAssertEqual(policy, again)
        XCTAssertNotEqual(policy.digest, trust.digest)
        XCTAssertEqual(policy.digest.count, 64)
        XCTAssertEqual(policy.fingerprint, "ref:" + policy.digest.prefix(16))
        XCTAssertNotEqual(OpenMedKeychainSecretProvider.account(for: policy), OpenMedKeychainSecretProvider.account(for: trust))

        let encoded = try JSONEncoder().encode(policy)
        XCTAssertEqual(try JSONDecoder().decode(OpenMedSecretReference.self, from: encoded), policy)
        XCTAssertEqual(try OpenMedSecretReference(purpose: .policySecret, digest: policy.digest), policy)
        for subject in [policy, trust, String(decoding: encoded, as: UTF8.self)] as [Any] {
            assertValueFree(subject)
        }

        for label in ["", " \n ", "a\u{0000}b", "line\nbreak", String(repeating: "x", count: 257)] {
            assertThrows(.invalidReference) { _ = try OpenMedSecretReference(purpose: .policySecret, label: label) }
        }
        for digest in ["", String(repeating: "A", count: 64), String(repeating: "g", count: 64), String(repeating: "a", count: 63)] {
            assertThrows(.invalidReference) { _ = try OpenMedSecretReference(purpose: .policySecret, digest: digest) }
        }
        let tampered = Data(#"{"purpose":"policy_secret","digest":"\#(Self.sentinel)"}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(OpenMedSecretReference.self, from: tampered)) { error in
            XCTAssertEqual(error as? OpenMedSecretProviderError, .invalidReference)
        }
    }

    func testLeakCheckFlagsValueBearingRenderings() {
        // Negative controls: shapes the provider must never produce.
        struct NaiveSecret {
            let label: String
            let bytes: Data
        }
        XCTAssertTrue(leaks(NaiveSecret(label: "study-\(Self.sentinel)", bytes: Self.secret)))
        XCTAssertTrue(leaks(NSError(domain: "keychain", code: -25300, userInfo: [NSLocalizedDescriptionKey: Self.service])))
        XCTAssertTrue(leaks(Self.secret.base64EncodedString()))
        XCTAssertTrue(leaks(String(decoding: Self.secret, as: UTF8.self)))
        XCTAssertFalse(leaks(OpenMedSecretProviderError.missing))
        XCTAssertFalse(leaks(Self.secret.count))
    }

    func testEveryErrorCodeIsValueFree() {
        XCTAssertEqual(OpenMedSecretProviderError.allCases.count, 16)
        for error in OpenMedSecretProviderError.allCases {
            XCTAssertEqual(error.errorDescription, error.rawValue)
            XCTAssertTrue(error.rawValue.hasPrefix("secret_"))
            XCTAssertEqual(error.localizedDescription, error.rawValue)
            assertValueFree(error)
        }
    }

    // MARK: - Optional real Keychain

    /// Exercises the system Keychain only when `OPENMED_KEYCHAIN_INTEGRATION=1`.
    ///
    /// An unsigned `swift test` runner normally lacks the data-protection
    /// Keychain entitlement, so the expected outcome there is the typed
    /// `missingEntitlement` code rather than a silent fallback. A signed host
    /// runs the full round trip and cleans up after itself.
    func testRealKeychainRoundTripWhenEnabled() throws {
        guard ProcessInfo.processInfo.environment["OPENMED_KEYCHAIN_INTEGRATION"] == "1" else {
            throw XCTSkip("set OPENMED_KEYCHAIN_INTEGRATION=1 to exercise the system Keychain")
        }
        let configuration = try OpenMedKeychainConfiguration(
            service: "org.openmed.tests.keychain.\(UUID().uuidString)",
            accessGroup: .applicationDefault
        )
        let provider = OpenMedKeychainSecretProvider(configuration: configuration)
        let reference = try OpenMedSecretReference(purpose: .policySecret, label: "integration-\(UUID().uuidString)")
        XCTAssertFalse(provider.protectionReport().secureHardwareClaimed)
        defer { _ = try? provider.delete(reference) }

        do {
            _ = try provider.put(Self.secret, for: reference)
        } catch let error as OpenMedSecretProviderError where error == .missingEntitlement || error == .unavailable {
            assertValueFree(error)
            throw XCTSkip("system Keychain unavailable to this runner: \(error.rawValue)")
        }
        XCTAssertEqual(try provider.get(reference).bytes, Self.secret)
        XCTAssertThrowsError(try provider.put(Self.secret, for: reference)) {
            XCTAssertEqual($0 as? OpenMedSecretProviderError, .duplicate)
        }
        XCTAssertEqual(try provider.rotate(reference, to: Self.rotatedSecret, expectedVersion: 1).version, 2)
        XCTAssertThrowsError(try provider.rotate(reference, to: Self.secret, expectedVersion: 1)) {
            XCTAssertEqual($0 as? OpenMedSecretProviderError, .rotated)
        }
        XCTAssertThrowsError(try provider.get(reference, expectedVersion: 1)) {
            XCTAssertEqual($0 as? OpenMedSecretProviderError, .rotated)
        }
        XCTAssertEqual(try provider.get(reference).bytes, Self.rotatedSecret)
        XCTAssertEqual(try provider.delete(reference), .deleted)
        XCTAssertThrowsError(try provider.get(reference)) {
            XCTAssertEqual($0 as? OpenMedSecretProviderError, .missing)
        }
        XCTAssertEqual(try provider.delete(reference), .alreadyAbsent)
    }
}
