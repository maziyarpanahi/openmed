import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/managed_import/vectors.json` against
/// `OpenMedManagedBundleImporter` and checks the Apple handoff adapters.
final class OpenMedManagedBundleImportTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDown() {
        for root in temporaryRoots { try? FileManager.default.removeItem(at: root) }
        temporaryRoots = []
        NoNetworkProtocol.reset()
        super.tearDown()
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "managed-import-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        temporaryRoots.append(root.resolvingSymlinksInPath())
        return root.resolvingSymlinksInPath()
    }

    private static let repoRoot: URL = {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root
    }()

    private lazy var vectors: [String: Any] = {
        let data = try! Data(contentsOf: Self.repoRoot.appending(path: "tests/fixtures/managed_import/vectors.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }()

    private var bundles: [String: [String: Any]] { vectors["bundles"] as! [String: [String: Any]] }
    private var archives: [String: [String: Any]] { vectors["archives"] as! [String: [String: Any]] }
    private var configs: [String: [String: Any]] { vectors["configs"] as! [String: [String: Any]] }
    private var bundleID: String { (vectors["identities"] as! [String: Any])["bundle_id"] as! String }

    private func generation(_ alias: String?) -> String? {
        alias.map { bundles[$0]!["generation"] as! String }
    }

    private func archiveData(_ alias: String) -> Data {
        Data(base64Encoded: archives[alias]!["base64"] as! String)!
    }

    private func writeBundleDirectory(_ alias: String, to directory: URL) throws {
        let bundle = bundles[alias]!
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data((bundle["manifest"] as! String).utf8).write(
            to: directory.appending(path: PortableBundle.manifestFilename))
        try Data((bundle["envelope"] as! String).utf8).write(
            to: directory.appending(path: PortableBundle.envelopeFilename))
        for entry in bundle["files"] as! [[String: Any]] {
            let file = directory.appending(path: entry["path"] as! String)
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data((entry["text"] as! String).utf8).write(to: file)
        }
    }

    private func target(_ variant: String) throws -> PortableBundleTarget {
        let targets = vectors["targets"] as! [String: [String: [String: Any]]]
        let value = targets["ios"]![variant]!
        return try PortableBundleTarget(
            platform: value["platform"] as! String, sdkName: value["sdk_name"] as! String,
            sdkVersion: value["sdk_version"] as! String, runtimes: value["runtimes"] as! [String],
            osVersion: value["os_version"] as? String, policyVersion: (value["policy_version"] as? NSNumber)?.int64Value,
            memoryBytes: (value["memory_bytes"] as? NSNumber)?.int64Value)
    }

    private func limits(_ value: [String: Any]) -> OpenMedManagedImportLimits {
        OpenMedManagedImportLimits(
            maxArchiveBytes: (value["max_archive_bytes"] as! NSNumber).int64Value,
            maxEntries: (value["max_entries"] as! NSNumber).intValue,
            maxTotalBytes: (value["max_total_bytes"] as! NSNumber).int64Value,
            maxCompressionRatio: (value["max_compression_ratio"] as! NSNumber).int64Value)
    }

    private struct FakeManagedConfiguration: OpenMedManagedConfigurationSource {
        let values: [String: Any]?
        func managedConfiguration() -> [String: Any]? { values }
    }

    private final class Box<T>: @unchecked Sendable {
        var value: T
        init(_ value: T) { self.value = value }
    }

    // MARK: - Shared vectors

    private final class Harness {
        let root: URL
        let cacheRoot: URL
        let stateRoot: URL
        let inbox: URL
        let clock = Box<Int64>(1_000_000)
        let available = Box<Int64?>(nil)
        var cache: OpenMedBundleCache!
        var importer: OpenMedManagedBundleImporter!
        let chunkBytes: Int
        var restarts = 0

        init(root: URL, chunkBytes: Int) throws {
            self.root = root
            cacheRoot = root.appending(path: "cache")
            stateRoot = root.appending(path: "state")
            inbox = root.appending(path: "inbox")
            self.chunkBytes = chunkBytes
            try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
            try start()
        }

        func start() throws {
            restarts += 1
            let clock = self.clock
            let available = self.available
            let options = OpenMedBundleCacheOptions(
                quotaBytes: 1 << 40, chunkBytes: chunkBytes,
                now: {
                    clock.value += 1
                    return clock.value
                }, availableBytes: { _ in available.value ?? Int64.max }, ownerID: "owner-\(restarts)")
            cache = try OpenMedBundleCache(rootDirectory: cacheRoot, options: options)
            importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: stateRoot)
        }
    }

    func testSharedVectors() throws {
        let scenarios = vectors["scenarios"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(scenarios.count, 20)
        for scenario in scenarios {
            try replay(scenario)
        }
    }

    private func replay(_ scenario: [String: Any]) throws {
        let id = scenario["id"] as! String
        let harness = try Harness(
            root: temporaryDirectory(), chunkBytes: (vectors["chunk_bytes"] as! NSNumber).intValue)
        let target = try target(scenario["target"] as! String)
        let scenarioLimits = limits(scenario["limits"] as! [String: Any])
        for (index, step) in (scenario["steps"] as! [[String: Any]]).enumerated() {
            let label = "\(id)#\(index)"
            let expect = step["expect"] as? [String: Any] ?? [:]
            switch step["op"] as! String {
            case "import":
                try runImport(step, expect: expect, harness: harness, target: target, limits: scenarioLimits, label)
            case "restart":
                try harness.start()
            case "recover":
                let report = harness.importer.recover()
                XCTAssertEqual(report.spoolPurged, (expect["spool_purged"] as! NSNumber).intValue, label)
            case "confirm":
                try harness.importer.confirm(bundleID: bundleID)
            case "state":
                let snapshot = try harness.cache.snapshot(bundleID: bundleID)
                XCTAssertEqual(snapshot.active, generation(expect["active"] as? String), "\(label) active")
                XCTAssertEqual(snapshot.lastGreen, generation(expect["last_green"] as? String), "\(label) last green")
                XCTAssertEqual(snapshot.rollback, generation(expect["rollback"] as? String), "\(label) rollback")
                let quarantined = (expect["quarantined"] as! [[String: Any]]).map {
                    OpenMedBundleQuarantineEntry(
                        generation: generation($0["generation"] as? String)!, reason: $0["reason"] as! String)
                }
                XCTAssertEqual(snapshot.quarantined, quarantined, "\(label) quarantine")
                XCTAssertEqual(
                    harness.importer.sequenceFloor(bundleID: bundleID),
                    (expect["sequence_floor"] as? NSNumber)?.int64Value, "\(label) floor")
                if let receipts = expect["receipts"] as? NSNumber {
                    XCTAssertEqual(harness.importer.receipts().count, receipts.intValue, "\(label) receipts")
                }
            default:
                XCTFail("unknown op in \(label)")
            }
        }
    }

    private func runImport(
        _ step: [String: Any], expect: [String: Any], harness: Harness, target: PortableBundleTarget,
        limits: OpenMedManagedImportLimits, _ label: String
    ) throws {
        harness.available.value = (step["storage_available_bytes"] as? NSNumber)?.int64Value
        var values = (step["config"] as? String).map { configs[$0]! }
        var explicit: OpenMedManagedBundleHandoff?
        switch step["handoff"] as! String {
        case "stream":
            explicit = .stream(InputStream(data: archiveData(step["archive"] as! String)))
        case "directory":
            let name = "\(step["bundle"] as! String)-\(UUID().uuidString)"
            try writeBundleDirectory(step["bundle"] as! String, to: harness.inbox.appending(path: name))
            values?[OpenMedManagedBundleConfiguration.locationKey] = name
        default:
            let name = "\(step["archive"] as! String)-\(UUID().uuidString).zip"
            try archiveData(step["archive"] as! String).write(to: harness.inbox.appending(path: name))
            values?[OpenMedManagedBundleConfiguration.locationKey] = name
        }

        let counts = Box<[String: Int]>([:])
        let cancelled = Box(false)
        let crashAt = step["crash_at"] as? [String: Any]
        let cancelAt = step["cancel_at"] as? [String: Any]
        let hook: (String) throws -> Void = { name in
            counts.value[name, default: 0] += 1
            let seen = counts.value[name]!
            if let crashAt, crashAt["checkpoint"] as? String == name,
                (crashAt["occurrence"] as! NSNumber).intValue == seen
            {
                throw OpenMedBundleCacheCrash(checkpoint: name)
            }
            if let cancelAt, cancelAt["checkpoint"] as? String == name,
                (cancelAt["occurrence"] as! NSNumber).intValue == seen
            {
                cancelled.value = true
            }
        }
        harness.cache.checkpointHook = hook
        harness.importer.checkpointHook = hook
        defer {
            harness.cache.checkpointHook = nil
            harness.importer.checkpointHook = nil
        }
        let health = step["health"] as? String
        let context = try PortableBundleContext(now: step["now"] as! String, target: target)
        let run = {
            try harness.importer.importManagedChecked(
                from: FakeManagedConfiguration(values: values), approvedRoots: [harness.inbox], handoff: explicit,
                context: context, keyPins: [], limits: limits, activation: .activate,
                activationCheck: health.map { value in { _ in value == "pass" } }, isCancelled: { cancelled.value })
        }
        if expect["crashed"] as? Bool == true {
            XCTAssertThrowsError(try run(), label) { XCTAssertTrue($0 is OpenMedBundleCacheCrash, label) }
            return
        }
        let receipt = try run()
        XCTAssertEqual(receipt.outcome.rawValue, expect["outcome"] as? String, "\(label) outcome")
        XCTAssertEqual(receipt.reason?.rawValue, expect["reason"] as? String, "\(label) reason")
        XCTAssertEqual(receipt.bundleReason?.rawValue, expect["bundle_reason"] as? String, "\(label) bundle reason")
        XCTAssertEqual(receipt.generation, generation(expect["generation"] as? String), "\(label) generation")
        XCTAssertEqual(
            receipt.previousActive, generation(expect["previous_active"] as? String), "\(label) previous active")
        XCTAssertEqual(receipt.active, generation(expect["active"] as? String), "\(label) active")
        XCTAssertEqual(receipt.lastGreen, generation(expect["last_green"] as? String), "\(label) last green")
        XCTAssertEqual(receipt.resumed, expect["resumed"] as? Bool, "\(label) resumed")
        XCTAssertEqual(receipt.filesVerified, (expect["files_verified"] as? NSNumber)?.intValue, "\(label) files")
        assertValueFree(receipt, roots: [harness.root], label)
    }

    private func assertValueFree(_ receipt: OpenMedManagedImportReceipt, roots: [URL], _ label: String) {
        let text = String(decoding: receipt.jsonData(), as: UTF8.self)
        XCTAssertTrue(text.hasPrefix("{"), label)
        for root in roots {
            XCTAssertFalse(text.contains(root.path), "\(label) leaks a path")
        }
        for needle in ["file:", "/private/", "/Users/", ".zip", "synthetic", "openmed.synthetic"] {
            XCTAssertFalse(text.contains(needle), "\(label) leaks \(needle)")
        }
    }

    // MARK: - Apple managed app configuration

    private func defaultConfiguration(location: String? = nil) -> [String: Any] {
        var values = configs["default"]!
        if let location { values[OpenMedManagedBundleConfiguration.locationKey] = location }
        return values
    }

    func testAppleManagedAppConfigurationAdapterReadsTheManagedDictionary() throws {
        let suite = "org.openmed.tests.managed-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let adapter = OpenMedManagedAppConfiguration(defaults: defaults)
        XCTAssertNil(adapter.managedConfiguration())

        let root = try temporaryDirectory()
        let inbox = root.appending(path: "inbox")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        try archiveData("v1").write(to: inbox.appending(path: "bundle.zip"))
        defaults.set(defaultConfiguration(location: "bundle.zip"), forKey: OpenMedManagedAppConfiguration.configurationKey)
        XCTAssertEqual(OpenMedManagedAppConfiguration.configurationKey, "com.apple.configuration.managed")

        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        let receipt = importer.importManaged(from: adapter, approvedRoots: [inbox], context: context)
        XCTAssertEqual(receipt.outcome, .activated)
        XCTAssertEqual(receipt.handoff, .file)
        XCTAssertEqual(receipt.trustDigest, OpenMedBundleCache.trustDigest(try policy("default")))
        let lease = try cache.open(bundleID: bundleID, verification: .full)
        defer { lease.release() }
        XCTAssertEqual(lease.generation, generation("v1"))
        XCTAssertEqual(importer.receipts(), [receipt])
        let again = importer.importManaged(from: adapter, approvedRoots: [inbox], context: context)
        XCTAssertEqual(again.outcome, .alreadyPresent)
    }

    private func policy(_ name: String) throws -> PortableBundleTrustPolicy {
        let policies = vectors["trust_policies"] as! [String: Any]
        return try PortableBundleTrustPolicy(json: JSONSerialization.data(withJSONObject: policies[name]!))
    }

    func testConfiguredLocationsStayLocalAndApproved() throws {
        let root = try temporaryDirectory()
        let inbox = root.appending(path: "inbox")
        let outside = root.appending(path: "outside")
        for directory in [inbox, outside] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try archiveData("v1").write(to: directory.appending(path: "bundle.zip"))
        }
        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        func run(_ location: String) -> OpenMedManagedImportReceipt {
            importer.importManaged(
                from: FakeManagedConfiguration(values: defaultConfiguration(location: location)), approvedRoots: [inbox],
                context: context)
        }
        for remote in ["https://mdm.example.invalid/bundle.zip", "http://10.0.0.1/b.zip", "ftp://host/b.zip"] {
            let receipt = run(remote)
            XCTAssertEqual(receipt.outcome, .unsupported, remote)
            XCTAssertEqual(receipt.reason, .handoffUnsupported, remote)
        }
        XCTAssertEqual(run(outside.appending(path: "bundle.zip").absoluteString).reason, .locationUnapproved)
        XCTAssertEqual(run("../outside/bundle.zip").reason, .locationUnapproved)
        try FileManager.default.createSymbolicLink(
            at: inbox.appending(path: "link.zip"), withDestinationURL: outside.appending(path: "bundle.zip"))
        XCTAssertEqual(run("link.zip").reason, .locationUnapproved)
        let pending = run("later.zip")
        XCTAssertEqual(pending.outcome, .deferred)
        XCTAssertEqual(pending.reason, .handoffPending)
        let absent = importer.importManaged(
            from: FakeManagedConfiguration(values: defaultConfiguration()), approvedRoots: [inbox], context: context)
        XCTAssertEqual(absent.reason, .handoffPending)
        XCTAssertEqual(run(inbox.appending(path: "bundle.zip").absoluteString).outcome, .activated)
        XCTAssertEqual(try cache.snapshot(bundleID: bundleID).active, generation("v1"))
    }

    func testBookmarkAndDirectHandoffsImportOffline() throws {
        let root = try temporaryDirectory()
        let file = root.appending(path: "picked.zip")
        try archiveData("v2").write(to: file)
        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        let configuration = try OpenMedManagedBundleConfiguration(managedValues: defaultConfiguration())
        let bookmark = try file.bookmarkData()
        let receipt = importer.importBundle(
            OpenMedManagedImportRequest(handoff: .bookmark(bookmark), configuration: configuration, context: context))
        XCTAssertEqual(receipt.outcome, .activated)
        XCTAssertEqual(receipt.handoff, .bookmark)
        let remote = importer.importBundle(
            OpenMedManagedImportRequest(
                handoff: .file(URL(string: "https://example.invalid/bundle.zip")!), configuration: configuration,
                context: context))
        XCTAssertEqual(remote.outcome, .unsupported)
        XCTAssertEqual(remote.reason, .handoffUnsupported)
        let unreadable = importer.importBundle(
            OpenMedManagedImportRequest(
                handoff: .bookmark(Data("not a bookmark".utf8)), configuration: configuration,
                context: context))
        XCTAssertEqual(unreadable.reason, .handoffUnreadable)
    }

    func testAppKeyPinsConstrainTheDeliveredTrustPolicy() throws {
        let root = try temporaryDirectory()
        let file = root.appending(path: "bundle.zip")
        try archiveData("v1").write(to: file)
        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        let pins = (vectors["identities"] as! [String: Any])["key_pins"] as! [String: String]
        let configuration = try OpenMedManagedBundleConfiguration(managedValues: defaultConfiguration())
        XCTAssertEqual(OpenMedManagedBundleConfiguration.pin(of: configuration.trustPolicy.keys[0]), pins["openmed-test-root-a"])
        let refused = importer.importBundle(
            OpenMedManagedImportRequest(
                handoff: .file(file), configuration: configuration, context: context,
                keyPins: [pins["openmed-test-root-b"]!]))
        XCTAssertEqual(refused.reason, .trustAnchorUnpinned)
        XCTAssertNil(refused.generation)
        let accepted = importer.importBundle(
            OpenMedManagedImportRequest(
                handoff: .file(file), configuration: configuration, context: context,
                keyPins: [pins["openmed-test-root-a"]!]))
        XCTAssertEqual(accepted.outcome, .activated)
    }

    // MARK: - Concurrency

    func testConcurrentReadersSeeOnlyCompleteGenerations() throws {
        let root = try temporaryDirectory()
        let cache = try OpenMedBundleCache(
            rootDirectory: root.appending(path: "cache"), options: OpenMedBundleCacheOptions(chunkBytes: 16))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        let configuration = try OpenMedManagedBundleConfiguration(managedValues: defaultConfiguration())
        for alias in ["v1", "v2"] { try archiveData(alias).write(to: root.appending(path: "\(alias).zip")) }
        func run(_ alias: String) -> OpenMedManagedImportReceipt {
            importer.importBundle(
                OpenMedManagedImportRequest(
                    handoff: .file(root.appending(path: "\(alias).zip")), configuration: configuration, context: context))
        }
        XCTAssertEqual(run("v1").outcome, .activated)
        let held = try cache.open(bundleID: bundleID)
        let expected: [String: [String: String]] = Dictionary(
            uniqueKeysWithValues: ["v1", "v2"].map { alias in
                (
                    generation(alias)!,
                    Dictionary(
                        uniqueKeysWithValues: (bundles[alias]!["files"] as! [[String: Any]]).map {
                            ($0["path"] as! String, $0["text"] as! String)
                        })
                )
            })
        let stop = Box(false)
        let failures = Box<[String]>([])
        let lock = NSLock()
        let reader = Thread {
            while !stop.value {
                guard let lease = try? cache.open(bundleID: self.bundleID) else { continue }
                for (path, text) in expected[lease.generation] ?? [:] {
                    let data = try? Data(contentsOf: lease.payloadDirectory.appending(path: path))
                    if data != Data(text.utf8) {
                        lock.lock()
                        failures.value.append(lease.generation)
                        lock.unlock()
                    }
                }
                lease.release()
            }
        }
        reader.start()
        let results = DispatchQueue.concurrentPerformResult(count: 3) { _ in run("v2") }
        stop.value = true
        while reader.isExecuting { usleep(1_000) }
        XCTAssertEqual(failures.value, [])
        XCTAssertEqual(results.filter { $0.outcome == .activated }.count, 1)
        XCTAssertEqual(results.filter { $0.outcome == .alreadyPresent }.count, 2)
        // The lease taken before the upgrade still reads v1 completely.
        for (path, text) in expected[generation("v1")!]! {
            XCTAssertEqual(try Data(contentsOf: held.payloadDirectory.appending(path: path)), Data(text.utf8))
        }
        held.release()
        XCTAssertEqual(try cache.snapshot(bundleID: bundleID).active, generation("v2"))
    }

    // MARK: - No network

    /// Records every URL request in the process and fails it.
    final class NoNetworkProtocol: URLProtocol {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var seen = 0

        static var requests: Int {
            lock.lock()
            defer { lock.unlock() }
            return seen
        }

        static func reset() {
            lock.lock()
            seen = 0
            lock.unlock()
        }

        override class func canInit(with request: URLRequest) -> Bool {
            lock.lock()
            seen += 1
            lock.unlock()
            return true
        }

        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
        }

        override func stopLoading() {}
    }

    func testImportsNeverTouchTheNetwork() throws {
        XCTAssertTrue(URLProtocol.registerClass(NoNetworkProtocol.self))
        defer { URLProtocol.unregisterClass(NoNetworkProtocol.self) }
        NoNetworkProtocol.reset()
        let root = try temporaryDirectory()
        let inbox = root.appending(path: "inbox")
        try FileManager.default.createDirectory(at: inbox, withIntermediateDirectories: true)
        try archiveData("v2").write(to: inbox.appending(path: "v2.zip"))
        try writeBundleDirectory("v1", to: inbox.appending(path: "v1"))
        let cache = try OpenMedBundleCache(rootDirectory: root.appending(path: "cache"))
        let importer = try OpenMedManagedBundleImporter(cache: cache, stateDirectory: root.appending(path: "state"))
        let context = try PortableBundleContext(now: "2026-11-07T00:00:00Z", target: target("default"))
        let source = { (location: String) in FakeManagedConfiguration(values: self.defaultConfiguration(location: location)) }
        XCTAssertEqual(importer.importManaged(from: source("v1"), approvedRoots: [inbox], context: context).outcome, .activated)
        XCTAssertEqual(
            importer.importManaged(from: source("v2.zip"), approvedRoots: [inbox], context: context).outcome, .activated)
        XCTAssertEqual(
            importer.importManaged(
                from: source("https://example.invalid/v3.zip"), approvedRoots: [inbox], context: context
            ).reason, .handoffUnsupported)
        _ = importer.recover()
        XCTAssertEqual(NoNetworkProtocol.requests, 0)

        // Source scan: the managed import, archive, cache, and verifier code
        // reference no networking API at all.
        let sources = Self.repoRoot.appending(path: "swift/OpenMedKit/Sources/OpenMedKit")
        for name in [
            "OpenMedManagedBundleImport.swift", "OpenMedBundleArchive.swift", "OpenMedBundleCache.swift",
            "PortableBundle.swift",
        ] {
            let text = try String(contentsOf: sources.appending(path: name), encoding: .utf8)
            for api in ["URLSession", "URLRequest", "NWConnection", "import Network", "CFStream", "socket("] {
                XCTAssertFalse(text.contains(api), "\(name) references \(api)")
            }
        }
    }
}

extension DispatchQueue {
    /// Runs `body` `count` times concurrently and returns the results in order.
    fileprivate static func concurrentPerformResult<T>(count: Int, _ body: (Int) -> T) -> [T] {
        var results = [T?](repeating: nil, count: count)
        let lock = NSLock()
        concurrentPerform(iterations: count) { index in
            let value = body(index)
            lock.lock()
            results[index] = value
            lock.unlock()
        }
        return results.map { $0! }
    }
}
