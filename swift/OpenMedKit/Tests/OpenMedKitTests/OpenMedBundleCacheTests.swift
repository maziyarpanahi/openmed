import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/bundle_cache/vectors.json` against `OpenMedBundleCache`.
final class OpenMedBundleCacheTests: XCTestCase {
    private var temporaryRoots: [URL] = []

    override func tearDown() {
        for root in temporaryRoots { try? FileManager.default.removeItem(at: root) }
        temporaryRoots = []
        super.tearDown()
    }

    private func temporaryDirectory(_ name: String = UUID().uuidString) throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(path: "bundle-cache-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        temporaryRoots.append(root)
        return root
    }

    private lazy var vectors: [String: Any] = {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try! Data(contentsOf: root.appending(path: "tests/fixtures/bundle_cache/vectors.json"))
        return try! JSONSerialization.jsonObject(with: data) as! [String: Any]
    }()

    private var bundles: [String: [String: Any]] { vectors["bundles"] as! [String: [String: Any]] }
    private var bundleID: String { (vectors["identities"] as! [String: Any])["bundle_id"] as! String }
    private func generation(_ alias: String) -> String { bundles[alias]!["generation"] as! String }

    private func files(_ alias: String) -> [String: Data] {
        var result: [String: Data] = [:]
        for entry in bundles[alias]!["files"] as! [[String: Any]] {
            result[entry["path"] as! String] = Data((entry["text"] as! String).utf8)
        }
        return result
    }

    private func policy(_ name: String) throws -> PortableBundleTrustPolicy {
        let policies = vectors["trust_policies"] as! [String: Any]
        return try PortableBundleTrustPolicy(json: JSONSerialization.data(withJSONObject: policies[name]!))
    }

    private func context() throws -> PortableBundleContext {
        try PortableBundleContext(now: (vectors["context"] as! [String: Any])["now"] as! String)
    }

    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Int64
        init(_ value: Int64) { self.value = value }
        var now: Int64 {
            get {
                lock.lock()
                defer { lock.unlock() }
                return value
            }
            set {
                lock.lock()
                value = newValue
                lock.unlock()
            }
        }
    }

    /// Serves synthetic bytes by range with scripted faults; never touches the network.
    private final class FakeSource: OpenMedBundleRangeSource {
        let identity: String
        private let files: [String: Data]
        private let failAfter: Int64?
        private let cancelAfter: Int64?
        private let corrupt: (String, Int)?
        private let extra: (String, Int)?
        private let ignoreRange: Bool
        private let onRead: () -> Void
        private let lock = NSLock()
        private var deliveredBytes: Int64 = 0

        init(
            files: [String: Data], identity: String, failAfter: Int64? = nil, cancelAfter: Int64? = nil,
            corrupt: (String, Int)? = nil, extra: (String, Int)? = nil, ignoreRange: Bool = false,
            onRead: @escaping () -> Void = {}
        ) {
            self.files = files
            self.identity = identity
            self.failAfter = failAfter
            self.cancelAfter = cancelAfter
            self.corrupt = corrupt
            self.extra = extra
            self.ignoreRange = ignoreRange
            self.onRead = onRead
        }

        var delivered: Int64 {
            lock.lock()
            defer { lock.unlock() }
            return deliveredBytes
        }

        var cancelled: Bool { cancelAfter.map { delivered >= $0 } ?? false }

        final class Reader: OpenMedBundleRangeReader {
            let bytes: [UInt8]
            var position: Int
            let source: FakeSource
            init(bytes: [UInt8], position: Int, source: FakeSource) {
                self.bytes = bytes
                self.position = position
                self.source = source
            }

            func read(upToCount count: Int) throws -> Data {
                source.onRead()
                if position >= bytes.count { return Data() }
                var allowed = Int64(count)
                if let failAfter = source.failAfter {
                    let left = failAfter - source.delivered
                    if left <= 0 { throw URLError(.networkConnectionLost) }
                    allowed = min(allowed, left)
                }
                let length = Int(min(allowed, Int64(bytes.count - position)))
                let data = Data(bytes[position..<(position + length)])
                position += length
                source.lock.lock()
                source.deliveredBytes += Int64(length)
                source.lock.unlock()
                return data
            }
        }

        func open(path: String, offset: Int64) throws -> OpenMedBundleRangeResponse {
            guard var data = files[path].map({ [UInt8]($0) }) else { throw URLError(.fileDoesNotExist) }
            if let corrupt, corrupt.0 == path { data[corrupt.1] ^= 0xff }
            if let extra, extra.0 == path { data += [UInt8](repeating: 0x58, count: extra.1) }
            let start = ignoreRange ? 0 : offset
            return OpenMedBundleRangeResponse(offset: start, reader: Reader(bytes: data, position: Int(start), source: self))
        }
    }

    private func request(
        _ alias: String, source: any OpenMedBundleRangeSource, policy name: String = "default", activate: Bool = true,
        confirm: Bool = true, cancelled: @escaping () -> Bool = { false }
    ) throws -> OpenMedBundleAcquireRequest {
        OpenMedBundleAcquireRequest(
            manifest: Data((bundles[alias]!["manifest"] as! String).utf8),
            envelope: Data((bundles[alias]!["envelope"] as! String).utf8),
            source: source, trustPolicy: try policy(name), context: try context(), activate: activate, confirm: confirm,
            isCancelled: cancelled)
    }

    // MARK: - Vector replay

    private static let aliasKeys: Set<String> = ["generation", "active", "last_green", "rollback", "evicted"]

    private func resolve(_ value: Any, key: String? = nil) -> Any {
        if let object = value as? [String: Any] {
            return object.reduce(into: [String: Any]()) { $0[$1.key] = resolve($1.value, key: $1.key) }
        }
        if let array = value as? [Any] { return array.map { resolve($0, key: key) } }
        if let text = value as? String, let key, Self.aliasKeys.contains(key) { return generation(text) }
        return value
    }

    private func canonical(_ value: Any) -> String {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .fragmentsAllowed])
        return String(decoding: data, as: UTF8.self)
    }

    private func optional(_ value: Any?) -> Any { value ?? NSNull() }

    private func failureObject(_ error: Error) -> [String: Any] {
        guard let error = error as? OpenMedBundleCacheError else {
            return ["ok": false, "reason": "unexpected", "bundle_reason": NSNull()]
        }
        return ["ok": false, "reason": error.reason.rawValue, "bundle_reason": optional(error.bundleReason?.rawValue)]
    }

    private final class Replay {
        let test: OpenMedBundleCacheTests
        let scenario: [String: Any]
        let root: URL
        let clock: Clock
        var owner = 1
        var available: Int64 = Int64.max / 4
        var leases: [String: OpenMedBundleLease] = [:]
        var cache: OpenMedBundleCache!

        init(test: OpenMedBundleCacheTests, scenario: [String: Any]) throws {
            self.test = test
            self.scenario = scenario
            root = try test.temporaryDirectory("replay")
            clock = Clock((test.vectors["clock"] as! [String: Any])["start_ms"] as! Int64)
            cache = try newCache()
        }

        var options: [String: Any] { scenario["options"] as! [String: Any] }

        func newCache() throws -> OpenMedBundleCache {
            let clock = self.clock
            let options = OpenMedBundleCacheOptions(
                quotaBytes: (self.options["quota_bytes"] as! NSNumber).int64Value,
                chunkBytes: self.options["chunk_bytes"] as! Int,
                lockTimeoutMilliseconds: 2_000,
                leaseTTLMilliseconds: (self.options["lease_ttl_ms"] as! NSNumber).int64Value,
                stagingMaxAgeMilliseconds: (self.options["staging_max_age_ms"] as! NSNumber).int64Value,
                now: { clock.now },
                availableBytes: { [unowned self] _ in self.available },
                ownerID: "owner-\(owner)")
            owner += 1
            return try OpenMedBundleCache(rootDirectory: root, options: options)
        }

        func installCrash(_ raw: Any?) {
            guard let crash = raw as? [String: Any] else {
                cache.checkpointHook = nil
                return
            }
            let name = crash["checkpoint"] as! String
            let occurrence = crash["occurrence"] as! Int
            var seen = 0
            cache.checkpointHook = { checkpoint in
                guard checkpoint == name else { return }
                seen += 1
                if seen == occurrence { throw OpenMedBundleCacheCrash(checkpoint: checkpoint) }
            }
        }

        func attempt(_ block: () throws -> [String: Any]) -> [String: Any] {
            do { return try block() } catch { return test.failureObject(error) }
        }

        func run() throws {
            let id = scenario["id"] as! String
            let step = (test.vectors["clock"] as! [String: Any])["step_ms"] as! Int64
            for (index, raw) in (scenario["steps"] as! [[String: Any]]).enumerated() {
                clock.now += step
                let op = raw["op"] as! String
                if let actual = try execute(raw) {
                    XCTAssertEqual(
                        test.canonical(actual), test.canonical(test.resolve(raw["expect"]!)), "\(id) step \(index) \(op)")
                }
            }
        }

        func execute(_ item: [String: Any]) throws -> [String: Any]? {
            let bundleID = test.bundleID
            switch item["op"] as! String {
            case "acquire": return try acquire(item)
            case "restart":
                clock.now += (item["advance_ms"] as! NSNumber).int64Value
                leases = [:]
                cache = try newCache()
                return nil
            case "recover":
                return [
                    "actions": cache.recover().actions.map {
                        ["action": $0.action.rawValue, "generation": test.optional($0.generation)] as [String: Any]
                    }
                ]
            case "activate":
                return attempt {
                    try cache.activate(
                        bundleID: bundleID, generation: test.generation(item["generation"] as! String),
                        confirm: item["confirm"] as! Bool)
                    return ["ok": true]
                }
            case "confirm":
                return attempt {
                    try cache.confirm(bundleID: bundleID)
                    return ["ok": true]
                }
            case "rollback":
                return attempt {
                    try cache.rollback(bundleID: bundleID, quarantineAbandoned: item["quarantine"] as! Bool)
                    return ["ok": true]
                }
            case "pin", "unpin":
                return attempt {
                    let target = test.generation(item["generation"] as! String)
                    if item["op"] as! String == "pin" {
                        try cache.pin(bundleID: bundleID, generation: target)
                    } else {
                        try cache.unpin(bundleID: bundleID, generation: target)
                    }
                    return ["ok": true]
                }
            case "lease":
                return attempt {
                    let lease = try cache.lease(
                        bundleID: bundleID, generation: (item["generation"] as? String).map { test.generation($0) })
                    leases[item["lease"] as! String] = lease
                    return ["ok": true, "generation": lease.generation]
                }
            case "release":
                return attempt {
                    try cache.release(leases[item["lease"] as! String]!)
                    return ["ok": true]
                }
            case "open":
                return attempt {
                    let lease = try cache.open(
                        bundleID: bundleID, verification: item["verification"] as! String == "full" ? .full : .metadata)
                    leases[item["lease"] as! String] = lease
                    return ["ok": true, "generation": lease.generation, "files_checked": lease.filesChecked]
                }
            case "evict":
                installCrash(item["crash_at"])
                defer { cache.checkpointHook = nil }
                let additional = (item["additional_bytes"] as! NSNumber).int64Value
                if (item["expect"] as! [String: Any])["crashed"] != nil {
                    XCTAssertThrowsError(try cache.evictChecked(additionalBytes: additional)) { error in
                        XCTAssertTrue(error is OpenMedBundleCacheCrash)
                    }
                    return ["crashed": true]
                }
                let decision = try cache.evictChecked(additionalBytes: additional)
                return ["evicted": decision.evicted, "satisfied": decision.satisfied, "used_bytes": decision.usedBytes]
            case "tamper":
                try tamper(item)
                return nil
            case "advance":
                clock.now += (item["ms"] as! NSNumber).int64Value
                return nil
            case "snapshot":
                return try snapshot()
            default:
                XCTFail("unknown op")
                return nil
            }
        }

        func acquire(_ item: [String: Any]) throws -> [String: Any] {
            let described = item["source"] as! [String: Any]
            let corrupt = (described["corrupt"] as? [String: Any]).map { ($0["path"] as! String, $0["offset"] as! Int) }
            let extra = (described["extra_bytes"] as? [String: Any]).map { ($0["path"] as! String, $0["count"] as! Int) }
            let source = FakeSource(
                files: test.files(described["serve"] as! String), identity: described["identity"] as! String,
                failAfter: (described["fail_after_bytes"] as? NSNumber)?.int64Value,
                cancelAfter: (described["cancel_after_bytes"] as? NSNumber)?.int64Value,
                corrupt: corrupt, extra: extra, ignoreRange: described["ignore_range"] as! Bool)
            let request = try test.request(
                item["bundle"] as! String, source: source, policy: item["policy"] as! String,
                activate: item["activate"] as! Bool, confirm: item["confirm"] as! Bool
            ) { source.cancelled }
            installCrash(item["crash_at"])
            if let bytes = item["storage_available_bytes"] as? NSNumber { available = bytes.int64Value }
            defer {
                available = Int64.max / 4
                cache.checkpointHook = nil
            }
            if (item["expect"] as! [String: Any])["crashed"] != nil {
                XCTAssertThrowsError(try cache.acquireChecked(request)) { error in
                    XCTAssertTrue(error is OpenMedBundleCacheCrash)
                }
                return ["crashed": true]
            }
            let result = try cache.acquireChecked(request)
            guard result.ok else {
                return [
                    "ok": false, "reason": test.optional(result.reason?.rawValue),
                    "bundle_reason": test.optional(result.bundleReason?.rawValue),
                ]
            }
            return [
                "ok": true, "generation": test.optional(result.generation), "state": test.optional(result.state?.rawValue),
                "from_cache": result.fromCache, "resumed": result.resumed,
                "resume_reason": test.optional(result.resumeReason?.rawValue),
                "bytes_transferred": result.bytesTransferred, "verified_bytes": result.verifiedBytes,
                "files_verified": result.filesVerified, "evicted": result.evicted,
            ]
        }

        func tamper(_ item: [String: Any]) throws {
            let area = item["area"] as! String == "ready" ? "generations" : "staging"
            let file = root.appending(path: OpenMedBundleCache.bundleKey(bundleID: test.bundleID))
                .appending(path: area).appending(path: test.generation(item["generation"] as! String))
                .appending(path: "payload").appending(path: item["path"] as! String)
            let handle = try FileHandle(forUpdating: file)
            defer { try? handle.close() }
            let offset = UInt64(item["offset"] as! Int)
            if item["mode"] as! String == "truncate" {
                try handle.truncate(atOffset: offset)
            } else {
                try handle.seek(toOffset: offset)
                let value = try XCTUnwrap(handle.read(upToCount: 1)?.first)
                try handle.seek(toOffset: offset)
                try handle.write(contentsOf: Data([value ^ 0xff]))
            }
        }

        func snapshot() throws -> [String: Any] {
            let snapshot = try cache.snapshot(bundleID: test.bundleID)
            return [
                "pointer": [
                    "active": test.optional(snapshot.active), "last_green": test.optional(snapshot.lastGreen),
                    "rollback": test.optional(snapshot.rollback),
                ],
                "generations": snapshot.generations.map {
                    [
                        "generation": $0.generation, "state": $0.state.rawValue, "total_bytes": $0.totalBytes,
                        "verified_bytes": $0.verifiedBytes, "protections": $0.protections.map(\.rawValue),
                    ] as [String: Any]
                },
                "quarantined": snapshot.quarantined.map { ["generation": $0.generation, "reason": $0.reason] },
                "used_bytes": snapshot.usedBytes,
            ]
        }
    }

    func testEverySharedScenarioReplaysExactly() throws {
        let scenarios = vectors["scenarios"] as! [[String: Any]]
        XCTAssertGreaterThanOrEqual(scenarios.count, 30)
        for scenario in scenarios {
            try Replay(test: self, scenario: scenario).run()
        }
    }

    func testIdentitiesAndVocabulariesMatchTheSharedVectors() throws {
        let identities = vectors["identities"] as! [String: Any]
        XCTAssertEqual(OpenMedBundleCache.bundleKey(bundleID: bundleID), identities["bundle_key"] as? String)
        for (name, digest) in identities["trust_digests"] as! [String: String] {
            XCTAssertEqual(OpenMedBundleCache.trustDigest(try policy(name)), digest, name)
        }
        for alias in ["v1", "v2", "v3", "v4"] {
            let contentID = PortableBundle.contentID(of: Data((bundles[alias]!["manifest"] as! String).utf8))
            XCTAssertEqual(contentID, bundles[alias]!["content_id"] as? String)
            XCTAssertEqual(OpenMedBundleCache.generationID(contentID: contentID), generation(alias))
        }
        XCTAssertEqual(vectors["reason_codes"] as? [String], OpenMedBundleCacheReason.allCases.map(\.rawValue))
        XCTAssertEqual(vectors["states"] as? [String], OpenMedBundleGenerationState.allCases.map(\.rawValue))
        XCTAssertEqual(vectors["recovery_actions"] as? [String], OpenMedBundleRecoveryAction.allCases.map(\.rawValue))
        XCTAssertEqual(vectors["resume_reasons"] as? [String], OpenMedBundleResumeReason.allCases.map(\.rawValue))
        XCTAssertEqual(
            vectors["checkpoints"] as? [String],
            [
                "staging_record_written", "chunk_committed", "verifying_record_written", "generation_committed",
                "pointer_committed", "eviction_committed",
            ])
    }

    // MARK: - Concurrency, offline, and value-free checks

    private func cache(_ root: URL, owner: String, lockTimeout: Int64 = 5_000) throws -> OpenMedBundleCache {
        try OpenMedBundleCache(
            rootDirectory: root,
            options: OpenMedBundleCacheOptions(chunkBytes: 16, lockTimeoutMilliseconds: lockTimeout, ownerID: owner))
    }

    func testConcurrentCallersConvergeOnOneVerifiedGeneration() throws {
        let root = try temporaryDirectory()
        let caches = try [cache(root, owner: "process-a"), cache(root, owner: "process-b"), cache(root, owner: "process-c")]
        let files = files("v1")
        let requests = try caches.map { _ in try request("v1", source: FakeSource(files: files, identity: "source-a")) }
        var results = [OpenMedBundleAcquireResult?](repeating: nil, count: caches.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: caches.count) { index in
            let result = caches[index].acquire(requests[index])
            lock.lock()
            results[index] = result
            lock.unlock()
        }
        let finished = results.compactMap { $0 }
        XCTAssertTrue(finished.allSatisfy(\.ok), "\(finished)")
        XCTAssertEqual(finished.filter { !$0.fromCache }.count, 1)
        XCTAssertEqual(Set(finished.compactMap(\.generation)), [generation("v1")])
        let snapshot = try caches[0].snapshot(bundleID: bundleID)
        XCTAssertEqual(snapshot.generations.map(\.generation), [generation("v1")])
        XCTAssertEqual(snapshot.generations.first?.state, .active)
        let lease = try caches[1].open(bundleID: bundleID, verification: .full)
        for (path, data) in files {
            XCTAssertEqual(try Data(contentsOf: lease.payloadDirectory.appending(path: path)), data)
        }
        lease.release()
    }

    func testLockWaitIsBoundedAndTyped() throws {
        let root = try temporaryDirectory()
        let holder = try cache(root, owner: "holder")
        let waiter = try cache(root, owner: "waiter", lockTimeout: 50)
        let entered = DispatchSemaphore(value: 0)
        let proceed = DispatchSemaphore(value: 0)
        var signalled = false
        let source = FakeSource(files: files("v1"), identity: "source-a") {
            if !signalled {
                signalled = true
                entered.signal()
                proceed.wait()
            }
        }
        let holderRequest = try request("v1", source: source)
        let done = DispatchSemaphore(value: 0)
        var held: OpenMedBundleAcquireResult?
        DispatchQueue.global().async {
            held = holder.acquire(holderRequest)
            done.signal()
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 10), .success)
        let refused = waiter.acquire(try request("v1", source: FakeSource(files: files("v1"), identity: "source-a")))
        XCTAssertEqual(refused.reason, .lockTimeout)
        XCTAssertThrowsError(try waiter.snapshot(bundleID: bundleID)) { error in
            XCTAssertEqual(error as? OpenMedBundleCacheError, OpenMedBundleCacheError(.lockTimeout))
        }
        proceed.signal()
        XCTAssertEqual(done.wait(timeout: .now() + 30), .success)
        XCTAssertEqual(held?.ok, true)
        XCTAssertTrue(waiter.acquire(try request("v1", source: FakeSource(files: files("v1"), identity: "source-a"))).fromCache)
    }

    /// Records any URL loading attempt made through the URL loading system.
    private final class AttemptCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0
        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }
        func increment() {
            lock.lock()
            count += 1
            lock.unlock()
        }
    }

    private final class NetworkTripwire: URLProtocol {
        static let counter = AttemptCounter()
        static var attempts: Int { counter.value }
        override class func canInit(with request: URLRequest) -> Bool {
            counter.increment()
            return false
        }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {}
        override func stopLoading() {}
    }

    func testPreparedBundleImportsAndOpensWithoutAnySource() throws {
        XCTAssertTrue(URLProtocol.registerClass(NetworkTripwire.self))
        defer {
            URLProtocol.unregisterClass(NetworkTripwire.self)
            XCTAssertEqual(NetworkTripwire.attempts, 0)
        }
        let prepared = try temporaryDirectory("prepared")
        for (path, data) in files("v1") {
            let target = prepared.appending(path: path)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: target)
        }
        let root = try temporaryDirectory()
        let imported = try cache(root, owner: "importer").acquire(
            request("v1", source: OpenMedDirectoryBundleSource(directory: prepared), activate: false))
        XCTAssertTrue(imported.ok)
        XCTAssertEqual(imported.state, .evictable)
        try FileManager.default.removeItem(at: prepared)

        let restarted = try cache(root, owner: "restarted")
        XCTAssertEqual(restarted.recover().actions, [])
        try restarted.activate(bundleID: bundleID, generation: generation("v1"))
        let lease = try restarted.open(bundleID: bundleID, verification: .full)
        XCTAssertEqual(lease.filesChecked, files("v1").count)
        lease.release()
    }

    func testMetadataResultsAndErrorsCarryNoPathsOrSourceIdentities() throws {
        let root = try temporaryDirectory("private-root-sentinel")
        let identity = "https://models.invalid/private?token=sentinel-credential"
        let cache = try cache(root, owner: "owner")
        let interrupted = cache.acquire(try request("v1", source: FakeSource(files: files("v1"), identity: identity, failAfter: 100)))
        XCTAssertEqual(interrupted.reason, .transferInterrupted)
        let result = cache.acquire(try request("v1", source: FakeSource(files: files("v1"), identity: identity)))
        XCTAssertTrue(result.ok)
        let lease = try cache.open(bundleID: bundleID)
        var rollbackError: Error?
        XCTAssertThrowsError(try cache.rollback(bundleID: bundleID)) { rollbackError = $0 }
        let surfaces = [
            "\(result)", "\(interrupted)", "\(lease)", "\(cache)", "\(try cache.snapshot(bundleID: bundleID))",
            "\(cache.recover())", "\(rollbackError!)", rollbackError!.localizedDescription,
        ]
        let sentinels = [root.path, "private-root-sentinel", "sentinel-credential", "models.invalid", "token="]
        for surface in surfaces {
            for sentinel in sentinels { XCTAssertFalse(surface.contains(sentinel), "leak of \(sentinel) in \(surface)") }
        }
        let records = (FileManager.default.enumerator(atPath: root.path)?.allObjects as? [String] ?? [])
            .filter { $0.hasSuffix(".json") && !$0.contains("/payload/") }
        XCTAssertGreaterThanOrEqual(records.count, 3)
        for record in records {
            let text = try String(contentsOf: root.appending(path: record), encoding: .utf8)
            for sentinel in sentinels { XCTAssertFalse(text.contains(sentinel), "leak of \(sentinel) in a record") }
        }
        lease.release()
    }

    // MARK: - OpenMedModelStore value-free errors

    private final class FailingHubProtocol: URLProtocol {
        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            let response = HTTPURLResponse(url: request.url!, statusCode: 503, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data())
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    func testModelStoreErrorsCarryNoURLsOrPaths() async throws {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FailingHubProtocol.self]
        let session = URLSession(configuration: configuration)
        let cacheRoot = try temporaryDirectory("private-store-sentinel")
        do {
            _ = try await OpenMedModelStore.downloadMLXModel(
                repoID: "OpenMed/Secret-Repo-Sentinel", revision: "main", cacheDirectory: cacheRoot, session: session)
            XCTFail("expected an HTTP failure")
        } catch let error as OpenMedModelStoreError {
            XCTAssertEqual(error, .httpError(statusCode: 503))
            XCTAssertEqual(error.reason, "http_status")
            for surface in ["\(error)", error.localizedDescription] {
                for sentinel in ["Secret-Repo-Sentinel", "private-store-sentinel", "https://", "huggingface"] {
                    XCTAssertFalse(surface.contains(sentinel), "leak of \(sentinel) in \(surface)")
                }
            }
        }
        let cases: [OpenMedModelStoreError] = [.invalidResponse, .missingManifest, .missingWeights, .invalidManifestPath("outside_snapshot")]
        for error in cases {
            XCTAssertFalse(error.localizedDescription.contains("/"), error.reason)
        }
        let bundleCache = try OpenMedModelStore.bundleCache(cacheDirectory: cacheRoot.appending(path: "bundles"))
        XCTAssertEqual(bundleCache.recover().actions, [])
    }
}
