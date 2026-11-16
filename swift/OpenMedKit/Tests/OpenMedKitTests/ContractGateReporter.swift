import CryptoKit
import Foundation
import XCTest

/// Records the Swift side of the cross-runtime contract gate.
///
/// `tests/fixtures/parity/contract_gate.json` names the shared parity
/// fixtures OpenMedKit must pass and the Swift test files that consume them.
/// When `OPENMED_CONTRACT_GATE_RESULTS_DIR` is set, this reporter writes
/// `<dir>/swift.json`, a result file matching
/// `tests/fixtures/parity/contract_gate_result.schema.json`, after every test
/// that consumed a fixture. When the variable is unset it does nothing, so the
/// parity tests behave exactly as before.
///
/// A consumer test class subclasses ``ContractGateTestCase`` and reads its
/// fixture through ``read(_:file:)``, which hashes the bytes the test actually
/// read. A test that read a fixture covers every case of it, unless it scopes
/// assertions with ``ContractGateTestCase/contractCase(_:of:file:_:)``: then it
/// covers only the cases it scoped, and a failure outside every scope fails all
/// of the fixture's cases. A case passes when at least one covering test passed
/// it and none failed it, the same rule as the Python runner.
///
/// The result carries fixture ids, digests, case ids derived from the fixture,
/// counts and reason codes only: never fixture text, assertion messages, paths
/// or timestamps. Case ids a test names that the fixture does not define are
/// dropped and reported as the fixture-level code `case_id_unknown`.
///
/// Tests run serially in one process; run `swift test` without `--parallel`
/// when writing a result file.
final class ContractGateReporter: @unchecked Sendable {
    static let runtime = "swift"
    static let resultsDirectoryVariable = "OPENMED_CONTRACT_GATE_RESULTS_DIR"
    static let manifestPath = "tests/fixtures/parity/contract_gate.json"
    static let resultFileName = "swift.json"

    /// The reporter every ``ContractGateTestCase`` uses, configured from the
    /// process environment.
    static let shared = ContractGateReporter(
        repositoryRoot: defaultRepositoryRoot(),
        resultsDirectory: ProcessInfo.processInfo.environment[resultsDirectoryVariable]
            .flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
    )

    /// One case-group rule from the manifest.
    struct CaseGroup: Equatable {
        let key: String
        let idField: String?
    }

    /// A fixture whose Swift cell is `required`.
    struct Fixture: Equatable {
        let id: String
        let path: String
        let caseGroups: [CaseGroup]
        let consumers: [String]
    }

    /// How one test covered one case, in the Python runner's precedence.
    enum Outcome: Hashable {
        case passed
        case failed
        case error
        case skipped
    }

    private struct Scope {
        let fixtureID: String
        let caseID: String
        let failuresBefore: Int
    }

    private struct TestState {
        var reads: Set<String> = []
        var scoped: [String: [String: Set<Outcome>]] = [:]
        var scopes: [Scope] = []
        var scopedFailures = 0
        var escaped: [Scope] = []
    }

    private struct FixtureRecord {
        var data: Data?
        var digests: [String] = []
        var coversAll: [Outcome] = []
        var cases: [String: Set<Outcome>] = [:]
        var passedTests = 0
        var failedTests = 0
        var skippedTests = 0
    }

    let repositoryRoot: URL
    let resultsDirectory: URL?
    private let manifestDigest: String?
    private let fixtures: [Fixture]
    private let lock = NSLock()
    private var current: TestState?
    private var records: [String: FixtureRecord] = [:]

    /// Creates a reporter.
    ///
    /// - Parameters:
    ///   - repositoryRoot: Checkout root that manifest and fixture paths are
    ///     relative to.
    ///   - resultsDirectory: Directory for `swift.json`, or `nil` to record
    ///     nothing.
    init(repositoryRoot: URL, resultsDirectory: URL?) {
        self.repositoryRoot = repositoryRoot.standardizedFileURL
        guard let resultsDirectory else {
            self.resultsDirectory = nil
            manifestDigest = nil
            fixtures = []
            return
        }
        let manifestURL = self.repositoryRoot.appending(path: Self.manifestPath)
        guard let data = try? Data(contentsOf: manifestURL), let parsed = Self.requiredFixtures(manifest: data) else {
            FileHandle.standardError.write(Data("contract_gate swift: manifest_unreadable\n".utf8))
            self.resultsDirectory = nil
            manifestDigest = nil
            fixtures = []
            return
        }
        self.resultsDirectory = resultsDirectory
        manifestDigest = Self.sha256(data)
        fixtures = parsed
    }

    var isEnabled: Bool { resultsDirectory != nil }

    // MARK: - Recording

    /// Reads a file and, when it is a required fixture read from one of its
    /// declared consumer files during a test, records that the test consumed
    /// it and the digest of the bytes read.
    ///
    /// - Parameters:
    ///   - url: File to read.
    ///   - file: Absolute path of the calling test source file.
    /// - Returns: The file's bytes.
    func read(_ url: URL, file: String) throws -> Data {
        let data = try Data(contentsOf: url)
        guard isEnabled, let path = relativePath(url), let consumer = relativePath(URL(fileURLWithPath: file)) else {
            return data
        }
        lock.withLock {
            guard let fixture = fixtures.first(where: { $0.path == path }), fixture.consumers.contains(consumer) else {
                return
            }
            var record = records[fixture.id] ?? FixtureRecord()
            let digest = Self.sha256(data)
            if record.data == nil { record.data = data }
            if !record.digests.contains(digest) { record.digests.append(digest) }
            records[fixture.id] = record
            current?.reads.insert(fixture.id)
        }
        return data
    }

    /// Runs `body` as the check of one case.
    ///
    /// The case fails when `failureCount` grows while `body` runs. When `body`
    /// throws without recording a failure, the case errors, or is skipped when
    /// the test ends skipped (`XCTSkip`). Errors propagate untouched, so the
    /// test stops exactly as it would without the scope; XCTest may report an
    /// uncaught error at this helper's line rather than the test's.
    ///
    /// - Parameters:
    ///   - caseID: `<group key>/<item id>`, `<group key>/<index>` or
    ///     `<group key>`.
    ///   - fixtureID: Manifest fixture id.
    ///   - file: Absolute path of the calling test source file.
    ///   - failureCount: The running test's failure count.
    ///   - body: The case's assertions.
    func check(
        _ caseID: String, of fixtureID: String, file: String, failureCount: @autoclosure () -> Int, _ body: () throws -> Void
    ) rethrows {
        guard beginScope(caseID, of: fixtureID, file: file, failures: failureCount()) else {
            try body()
            return
        }
        var completed = false
        defer { endScope(failures: failureCount(), threw: !completed) }
        try body()
        completed = true
    }

    /// Starts recording a test.
    func testWillStart() {
        guard isEnabled else { return }
        lock.withLock { current = TestState() }
    }

    /// Finishes the running test, folds its coverage into the result and
    /// rewrites the result file when the test consumed a fixture.
    ///
    /// - Parameters:
    ///   - failureCount: The test's total failure count.
    ///   - skipped: Whether the test was skipped.
    func testDidFinish(failureCount: Int, skipped: Bool) {
        guard isEnabled else { return }
        let consumed: Bool = lock.withLock {
            guard var state = current else { return false }
            current = nil
            // A scope left open was interrupted by a failure that stopped the test.
            for scope in state.scopes {
                state.scoped[scope.fixtureID, default: [:]][scope.caseID, default: []].insert(.failed)
            }
            // An error thrown out of a scope either ended the test skipped
            // (XCTSkip) or was recorded as one failure when it left the test.
            let thrown: Outcome = skipped ? .skipped : .error
            for scope in state.escaped {
                state.scoped[scope.fixtureID, default: [:]][scope.caseID, default: []].insert(thrown)
            }
            let escapedErrors = thrown == .error ? state.escaped.count : 0
            let unscoped = max(0, failureCount - state.scopedFailures - escapedErrors)
            let overall: Outcome = failureCount > 0 ? .failed : (skipped ? .skipped : .passed)
            let consumedIDs = state.reads.union(state.scoped.keys)
            for fixtureID in consumedIDs {
                var record = records[fixtureID] ?? FixtureRecord()
                switch overall {
                case .passed: record.passedTests += 1
                case .skipped: record.skippedTests += 1
                case .failed, .error: record.failedTests += 1
                }
                if let cases = state.scoped[fixtureID], !cases.isEmpty {
                    if unscoped > 0 {
                        record.coversAll.append(.failed)
                    } else {
                        for (caseID, outcomes) in cases {
                            record.cases[caseID, default: []].formUnion(outcomes)
                        }
                    }
                } else {
                    record.coversAll.append(overall)
                }
                records[fixtureID] = record
            }
            return !consumedIDs.isEmpty
        }
        if consumed { write() }
    }

    private func beginScope(_ caseID: String, of fixtureID: String, file: String, failures: Int) -> Bool {
        guard isEnabled, let consumer = relativePath(URL(fileURLWithPath: file)) else { return false }
        return lock.withLock {
            guard current != nil,
                let fixture = fixtures.first(where: { $0.id == fixtureID }),
                fixture.consumers.contains(consumer)
            else { return false }
            current?.scopes.append(Scope(fixtureID: fixtureID, caseID: caseID, failuresBefore: failures))
            return true
        }
    }

    private func endScope(failures: Int, threw: Bool) {
        lock.withLock {
            guard var state = current, let scope = state.scopes.popLast() else { return }
            let recorded = max(0, failures - scope.failuresBefore)
            state.scopedFailures += recorded
            if recorded > 0 {
                state.scoped[scope.fixtureID, default: [:]][scope.caseID, default: []].insert(.failed)
            } else if threw {
                // An error that already recorded a failure, such as a failed
                // XCTUnwrap, is handled above. Any other error is recorded by
                // XCTest once it leaves the test, or ends the test skipped.
                state.escaped.append(scope)
            } else {
                state.scoped[scope.fixtureID, default: [:]][scope.caseID, default: []].insert(.passed)
            }
            current = state
        }
    }

    // MARK: - Result

    /// Builds the result document from everything recorded so far.
    ///
    /// - Returns: The result, or `nil` when the reporter is disabled.
    func result() -> ContractGateResult? {
        guard isEnabled, let manifestDigest else { return nil }
        let snapshot = lock.withLock { records }
        return ContractGateResult(
            runtime: Self.runtime,
            manifestSHA256: manifestDigest,
            fixtures: fixtures.map { fixtureResult($0, record: snapshot[$0.id]) }
        )
    }

    /// Rewrites `<results directory>/swift.json` from the current result.
    func write() {
        guard let resultsDirectory, let result = result() else { return }
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
            var data = try encoder.encode(result)
            data.append(0x0A)
            try FileManager.default.createDirectory(at: resultsDirectory, withIntermediateDirectories: true)
            try data.write(to: resultsDirectory.appending(path: Self.resultFileName), options: .atomic)
        } catch {
            FileHandle.standardError.write(Data("contract_gate swift: result_unwritable\n".utf8))
        }
    }

    private func fixtureResult(_ fixture: Fixture, record: FixtureRecord?) -> ContractGateFixtureResult {
        var result = ContractGateFixtureResult(fixtureID: fixture.id, fixtureSHA256: String(repeating: "0", count: 64))
        let bytes = record?.data ?? (try? Data(contentsOf: repositoryRoot.appending(path: fixture.path)))
        guard let bytes else {
            result.errors = ["fixture_unreadable"]
            return result
        }
        result.fixtureSHA256 = Self.sha256(bytes)
        guard let document = try? JSONSerialization.jsonObject(with: bytes),
            let caseIDs = Self.caseIDs(groups: fixture.caseGroups, document: document)
        else {
            result.errors = ["fixture_unreadable"]
            return result
        }
        let present = fixture.consumers.allSatisfy {
            FileManager.default.fileExists(atPath: repositoryRoot.appending(path: $0).path)
        }
        guard present, !fixture.consumers.isEmpty else {
            result.failed = caseIDs.map { ContractGateCaseOutcome(caseID: $0, reason: "consumer_missing") }
            return result
        }
        guard let record, record.passedTests + record.failedTests + record.skippedTests > 0 else {
            result.failed = caseIDs.map { ContractGateCaseOutcome(caseID: $0, reason: "consumer_not_collected") }
            return result
        }
        // A failure under a case id the fixture does not define cannot be
        // attributed, so it fails every case; the id itself is never written.
        let unknown = record.cases.filter { !caseIDs.contains($0.key) }
        let all = Set(record.coversAll).union(unknown.values.joined().filter { $0 == .failed || $0 == .error })
        for caseID in caseIDs {
            let outcomes = all.union(record.cases[caseID] ?? [])
            if outcomes.contains(.failed) {
                result.failed.append(ContractGateCaseOutcome(caseID: caseID, reason: "consumer_failed"))
            } else if outcomes.contains(.error) {
                result.failed.append(ContractGateCaseOutcome(caseID: caseID, reason: "consumer_error"))
            } else if outcomes.contains(.passed) {
                result.passed.append(caseID)
            } else if outcomes.contains(.skipped) {
                result.skipped.append(ContractGateCaseOutcome(caseID: caseID, reason: "consumer_skipped"))
            } else {
                result.failed.append(ContractGateCaseOutcome(caseID: caseID, reason: "case_not_exercised"))
            }
        }
        var errors: [String] = []
        if record.digests.count > 1 { errors.append("fixture_digest_inconsistent") }
        if !unknown.isEmpty { errors.append("case_id_unknown") }
        result.errors = errors.isEmpty ? nil : errors
        result.tests = ContractGateTestCounts(
            passed: record.passedTests, failed: record.failedTests, skipped: record.skippedTests)
        return result
    }

    // MARK: - Manifest and case ids

    /// Parses the manifest and returns the fixtures whose Swift cell is
    /// `required`, in manifest order.
    ///
    /// - Parameter manifest: The `contract_gate.json` bytes.
    /// - Returns: The required fixtures, or `nil` when the manifest is malformed.
    static func requiredFixtures(manifest: Data) -> [Fixture]? {
        guard let document = try? JSONSerialization.jsonObject(with: manifest) as? [String: Any],
            document["schema"] as? String == "openmed.contract_gate.manifest",
            let entries = document["fixtures"] as? [[String: Any]]
        else { return nil }
        var fixtures: [Fixture] = []
        for entry in entries {
            guard let id = entry["id"] as? String,
                let path = entry["path"] as? String,
                let rawGroups = entry["case_groups"] as? [[String: Any]],
                let cell = (entry["runtimes"] as? [String: Any])?[runtime] as? [String: Any],
                let status = cell["status"] as? String
            else { return nil }
            guard status == "required" else { continue }
            var groups: [CaseGroup] = []
            for raw in rawGroups {
                guard let key = raw["key"] as? String else { return nil }
                groups.append(CaseGroup(key: key, idField: raw["id_field"] as? String))
            }
            guard let consumers = cell["consumers"] as? [String] else { return nil }
            fixtures.append(Fixture(id: id, path: path, caseGroups: groups, consumers: consumers))
        }
        return fixtures
    }

    /// Derives a fixture's case ids with the manifest rules.
    ///
    /// A list group with an id field yields `<key>/<item id>`, a list group
    /// without one yields `<key>/<index>`, and an object group yields `<key>`.
    ///
    /// - Parameters:
    ///   - groups: The fixture's case groups.
    ///   - document: The decoded fixture.
    /// - Returns: Case ids in fixture order, or `nil` when a group is missing,
    ///   an item id is invalid, or an id repeats.
    static func caseIDs(groups: [CaseGroup], document: Any) -> [String]? {
        guard let document = document as? [String: Any] else { return nil }
        var ids: [String] = []
        for group in groups {
            let value = document[group.key]
            if value is [String: Any] {
                ids.append(group.key)
                continue
            }
            guard let items = value as? [Any] else { return nil }
            for (index, item) in items.enumerated() {
                guard let field = group.idField else {
                    ids.append("\(group.key)/\(index)")
                    continue
                }
                guard let raw = (item as? [String: Any])?[field] as? String, isCasePart(raw) else { return nil }
                ids.append("\(group.key)/\(raw)")
            }
        }
        return Set(ids).count == ids.count ? ids : nil
    }

    /// Whether `value` matches `[A-Za-z0-9][A-Za-z0-9._-]{0,95}`.
    static func isCasePart(_ value: String) -> Bool {
        let scalars = Array(value.unicodeScalars)
        guard let first = scalars.first, scalars.count <= 96, isASCIIAlphanumeric(first) else { return false }
        return scalars.dropFirst().allSatisfy { isASCIIAlphanumeric($0) || $0 == "." || $0 == "_" || $0 == "-" }
    }

    private static func isASCIIAlphanumeric(_ scalar: Unicode.Scalar) -> Bool {
        ("a"..."z").contains(scalar) || ("A"..."Z").contains(scalar) || ("0"..."9").contains(scalar)
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func relativePath(_ url: URL) -> String? {
        let root = repositoryRoot.path
        let path = url.standardizedFileURL.path
        guard path.hasPrefix(root + "/") else { return nil }
        return String(path.dropFirst(root.count + 1))
    }

    private static func defaultRepositoryRoot() -> URL {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return url
    }
}

extension ContractGateReporter {
    /// Reads a parity fixture through the shared reporter.
    ///
    /// - Parameters:
    ///   - url: File to read.
    ///   - file: The calling test source file; leave the default.
    /// - Returns: The file's bytes.
    static func read(_ url: URL, file: StaticString = #filePath) throws -> Data {
        try shared.read(url, file: "\(file)")
    }
}

/// Base class for test classes that consume a contract-gate fixture.
///
/// It brackets every test with the shared reporter, so a fixture read through
/// ``ContractGateReporter/read(_:file:)`` is attributed to the running test.
class ContractGateTestCase: XCTestCase {
    /// The reporter this class records into.
    class var contractGateReporter: ContractGateReporter { .shared }

    override func invokeTest() {
        let reporter = Self.contractGateReporter
        reporter.testWillStart()
        super.invokeTest()
        reporter.testDidFinish(
            failureCount: testRun?.totalFailureCount ?? 0,
            skipped: testRun?.hasBeenSkipped ?? false
        )
    }

    /// Runs `body` as the check of one fixture case.
    ///
    /// - Parameters:
    ///   - caseID: `<group key>/<item id>`, `<group key>/<index>` or
    ///     `<group key>`, exactly as the manifest derives it.
    ///   - fixtureID: Manifest fixture id.
    ///   - file: The calling test source file; leave the default.
    ///   - body: The case's assertions, unchanged.
    func contractCase(
        _ caseID: String, of fixtureID: String, file: StaticString = #filePath, _ body: () throws -> Void
    ) rethrows {
        try Self.contractGateReporter.check(
            caseID, of: fixtureID, file: "\(file)", failureCount: testRun?.totalFailureCount ?? 0, body)
    }
}

/// One runtime's contract-gate result, as written to `swift.json`.
struct ContractGateResult: Codable, Equatable {
    var schema = "openmed.contract_gate.result"
    var schemaVersion = 1
    var runtime: String
    var manifestSHA256: String
    var fixtures: [ContractGateFixtureResult]

    enum CodingKeys: String, CodingKey {
        case schema
        case schemaVersion = "schema_version"
        case runtime
        case manifestSHA256 = "manifest_sha256"
        case fixtures
    }
}

/// One fixture's case outcomes.
struct ContractGateFixtureResult: Codable, Equatable {
    var fixtureID: String
    var fixtureSHA256: String
    var passed: [String] = []
    var failed: [ContractGateCaseOutcome] = []
    var skipped: [ContractGateCaseOutcome] = []
    var errors: [String]?
    var tests: ContractGateTestCounts?

    enum CodingKeys: String, CodingKey {
        case fixtureID = "fixture_id"
        case fixtureSHA256 = "fixture_sha256"
        case passed
        case failed
        case skipped
        case errors
        case tests
    }
}

/// A failed or skipped case and its reason code.
struct ContractGateCaseOutcome: Codable, Equatable {
    var caseID: String
    var reason: String

    enum CodingKeys: String, CodingKey {
        case caseID = "case_id"
        case reason
    }
}

/// Test executions that covered a fixture; errors count as failed.
struct ContractGateTestCounts: Codable, Equatable {
    var passed: Int
    var failed: Int
    var skipped: Int
}
