import Foundation
import XCTest

/// Drives ``ContractGateReporter`` over a synthetic checkout: a manifest, its
/// fixtures and consumer files in a temporary directory, with test starts,
/// failures and skips simulated instead of run.
final class ContractGateReporterTests: XCTestCase {
    /// Synthetic planted text: it must never reach a result file.
    private static let sentinel = "Quorble Zyxwick MRN-PLANTED-0042"

    private struct SyntheticError: Error {}

    private final class Failures {
        var count = 0
    }

    private var root: URL!
    private var results: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appending(path: "openmed-contract-gate-\(UUID().uuidString)", directoryHint: .isDirectory)
        results = root.appending(path: "results", directoryHint: .isDirectory)
        try write(["version": 1, "cases": [fixtureCase("a1"), fixtureCase("a2"), fixtureCase("a3")]], to: "fx/alpha.json")
        try write(["items": [["value": Self.sentinel], ["value": 2]] as [[String: Any]], "settings": [String: Any]()], to: "fx/beta.json")
        try write(["cases": [fixtureCase("g1")]], to: "fx/gamma.json")
        try write(["cases": [fixtureCase("d1")]], to: "fx/delta.json")
        try FileManager.default.createDirectory(at: url("Tests"), withIntermediateDirectories: true)
        for consumer in ["Tests/AlphaTests.swift", "Tests/BetaTests.swift", "Tests/OtherTests.swift"] {
            try Data().write(to: url(consumer))
        }
        try writeManifest()
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    // MARK: - Recording

    func testDisabledReporterReadsWithoutRecording() throws {
        let reporter = ContractGateReporter(repositoryRoot: root, resultsDirectory: nil)
        var ran = false

        reporter.testWillStart()
        let data = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.check("cases/a1", of: "alpha", file: path("Tests/AlphaTests.swift"), failureCount: 0) {
            ran = true
        }
        reporter.testDidFinish(failureCount: 0, skipped: false)

        XCTAssertEqual(data, try Data(contentsOf: url("fx/alpha.json")))
        XCTAssertTrue(ran)
        XCTAssertFalse(reporter.isEnabled)
        XCTAssertNil(reporter.result())
        XCTAssertFalse(FileManager.default.fileExists(atPath: results.path))
    }

    func testUnscopedTestCoversEveryCaseWithTheDigestItRead() throws {
        let reporter = makeReporter()
        let manifest = try Data(contentsOf: url(ContractGateReporter.manifestPath))

        reporter.testWillStart()
        let read = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.testDidFinish(failureCount: 0, skipped: false)
        // A later change on disk must not replace the digest of the bytes read.
        try write(["version": 1, "cases": [fixtureCase("a1")]], to: "fx/alpha.json")

        let result = try XCTUnwrap(reporter.result())
        XCTAssertEqual(result.schema, "openmed.contract_gate.result")
        XCTAssertEqual(result.schemaVersion, 1)
        XCTAssertEqual(result.runtime, "swift")
        XCTAssertEqual(result.manifestSHA256, ContractGateReporter.sha256(manifest))
        XCTAssertEqual(result.fixtures.map(\.fixtureID), ["alpha", "beta", "delta"])
        let alpha = try fixture("alpha", in: result)
        XCTAssertEqual(alpha.fixtureSHA256, ContractGateReporter.sha256(read))
        XCTAssertEqual(alpha.passed, ["cases/a1", "cases/a2", "cases/a3"])
        XCTAssertEqual(alpha.failed, [])
        XCTAssertNil(alpha.errors)
        XCTAssertEqual(alpha.tests, ContractGateTestCounts(passed: 1, failed: 0, skipped: 0))
        let beta = try fixture("beta", in: result)
        XCTAssertEqual(
            beta.failed.map(\.caseID), ["items/0", "items/1", "settings"])
        XCTAssertEqual(Set(beta.failed.map(\.reason)), ["consumer_not_collected"])
        XCTAssertEqual(beta.fixtureSHA256, ContractGateReporter.sha256(try Data(contentsOf: url("fx/beta.json"))))
        XCTAssertNil(beta.tests)
        let delta = try fixture("delta", in: result)
        XCTAssertEqual(delta.failed, [ContractGateCaseOutcome(caseID: "cases/d1", reason: "consumer_missing")])
    }

    func testScopedCasesCoverOnlyThemselves() throws {
        let reporter = makeReporter()
        let failures = Failures()

        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        scope(reporter, "cases/a1", failures) {}
        scope(reporter, "cases/a2", failures) { failures.count += 1 }
        reporter.testDidFinish(failureCount: failures.count, skipped: false)

        let alpha = try fixture("alpha", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(alpha.passed, ["cases/a1"])
        XCTAssertEqual(
            alpha.failed,
            [
                ContractGateCaseOutcome(caseID: "cases/a2", reason: "consumer_failed"),
                ContractGateCaseOutcome(caseID: "cases/a3", reason: "case_not_exercised"),
            ])
        XCTAssertEqual(alpha.tests, ContractGateTestCounts(passed: 0, failed: 1, skipped: 0))
    }

    func testFailureOutsideEveryScopeFailsAllCases() throws {
        let reporter = makeReporter()
        let failures = Failures()

        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        scope(reporter, "cases/a1", failures) {}
        failures.count += 1
        reporter.testDidFinish(failureCount: failures.count, skipped: false)

        let alpha = try fixture("alpha", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(alpha.passed, [])
        XCTAssertEqual(alpha.failed.map(\.caseID), ["cases/a1", "cases/a2", "cases/a3"])
        XCTAssertEqual(Set(alpha.failed.map(\.reason)), ["consumer_failed"])
    }

    func testThrownErrorsAndSkipsAreRethrownAndRecorded() throws {
        let reporter = makeReporter()
        let failures = Failures()

        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        XCTAssertThrowsError(
            try reporter.check("cases/a1", of: "alpha", file: path("Tests/AlphaTests.swift"), failureCount: 0) {
                throw SyntheticError()
            }
        ) { XCTAssertTrue($0 is SyntheticError) }
        // XCTest records the error once it leaves the test method.
        reporter.testDidFinish(failureCount: 1, skipped: false)

        reporter.testWillStart()
        XCTAssertThrowsError(
            try reporter.check("cases/a2", of: "alpha", file: path("Tests/AlphaTests.swift"), failureCount: 0) {
                throw XCTSkip("synthetic")
            }
        ) { XCTAssertTrue($0 is XCTSkip) }
        reporter.testDidFinish(failureCount: 0, skipped: true)

        reporter.testWillStart()
        scope(reporter, "cases/a3", failures) {}
        reporter.testDidFinish(failureCount: 0, skipped: false)

        let alpha = try fixture("alpha", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(alpha.passed, ["cases/a3"])
        XCTAssertEqual(alpha.failed, [ContractGateCaseOutcome(caseID: "cases/a1", reason: "consumer_error")])
        XCTAssertEqual(alpha.skipped, [ContractGateCaseOutcome(caseID: "cases/a2", reason: "consumer_skipped")])
        XCTAssertEqual(alpha.tests, ContractGateTestCounts(passed: 1, failed: 1, skipped: 1))
    }

    func testOneFailingTestFailsACaseThatAnotherTestPassed() throws {
        let reporter = makeReporter()
        let failures = Failures()

        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.testDidFinish(failureCount: 0, skipped: false)
        reporter.testWillStart()
        scope(reporter, "cases/a2", failures) { failures.count += 1 }
        reporter.testDidFinish(failureCount: failures.count, skipped: false)

        let alpha = try fixture("alpha", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(alpha.passed, ["cases/a1", "cases/a3"])
        XCTAssertEqual(alpha.failed, [ContractGateCaseOutcome(caseID: "cases/a2", reason: "consumer_failed")])
    }

    func testObjectAndIndexGroupsAndTheRealFailureCount() throws {
        let reporter = makeReporter()
        let file = path("Tests/BetaTests.swift")

        reporter.testWillStart()
        _ = try reporter.read(url("fx/beta.json"), file: file)
        for (index, _) in ["first", "second"].enumerated() {
            reporter.check("items/\(index)", of: "beta", file: file, failureCount: testRun?.totalFailureCount ?? 0) {
                XCTAssertTrue(true)
            }
        }
        reporter.check("settings", of: "beta", file: file, failureCount: testRun?.totalFailureCount ?? 0) {}
        reporter.testDidFinish(failureCount: 0, skipped: false)

        let beta = try fixture("beta", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(beta.passed, ["items/0", "items/1", "settings"])
        XCTAssertEqual(beta.failed, [])
    }

    func testReadsOutsideDeclaredConsumersOrTestsAreNotCoverage() throws {
        let reporter = makeReporter()

        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/OtherTests.swift"))
        reporter.check("cases/a1", of: "alpha", file: path("Tests/OtherTests.swift"), failureCount: 0) {}
        reporter.check("cases/g1", of: "gamma", file: path("Tests/AlphaTests.swift"), failureCount: 0) {}
        reporter.testDidFinish(failureCount: 0, skipped: false)
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))

        let result = try XCTUnwrap(reporter.result())
        XCTAssertFalse(FileManager.default.fileExists(atPath: resultFile.path))
        let alpha = try fixture("alpha", in: result)
        XCTAssertEqual(Set(alpha.failed.map(\.reason)), ["consumer_not_collected"])
        XCTAssertFalse(result.fixtures.contains { $0.fixtureID == "gamma" })
    }

    func testFixtureBytesThatChangeDuringTheRunAreReported() throws {
        let reporter = makeReporter()

        reporter.testWillStart()
        let first = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.testDidFinish(failureCount: 0, skipped: false)
        try write(["version": 2, "cases": [fixtureCase("a1"), fixtureCase("a2"), fixtureCase("a3")]], to: "fx/alpha.json")
        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.testDidFinish(failureCount: 0, skipped: false)

        let alpha = try fixture("alpha", in: XCTUnwrap(reporter.result()))
        XCTAssertEqual(alpha.fixtureSHA256, ContractGateReporter.sha256(first))
        XCTAssertEqual(alpha.errors, ["fixture_digest_inconsistent"])
    }

    func testUnreadableFixturesAreFixtureLevelErrors() throws {
        try Data("{\"cases\": [".utf8).write(to: url("fx/alpha.json"))
        try FileManager.default.removeItem(at: url("fx/beta.json"))
        let reporter = makeReporter()

        let result = try XCTUnwrap(reporter.result())
        let alpha = try fixture("alpha", in: result)
        XCTAssertEqual(alpha.errors, ["fixture_unreadable"])
        XCTAssertEqual(alpha.fixtureSHA256, ContractGateReporter.sha256(try Data(contentsOf: url("fx/alpha.json"))))
        XCTAssertEqual(alpha.passed + alpha.failed.map(\.caseID), [])
        let beta = try fixture("beta", in: result)
        XCTAssertEqual(beta.errors, ["fixture_unreadable"])
        XCTAssertEqual(beta.fixtureSHA256, String(repeating: "0", count: 64))
    }

    func testMalformedManifestDisablesTheReporter() throws {
        try Data("{\"schema\": \"openmed.contract_gate.manifest\", \"fixtures\": [{}]}".utf8)
            .write(to: url(ContractGateReporter.manifestPath))

        let reporter = makeReporter()

        XCTAssertFalse(reporter.isEnabled)
        reporter.testWillStart()
        _ = try reporter.read(url("fx/alpha.json"), file: path("Tests/AlphaTests.swift"))
        reporter.testDidFinish(failureCount: 0, skipped: false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: resultFile.path))
    }

    // MARK: - Result file

    func testResultFileHoldsCaseIDsAndReasonCodesOnly() throws {
        let reporter = makeReporter()
        let failures = Failures()

        reporter.testWillStart()
        let read = try reporter.read(url("fx/beta.json"), file: path("Tests/BetaTests.swift"))
        XCTAssertTrue(String(decoding: read, as: UTF8.self).contains(Self.sentinel))
        reporter.check(
            "items/\(Self.sentinel)", of: "beta", file: path("Tests/BetaTests.swift"), failureCount: failures.count
        ) { failures.count += 1 }
        reporter.testDidFinish(failureCount: failures.count, skipped: false)

        let data = try Data(contentsOf: resultFile)
        let text = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(text.hasSuffix("}\n"))
        XCTAssertFalse(text.contains(Self.sentinel))
        XCTAssertFalse(text.contains("Quorble"))
        XCTAssertFalse(text.contains(root.path))
        XCTAssertFalse(text.contains("fx/"))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["schema", "schema_version", "runtime", "manifest_sha256", "fixtures"])
        let decoded = try JSONDecoder().decode(ContractGateResult.self, from: data)
        XCTAssertEqual(decoded, reporter.result())
        let beta = try fixture("beta", in: decoded)
        XCTAssertEqual(beta.errors, ["case_id_unknown"])
        // The unknown case is dropped; the test's failure still fails every case.
        XCTAssertEqual(beta.failed.map(\.caseID), ["items/0", "items/1", "settings"])
        XCTAssertEqual(Set(beta.failed.map(\.reason)), ["consumer_failed"])
        let codes = decoded.fixtures.flatMap { $0.failed.map(\.reason) + $0.skipped.map(\.reason) + ($0.errors ?? []) }
        XCTAssertTrue(codes.allSatisfy { $0.range(of: "^[a-z][a-z0-9_]{0,63}$", options: .regularExpression) != nil })
        let caseIDs = decoded.fixtures.flatMap { $0.passed + $0.failed.map(\.caseID) + $0.skipped.map(\.caseID) }
        XCTAssertTrue(
            caseIDs.allSatisfy {
                $0.range(of: "^[a-z][a-z0-9_]{0,63}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?$", options: .regularExpression)
                    != nil
            })
    }

    // MARK: - Case ids

    func testCaseIDsFollowTheManifestRules() {
        typealias Group = ContractGateReporter.CaseGroup
        let groups = [Group(key: "cases", idField: "id"), Group(key: "tokens", idField: nil), Group(key: "segment", idField: nil)]
        let document: [String: Any] = [
            "cases": [["id": "cjk-01"], ["id": "Deva_2.v1"]],
            "tokens": [["field": "start"], "free", 3] as [Any],
            "segment": ["text": Self.sentinel],
        ]

        XCTAssertEqual(
            ContractGateReporter.caseIDs(groups: groups, document: document),
            ["cases/cjk-01", "cases/Deva_2.v1", "tokens/0", "tokens/1", "tokens/2", "segment"])
        for invalid in ["", "-leading", "has space", "slash/inside", "é", String(repeating: "a", count: 97)] {
            XCTAssertNil(
                ContractGateReporter.caseIDs(groups: [groups[0]], document: ["cases": [["id": invalid]]]),
                "invalid id length=\(invalid.count)")
        }
        XCTAssertNotNil(
            ContractGateReporter.caseIDs(groups: [groups[0]], document: ["cases": [["id": String(repeating: "a", count: 96)]]]))
        XCTAssertNil(ContractGateReporter.caseIDs(groups: [groups[0]], document: ["cases": [["id": 7]]]))
        XCTAssertNil(ContractGateReporter.caseIDs(groups: [groups[0]], document: ["cases": [["id": "x"], ["id": "x"]]]))
        XCTAssertNil(ContractGateReporter.caseIDs(groups: [groups[0]], document: ["other": [Any]()]))
        XCTAssertNil(ContractGateReporter.caseIDs(groups: [groups[0]], document: ["cases": "text"]))
        XCTAssertNil(ContractGateReporter.caseIDs(groups: [groups[0]], document: [1, 2]))
    }

    func testCommittedManifestNamesSwiftConsumersWithDerivableCases() throws {
        var checkout = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { checkout.deleteLastPathComponent() }
        let manifest = try Data(contentsOf: checkout.appending(path: ContractGateReporter.manifestPath))

        let fixtures = try XCTUnwrap(ContractGateReporter.requiredFixtures(manifest: manifest))

        XCTAssertFalse(fixtures.isEmpty)
        XCTAssertTrue(fixtures.contains { $0.id == "offset_contract" })
        for fixture in fixtures {
            XCTAssertFalse(fixture.consumers.isEmpty, fixture.id)
            for consumer in fixture.consumers {
                XCTAssertTrue(consumer.hasPrefix("swift/OpenMedKit/Tests/"), fixture.id)
                XCTAssertTrue(FileManager.default.fileExists(atPath: checkout.appending(path: consumer).path), fixture.id)
            }
            let data = try Data(contentsOf: checkout.appending(path: fixture.path))
            let ids = try XCTUnwrap(
                ContractGateReporter.caseIDs(groups: fixture.caseGroups, document: JSONSerialization.jsonObject(with: data)),
                fixture.id)
            XCTAssertFalse(ids.isEmpty, fixture.id)
        }
    }

    // MARK: - Helpers

    private var resultFile: URL { results.appending(path: ContractGateReporter.resultFileName) }

    private func makeReporter() -> ContractGateReporter {
        ContractGateReporter(repositoryRoot: root, resultsDirectory: results)
    }

    private func scope(
        _ reporter: ContractGateReporter, _ caseID: String, _ failures: Failures, _ body: () -> Void
    ) {
        reporter.check(caseID, of: "alpha", file: path("Tests/AlphaTests.swift"), failureCount: failures.count, body)
    }

    private func fixture(_ id: String, in result: ContractGateResult) throws -> ContractGateFixtureResult {
        try XCTUnwrap(result.fixtures.first { $0.fixtureID == id }, id)
    }

    private func fixtureCase(_ id: String) -> [String: Any] {
        ["id": id, "text": Self.sentinel]
    }

    private func url(_ relative: String) -> URL {
        root.appending(path: relative)
    }

    private func path(_ relative: String) -> String {
        url(relative).path
    }

    private func write(_ object: [String: Any], to relative: String) throws {
        let target = url(relative)
        try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]).write(to: target)
    }

    private func writeManifest() throws {
        func cell(_ consumers: [String]) -> [String: Any] {
            ["status": "required", "consumers": consumers]
        }
        let notApplicable: [String: Any] = ["status": "not_applicable", "reason": "Synthetic Python-only contract."]
        let fixtures: [[String: Any]] = [
            [
                "id": "alpha", "path": "fx/alpha.json",
                "case_groups": [["key": "cases", "id_field": "id"]],
                "runtimes": ["python": cell(["tests/test_alpha.py"]), "swift": cell(["Tests/AlphaTests.swift"])],
            ],
            [
                "id": "beta", "path": "fx/beta.json",
                "case_groups": [["key": "items"], ["key": "settings"]],
                "runtimes": ["python": cell(["tests/test_beta.py"]), "swift": cell(["Tests/BetaTests.swift"])],
            ],
            [
                "id": "gamma", "path": "fx/gamma.json",
                "case_groups": [["key": "cases", "id_field": "id"]],
                "runtimes": ["python": cell(["tests/test_gamma.py"]), "swift": notApplicable],
            ],
            [
                "id": "delta", "path": "fx/delta.json",
                "case_groups": [["key": "cases", "id_field": "id"]],
                "runtimes": ["python": cell(["tests/test_delta.py"]), "swift": cell(["Tests/MissingTests.swift"])],
            ],
        ]
        try write(
            ["schema": "openmed.contract_gate.manifest", "version": 1, "fixtures": fixtures],
            to: ContractGateReporter.manifestPath)
    }
}
