import Foundation
import OpenMedKit

// Runs offline checks against the installed OpenMedKit package and prints a
// value-free JSON report: check ids, pass/fail and failure codes only.

struct ConsumerReport: Encodable {
    let lane: String
    let checks: [CheckResult]
    let networkRequests: Int

    enum CodingKeys: String, CodingKey {
        case lane
        case checks
        case networkRequests = "network_requests"
    }
}

DenyNetworkURLProtocol.install()
let runner = CheckRunner()

do {
    let contract = try ConsumerContract.load()
    runner.run("contract.synthetic_v1") {
        try expect(contract.schemaVersion == 1, "schema_version")
        try expect(contract.synthetic, "not_synthetic")
    }
    InstalledPackageChecks.policyProfiles(contract, runner: runner)
    InstalledPackageChecks.offsets(contract, runner: runner)
    InstalledPackageChecks.redaction(contract, runner: runner)
    await InstalledPackageChecks.modelStore(contract, runner: runner)
    runner.run("migration.typed_incompatibilities") {
        try expect(
            MigrationExamples.incompatibilityCode(
                for: PolicyError.unknownProfile(contract.unknownProfile, allowed: [])
            ) == "unknown_policy_profile",
            "unknown_profile_code"
        )
        try expect(
            MigrationExamples.incompatibilityCode(
                for: OpenMedModelStoreError.invalidManifestPath("outside_snapshot")
            ) == "unsafe_manifest_path",
            "unsafe_manifest_code"
        )
        let emptyCache = FileManager.default.temporaryDirectory.appending(
            path: "openmed-installed-consumer-empty-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        defer { try? FileManager.default.removeItem(at: emptyCache) }
        try expect(
            try MigrationExamples.stagedSnapshot(
                repoID: contract.modelFixture.repoID,
                cacheDirectory: emptyCache
            ) == nil,
            "missing_snapshot_resolved"
        )
    }
} catch {
    runner.run("contract.load") { throw CheckFailure(code: "contract_unreadable") }
}

let packageRequests = DenyNetworkURLProtocol.requestCount
runner.run("network.no_requests") {
    try expect(packageRequests == 0, "network_requested")
}
// Negative control: the guard must intercept a shared-session request, or the
// zero-request check above would prove nothing.
await runner.run("network.guard_intercepts") {
    guard let probe = URL(string: "https://openmed-consumer.invalid/probe") else {
        throw CheckFailure(code: "probe_url")
    }
    do {
        _ = try await URLSession.shared.data(from: probe)
        throw CheckFailure(code: "probe_not_blocked")
    } catch let error as URLError where error.code == .notConnectedToInternet {
        try expect(DenyNetworkURLProtocol.requestCount > packageRequests, "probe_not_counted")
    }
}

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let report = ConsumerReport(
    lane: "swiftpm",
    checks: runner.results,
    networkRequests: packageRequests
)
if let data = try? encoder.encode(report) {
    print(String(decoding: data, as: UTF8.self))
}
exit(runner.allPassed ? 0 : 1)
