import Foundation
import OpenMedKit

/// A value-free failure code. Codes name fixture cases and checks, never text.
struct CheckFailure: Error {
    let code: String
}

struct CheckResult: Encodable {
    let id: String
    let status: String
    let code: String
}

func expect(_ condition: @autoclosure () throws -> Bool, _ code: String) throws {
    guard try condition() else {
        throw CheckFailure(code: code)
    }
}

final class CheckRunner {
    private(set) var results: [CheckResult] = []

    var allPassed: Bool {
        !results.isEmpty && results.allSatisfy { $0.status == "pass" }
    }

    func run(_ id: String, _ body: () throws -> Void) {
        do {
            try body()
            results.append(CheckResult(id: id, status: "pass", code: "ok"))
        } catch {
            results.append(CheckResult(id: id, status: "fail", code: Self.code(for: error)))
        }
    }

    func run(_ id: String, _ body: () async throws -> Void) async {
        do {
            try await body()
            results.append(CheckResult(id: id, status: "pass", code: "ok"))
        } catch {
            results.append(CheckResult(id: id, status: "fail", code: Self.code(for: error)))
        }
    }

    private static func code(for error: Error) -> String {
        if let failure = error as? CheckFailure {
            return failure.code
        }
        return "unexpected_error:\(String(describing: type(of: error)))"
    }
}

/// Fails and counts every URL request made through `URLSession.shared`.
final class DenyNetworkURLProtocol: URLProtocol {
    private static let lock = NSLock()
    private static var count = 0

    static var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    static func install() {
        URLProtocol.registerClass(DenyNetworkURLProtocol.self)
    }

    override class func canInit(with request: URLRequest) -> Bool {
        lock.lock()
        count += 1
        lock.unlock()
        return true
    }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
    }

    override func stopLoading() {}
}

enum InstalledPackageChecks {
    static func policyProfiles(_ contract: ConsumerContract, runner: CheckRunner) {
        runner.run("policy.bundled_profiles_load") {
            try expect(!Policy.bundledProfileNames.isEmpty, "no_bundled_profiles")
            for name in Policy.bundledProfileNames {
                let policy = try Policy(named: name)
                try expect(policy.name == name, "\(name):name_mismatch")
                try expect(policy.schemaVersion == 1, "\(name):schema_version")
            }
            let defaultPolicy = try Policy(named: Policy.defaultName)
            try expect(defaultPolicy.name == "hipaa_safe_harbor", "default_profile_changed")
        }
        runner.run("policy.aliases_resolve") {
            for (alias, canonical) in contract.profileAliases.sorted(by: { $0.key < $1.key }) {
                try expect(try Policy(named: alias).name == canonical, "\(alias):alias_mismatch")
            }
        }
        runner.run("policy.unknown_profile_typed_error") {
            do {
                _ = try Policy(named: contract.unknownProfile)
            } catch PolicyError.unknownProfile(_, let allowed) {
                try expect(allowed.contains(Policy.defaultName), "allowed_list_incomplete")
                return
            }
            throw CheckFailure(code: "unknown_profile_accepted")
        }
        runner.run("policy.label_actions") {
            for policyCase in contract.policyCases {
                let policy = try Policy(named: policyCase.profile)
                for (label, action) in policyCase.labelActions.sorted(by: { $0.key < $1.key }) {
                    try expect(
                        policy.action(for: label).rawValue == action,
                        "\(policyCase.profile):\(label):action_mismatch"
                    )
                }
            }
        }
    }

    static func offsets(_ contract: ConsumerContract, runner: CheckRunner) {
        runner.run("offsets.unicode_scalar_contract") {
            try expect(contract.offsetUnit == "unicode_scalar", "offset_unit")
            for offsetCase in contract.offsetCases {
                let text = offsetCase.text
                let id = offsetCase.id
                try expect(
                    PostProcessing.unicodeScalarCount(in: text) == offsetCase.scalarLength,
                    "\(id):scalar_length"
                )
                try expect(text.utf16.count == offsetCase.utf16Length, "\(id):utf16_length")
                guard
                    let range = PostProcessing.scalarRange(
                        in: text,
                        start: offsetCase.start,
                        end: offsetCase.end
                    )
                else {
                    throw CheckFailure(code: "\(id):scalar_range")
                }
                let utf16Start = text.utf16.distance(from: text.startIndex, to: range.lowerBound)
                let utf16End = text.utf16.distance(from: text.startIndex, to: range.upperBound)
                try expect(utf16Start == offsetCase.utf16Start, "\(id):utf16_start")
                try expect(utf16End == offsetCase.utf16End, "\(id):utf16_end")
                let entity = EntityPrediction(
                    label: "FIXTURE",
                    text: PostProcessing.scalarSubstring(
                        text,
                        start: offsetCase.start,
                        end: offsetCase.end
                    ),
                    confidence: 1,
                    start: offsetCase.start,
                    end: offsetCase.end
                )
                try expect(entity.range(in: text) == range, "\(id):entity_range")
                try expect(
                    PostProcessing.replacingScalarSpan(
                        in: text,
                        start: offsetCase.start,
                        end: offsetCase.end,
                        with: offsetCase.replacement
                    ) == offsetCase.expectedRedacted,
                    "\(id):redaction"
                )
            }
        }
    }

    static func redaction(_ contract: ConsumerContract, runner: CheckRunner) {
        runner.run("redaction.policy_actions") {
            for redactionCase in contract.redactionCases {
                let policy = try Policy(named: redactionCase.profile)
                var redacted = redactionCase.text
                for entity in redactionCase.entities.sorted(by: { $0.start > $1.start }) {
                    let replacement: String
                    switch policy.action(for: entity.label) {
                    case .keep:
                        continue
                    case .mask, .redact:
                        replacement = "[\(Policy.canonicalLabel(for: entity.label))]"
                    case .replace, .remove, .hash:
                        throw CheckFailure(code: "\(redactionCase.id):unexpected_action")
                    }
                    guard
                        let next = PostProcessing.replacingScalarSpan(
                            in: redacted,
                            start: entity.start,
                            end: entity.end,
                            with: replacement
                        )
                    else {
                        throw CheckFailure(code: "\(redactionCase.id):span")
                    }
                    redacted = next
                }
                try expect(
                    redacted == redactionCase.expectedRedacted,
                    "\(redactionCase.id):redaction_mismatch"
                )
            }
        }
        runner.run("result.python_schema_roundtrip") {
            guard let redactionCase = contract.redactionCases.first else {
                throw CheckFailure(code: "no_redaction_case")
            }
            let entities = redactionCase.entities.map { entity in
                EntityPrediction(
                    label: entity.label,
                    text: PostProcessing.scalarSubstring(
                        redactionCase.text,
                        start: entity.start,
                        end: entity.end
                    ),
                    confidence: entity.confidence,
                    start: entity.start,
                    end: entity.end
                )
            }
            let result = DeidentificationResult(
                originalText: redactionCase.text,
                deidentifiedText: redactionCase.expectedRedacted,
                entities: entities,
                method: DeidentificationMethod.mask.rawValue,
                timestamp: Date(timeIntervalSince1970: 0)
            )
            let data = try result.encode()
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            try expect(object?["num_entities_redacted"] as? Int == entities.count, "entity_count")
            try expect(object?["method"] as? String == "mask", "method")
            try expect(object?["pii_entities"] is [Any], "pii_entities")
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            try expect(
                try decoder.decode(DeidentificationResult.self, from: data) == result,
                "roundtrip"
            )
        }
    }

    static func modelStore(_ contract: ConsumerContract, runner: CheckRunner) async {
        let fixture = contract.modelFixture
        let cacheRoot = FileManager.default.temporaryDirectory.appending(
            path: "openmed-installed-consumer-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        defer { try? FileManager.default.removeItem(at: cacheRoot) }

        await runner.run("model_store.offline_legacy_snapshot") {
            let directory = try OpenMedModelStore.cachedMLXModelDirectory(
                repoID: fixture.repoID,
                revision: fixture.revision,
                cacheDirectory: cacheRoot
            )
            try expect(
                try state(fixture.repoID, fixture.revision, cacheRoot) == .missing,
                "initial_state"
            )
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            for (index, file) in fixture.mlxLegacyFiles.enumerated() {
                try Data(file.content.utf8).write(to: directory.appending(path: file.path))
                let expected: OpenMedMLXModelCacheState =
                    index == fixture.mlxLegacyFiles.count - 1 ? .ready : .partial
                try expect(
                    try state(fixture.repoID, fixture.revision, cacheRoot) == expected,
                    "state_after_\(file.path)"
                )
            }
            try expect(
                try OpenMedModelStore.isMLXModelCached(
                    repoID: fixture.repoID,
                    revision: fixture.revision,
                    cacheDirectory: cacheRoot
                ),
                "cached_flag"
            )
            let requestsBefore = DenyNetworkURLProtocol.requestCount
            let resolved = try await OpenMedModelStore.downloadMLXModel(
                repoID: fixture.repoID,
                revision: fixture.revision,
                cacheDirectory: cacheRoot
            )
            try expect(
                resolved.standardizedFileURL.path == directory.standardizedFileURL.path,
                "cache_hit_directory"
            )
            try expect(
                DenyNetworkURLProtocol.requestCount == requestsBefore,
                "cache_hit_used_network"
            )
        }

        runner.run("model_store.unsafe_manifest_typed_error") {
            let repoID = fixture.repoID + "-unsafe"
            let directory = try OpenMedModelStore.cachedMLXModelDirectory(
                repoID: repoID,
                revision: fixture.revision,
                cacheDirectory: cacheRoot
            )
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            for file in fixture.mlxLegacyFiles {
                try Data(file.content.utf8).write(to: directory.appending(path: file.path))
            }
            let manifest: [String: Any] = [
                "format": "openmed-mlx",
                "format_version": 1,
                "task": "token-classification",
                "family": "bert",
                "source_model_id": fixture.repoID,
                "config_path": fixture.unsafeManifestConfigPath,
                "preferred_weights": "weights.safetensors",
                "fallback_weights": [String](),
                "available_weights": ["weights.safetensors"],
                "weights_format": "safetensors",
                "tokenizer": ["path": ".", "files": [String]()],
            ]
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: directory.appending(path: "openmed-mlx.json"))
            do {
                _ = try state(repoID, fixture.revision, cacheRoot)
            } catch OpenMedModelStoreError.invalidManifestPath {
                return
            }
            throw CheckFailure(code: "unsafe_manifest_accepted")
        }
    }

    private static func state(
        _ repoID: String,
        _ revision: String,
        _ cacheRoot: URL
    ) throws -> OpenMedMLXModelCacheState {
        try OpenMedModelStore.mlxModelCacheState(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheRoot
        )
    }
}
