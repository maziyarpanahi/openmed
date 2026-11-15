import Foundation
import XCTest

@testable import OpenMedKit

final class ExecutionPlanParityTests: XCTestCase {
    private func vectors() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/device_capabilities/vectors.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func json(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed])
    }

    private func canonical(_ value: Any) throws -> Data {
        try PortableBundle.canonicalJSON(try json(value))
    }

    private func entry(_ vectors: [String: Any], _ group: String, _ name: String) throws -> [String: Any] {
        let items = try XCTUnwrap(vectors[group] as? [String: [String: Any]])
        return try XCTUnwrap(items[name], name)
    }

    private func profile(_ vectors: [String: Any], _ name: String) throws -> DeviceCapabilityProfile {
        try DeviceCapabilityProfile(json: try json(try XCTUnwrap(try entry(vectors, "profiles", name)["document"])))
    }

    private func constraints(_ vectors: [String: Any], _ name: String) throws -> ExecutionConstraints {
        try ExecutionConstraints(json: try json(try XCTUnwrap(try entry(vectors, "constraints", name)["document"])))
    }

    private func verification(_ raw: [String: Any]) throws -> PortableBundleVerification {
        PortableBundleVerification(
            ok: try XCTUnwrap(raw["ok"] as? Bool),
            reason: (raw["reason"] as? String).flatMap(PortableBundleReason.init(rawValue:)),
            contentID: raw["content_id"] as? String,
            validSignatures: try XCTUnwrap(raw["valid_signatures"] as? Int),
            eligibleRuntimes: try XCTUnwrap(raw["eligible_runtimes"] as? [Int]),
            filesChecked: try XCTUnwrap(raw["files_checked"] as? Int),
            filesVerified: try XCTUnwrap(raw["files_verified"] as? Bool))
    }

    private struct Inputs {
        let manifest: Data
        let verification: PortableBundleVerification
        let profile: DeviceCapabilityProfile
        let constraints: ExecutionConstraints
    }

    private func inputs(_ vectors: [String: Any], _ item: [String: Any]) throws -> Inputs {
        let manifest = try entry(vectors, "manifests", try XCTUnwrap(item["manifest"] as? String))
        return Inputs(
            manifest: Data(try XCTUnwrap(manifest["text"] as? String).utf8),
            verification: try verification(try XCTUnwrap(item["verification"] as? [String: Any])),
            profile: try profile(vectors, try XCTUnwrap(item["profile"] as? String)),
            constraints: try constraints(vectors, try XCTUnwrap(item["constraints"] as? String)))
    }

    private func resolve(_ inputs: Inputs) -> ExecutionPlan {
        ExecutionPlanResolver.resolve(
            manifest: inputs.manifest, verification: inputs.verification, profile: inputs.profile,
            constraints: inputs.constraints)
    }

    private func cases(_ vectors: [String: Any]) throws -> [[String: Any]] {
        try XCTUnwrap(vectors["cases"] as? [[String: Any]])
    }

    private func assertNoSentinels(_ data: Data, _ vectors: [String: Any], _ label: String) throws {
        let text = String(decoding: data, as: UTF8.self)
        for sentinel in try XCTUnwrap(vectors["identity_sentinels"] as? [String]) {
            XCTAssertFalse(text.contains(sentinel), label)
        }
    }

    func testAppleAdapterNormalizesSharedProbes() throws {
        let vectors = try vectors()
        let items = try XCTUnwrap(vectors["normalization"] as? [[String: Any]]).filter { $0["kit"] as? String == "apple" }
        XCTAssertGreaterThanOrEqual(items.count, 5)
        for item in items {
            let id = try XCTUnwrap(item["id"] as? String)
            let probe = try XCTUnwrap(item["probe"] as? [String: Any])
            let platform = try XCTUnwrap(OpenMedApplePlatform(rawValue: try XCTUnwrap(probe["platform"] as? String)))
            let configuration = PlatformModelConfiguration.configuration(for: platform)
            let raw = try XCTUnwrap(probe["configuration"] as? [String: Any])
            XCTAssertEqual(raw["maximum_tier"] as? String, configuration.maximumTier.rawValue, id)
            XCTAssertEqual(raw["maximum_resident_memory_mb"] as? Int, configuration.maximumResidentMemoryMB, id)
            XCTAssertEqual(raw["maximum_sequence_length"] as? Int, configuration.maximumSequenceLength, id)
            XCTAssertEqual(raw["requires_int8"] as? Bool, configuration.requiresINT8, id)
            let profile = try DeviceCapabilityProfile.apple(
                configuration: configuration, osVersion: probe["os_version"] as? String,
                mlxAvailable: try XCTUnwrap(probe["mlx_available"] as? Bool),
                architecture: try XCTUnwrap(probe["architecture"] as? String),
                sdkVersion: try XCTUnwrap(probe["sdk_version"] as? String),
                storageFreeBytes: (probe["storage_free_bytes"] as? NSNumber)?.int64Value,
                operatorSets: ((probe["operator_sets"] as? [String: Int]) ?? [:]).mapValues(Int64.init),
                network: (probe["network"] as? String) ?? "unknown")
            XCTAssertEqual(profile.canonicalJSON(), try canonical(try XCTUnwrap(item["expected"])), id)
            try assertNoSentinels(profile.canonicalJSON(), vectors, id)
        }
    }

    func testCurrentAppleProfileIsCoarse() throws {
        let profile = try DeviceCapabilityProfile.currentApple(sdkVersion: "4.0.0")
        XCTAssertEqual(profile.platform, OpenMedApplePlatform.current.rawValue.lowercased())
        XCTAssertLessThanOrEqual(profile.osVersion?.split(separator: ".").count ?? 0, 2)
        XCTAssertTrue(profile.providers.contains("coreml"))
        let reparsed = try DeviceCapabilityProfile(json: profile.canonicalJSON())
        XCTAssertEqual(reparsed, profile)
    }

    func testProfileDocumentsRoundTrip() throws {
        let vectors = try vectors()
        for (name, raw) in try XCTUnwrap(vectors["profiles"] as? [String: [String: Any]]) {
            let profile = try profile(vectors, name)
            XCTAssertEqual(profile.canonicalJSON(), try canonical(try XCTUnwrap(raw["document"])), name)
            XCTAssertEqual(profile.digest, raw["digest"] as? String, name)
            try assertNoSentinels(profile.canonicalJSON(), vectors, name)
        }
    }

    func testConstraintDocumentsRoundTrip() throws {
        let vectors = try vectors()
        for (name, raw) in try XCTUnwrap(vectors["constraints"] as? [String: [String: Any]]) {
            let constraints = try constraints(vectors, name)
            XCTAssertEqual(constraints.canonicalJSON(), try canonical(try XCTUnwrap(raw["document"])), name)
            XCTAssertEqual(constraints.digest, raw["digest"] as? String, name)
        }
        XCTAssertEqual(ExecutionConstraints.defaults, try constraints(vectors, "default"))
    }

    func testResolverReproducesEveryGoldenPlan() throws {
        let vectors = try vectors()
        for item in try cases(vectors) {
            let id = try XCTUnwrap(item["id"] as? String)
            let plan = resolve(try inputs(vectors, item))
            XCTAssertEqual(
                String(decoding: plan.canonicalJSON(), as: UTF8.self),
                String(decoding: try canonical(try XCTUnwrap(item["expected"])), as: UTF8.self), id)
            try assertNoSentinels(plan.canonicalJSON(), vectors, id)
        }
    }

    func testInvalidationVectors() throws {
        let vectors = try vectors()
        let byID = Dictionary(uniqueKeysWithValues: try cases(vectors).map { ($0["id"] as? String ?? "", $0) })
        for item in try XCTUnwrap(vectors["invalidations"] as? [[String: Any]]) {
            let planCase = try XCTUnwrap(byID[try XCTUnwrap(item["plan"] as? String)])
            let current = try inputs(vectors, try XCTUnwrap(byID[try XCTUnwrap(item["inputs"] as? String)]))
            let plan = resolve(try inputs(vectors, planCase))
            let changed = ExecutionPlanResolver.invalidation(
                of: plan, manifest: current.manifest, verification: current.verification, profile: current.profile,
                constraints: current.constraints)
            XCTAssertEqual(changed, item["expected"] as? [String])
        }
    }

    func testVocabularyMatchesVectors() throws {
        let vocabulary = try XCTUnwrap(try vectors()["vocabulary"] as? [String: Any])
        XCTAssertEqual(vocabulary["reason_codes"] as? [String], ExecutionPlanReason.allCases.map(\.rawValue))
        XCTAssertEqual(
            vocabulary["provider_accelerators"] as? [String: String], DeviceCapabilityVocabulary.providerAccelerators)
        XCTAssertEqual(
            vocabulary["platform_providers"] as? [String: [String]],
            DeviceCapabilityVocabulary.platformProviders.mapValues { $0.sorted() })
        XCTAssertEqual(
            vocabulary["format_providers"] as? [String: [String]],
            ExecutionPlanResolver.formatProviders.mapValues { $0.sorted() })
        XCTAssertEqual(vocabulary["default_providers"] as? [String: [String]], ExecutionPlanResolver.defaultProviders)
        XCTAssertEqual(vocabulary["features"] as? [String], DeviceCapabilityVocabulary.features.sorted())
        XCTAssertEqual(vocabulary["cpu_architectures"] as? [String], DeviceCapabilityVocabulary.cpuArchitectures.sorted())
        XCTAssertEqual(vocabulary["model_tiers"] as? [String], DeviceCapabilityVocabulary.modelTiers)
        XCTAssertEqual(vocabulary["network_states"] as? [String], DeviceCapabilityVocabulary.networkStates.sorted())
        XCTAssertEqual(vocabulary["step_roles"] as? [String], ExecutionPlanResolver.stepRoles)
        XCTAssertEqual(vocabulary["assumptions"] as? [String], ExecutionPlanResolver.assumptions)
        XCTAssertEqual(vocabulary["invalidation_components"] as? [String], ExecutionPlanResolver.invalidationComponents)
        XCTAssertEqual(vocabulary["execution_extension"] as? String, ExecutionPlanResolver.extensionName)
        XCTAssertEqual((vocabulary["resolver_version"] as? NSNumber)?.int64Value, ExecutionPlanResolver.resolverVersion)
        let ladder = try XCTUnwrap(vocabulary["budget_ladder"] as? [String: NSNumber])
        XCTAssertEqual(ladder["min_bytes"]?.int64Value, DeviceCapabilityVocabulary.minBudgetBytes)
        XCTAssertEqual(ladder["max_bytes"]?.int64Value, DeviceCapabilityVocabulary.maxBudgetBytes)
        let sdkNames = try XCTUnwrap(vocabulary["sdk_names"] as? [String: String])
        XCTAssertEqual(sdkNames["apple"], DeviceCapabilityVocabulary.appleSDKName)
    }

    func testClosedDocumentsRejectIdentityBearingMembers() throws {
        let vectors = try vectors()
        let base = try XCTUnwrap(try entry(vectors, "profiles", "apple_ios")["document"] as? [String: Any])
        let sentinels = try XCTUnwrap(vectors["identity_sentinels"] as? [String])
        let mutations: [[String: Any]] = [
            ["device_id": sentinels[1]], ["sdk_name": sentinels[0]], ["providers": ["webgpu"]],
            ["memory_budget_bytes": 367_001_600], ["os_version": "17.4.1"], ["accelerators": ["cpu"]],
            ["network": sentinels[3]],
        ]
        for mutation in mutations {
            let document = base.merging(mutation) { _, new in new }
            XCTAssertThrowsError(try DeviceCapabilityProfile(json: try json(document))) { error in
                XCTAssertEqual(error as? DeviceCapabilityError, .profileInvalid)
            }
        }
        let defaults = try XCTUnwrap(try entry(vectors, "constraints", "default")["document"] as? [String: Any])
        for mutation: [String: Any] in [["remote_endpoint": "x"], ["max_steps": 0], ["disallowed_providers": ["cloud"]]] {
            XCTAssertThrowsError(try ExecutionConstraints(json: try json(defaults.merging(mutation) { _, new in new })))
        }
        XCTAssertThrowsError(try ExecutionConstraints(localOnly: true, maxSteps: 17))
    }

    func testBudgetLadderAndOSVersionsStayCoarse() throws {
        XCTAssertEqual(try DeviceCapabilityVocabulary.budgetBucket(nil), nil)
        XCTAssertEqual(try DeviceCapabilityVocabulary.budgetBucket(150 << 20), 128 << 20)
        XCTAssertEqual(try DeviceCapabilityVocabulary.budgetBucket(900 << 20), 512 << 20)
        XCTAssertEqual(try DeviceCapabilityVocabulary.budgetBucket(1 << 45), 1 << 40)
        XCTAssertEqual(try DeviceCapabilityVocabulary.budgetBucket(1), 0)
        XCTAssertThrowsError(try DeviceCapabilityVocabulary.budgetBucket(-1))
        XCTAssertEqual(try DeviceCapabilityVocabulary.coarseOSVersion("17.4.1"), "17.4")
        XCTAssertThrowsError(try DeviceCapabilityVocabulary.coarseOSVersion("17.4-beta"))
    }
}
