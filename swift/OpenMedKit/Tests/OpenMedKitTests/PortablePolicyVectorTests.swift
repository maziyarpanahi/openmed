import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

/// Runs the shared portable policy vectors in
/// `tests/fixtures/parity/portable_policy_vectors.json` (derived from the
/// Python reference engine) through the Swift policy loader and engine.
///
/// Failure messages carry case identifiers, field names, and reason codes
/// only.
final class PortablePolicyVectorTests: XCTestCase {
    static let vectorsPath = "tests/fixtures/parity/portable_policy_vectors.json"
    static let androidPolicies = "android/openmedkit/src/main/assets/policies"
    static let swiftPolicies = "swift/OpenMedKit/Sources/OpenMedKit/Resources/policies"

    func testBundledResourcesAreByteIdenticalAcrossKits() throws {
        let vectors = try Self.vectors()
        let manifestData = try XCTUnwrap(Policy.bundledResource(Policy.manifestFile))
        XCTAssertEqual(Self.sha256Hex(manifestData), vectors["manifest_sha256"] as? String)
        let manifest = try Policy.decodeManifest(manifestData)
        XCTAssertEqual(manifest.profiles, vectors["profiles"] as? [String: String])
        XCTAssertEqual(manifest.excludedProfiles, vectors["excluded_profiles"] as? [String: String])
        XCTAssertEqual(Policy.bundledProfileNames, manifest.profiles.keys.sorted())
        XCTAssertEqual(Policy.supportedCapabilities, vectors["supported_capabilities"] as? [String])

        let files =
            manifest.profiles.keys.map { "\($0).json" } + [
                Policy.manifestFile, Policy.taxonomyFile,
            ]
        for file in files {
            let bundled = try XCTUnwrap(Policy.bundledResource(file), file)
            XCTAssertEqual(bundled, try Self.repositoryBytes("\(Self.swiftPolicies)/\(file)"), file)
            XCTAssertEqual(bundled, try Self.repositoryBytes("\(Self.androidPolicies)/\(file)"), file)
        }
        for (name, digest) in manifest.profiles {
            XCTAssertEqual(Policy.sha256Digest(try XCTUnwrap(Policy.bundledResource("\(name).json"))), digest)
        }
        for name in manifest.excludedProfiles.keys {
            XCTAssertNil(Policy.bundledResource("\(name).json"), name)
        }
        let consumers = try XCTUnwrap(vectors["consumers"] as? [[String: String]])
        XCTAssertTrue(
            consumers.contains {
                $0["surface"] == "swift" && $0["test"]?.hasSuffix("PortablePolicyVectorTests.swift") == true
            }
        )
    }

    func testTaxonomyIsTheKitParityVocabulary() throws {
        let taxonomy = try XCTUnwrap(Policy.bundledTaxonomy)
        XCTAssertEqual(KitParity.canonicalLabels, taxonomy.labels)
        for label in ["DIET_TYPE", "FEEDING_ROUTE", "NUTRITIONAL_STATUS", "NUTRITION_TARGET"] {
            XCTAssertEqual(Policy.canonicalLabel(for: label), label)
        }
        XCTAssertEqual(Policy.canonicalLabel(for: "patient_name"), "PERSON")
        XCTAssertEqual(Policy.canonicalLabel(for: "B-PATIENT_NAME"), "PERSON")
        XCTAssertEqual(Policy.canonicalLabel(for: "medical_record_number"), "ID_NUM")
        XCTAssertEqual(Policy.canonicalLabel(for: "season_hint"), "OTHER")
    }

    func testDecisionVectorsMatchThePythonReference() throws {
        let vectors = try Self.vectors()
        let cases = try XCTUnwrap(vectors["decision_cases"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(cases.count, 25)
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let outcome = try Self.decide(item, vectors: vectors)
            let serialized = try KitParity.serialize(outcome.result)

            XCTAssertEqual(serialized, expected["result_json"] as? String, "case=\(id) result bytes")
            XCTAssertEqual(
                Self.sha256Hex(Data(serialized.utf8)),
                expected["result_sha256"] as? String,
                "case=\(id)"
            )
            XCTAssertEqual(
                Array(outcome.result.deidentifiedText.utf8),
                Array(((expected["deidentified_text"] as? String) ?? "").utf8),
                "case=\(id) de-identified UTF-8 bytes"
            )
            XCTAssertEqual(outcome.policyDigest, expected["policy_digest"] as? String, "case=\(id)")
            let decisions = try XCTUnwrap(expected["decisions"] as? [[String: Any]])
            XCTAssertEqual(outcome.decisions.count, decisions.count, "case=\(id) decision count")
            for (index, (actual, want)) in zip(outcome.decisions, decisions).enumerated() {
                XCTAssertEqual(Self.record(actual), Self.normalized(want), "case=\(id) decision=\(index)")
            }
        }
    }

    func testErrorVectorsFailWithValueFreeReasons() throws {
        let vectors = try Self.vectors()
        for item in try XCTUnwrap(vectors["error_cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: String])
            XCTAssertThrowsError(try Self.decide(item, vectors: vectors), id) { error in
                let policyError = error as? PolicyError
                XCTAssertEqual(policyError?.code, expected["code"], id)
                XCTAssertEqual(policyError?.fieldPath, expected["field_path"], id)
                if let text = item["text"] as? String, !text.isEmpty {
                    XCTAssertFalse(String(describing: error).contains(text), id)
                }
            }
        }
    }

    func testLoadVectorsPinDigestsAndFailClosed() throws {
        let vectors = try Self.vectors()
        for item in try XCTUnwrap(vectors["load_cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            if let code = expected["code"] as? String {
                XCTAssertThrowsError(try Self.runLoadCase(item), id) { error in
                    XCTAssertEqual((error as? PolicyError)?.code, code, id)
                    XCTAssertEqual((error as? PolicyError)?.fieldPath, expected["field_path"] as? String, id)
                }
                continue
            }
            let policy = try Self.runLoadCase(item)
            XCTAssertEqual(policy.name, expected["name"] as? String, id)
            XCTAssertEqual(policy.digest, expected["digest"] as? String, id)
            XCTAssertEqual(policy.requiredCapabilities, expected["required_capabilities"] as? [String], id)
        }
    }

    func testRuntimeWithoutStrictCapabilityRefusesStrictProfiles() throws {
        let supported = Policy.supportedCapabilities.filter { !$0.hasPrefix("strict") }
        XCTAssertThrowsError(try Policy.load(named: "strict_no_leak", supportedCapabilities: supported)) {
            XCTAssertEqual(($0 as? PolicyError)?.code, "capability_missing")
        }
        XCTAssertNoThrow(try Policy.load(named: "hipaa_safe_harbor", supportedCapabilities: supported))
    }

    func testSurrogatesMatchAnIndependentHMAC() throws {
        let vectors = try Self.vectors()
        let keys = try XCTUnwrap(vectors["surrogate_test_keys"] as? [String: String])
        let key = Self.bytes(hex: try XCTUnwrap(keys["synthetic_test_key_a"]))
        var message = Data("openmed.portable_surrogate.v1".utf8)
        message.append(0)
        message.append(Data("PERSON".utf8))
        message.append(0)
        message.append(Data("Ada Quill".utf8))
        let mac = HMAC<SHA256>.authenticationCode(for: message, using: SymmetricKey(data: key))
        let expected = Data(mac).prefix(8).map { String(format: "%02x", $0) }.joined()

        XCTAssertEqual(
            PolicyEngine.surrogateToken(key: key, canonicalLabel: "PERSON", surface: "Ada Quill"),
            expected
        )
        XCTAssertNotEqual(
            PolicyEngine.surrogateToken(
                key: Self.bytes(hex: try XCTUnwrap(keys["synthetic_test_key_b"])),
                canonicalLabel: "PERSON",
                surface: "Ada Quill"
            ),
            expected
        )
    }

    // MARK: - Helpers

    private static func decide(_ item: [String: Any], vectors: [String: Any]) throws -> PolicyDecisionResult {
        let profile = item["profile"] as? String ?? ""
        let text = item["text"] as? String ?? ""
        let detections = (item["detections"] as? [[String: Any]] ?? []).map { raw in
            PolicyDetection(
                label: raw["label"] as? String ?? "",
                start: raw["start"] as? Int ?? 0,
                end: raw["end"] as? Int ?? 0,
                confidencePPM: raw["confidence_ppm"] as? Int ?? 0,
                source: raw["source"] as? String ?? "model",
                validated: raw["validated"] as? Bool ?? true
            )
        }
        var key: Data?
        if let name = item["surrogate_key"] as? String,
            let hex = (vectors["surrogate_test_keys"] as? [String: String])?[name]
        {
            key = bytes(hex: hex)
        }
        return try PolicyEngine.decide(
            policy: try Policy(named: profile),
            text: text,
            detections: detections,
            overrides: item["overrides"] as? [String: String] ?? [:],
            surrogateKey: key
        )
    }

    private static func runLoadCase(_ item: [String: Any]) throws -> Policy {
        let profile = item["profile"] as? String ?? ""
        if let patch = item["profile_patch"] as? [String: Any] {
            let policy = try Policy(named: profile)
            let data = try XCTUnwrap(Policy.bundledResource("\(policy.name).json"))
            var document = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            for (field, value) in patch {
                document[field] = value
            }
            return try Policy.decodeProfile(
                try JSONSerialization.data(withJSONObject: document),
                name: policy.name,
                digest: policy.digest,
                taxonomy: policy.taxonomy
            )
        }
        var resources: Policy.ResourceProvider = Policy.bundledResource
        if let patch = item["manifest_patch"] as? [String: Any] {
            let data = try XCTUnwrap(Policy.bundledResource(Policy.manifestFile))
            var manifest = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            for (field, value) in patch {
                if let nested = value as? [String: Any], var current = manifest[field] as? [String: Any] {
                    current.merge(nested) { _, new in new }
                    manifest[field] = current
                } else {
                    manifest[field] = value
                }
            }
            let patched = try JSONSerialization.data(withJSONObject: manifest)
            resources = { $0 == Policy.manifestFile ? patched : Policy.bundledResource($0) }
        }
        return try Policy.load(
            named: profile,
            expectedDigest: item["expected_digest"] as? String,
            resources: resources
        )
    }

    private static func record(_ decision: PolicyDecision) -> [String: String] {
        var fields: [String: String] = [
            "action": decision.action,
            "canonical_label": decision.canonicalLabel,
            "confidence_ppm": String(decision.confidencePPM),
            "end": String(decision.end),
            "escalate": String(decision.escalate),
            "inputs": decision.inputs.map(String.init).joined(separator: ","),
            "keep_floor_ppm": String(decision.keepFloorPPM),
            "kind": decision.kind,
            "label": decision.label,
            "outcome": decision.outcome,
            "recognized": String(decision.recognized),
            "start": String(decision.start),
            "strict_upgrade": String(decision.strictUpgrade),
        ]
        if let span = decision.span {
            fields["span"] = String(span)
        }
        return fields
    }

    private static func normalized(_ raw: [String: Any]) -> [String: String] {
        var fields: [String: String] = [:]
        for (key, value) in raw {
            switch value {
            case let number as NSNumber where CFGetTypeID(number) == CFBooleanGetTypeID():
                fields[key] = String(number.boolValue)
            case let items as [Int]:
                fields[key] = items.map(String.init).joined(separator: ",")
            case let number as Int:
                fields[key] = String(number)
            default:
                fields[key] = String(describing: value)
            }
        }
        return fields
    }

    private static func vectors() throws -> [String: Any] {
        let data = try repositoryBytes(vectorsPath)
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func repositoryBytes(_ path: String) throws -> Data {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return try Data(contentsOf: url.appending(path: path))
    }

    private static func bytes(hex: String) -> Data {
        var data = Data()
        var index = hex.startIndex
        while index < hex.endIndex {
            let next = hex.index(index, offsetBy: 2)
            data.append(UInt8(hex[index..<next], radix: 16) ?? 0)
            index = next
        }
        return data
    }

    private static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
