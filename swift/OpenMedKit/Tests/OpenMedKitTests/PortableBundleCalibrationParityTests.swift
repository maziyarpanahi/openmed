import Foundation
import XCTest

@testable import OpenMedKit

final class PortableBundleCalibrationParityTests: XCTestCase {
    private func vectors() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/portable_bundle_calibration/vectors.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func manifest(_ vectors: [String: Any], _ key: String) throws -> Data {
        let manifests = try XCTUnwrap(vectors["manifests"] as? [String: [String: Any]])
        let entry = try XCTUnwrap(manifests[key], key)
        if let text = entry["text"] as? String { return Data(text.utf8) }
        let base = Array(try manifest(vectors, try XCTUnwrap(entry["base"] as? String)))
        let at = try XCTUnwrap(entry["at"] as? Int)
        let delete = try XCTUnwrap(entry["delete"] as? Int)
        let insert = Array(try XCTUnwrap(entry["insert"] as? String).utf8)
        return Data(base[..<at] + insert + base[(at + delete)...])
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

    private func context(_ vectors: [String: Any], understood: [String]) throws -> PortableBundleContext {
        let value = try XCTUnwrap(vectors["target"] as? [String: Any])
        let target = try PortableBundleTarget(
            platform: try XCTUnwrap(value["platform"] as? String),
            sdkName: try XCTUnwrap(value["sdk_name"] as? String),
            sdkVersion: try XCTUnwrap(value["sdk_version"] as? String),
            runtimes: try XCTUnwrap(value["runtimes"] as? [String]),
            osVersion: value["os_version"] as? String,
            policyVersion: (value["policy_version"] as? NSNumber)?.int64Value,
            memoryBytes: (value["memory_bytes"] as? NSNumber)?.int64Value,
            storageBytes: (value["storage_bytes"] as? NSNumber)?.int64Value)
        return try PortableBundleContext(
            now: try XCTUnwrap(vectors["now"] as? String), target: target, understoodExtensions: Set(understood))
    }

    private func tree(_ vectors: [String: Any], _ item: [String: Any]) throws -> [String: Data] {
        var files: [String: Data] = [:]
        for entry in try XCTUnwrap(vectors["tree"] as? [[String: String]]) {
            files[try XCTUnwrap(entry["path"])] = Data(try XCTUnwrap(entry["text"]).utf8)
        }
        for entry in item["tree_set"] as? [[String: String]] ?? [] {
            files[try XCTUnwrap(entry["path"])] = Data(try XCTUnwrap(entry["text"]).utf8)
        }
        return files
    }

    private func request(_ raw: [String: Any]) throws -> DecisionCalibrationRequest {
        try DecisionCalibrationRequest(
            calibrationID: try XCTUnwrap(raw["calibration_id"] as? String),
            runtimeID: try XCTUnwrap(raw["runtime_id"] as? String),
            decisionSchemaVersion: try XCTUnwrap(raw["decision_schema_version"] as? String),
            modelID: raw["model_id"] as? String,
            modelRevision: raw["model_revision"] as? String,
            runtimeDigest: raw["runtime_digest"] as? String,
            scopeKind: raw["scope_kind"] as? String,
            scopeDigest: raw["scope_digest"] as? String,
            profileDigest: raw["profile_digest"] as? String)
    }

    func testVectorsShareTheReasonVocabulary() throws {
        let corpus = try vectors()
        XCTAssertEqual(corpus["synthetic_only"] as? Bool, true)
        XCTAssertEqual(corpus["test_only_keys"] as? Bool, true)
        XCTAssertEqual(corpus["extension"] as? String, PortableBundleCalibration.extensionName)
        XCTAssertEqual((corpus["extension_version"] as? NSNumber)?.int64Value, PortableBundleCalibration.extensionVersion)
        XCTAssertEqual(corpus["reason_codes"] as? [String], DecisionCalibrationReason.allCases.map(\.rawValue))
    }

    func testVerificationCasesMatchThePythonReference() throws {
        let corpus = try vectors()
        let policy = try PortableBundleTrustPolicy(
            json: try JSONSerialization.data(withJSONObject: try XCTUnwrap(corpus["trust_policy"])))
        let cases = try XCTUnwrap(corpus["verification_cases"] as? [[String: Any]])
        XCTAssertGreaterThan(cases.count, 10)
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let result = PortableBundle.verify(
                manifest: try manifest(corpus, try XCTUnwrap(item["manifest"] as? String)),
                envelope: Data(try XCTUnwrap(item["envelope"] as? String).utf8),
                trustPolicy: policy,
                context: try context(corpus, understood: try XCTUnwrap(item["understood_extensions"] as? [String])),
                files: try tree(corpus, item))
            XCTAssertEqual(result.ok, expected["ok"] as? Bool, id)
            XCTAssertEqual(result.reason?.rawValue, expected["reason"] as? String, id)
            XCTAssertEqual(result.contentID, expected["content_id"] as? String, id)
            XCTAssertEqual(result.validSignatures, expected["valid_signatures"] as? Int, id)
            XCTAssertEqual(result.eligibleRuntimes, expected["eligible_runtimes"] as? [Int], id)
            XCTAssertEqual(result.filesChecked, expected["files_checked"] as? Int, id)
            XCTAssertEqual(result.filesVerified, expected["files_verified"] as? Bool, id)
        }
    }

    func testReadCasesMatchThePythonReference() throws {
        let corpus = try vectors()
        let cases = try XCTUnwrap(corpus["read_cases"] as? [[String: Any]])
        XCTAssertGreaterThan(cases.count, 40)
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let result = PortableBundleCalibration.read(
                manifest: try manifest(corpus, try XCTUnwrap(item["manifest"] as? String)),
                verification: try verification(try XCTUnwrap(item["verification"] as? [String: Any])))
            XCTAssertEqual(result.ok, expected["ok"] as? Bool, id)
            XCTAssertEqual(result.reason?.rawValue, expected["reason"] as? String, id)
            XCTAssertEqual(result.contentID, expected["content_id"] as? String, id)
            XCTAssertEqual(
                result.profiles.map { String(decoding: $0.canonicalJSON(), as: UTF8.self) },
                try XCTUnwrap(expected["profiles"] as? [String]), id)
        }
    }

    func testResolveCasesMatchThePythonReference() throws {
        let corpus = try vectors()
        let cases = try XCTUnwrap(corpus["resolve_cases"] as? [[String: Any]])
        XCTAssertGreaterThan(cases.count, 20)
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let raw = try XCTUnwrap(item["request"] as? [String: Any])
            if let error = expected["error"] as? String {
                XCTAssertThrowsError(try request(raw), id) {
                    XCTAssertEqual(String(describing: $0), error, id)
                }
                continue
            }
            let selection = PortableBundleCalibration.resolve(
                manifest: try manifest(corpus, try XCTUnwrap(item["manifest"] as? String)),
                verification: try verification(try XCTUnwrap(item["verification"] as? [String: Any])),
                request: try request(raw))
            XCTAssertEqual(selection.ok, expected["ok"] as? Bool, id)
            XCTAssertEqual(selection.reason?.rawValue, expected["reason"] as? String, id)
            XCTAssertEqual(selection.contentID, expected["content_id"] as? String, id)
            XCTAssertEqual(selection.profile?.profileDigest, expected["profile_digest"] as? String, id)
        }
    }

    func testGoldenDigestsAndTypedProfiles() throws {
        let corpus = try vectors()
        let golden = try XCTUnwrap(corpus["golden"] as? [String: Any])
        let data = try manifest(corpus, "golden")
        XCTAssertEqual(PortableBundle.contentID(of: data), golden["content_id"] as? String)
        for (runtimeID, digest) in try XCTUnwrap(golden["runtime_digests"] as? [String: String]) {
            XCTAssertEqual(PortableBundleCalibration.runtimeDigest(manifest: data, runtimeID: runtimeID), digest)
        }
        XCTAssertNil(PortableBundleCalibration.runtimeDigest(manifest: data, runtimeID: "tflite-int8"))
        let verified = PortableBundleVerification(
            ok: true, reason: nil, contentID: PortableBundle.contentID(of: data), validSignatures: 1, eligibleRuntimes: [1, 2],
            filesChecked: 10, filesVerified: true)
        let read = PortableBundleCalibration.read(manifest: data, verification: verified)
        XCTAssertEqual(read.profiles.map(\.profileDigest), golden["profile_digests"] as? [String])
        let triage = try XCTUnwrap(read.profiles.first)
        XCTAssertEqual(triage.method, .deterministicMargin)
        XCTAssertEqual(triage.minimumConfidence, 0.7)
        XCTAssertEqual(triage.minimumMargin, 0.15)
        XCTAssertEqual(triage.runtimeIDs, ["coreml-fp16", "onnx-fp32"])
        XCTAssertEqual(read.profiles[1].method, .temperatureScaling(temperaturePPM: 1_250_000))
        XCTAssertEqual(read.profiles[2].method, .plattScaling(slopePPM: -1_800_000, interceptPPM: 250_000))
        guard case .isotonicRegression(let points) = read.profiles[3].method else { return XCTFail("isotonic") }
        XCTAssertEqual(points[1], DecisionCalibrationPoint(inputPPM: 250_000, outputPPM: 100_000))
        XCTAssertEqual(read.profiles[3].scopeKind, "labels")
    }

    func testErrorsCarryOnlyReasonCodes() throws {
        XCTAssertThrowsError(
            try DecisionCalibrationRequest(
                calibrationID: "openmed.synthetic.triage.v1", runtimeID: "onnx-fp32", decisionSchemaVersion: "1.1.0",
                modelRevision: "SENTINEL-REVISION")
        ) {
            XCTAssertEqual(String(describing: $0), "calibration_request_invalid")
        }
    }
}
