import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

final class PortableBundleParityTests: XCTestCase {
    private func vectors() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/portable_bundle/vectors.json"))
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

    private func optionalInt(_ value: Any?) -> Int64? {
        (value as? NSNumber).map { $0.int64Value }
    }

    private func context(_ raw: [String: Any]) throws -> PortableBundleContext {
        var target: PortableBundleTarget?
        if let value = raw["target"] as? [String: Any] {
            target = try PortableBundleTarget(
                platform: try XCTUnwrap(value["platform"] as? String),
                sdkName: try XCTUnwrap(value["sdk_name"] as? String),
                sdkVersion: try XCTUnwrap(value["sdk_version"] as? String),
                runtimes: try XCTUnwrap(value["runtimes"] as? [String]),
                osVersion: value["os_version"] as? String,
                policyVersion: optionalInt(value["policy_version"]),
                memoryBytes: optionalInt(value["memory_bytes"]),
                storageBytes: optionalInt(value["storage_bytes"]))
        }
        let floors = try XCTUnwrap(raw["sequence_floors"] as? [String: NSNumber]).mapValues { $0.int64Value }
        return try PortableBundleContext(
            now: try XCTUnwrap(raw["now"] as? String),
            target: target,
            sequenceFloors: floors,
            understoodExtensions: Set(try XCTUnwrap(raw["understood_extensions"] as? [String])),
            maxManifestBytes: try XCTUnwrap(raw["max_manifest_bytes"] as? Int),
            maxEnvelopeBytes: try XCTUnwrap(raw["max_envelope_bytes"] as? Int),
            maxTotalBytes: try XCTUnwrap(optionalInt(raw["max_total_bytes"])),
            maxFiles: try XCTUnwrap(raw["max_files"] as? Int))
    }

    private func policy(_ vectors: [String: Any], _ name: String) throws -> PortableBundleTrustPolicy {
        let policies = try XCTUnwrap(vectors["trust_policies"] as? [String: Any])
        let data = try JSONSerialization.data(withJSONObject: try XCTUnwrap(policies[name]))
        return try PortableBundleTrustPolicy(json: data)
    }

    private func tree(_ vectors: [String: Any], _ item: [String: Any]) throws -> [String: Data]? {
        guard let name = item["tree"] as? String else { return nil }
        let trees = try XCTUnwrap(vectors["trees"] as? [String: [[String: String]]])
        var files: [String: Data] = [:]
        for entry in try XCTUnwrap(trees[name]) {
            files[try XCTUnwrap(entry["path"])] = Data(try XCTUnwrap(entry["text"]).utf8)
        }
        for path in try XCTUnwrap(item["tree_remove"] as? [String]) { files.removeValue(forKey: path) }
        for entry in try XCTUnwrap(item["tree_set"] as? [[String: String]]) {
            files[try XCTUnwrap(entry["path"])] = Data(try XCTUnwrap(entry["text"]).utf8)
        }
        return files
    }

    private func run(_ vectors: [String: Any], _ item: [String: Any]) throws -> PortableBundleVerification {
        let envelopes = try XCTUnwrap(vectors["envelopes"] as? [String: String])
        let contexts = try XCTUnwrap(vectors["contexts"] as? [String: [String: Any]])
        return PortableBundle.verify(
            manifest: try manifest(vectors, try XCTUnwrap(item["manifest"] as? String)),
            envelope: Data(try XCTUnwrap(envelopes[try XCTUnwrap(item["envelope"] as? String)]).utf8),
            trustPolicy: try policy(vectors, try XCTUnwrap(item["trust_policy"] as? String)),
            context: try context(try XCTUnwrap(contexts[try XCTUnwrap(item["context"] as? String)])),
            files: try tree(vectors, item))
    }

    func testVectorsAreSyntheticAndShareTheReasonVocabulary() throws {
        let corpus = try vectors()
        XCTAssertEqual(corpus["synthetic_only"] as? Bool, true)
        XCTAssertEqual(corpus["test_only_keys"] as? Bool, true)
        XCTAssertEqual(corpus["payload_type"] as? String, PortableBundle.payloadType)
        XCTAssertEqual(corpus["reason_codes"] as? [String], PortableBundleReason.allCases.map(\.rawValue))
    }

    func testEveryGoldenCaseMatchesThePythonReference() throws {
        let corpus = try vectors()
        let cases = try XCTUnwrap(corpus["cases"] as? [[String: Any]])
        XCTAssertGreaterThan(cases.count, 50)
        for item in cases {
            let id = try XCTUnwrap(item["id"] as? String)
            let expected = try XCTUnwrap(item["expected"] as? [String: Any])
            let result = try run(corpus, item)
            XCTAssertEqual(result.ok, expected["ok"] as? Bool, id)
            XCTAssertEqual(result.reason?.rawValue, expected["reason"] as? String, id)
            XCTAssertEqual(result.contentID, expected["content_id"] as? String, id)
            XCTAssertEqual(result.validSignatures, expected["valid_signatures"] as? Int, id)
            XCTAssertEqual(result.eligibleRuntimes, expected["eligible_runtimes"] as? [Int], id)
            XCTAssertEqual(result.filesChecked, expected["files_checked"] as? Int, id)
            XCTAssertEqual(result.filesVerified, expected["files_verified"] as? Bool, id)
        }
    }

    func testGoldenIdentityAndSigningInput() throws {
        let corpus = try vectors()
        let golden = try XCTUnwrap(corpus["golden"] as? [String: String])
        let data = try manifest(corpus, "golden")
        XCTAssertEqual(PortableBundle.contentID(of: data), golden["content_id"])
        let digest = SHA256.hash(data: PortableBundle.signingInput(for: data)).map { String(format: "%02x", $0) }
        XCTAssertEqual(digest.joined(), golden["signing_input_sha256"])
    }

    func testCanonicalizationVectors() throws {
        let corpus = try vectors()
        for item in try XCTUnwrap(corpus["canonicalization"] as? [[String: String]]) {
            let canonical = try PortableBundle.canonicalJSON(Data(try XCTUnwrap(item["input"]).utf8))
            XCTAssertEqual(String(decoding: canonical, as: UTF8.self), item["canonical"], item["id"] ?? "")
        }
        for item in try XCTUnwrap(corpus["canonicalization_rejects"] as? [[String: String]]) {
            XCTAssertThrowsError(try PortableBundle.canonicalJSON(Data(try XCTUnwrap(item["input"]).utf8)), item["id"] ?? "") {
                XCTAssertEqual($0 as? PortableBundleError, PortableBundleError(.manifestNotCanonical))
            }
        }
    }

    func testCanonicalKeysAreNotMergedByUnicodeEquivalence() throws {
        let decomposed = Data("{\"e\\u0301\":1,\"\\u00e9\":2}".utf8)
        let canonical = try PortableBundle.canonicalJSON(decomposed)
        XCTAssertEqual(String(decoding: canonical, as: UTF8.self), "{\"e\u{0301}\":1,\"\u{00e9}\":2}")
        XCTAssertThrowsError(try PortableBundle.canonicalJSON(Data("{\"a\":1,\"a\":2}".utf8)))
    }

    func testUnknownNoncriticalExtensionsRoundTripByteForByte() throws {
        let corpus = try vectors()
        let golden = try manifest(corpus, "golden")
        XCTAssertNoThrow(try PortableBundle.validateManifest(golden))
        XCTAssertEqual(try PortableBundle.canonicalJSON(golden), golden)
        let critical = try manifest(corpus, "valid_critical_extension_understood")
        XCTAssertThrowsError(try PortableBundle.validateManifest(critical)) {
            XCTAssertEqual($0 as? PortableBundleError, PortableBundleError(.unknownCriticalExtension))
        }
        XCTAssertNoThrow(try PortableBundle.validateManifest(critical, understoodExtensions: ["org.openmed.test.note"]))
    }

    func testRFC8032VectorsVerifyThroughCryptoKit() throws {
        let corpus = try vectors()
        func bytes(_ hex: String) -> Data {
            var output = Data()
            var index = hex.startIndex
            while index < hex.endIndex {
                let next = hex.index(index, offsetBy: 2)
                output.append(UInt8(hex[index..<next], radix: 16) ?? 0)
                index = next
            }
            return output
        }
        for item in try XCTUnwrap(corpus["ed25519_rfc8032"] as? [[String: String]]) {
            XCTAssertTrue(
                try PortableBundle.ed25519Verify(
                    publicKey: bytes(try XCTUnwrap(item["public_hex"])),
                    signature: bytes(try XCTUnwrap(item["signature_hex"])),
                    message: bytes(try XCTUnwrap(item["message_hex"]))),
                item["name"] ?? "")
        }
    }

    func testDirectoryVerificationStreamsAndRefusesLinks() throws {
        let corpus = try vectors()
        let root = FileManager.default.temporaryDirectory.appending(path: "openmed-portable-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let trees = try XCTUnwrap(corpus["trees"] as? [String: [[String: String]]])
        for entry in try XCTUnwrap(trees["golden"]) {
            let url = root.appending(path: try XCTUnwrap(entry["path"]))
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try Data(try XCTUnwrap(entry["text"]).utf8).write(to: url)
        }
        let envelopes = try XCTUnwrap(corpus["envelopes"] as? [String: String])
        try manifest(corpus, "golden").write(to: root.appending(path: PortableBundle.manifestFilename))
        try Data(try XCTUnwrap(envelopes["golden"]).utf8).write(to: root.appending(path: PortableBundle.envelopeFilename))
        let contexts = try XCTUnwrap(corpus["contexts"] as? [String: [String: Any]])
        let verificationContext = try context(try XCTUnwrap(contexts["default"]))
        let trust = try policy(corpus, "default")
        let accepted = PortableBundle.verifyDirectory(at: root, trustPolicy: trust, context: verificationContext)
        XCTAssertTrue(accepted.ok)
        XCTAssertTrue(accepted.eligibleForActivation)
        XCTAssertEqual(accepted.filesChecked, 14)
        try FileManager.default.createSymbolicLink(
            at: root.appending(path: "link.txt"), withDestinationURL: root.appending(path: "LICENSE"))
        let linked = PortableBundle.verifyDirectory(at: root, trustPolicy: trust, context: verificationContext)
        XCTAssertEqual(linked.reason, .pathUnsafe)
    }

    func testCallerInputErrorsCarryOnlyReasonCodes() throws {
        XCTAssertThrowsError(try PortableBundleTrustPolicy(json: Data("{\"schema\":\"other\"}".utf8))) {
            XCTAssertEqual(String(describing: $0), "trust_policy_invalid")
        }
        XCTAssertThrowsError(try PortableBundleContext(now: "yesterday")) {
            XCTAssertEqual(String(describing: $0), "context_invalid")
        }
        let key = try PortableBundleTrustedKey(
            kid: "unit-test", publicKey: Data(repeating: 7, count: 32), notBefore: "2026-01-01T00:00:00Z",
            notAfter: "2027-01-01T00:00:00Z")
        XCTAssertFalse(String(describing: key).contains("BwcH"))
    }
}
