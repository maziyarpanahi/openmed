import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/parity/error_codes_v1.json`: every scenario whose
/// Swift cell is `required` runs with the fixture's synthetic inputs and must
/// fail with an ``OpenMedCodedError`` carrying the scenario's portable code.
/// Only the code is part of the contract; each error keeps its own reason code.
final class ErrorCodeParityTests: XCTestCase {
    private typealias Trigger = (_ inputs: [String: Any]) throws -> Void

    /// One trigger per scenario id, with the native reason code it still raises.
    private var triggers: [String: (trigger: Trigger, reason: String)] {
        var triggers: [String: (trigger: Trigger, reason: String)] = [
            "empty_hash_secret": (
                { inputs in
                    let secret = try XCTUnwrap(inputs["secret"] as? String)
                    _ = try PolicyEngine.decide(
                        policy: Policy(named: Policy.defaultName), text: "synthetic note", detections: [],
                        surrogateKey: Data(secret.utf8))
                }, "surrogate_key_invalid"
            ),
            "invalid_offsets": (
                { inputs in
                    let length = try XCTUnwrap(inputs["text_length"] as? Int)
                    let detection = PolicyDetection(
                        label: "NAME", start: try XCTUnwrap(inputs["start"] as? Int), end: try XCTUnwrap(inputs["end"] as? Int),
                        confidencePPM: 900_000)
                    _ = try PolicyEngine.decide(
                        policy: Policy(named: Policy.defaultName), text: String(repeating: "x", count: length),
                        detections: [detection])
                }, "detection_offset_invalid"
            ),
            "unknown_policy_profile": (
                { inputs in _ = try Policy(named: try XCTUnwrap(inputs["profile"] as? String)) }, "profile_unknown"
            ),
        ]
        #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
            // OpenMedKit is local-first: without local assets nothing is fetched.
            triggers["model_missing_offline"] = (
                { inputs in
                    XCTAssertEqual(inputs["offline"] as? Bool, true)
                    let missing = FileManager.default.temporaryDirectory.appending(
                        path: "openmed-error-codes-\(UUID().uuidString)", directoryHint: .isDirectory)
                    _ = try OpenMed(
                        backend: .coreML(
                            modelURL: missing.appending(path: "model.mlmodelc"),
                            id2labelURL: missing.appending(path: "id2label.json"),
                            tokenizerName: try XCTUnwrap(inputs["model_id"] as? String),
                            tokenizerFolderURL: missing))
                }, "tokenizer_assets_missing"
            )
        #else
            triggers["model_missing_offline"] = (
                { _ in
                    let missing = FileManager.default.temporaryDirectory.appending(
                        path: "openmed-error-codes-\(UUID().uuidString)", directoryHint: .isDirectory)
                    _ = try OpenMedMLXArtifact(modelDirectoryURL: missing)
                }, "unreadable_file"
            )
        #endif
        return triggers
    }

    func testEveryRequiredSwiftScenarioRaisesItsCode() throws {
        let fixture = try loadFixture()
        let scenarios = try XCTUnwrap(fixture["scenarios"] as? [[String: Any]])
        var ran = 0

        for scenario in scenarios {
            let id = try XCTUnwrap(scenario["id"] as? String)
            let expected = try XCTUnwrap((scenario["code"] as? String).flatMap(OpenMedErrorCode.init(rawValue:)), id)
            let cell = try XCTUnwrap((scenario["runtimes"] as? [String: [String: Any]])?["swift"], id)
            guard cell["status"] as? String == "required" else { continue }
            let inputs = try XCTUnwrap(scenario["inputs"] as? [String: Any], id)
            let entry = try XCTUnwrap(triggers[id], "no Swift trigger for required scenario \(id)")

            XCTAssertThrowsError(try entry.trigger(inputs), id) { error in
                XCTAssertEqual((error as? any OpenMedCodedError)?.openMedErrorCode, expected, id)
                XCTAssertEqual(Self.reasonCode(error), entry.reason, id)
                self.assertValueFree(error, inputs: inputs, id)
            }
            ran += 1
        }
        XCTAssertEqual(ran, triggers.count)
    }

    func testSwiftCellsAreRequiredOrSayWhyNot() throws {
        let fixture = try loadFixture()
        let scenarios = try XCTUnwrap(fixture["scenarios"] as? [[String: Any]])
        for scenario in scenarios {
            let id = try XCTUnwrap(scenario["id"] as? String)
            let cell = try XCTUnwrap((scenario["runtimes"] as? [String: [String: Any]])?["swift"], id)
            switch cell["status"] as? String {
            case "required":
                XCTAssertNotNil(triggers[id], id)
                XCTAssertFalse((cell["entry"] as? [String] ?? []).isEmpty, id)
            case "pending", "not_applicable":
                XCTAssertFalse((cell["reason"] as? String ?? "").isEmpty, id)
                XCTAssertNil(triggers[id], id)
            default:
                XCTFail("unknown status for \(id)")
            }
        }
    }

    func testFixtureCodesAreTheGeneratedSwiftCodes() throws {
        let fixture = try loadFixture()
        let codes = try XCTUnwrap(fixture["codes"] as? [[String: Any]])
        XCTAssertEqual(codes.compactMap { $0["code"] as? String }, OpenMedErrorCode.allCases.map(\.rawValue))
        for entry in codes {
            let code = try XCTUnwrap((entry["code"] as? String).flatMap(OpenMedErrorCode.init(rawValue:)))
            XCTAssertEqual(entry["retryable"] as? Bool, code.isRetryable, code.rawValue)
        }
    }

    func testPolicyReasonCodesMapToPortableFamilies() {
        let table: [(String, OpenMedErrorCode)] = [
            ("surrogate_key_invalid", .configurationError),
            ("detection_label_invalid", .inputError),
            ("detection_offset_invalid", .inputError),
            ("detection_offset_out_of_bounds", .inputError),
            ("detection_confidence_invalid", .inputError),
            ("detection_source_invalid", .inputError),
            ("capability_missing", .capabilityError),
            ("profile_unknown", .policyError),
            ("profile_invalid", .policyError),
            ("manifest_invalid", .policyError),
            ("digest_mismatch", .policyError),
            ("override_label_unknown", .policyError),
        ]
        for (reason, expected) in table {
            let error = PolicyError(reason, "$.synthetic")
            XCTAssertEqual(error.openMedErrorCode, expected, reason)
            // The native code and rendering are unchanged.
            XCTAssertEqual(error.code, reason)
            XCTAssertEqual(error.description, "\(reason) at $.synthetic")
        }
    }

    func testModelAndTokenizerErrorsMapToPortableFamilies() {
        let store: [OpenMedModelStoreError] = [
            .invalidResponse, .httpError(statusCode: 404), .missingManifest, .missingWeights,
            .invalidManifestPath("outside_snapshot"),
        ]
        XCTAssertTrue(store.allSatisfy { $0.openMedErrorCode == .modelLoadError })
        let tokenizer: [(OpenMedTokenizerError, OpenMedErrorCode)] = [
            (.missingLocalAssets, .modelLoadError), (.downloadFailed(statusCode: nil), .modelLoadError),
            (.sizeMismatch, .modelLoadError), (.digestMismatch, .modelLoadError),
            (.cacheUnavailable, .modelLoadError), (.invalidAssets, .modelLoadError),
            (.pinnedRevisionRequired, .configurationError), (.identityMismatch, .configurationError),
        ]
        for (error, expected) in tokenizer {
            XCTAssertEqual(error.openMedErrorCode, expected, error.reasonCode)
        }
        let artifact: [(OpenMedMLXArtifactError, OpenMedErrorCode)] = [
            (.unreadableFile(.config), .modelLoadError), (.malformedFile(.manifest), .modelLoadError),
            (.invalidManifestFormat, .modelLoadError), (.missingTokenizerAssets, .modelLoadError),
            (.missingTokenizerReference, .modelLoadError),
            (.unsupportedArchitecture(task: nil, family: nil), .capabilityError),
        ]
        for (error, expected) in artifact {
            XCTAssertEqual((error as any OpenMedCodedError).openMedErrorCode, expected)
        }
        XCTAssertThrowsError(
            try OpenMedMLXArtifact(
                modelDirectoryURL: FileManager.default.temporaryDirectory.appending(
                    path: "openmed-error-codes-\(UUID().uuidString)", directoryHint: .isDirectory))
        ) { XCTAssertEqual(($0 as? any OpenMedCodedError)?.openMedErrorCode, .modelLoadError) }
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        func testAbsentMLXModelIsAModelLoadErrorWhereMLXRuns() throws {
            let missing = FileManager.default.temporaryDirectory.appending(
                path: "openmed-error-codes-\(UUID().uuidString)", directoryHint: .isDirectory)
            let expected: OpenMedErrorCode = OpenMedMLXRuntimeSupport.isSupported ? .modelLoadError : .capabilityError
            XCTAssertThrowsError(try OpenMed(backend: .mlx(modelDirectoryURL: missing))) {
                XCTAssertEqual(($0 as? any OpenMedCodedError)?.openMedErrorCode, expected)
            }
            // Where MLX cannot run, such as the iOS Simulator, the refusal is a
            // capability_error before any file is read.
            try OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                XCTAssertThrowsError(try OpenMed(backend: .mlx(modelDirectoryURL: missing))) {
                    XCTAssertEqual($0 as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                    XCTAssertEqual(($0 as? any OpenMedCodedError)?.openMedErrorCode, .capabilityError)
                }
            }
        }
    #endif

    // MARK: - Helpers

    private static func reasonCode(_ error: any Error) -> String? {
        switch error {
        case let error as PolicyError: return error.code
        case let error as OpenMedTokenizerError: return error.reasonCode
        case let error as OpenMedModelStoreError: return error.reason
        case let error as OpenMedMLXArtifactError:
            if case .unreadableFile = error { return "unreadable_file" }
            return nil
        default: return nil
        }
    }

    /// Error renderings never contain the scenario's string inputs.
    private func assertValueFree(_ error: any Error, inputs: [String: Any], _ id: String) {
        let renderings = [
            String(describing: error), String(reflecting: error), error.localizedDescription,
            (error as NSError).description,
        ]
        let values = inputs.values.compactMap { $0 as? String }.filter { !$0.isEmpty }
        for text in renderings {
            for value in values {
                XCTAssertFalse(text.contains(value), "\(id) rendering echoes an input")
            }
        }
    }

    private func loadFixture() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/error_codes_v1.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
