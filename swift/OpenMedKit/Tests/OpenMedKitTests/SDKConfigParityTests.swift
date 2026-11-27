import Foundation
import XCTest

@testable import OpenMedKit

/// Replays `tests/fixtures/parity/sdk_config_v1.json` through
/// ``OpenMedSDKConfig`` and checks the type against `sdk_config.schema.json`.
final class SDKConfigParityTests: XCTestCase {
    private typealias Field = OpenMedSDKConfig.Field

    func testValidVectorsDecodeToTheirResolvedForm() throws {
        let fixture = try loadJSON("tests/fixtures/parity/sdk_config_v1.json")
        let valid = try XCTUnwrap(fixture["valid"] as? [[String: Any]])
        XCTAssertFalse(valid.isEmpty)

        for vector in valid {
            let id = try XCTUnwrap(vector["id"] as? String)
            let resolved = try XCTUnwrap(vector["resolved"] as? [String: Any], id)
            let config = try OpenMedSDKConfig.load(try document(vector))

            assertConfig(config, matches: resolved, id)
            XCTAssertEqual(OpenMedSDKConfig.validate(try document(vector)), [], id)
            XCTAssertEqual(config.allowNetworkAccess, !config.localOnly, id)
            // Encoding writes all six keys, the resolved form, and decodes back.
            let encoded = try JSONEncoder().encode(config)
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any], id)
            XCTAssertTrue(NSDictionary(dictionary: object).isEqual(to: resolved), id)
            XCTAssertEqual(try OpenMedSDKConfig.load(encoded), config, id)
        }
    }

    func testInvalidVectorsFailWithTheSharedOrderedIssues() throws {
        let fixture = try loadJSON("tests/fixtures/parity/sdk_config_v1.json")
        let invalid = try XCTUnwrap(fixture["invalid"] as? [[String: Any]])
        XCTAssertFalse(invalid.isEmpty)
        let fieldNames = Set(Field.allCases.map(\.rawValue))
        let codes = Set(OpenMedSDKConfigIssue.Code.allCases.map(\.rawValue))

        for vector in invalid {
            let id = try XCTUnwrap(vector["id"] as? String)
            let expected = try XCTUnwrap(vector["errors"] as? [[String: Any]], id).map { error in
                OpenMedSDKConfigIssue(
                    code: OpenMedSDKConfigIssue.Code(rawValue: error["code"] as? String ?? "")!,
                    field: (error["field"] as? String).flatMap(Field.init(rawValue:)))
            }
            let data = try document(vector)

            XCTAssertEqual(OpenMedSDKConfig.validate(data), expected, id)
            XCTAssertThrowsError(try OpenMedSDKConfig.load(data), id) { error in
                let failure = error as? OpenMedSDKConfigError
                XCTAssertEqual(failure?.issues, expected, id)
                // Descriptions carry codes and portable key names only.
                let description = failure?.localizedDescription ?? ""
                XCTAssertTrue(description.hasPrefix("invalid SDK configuration: "), id)
                let tokens = description.dropFirst("invalid SDK configuration: ".count)
                    .split(separator: ",").flatMap { $0.trimmingCharacters(in: .whitespaces).split(separator: ":") }
                XCTAssertTrue(tokens.allSatisfy { codes.contains(String($0)) || fieldNames.contains(String($0)) }, id)
            }
            XCTAssertThrowsError(try JSONDecoder().decode(OpenMedSDKConfig.self, from: data), id) { error in
                XCTAssertEqual((error as? OpenMedSDKConfigError)?.issues, expected, id)
            }
        }
    }

    func testRejectedValuesAndUnknownKeysNeverReachTheError() throws {
        let secret = "SYNTHETIC_PRIVATE_SENTINEL_5387"
        let inputs = [
            #"{"\#(secret)": true}"#,
            #"{"model_id": "\#(secret) /x"}"#,
            #"{"model_revision": "\#(secret)"}"#,
            #"{"policy_profile": "\#(secret)"}"#,
            #"{"confidence_threshold": "\#(secret)"}"#,
            #"["\#(secret)"]"#,
            #"{"local_only": "\#(secret)""#,
        ]
        for input in inputs {
            XCTAssertThrowsError(try OpenMedSDKConfig.load(Data(input.utf8))) { error in
                for text in [String(describing: error), String(reflecting: error), error.localizedDescription] {
                    XCTAssertFalse(text.contains(secret), text)
                    XCTAssertFalse(text.contains("SYNTHETIC"), text)
                }
            }
        }
    }

    func testNonJSONAndNonFiniteInputsFailClosed() {
        let notObject = [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)]
        XCTAssertEqual(OpenMedSDKConfig.validate(Data()), notObject)
        XCTAssertEqual(OpenMedSDKConfig.validate(Data("{".utf8)), notObject)
        XCTAssertEqual(OpenMedSDKConfig.validate(Data("{\"local_only\": tru}".utf8)), notObject)
        XCTAssertEqual(
            OpenMedSDKConfig.validate(Data("{\"confidence_threshold\": 1e400}".utf8)),
            [OpenMedSDKConfigIssue(code: .outOfRange, field: .confidenceThreshold)])
        XCTAssertEqual(
            OpenMedSDKConfig.validate(Data("{\"local_only\": 1e400, \"model_id\": [], \"onnx_variant\": {}}".utf8)),
            [
                OpenMedSDKConfigIssue(code: .wrongType, field: .localOnly),
                OpenMedSDKConfigIssue(code: .wrongType, field: .modelID),
                OpenMedSDKConfigIssue(code: .wrongType, field: .onnxVariant),
            ])
    }

    /// Text resolves as it does in Python's `json.loads`, the Kotlin kit and
    /// `JSON.parse`, not as Foundation's more lenient `JSONDecoder` reads it.
    func testDocumentTextResolvesLikeTheOtherRuntimes() throws {
        func load(_ text: String) throws -> OpenMedSDKConfig { try OpenMedSDKConfig.load(Data(text.utf8)) }
        func validate(_ text: String) -> [OpenMedSDKConfigIssue] { OpenMedSDKConfig.validate(Data(text.utf8)) }

        // The last value of a repeated key wins, so this document stays offline.
        let offline = try load(#"{"local_only": false, "local_only": true}"#)
        XCTAssertTrue(offline.localOnly)
        XCTAssertFalse(offline.allowNetworkAccess)
        XCTAssertTrue(try load(#"{"local_only": true, "local_only": false}"#).allowNetworkAccess)
        XCTAssertTrue(try load(#"{"local_only": false, "local_only": true}"#).localOnly)
        XCTAssertEqual(validate(#"{"local_only": "yes", "local_only": true}"#), [])
        XCTAssertEqual(
            validate(#"{"local_only": true, "local_only": "yes"}"#),
            [OpenMedSDKConfigIssue(code: .wrongType, field: .localOnly)])
        XCTAssertEqual(try load(#"{"onnx_variant": "fp16", "onnx_variant": "int8"}"#).onnxVariant, .int8)

        // A number below the smallest Double is 0, as float() makes it.
        XCTAssertEqual(validate(#"{"confidence_threshold": 1e-400}"#), [])
        XCTAssertEqual(try load(#"{"confidence_threshold": 1e-400}"#).confidenceThreshold, 0)
        XCTAssertEqual(try load(#"{"confidence_threshold": -1E-400}"#).confidenceThreshold, 0)
        XCTAssertEqual(
            validate(#"{"confidence_threshold": 2e400}"#),
            [OpenMedSDKConfigIssue(code: .outOfRange, field: .confidenceThreshold)])

        // Escapes decode; a lone surrogate is still a string.
        XCTAssertEqual(try load(#"{"policy_profile": "strict_no_leak"}"#).policyProfile, .strictNoLeak)
        XCTAssertEqual(
            validate(#"{"model_id": "\ud800", "onnx_variant": "😀"}"#),
            [
                OpenMedSDKConfigIssue(code: .invalidFormat, field: .modelID),
                OpenMedSDKConfigIssue(code: .notAllowedValue, field: .onnxVariant),
            ])

        // Anything that is not RFC 8259 JSON is not a JSON object.
        let notObject = [OpenMedSDKConfigIssue(code: .configNotObject, field: nil)]
        let refused = [
            "\u{FEFF}{}", "\u{FEFF}{\"local_only\": true}", #"{"local_only": true,}"#, #"{"model_id": [null,]}"#,
            #"{,}"#, #"{"local_only": true,, "model_id": null}"#, #"{"confidence_threshold": NaN}"#,
            #"{"confidence_threshold": Infinity}"#, #"{"confidence_threshold": -Infinity}"#, #"{local_only: true}"#,
            #"{'local_only': true}"#, #"{"local_only": true} // offline"#, #"/* offline */ {}"#,
            #"{"local_only": true} {}"#, #"{"local_only": true}x"#, #"{"local_only": True}"#,
            #"{"confidence_threshold": 01}"#, #"{"confidence_threshold": .5}"#, #"{"confidence_threshold": +0.5}"#,
            #"{"confidence_threshold": 1.}"#, #"{"confidence_threshold": 1e}"#, #"{"confidence_threshold": 0x1}"#,
            #"{"model_id": "\x"}"#, #"{"model_id": "\u12"}"#, "{\"model_id\": \"a\tb\"}", "{\"local_only\" true}",
            "{\"local_only\":\u{A0}true}", "",
        ]
        for text in refused { XCTAssertEqual(validate(text), notObject, text) }
        // Invalid UTF-8, an encoded surrogate and an overlong encoding.
        for bytes: [UInt8] in [[0xFF], [0xED, 0xA0, 0x80], [0xC0, 0xAF]] {
            let data = Data(#"{"model_id": ""#.utf8) + bytes + Data(#""}"#.utf8)
            XCTAssertEqual(OpenMedSDKConfig.validate(data), notObject, "\(bytes)")
        }
        XCTAssertEqual(
            validate("\t{\r\n\"model_id\" : \"a\u{00E9}\" }\n"), [OpenMedSDKConfigIssue(code: .invalidFormat, field: .modelID)])

        // Nesting is checked without recursion and refused beyond 512 levels.
        func nested(_ depth: Int) -> String {
            #"{"model_id": "# + String(repeating: #"[{"a": "#, count: depth / 2) + String(repeating: "[", count: depth % 2)
                + "1" + String(repeating: "]", count: depth % 2) + String(repeating: "}]", count: depth / 2) + "}"
        }
        XCTAssertEqual(validate(nested(511)), [OpenMedSDKConfigIssue(code: .wrongType, field: .modelID)])
        XCTAssertEqual(validate(nested(512)), notObject)
        XCTAssertEqual(validate(nested(100_000)), notObject)
    }

    func testTypedInitializerValidatesLikeADocument() throws {
        let config = try OpenMedSDKConfig(
            confidenceThreshold: 0.25, localOnly: false, modelID: "synthetic-org/synthetic-model",
            modelRevision: "main", onnxVariant: .int8, policyProfile: .strictNoLeak)
        XCTAssertTrue(config.allowNetworkAccess)
        XCTAssertEqual(try OpenMedSDKConfig.load(JSONEncoder().encode(config)), config)
        XCTAssertEqual(try OpenMedSDKConfig(), OpenMedSDKConfig.defaults)

        let cases: [(() throws -> OpenMedSDKConfig, OpenMedSDKConfigIssue)] = [
            ({ try OpenMedSDKConfig(confidenceThreshold: 1.5) }, .init(code: .outOfRange, field: .confidenceThreshold)),
            ({ try OpenMedSDKConfig(confidenceThreshold: .nan) }, .init(code: .outOfRange, field: .confidenceThreshold)),
            ({ try OpenMedSDKConfig(modelID: "../synthetic") }, .init(code: .invalidFormat, field: .modelID)),
            ({ try OpenMedSDKConfig(modelID: "a/b/c") }, .init(code: .invalidFormat, field: .modelID)),
            ({ try OpenMedSDKConfig(modelID: "a", modelRevision: "a/b") }, .init(code: .invalidFormat, field: .modelRevision)),
            ({ try OpenMedSDKConfig(modelRevision: "main") }, .init(code: .missingDependency, field: .modelRevision)),
        ]
        for (make, issue) in cases {
            XCTAssertThrowsError(try make()) { XCTAssertEqual(($0 as? OpenMedSDKConfigError)?.issues, [issue]) }
        }
        XCTAssertNoThrow(try OpenMedSDKConfig(modelID: String(repeating: "a", count: 96) + "/" + String(repeating: "b", count: 96)))
        XCTAssertNoThrow(try OpenMedSDKConfig(modelID: "a", modelRevision: String(repeating: "c", count: 128)))
        XCTAssertThrowsError(try OpenMedSDKConfig(modelID: String(repeating: "a", count: 97)))
        XCTAssertThrowsError(try OpenMedSDKConfig(modelID: "a", modelRevision: String(repeating: "c", count: 129)))
    }

    func testVocabularyMatchesTheVectorsTheSchemaAndTheBundledPolicies() throws {
        let fixture = try loadJSON("tests/fixtures/parity/sdk_config_v1.json")
        XCTAssertEqual(fixture["sdk_config_schema"] as? String, OpenMedSDKConfig.schemaID)
        XCTAssertEqual(fixture["sdk_config_version"] as? Int, OpenMedSDKConfig.version)
        XCTAssertEqual(fixture["fields"] as? [String], Field.allCases.map(\.rawValue))
        XCTAssertEqual(fixture["issue_codes"] as? [String], OpenMedSDKConfigIssue.Code.allCases.map(\.rawValue))
        assertConfig(OpenMedSDKConfig.defaults, matches: try XCTUnwrap(fixture["defaults"] as? [String: Any]), "defaults")

        let schema = try loadJSON("openmed/core/schemas/json/sdk_config.schema.json")
        XCTAssertEqual(schema["$id"] as? String, OpenMedSDKConfig.schemaID)
        XCTAssertEqual(schema["x-sdk-config-version"] as? Int, OpenMedSDKConfig.version)
        XCTAssertEqual(schema["additionalProperties"] as? Bool, false)
        let properties = try XCTUnwrap(schema["properties"] as? [String: [String: Any]])
        XCTAssertEqual(Set(properties.keys), Set(Field.allCases.map(\.rawValue)))
        XCTAssertEqual(properties["onnx_variant"]?["enum"] as? [String], OpenMedSDKConfig.ONNXVariant.allCases.map(\.rawValue))
        XCTAssertEqual(properties["policy_profile"]?["enum"] as? [String], OpenMedSDKConfig.PolicyProfile.allCases.map(\.rawValue))
        XCTAssertEqual(properties["local_only"]?["default"] as? Bool, true)
        XCTAssertEqual(Policy.bundledProfileNames, OpenMedSDKConfig.PolicyProfile.allCases.map(\.rawValue))
    }

    // MARK: - Helpers

    private func assertConfig(_ config: OpenMedSDKConfig, matches resolved: [String: Any], _ id: String) {
        XCTAssertEqual(Set(resolved.keys), Set(Field.allCases.map(\.rawValue)), id)
        XCTAssertEqual(config.confidenceThreshold, (resolved["confidence_threshold"] as? NSNumber)?.doubleValue, id)
        XCTAssertEqual(config.localOnly, resolved["local_only"] as? Bool, id)
        XCTAssertEqual(config.modelID, resolved["model_id"] as? String, id)
        XCTAssertEqual(config.modelRevision, resolved["model_revision"] as? String, id)
        XCTAssertEqual(config.onnxVariant.rawValue, resolved["onnx_variant"] as? String, id)
        XCTAssertEqual(config.policyProfile.rawValue, resolved["policy_profile"] as? String, id)
        if config.modelID == nil { XCTAssertTrue(resolved["model_id"] is NSNull, id) }
        if config.modelRevision == nil { XCTAssertTrue(resolved["model_revision"] is NSNull, id) }
    }

    /// The vector's `config` value, serialized again exactly as parsed.
    private func document(_ vector: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: vector["config"] ?? NSNull(), options: [.fragmentsAllowed])
    }

    private func loadJSON(_ path: String) throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: path))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    /// Fails and counts every request, so the tests can prove zero traffic.
    private final class SDKConfigURLProtocol: URLProtocol {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var count = 0

        static var requestCount: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }

        static func reset() {
            lock.lock()
            defer { lock.unlock() }
            count = 0
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            Self.lock.lock()
            Self.count += 1
            Self.lock.unlock()
            client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
        }
        override func stopLoading() {}
    }

    /// The config initializers map `local_only` onto `allowNetworkAccess`.
    extension SDKConfigParityTests {
        func testStubObservesSharedSessionRequests() async {
            installStub()
            // Negative control: the zero-request checks below only mean
            // something if the stub sees shared-session traffic.
            _ = try? await URLSession.shared.data(from: URL(string: "https://sdk-config-stub.invalid/probe")!)
            XCTAssertEqual(SDKConfigURLProtocol.requestCount, 1)
        }

        func testLocalOnlyConfigMakesZeroRequestsDuringInitialization() async throws {
            installStub()
            let config = try OpenMedSDKConfig.load(Data("{}".utf8))
            XCTAssertTrue(config.localOnly)
            try await assertLocalOnlyInitializationMakesZeroRequests(config)
        }

        func testRepeatedLocalOnlyKeyKeepsTheLastValueAndMakesZeroRequests() async throws {
            installStub()
            // Python, Kotlin and the web kit keep the last value; so must Swift.
            let config = try OpenMedSDKConfig.load(Data(#"{"local_only": false, "local_only": true}"#.utf8))
            XCTAssertTrue(config.localOnly)
            try await assertLocalOnlyInitializationMakesZeroRequests(config)
        }

        /// A pin would allow a download; local_only still refuses it.
        private func assertLocalOnlyInitializationMakesZeroRequests(_ config: OpenMedSDKConfig) async throws {
            let pin = try OpenMedTokenizerPin(
                repositoryID: "OpenMed/Synthetic-SDK-Config-Tokenizer",
                revision: "fedcba9876543210fedcba9876543210fedcba98",
                files: ["tokenizer.json", "tokenizer_config.json"].map {
                    OpenMedTokenizerPin.File(name: $0, sha256: String(repeating: "0", count: 64), byteCount: 2)
                })

            XCTAssertThrowsError(try OpenMed(backend: backend(), config: config, tokenizerPin: pin)) {
                XCTAssertEqual($0 as? OpenMedTokenizerError, .missingLocalAssets)
            }
            do {
                _ = try await OpenMed.load(backend: backend(), config: config, tokenizerPin: pin)
                XCTFail("local-only load reached the network path")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .missingLocalAssets)
            }
            XCTAssertEqual(SDKConfigURLProtocol.requestCount, 0)
        }

        func testNetworkOptInReachesTheExistingAcquisitionRules() async throws {
            installStub()
            let config = try OpenMedSDKConfig.load(Data(#"{"local_only": false}"#.utf8))
            XCTAssertTrue(config.allowNetworkAccess)

            // allowNetworkAccess without a pin fails before any request.
            XCTAssertThrowsError(try OpenMed(backend: backend(), config: config)) {
                XCTAssertEqual($0 as? OpenMedTokenizerError, .pinnedRevisionRequired)
            }
            do {
                _ = try await OpenMed.load(backend: backend(), config: config)
                XCTFail("network opt-in without a pin succeeded")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .pinnedRevisionRequired)
            }
            XCTAssertEqual(SDKConfigURLProtocol.requestCount, 0)
        }

        /// Registers the counting stub for this test only.
        private func installStub() {
            SDKConfigURLProtocol.reset()
            URLProtocol.registerClass(SDKConfigURLProtocol.self)
            addTeardownBlock { URLProtocol.unregisterClass(SDKConfigURLProtocol.self) }
        }

        /// A Core ML backend whose model and tokenizer assets do not exist.
        private func backend() -> OpenMedBackend {
            let missing = FileManager.default.temporaryDirectory.appending(
                path: "openmed-sdk-config-missing-\(UUID().uuidString)", directoryHint: .isDirectory)
            return .coreML(
                modelURL: missing.appending(path: "Missing.mlmodelc"),
                id2labelURL: missing.appending(path: "id2label.json"))
        }
    }
#endif
