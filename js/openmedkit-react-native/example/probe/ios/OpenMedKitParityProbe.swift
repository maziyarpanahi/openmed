import CryptoKit
import Foundation
import OpenMedKit
import React
import Security

/// Example-only native module that calls OpenMedKit directly.
///
/// The React Native parity harness compares `OpenMedKitRN` bridge results with
/// the results of this module. It never goes through `OpenMedKitRN`: it loads
/// its own `OpenMed` runtime from the same verified local bundle, offline, and
/// returns results as a JSON string. Rejections and error records carry fixed
/// codes only, never input text, file paths, or native error messages.
@objc(OpenMedKitParityProbe)
final class OpenMedKitParityProbe: NSObject, RCTBridgeModule {
    private let queue = DispatchQueue(label: "org.openmed.openmedkit.parityprobe")
    private var runtime: OpenMed?
    private var runtimeKey: String?
    private var fixture: VerifiedFixture?

    @objc
    static func moduleName() -> String! {
        "OpenMedKitParityProbe"
    }

    @objc
    static func requiresMainQueueSetup() -> Bool {
        false
    }

    @objc(describeEnvironment:rejecter:)
    func describeEnvironment(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        resolve([
            "platform": "ios",
            "runtime_backend": "coreml",
            "device_class": Self.deviceClass,
            "os_version": "\(version.majorVersion).\(version.minorVersion).\(version.patchVersion)",
            "build_configuration": Self.buildConfiguration,
        ])
    }

    @objc(prepareModelFixture:rejecter:)
    func prepareModelFixture(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let fixture = try FixtureVerifier.verify(root: try Self.fixtureRoot())
                self.fixture = fixture
                resolve(fixture.record)
            } catch {
                reject(Self.code(of: error, fallback: "model_fixture_unverified"), "model fixture verification failed", nil)
            }
        }
    }

    @objc(createEphemeralHashSecret:rejecter:)
    func createEphemeralHashSecret(
        _ resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            reject("ephemeral_secret_unavailable", "secure random bytes unavailable", nil)
            return
        }
        resolve(bytes.map { String(format: "%02x", $0) }.joined())
    }

    @objc(runDirect:resolver:rejecter:)
    func runDirect(
        _ requestJSON: String,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            var operation = "unknown"
            let response: [String: Any]
            do {
                let request = try DirectRequest(json: requestJSON)
                operation = request.operation
                guard let fixture = self.fixture, request.model.matches(fixture) else {
                    throw ProbeError.modelUnverified
                }
                let runtime = try self.runtime(for: request.model)
                response = ["ok": true, "result": try Self.run(request, runtime: runtime)]
            } catch {
                response = [
                    "ok": false,
                    "error": ["code": Self.code(of: error, fallback: "direct_\(operation)_failed")],
                ]
            }
            resolve(Self.jsonString(response))
        }
    }

    @objc(writeReport:resolver:rejecter:)
    func writeReport(
        _ reportJSON: String,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let url = try Self.documentsDirectory()
                    .appendingPathComponent("openmed-rn-parity-report.json")
                try Data(reportJSON.utf8).write(to: url, options: .atomic)
                resolve(true)
            } catch {
                reject("report_write_failed", "parity report could not be written", nil)
            }
        }
    }

    // MARK: - Direct OpenMedKit calls

    private func runtime(for model: DirectModel) throws -> OpenMed {
        if let runtime, runtimeKey == model.cacheKey {
            return runtime
        }
        runtime = nil
        runtimeKey = nil
        let loaded: OpenMed
        switch model.backend {
        case "coreml":
            guard let id2LabelPath = model.id2LabelPath else {
                throw ProbeError.modelUnverified
            }
            loaded = try OpenMed(
                backend: .coreML(
                    modelURL: URL(fileURLWithPath: model.modelPath),
                    id2labelURL: URL(fileURLWithPath: id2LabelPath),
                    tokenizerName: model.tokenizerName
                        ?? "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                    tokenizerFolderURL: model.tokenizerFolderPath.map(URL.init(fileURLWithPath:))
                ),
                allowNetworkAccess: false
            )
        case "mlx":
            loaded = try OpenMed(
                backend: .mlx(modelDirectoryURL: URL(fileURLWithPath: model.modelPath)),
                allowNetworkAccess: false
            )
        default:
            throw ProbeError.unsupportedBackend
        }
        runtime = loaded
        runtimeKey = model.cacheKey
        return loaded
    }

    private static func run(_ request: DirectRequest, runtime: OpenMed) throws -> [String: Any] {
        let options = request.options
        let text = request.text
        switch request.operation {
        case "analyzeText":
            let entities = try runtime.analyzeText(
                text,
                confidenceThreshold: options.confidenceThreshold
            )
            return ["spans": entities.map { entitySpan($0, text: text, options: options) }]
        case "extractPii":
            let entities = try runtime.extractPII(
                text,
                confidenceThreshold: options.confidenceThreshold,
                useSmartMerging: options.useSmartMerging
            )
            return ["spans": entities.map { entitySpan($0, text: text, options: options) }]
        case "deidentify":
            let result = try runtime.deidentify(
                text,
                policy: options.policy ?? Policy.defaultName,
                confidenceThreshold: options.confidenceThreshold,
                useSmartMerging: options.useSmartMerging
            )
            return [
                "text": text,
                "deidentifiedText": result.redactedText,
                "spans": result.actions.map {
                    span(
                        label: $0.label,
                        canonicalLabel: $0.canonicalLabel,
                        start: $0.start,
                        end: $0.end,
                        confidence: $0.confidence,
                        action: $0.action.rawValue,
                        replacement: $0.replacement,
                        text: text,
                        options: options
                    )
                },
            ]
        default:
            throw ProbeError.requestInvalid
        }
    }

    private static func entitySpan(
        _ entity: EntityPrediction,
        text: String,
        options: DirectOptions
    ) -> [String: Any] {
        span(
            label: entity.label,
            canonicalLabel: Policy.canonicalLabel(for: entity.label),
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence,
            action: "keep",
            replacement: nil,
            text: text,
            options: options
        )
    }

    /// The documented span schema v1, built from direct native values.
    private static func span(
        label: String,
        canonicalLabel: String,
        start: Int,
        end: Int,
        confidence: Float,
        action: String,
        replacement: String?,
        text: String,
        options: DirectOptions
    ) -> [String: Any] {
        [
            "schema_version": 1,
            "doc_id": options.docID,
            "start": start,
            "end": end,
            "text_hash": hmacTextHash(
                scalarSubstring(text, start: start, end: end),
                secret: options.hashSecret
            ),
            "entity_type": label,
            "canonical_label": canonicalLabel,
            "policy_label": policyLabel(for: canonicalLabel),
            "regulatory_tags": [String](),
            "score": Double(confidence),
            "detector": "openmedkit-ios",
            "evidence": ["bridge": "react-native", "runtime": "OpenMedKit"],
            "action": action,
            "replacement": replacement ?? NSNull(),
            "reversible_id": NSNull(),
            "section": NSNull(),
            "metadata": options.metadata,
        ]
    }

    /// Slice by Unicode scalar offsets, the OpenMed offset unit.
    private static func scalarSubstring(_ text: String, start: Int, end: Int) -> String {
        let scalars = Array(text.unicodeScalars)
        guard start >= 0, end > start, end <= scalars.count else {
            return ""
        }
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars[start..<end])
        return String(view)
    }

    private static func hmacTextHash(_ surface: String, secret: SymmetricKey) -> String {
        let signature = HMAC<SHA256>.authenticationCode(for: Data(surface.utf8), using: secret)
        return "hmac-sha256:" + signature.map { String(format: "%02x", $0) }.joined()
    }

    /// Mirrors the private policy-label table in OpenMedKit `Policy.swift`
    /// exactly (a test keeps them equal). Swift currently lists fewer clinical
    /// concept labels than Kotlin; the Swift canonicalizer never emits the
    /// missing ones, and cross-kit table equivalence belongs to #3643.
    private static func policyLabel(for canonicalLabel: String) -> String {
        if clinicalConceptLabels.contains(canonicalLabel) {
            return "CLINICAL_CONCEPT"
        }
        if quasiIdentifierLabels.contains(canonicalLabel) {
            return "QUASI_IDENTIFIER"
        }
        return "DIRECT_IDENTIFIER"
    }

    private static let clinicalConceptLabels: Set<String> = [
        "MICROORGANISM",
        "ANTIBIOTIC",
        "SUSCEPTIBILITY",
        "CONDITION",
        "MEDICATION",
        "LAB_TEST",
        "PROCEDURE",
        "BODY_SITE",
        "OTHER",
    ]

    private static let quasiIdentifierLabels: Set<String> = [
        "LOCATION",
        "ZIPCODE",
        "ORDINAL_DIRECTION",
        "DATE",
        "TIME",
        "AGE",
        "CREDIT_CARD_ISSUER",
        "AMOUNT",
        "CURRENCY",
        "GENDER",
        "EYE_COLOR",
        "HEIGHT",
        "ORGANIZATION",
        "JOB_TITLE",
        "JOB_DEPARTMENT",
        "OCCUPATION",
    ]

    // MARK: - Helpers

    private static var deviceClass: String {
        #if targetEnvironment(simulator)
            return "simulator"
        #else
            return "device"
        #endif
    }

    private static var buildConfiguration: String {
        #if DEBUG
            return "debug"
        #else
            return "release"
        #endif
    }

    private static func documentsDirectory() throws -> URL {
        guard
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        else {
            throw ProbeError.fixtureMissing
        }
        return url
    }

    private static func fixtureRoot() throws -> URL {
        try documentsDirectory().appendingPathComponent("openmed-parity-fixture", isDirectory: true)
    }

    private static func code(of error: Error, fallback: String) -> String {
        (error as? ProbeError)?.rawValue ?? fallback
    }

    private static func jsonString(_ object: [String: Any]) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys, .withoutEscapingSlashes]
            ),
            let string = String(data: data, encoding: .utf8)
        else {
            return #"{"ok":false,"error":{"code":"direct_serialization_failed"}}"#
        }
        return string
    }
}

private enum ProbeError: String, Error {
    case requestInvalid = "direct_request_invalid"
    case modelUnverified = "direct_model_unverified"
    case unsupportedBackend = "direct_backend_unsupported"
    case fixtureMissing = "model_fixture_missing"
    case fixtureManifestInvalid = "model_fixture_manifest_invalid"
    case fixtureFileSetMismatch = "model_fixture_file_set_mismatch"
    case fixtureDigestMismatch = "model_fixture_digest_mismatch"
    case fixtureBundleDigestMismatch = "model_fixture_bundle_digest_mismatch"
    case fixtureSymlink = "model_fixture_symlink"
}

private struct DirectModel {
    let cacheKey: String
    let modelPath: String
    let backend: String
    let id2LabelPath: String?
    let tokenizerFolderPath: String?
    let tokenizerName: String?

    func matches(_ fixture: VerifiedFixture) -> Bool {
        modelPath == fixture.modelPath
            && backend == fixture.backend
            && id2LabelPath == fixture.id2LabelPath
            && tokenizerFolderPath == fixture.tokenizerFolderPath
    }
}

private struct DirectOptions {
    let docID: String
    let hashSecret: SymmetricKey
    let metadata: [String: Any]
    let confidenceThreshold: Float
    let useSmartMerging: Bool
    let policy: String?
}

private struct DirectRequest {
    let operation: String
    let text: String
    let options: DirectOptions
    let model: DirectModel

    init(json: String) throws {
        guard
            let object = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
            let root = object as? [String: Any],
            let operation = root["operation"] as? String,
            let text = root["text"] as? String,
            let options = root["options"] as? [String: Any],
            let model = root["model"] as? [String: Any],
            let docID = options["docId"] as? String,
            let secret = options["hashSecret"] as? String, !secret.isEmpty,
            let threshold = (options["confidenceThreshold"] as? NSNumber)?.floatValue,
            let useSmartMerging = options["useSmartMerging"] as? Bool,
            let cacheKey = model["cacheKey"] as? String,
            let modelPath = model["modelPath"] as? String,
            let backend = model["backend"] as? String
        else {
            throw ProbeError.requestInvalid
        }
        self.operation = operation
        self.text = text
        self.options = DirectOptions(
            docID: docID,
            hashSecret: SymmetricKey(data: Data(secret.utf8)),
            metadata: options["metadata"] as? [String: Any] ?? [:],
            confidenceThreshold: threshold,
            useSmartMerging: useSmartMerging,
            policy: options["policy"] as? String
        )
        self.model = DirectModel(
            cacheKey: cacheKey,
            modelPath: modelPath,
            backend: backend,
            id2LabelPath: model["id2LabelPath"] as? String,
            tokenizerFolderPath: model["tokenizerFolderPath"] as? String,
            tokenizerName: model["tokenizerName"] as? String
        )
    }
}

private struct VerifiedFixture {
    let modelPath: String
    let backend: String
    let id2LabelPath: String
    let tokenizerFolderPath: String?
    let tokenizerName: String?
    let bundleDigest: String
    let fileCount: Int

    var record: [String: Any] {
        var record: [String: Any] = [
            "modelPath": modelPath,
            "backend": backend,
            "id2LabelPath": id2LabelPath,
            "bundleDigest": bundleDigest,
            "fileCount": fileCount,
        ]
        if let tokenizerFolderPath {
            record["tokenizerFolderPath"] = tokenizerFolderPath
        }
        if let tokenizerName {
            record["tokenizerName"] = tokenizerName
        }
        return record
    }
}

/// Verifies a staged fixture exactly as `example/src/parity/fixture.ts` defines it.
private enum FixtureVerifier {
    private struct Manifest: Decodable {
        struct File: Decodable {
            let path: String
            let sha256: String
            let bytes: Int
        }

        let schema: String
        let version: Int
        let platform: String
        let backend: String
        let modelPath: String
        let id2labelPath: String
        let tokenizerFolderPath: String?
        let tokenizerName: String?
        let files: [File]
        let bundleDigest: String

        enum CodingKeys: String, CodingKey {
            case schema
            case version
            case platform
            case backend
            case modelPath = "model_path"
            case id2labelPath = "id2label_path"
            case tokenizerFolderPath = "tokenizer_folder_path"
            case tokenizerName = "tokenizer_name"
            case files
            case bundleDigest = "bundle_digest"
        }
    }

    static func verify(root: URL) throws -> VerifiedFixture {
        guard
            let data = try? Data(contentsOf: root.appendingPathComponent("openmed-parity-fixture.json"))
        else {
            throw ProbeError.fixtureMissing
        }
        guard
            let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
            manifest.schema == "openmed.rn.parity.model_fixture",
            manifest.version == 1,
            manifest.platform == "ios",
            !manifest.files.isEmpty,
            manifest.files.allSatisfy({ isSafeRelativePath($0.path) }),
            isSafeRelativePath(manifest.modelPath),
            isSafeRelativePath(manifest.id2labelPath)
        else {
            throw ProbeError.fixtureManifestInvalid
        }

        let resolvedRoot = root.resolvingSymlinksInPath()
        let listed = Set(manifest.files.map(\.path))
        guard listed.count == manifest.files.count else {
            throw ProbeError.fixtureManifestInvalid
        }
        guard try regularFiles(under: resolvedRoot, prefix: "model") == listed else {
            throw ProbeError.fixtureFileSetMismatch
        }
        for file in manifest.files {
            let (digest, bytes) = try sha256(of: resolvedRoot.appendingPathComponent(file.path))
            guard digest == file.sha256, bytes == file.bytes else {
                throw ProbeError.fixtureDigestMismatch
            }
        }
        let lines = manifest.files
            .sorted { Array($0.path.utf8).lexicographicallyPrecedes(Array($1.path.utf8)) }
            .map { "\($0.path)\t\($0.sha256)\t\($0.bytes)\n" }
            .joined()
        let digest =
            "sha256:"
            + SHA256.hash(data: Data(lines.utf8)).map { String(format: "%02x", $0) }.joined()
        guard digest == manifest.bundleDigest else {
            throw ProbeError.fixtureBundleDigestMismatch
        }

        return VerifiedFixture(
            modelPath: resolvedRoot.appendingPathComponent(manifest.modelPath).path,
            backend: manifest.backend,
            id2LabelPath: resolvedRoot.appendingPathComponent(manifest.id2labelPath).path,
            tokenizerFolderPath: manifest.tokenizerFolderPath.map {
                resolvedRoot.appendingPathComponent($0).path
            },
            tokenizerName: manifest.tokenizerName,
            bundleDigest: manifest.bundleDigest,
            fileCount: manifest.files.count
        )
    }

    private static func isSafeRelativePath(_ path: String) -> Bool {
        let segments = path.split(separator: "/", omittingEmptySubsequences: false)
        return segments.first == "model"
            && segments.allSatisfy { segment in
                !segment.isEmpty && segment != "." && segment != ".."
                    && segment.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || "._-".contains($0)) }
            }
    }

    private static func regularFiles(under root: URL, prefix: String) throws -> Set<String> {
        let base = root.appendingPathComponent(prefix)
        guard
            let enumerator = FileManager.default.enumerator(
                at: base,
                includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]
            )
        else {
            throw ProbeError.fixtureMissing
        }
        let basePath = base.resolvingSymlinksInPath().path + "/"
        var files = Set<String>()
        for case let url as URL in enumerator {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            if values.isSymbolicLink == true {
                throw ProbeError.fixtureSymlink
            }
            guard values.isRegularFile == true else {
                continue
            }
            let path = url.resolvingSymlinksInPath().path
            guard path.hasPrefix(basePath) else {
                throw ProbeError.fixtureFileSetMismatch
            }
            files.insert(prefix + "/" + String(path.dropFirst(basePath.count)))
        }
        return files
    }

    private static func sha256(of url: URL) throws -> (String, Int) {
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw ProbeError.fixtureFileSetMismatch
        }
        defer { try? handle.close() }
        var hasher = SHA256()
        var total = 0
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
            total += chunk.count
        }
        let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return (digest, total)
    }
}
