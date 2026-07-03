import CryptoKit
import Foundation
import OpenMedKit
import React

@objc(OpenMedKitRN)
final class OpenMedKitRN: NSObject, RCTBridgeModule {
    private let queue = DispatchQueue(label: "org.openmed.openmedkit.reactnative")
    private var runtime: OpenMed?
    private var loadedCacheKey: String?

    @objc
    static func moduleName() -> String! {
        "OpenMedKitRN"
    }

    @objc
    static func requiresMainQueueSetup() -> Bool {
        false
    }

    @objc(loadModel:resolver:rejecter:)
    func loadModel(
        _ options: NSDictionary,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let modelPath = try Self.requiredString(options, "modelPath")
                let backendName = Self.string(options, "backend") ?? "mlx"
                let cacheKey = Self.string(options, "cacheKey") ?? "\(backendName):\(modelPath)"

                if self.loadedCacheKey == cacheKey, self.runtime != nil {
                    resolve([
                        "cacheKey": cacheKey,
                        "modelPath": modelPath,
                        "backend": backendName,
                        "platform": "ios",
                        "loaded": false,
                    ])
                    return
                }

                self.runtime = try OpenMed(backend: Self.backend(from: options))
                self.loadedCacheKey = cacheKey
                resolve([
                    "cacheKey": cacheKey,
                    "modelPath": modelPath,
                    "backend": backendName,
                    "platform": "ios",
                    "loaded": true,
                ])
            } catch {
                reject("openmedkit_load_failed", error.localizedDescription, error)
            }
        }
    }

    @objc(analyzeText:options:resolver:rejecter:)
    func analyzeText(
        _ text: String,
        options: NSDictionary?,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let runtime = try self.requireRuntime()
                let bridgeOptions = BridgeOptions(options)
                let entities = try runtime.analyzeText(
                    text,
                    confidenceThreshold: bridgeOptions.confidenceThreshold
                )
                resolve(entities.map {
                    Self.spanDictionary(
                        entity: $0,
                        text: text,
                        options: bridgeOptions,
                        action: "keep",
                        replacement: nil
                    )
                })
            } catch {
                reject("openmedkit_analyze_failed", error.localizedDescription, error)
            }
        }
    }

    @objc(extractPii:options:resolver:rejecter:)
    func extractPii(
        _ text: String,
        options: NSDictionary?,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let runtime = try self.requireRuntime()
                let bridgeOptions = BridgeOptions(options)
                let entities = try runtime.extractPII(
                    text,
                    confidenceThreshold: bridgeOptions.confidenceThreshold,
                    useSmartMerging: bridgeOptions.useSmartMerging
                )
                resolve(entities.map {
                    Self.spanDictionary(
                        entity: $0,
                        text: text,
                        options: bridgeOptions,
                        action: "keep",
                        replacement: nil
                    )
                })
            } catch {
                reject("openmedkit_extract_failed", error.localizedDescription, error)
            }
        }
    }

    @objc(deidentify:options:resolver:rejecter:)
    func deidentify(
        _ text: String,
        options: NSDictionary?,
        resolver resolve: @escaping RCTPromiseResolveBlock,
        rejecter reject: @escaping RCTPromiseRejectBlock
    ) {
        queue.async {
            do {
                let runtime = try self.requireRuntime()
                let bridgeOptions = BridgeOptions(options)
                let result = try runtime.deidentify(
                    text,
                    policy: bridgeOptions.policy,
                    confidenceThreshold: bridgeOptions.confidenceThreshold,
                    useSmartMerging: bridgeOptions.useSmartMerging
                )
                resolve([
                    "text": text,
                    "deidentifiedText": result.redactedText,
                    "spans": result.actions.map {
                        Self.spanDictionary(
                            action: $0,
                            text: text,
                            options: bridgeOptions
                        )
                    },
                ])
            } catch {
                reject("openmedkit_deidentify_failed", error.localizedDescription, error)
            }
        }
    }

    private func requireRuntime() throws -> OpenMed {
        guard let runtime else {
            throw BridgeError.modelNotLoaded
        }
        return runtime
    }

    private static func backend(from options: NSDictionary) throws -> OpenMedBackend {
        let modelPath = try requiredString(options, "modelPath")
        let backendName = string(options, "backend") ?? "mlx"

        switch backendName {
        case "mlx":
            return .mlx(modelDirectoryURL: URL(fileURLWithPath: modelPath))
        case "coreml":
            let id2LabelPath = try requiredString(options, "id2LabelPath")
            let tokenizerName =
                string(options, "tokenizerName")
                ?? "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
            let tokenizerFolderPath = string(options, "tokenizerFolderPath")
            return .coreML(
                modelURL: URL(fileURLWithPath: modelPath),
                id2labelURL: URL(fileURLWithPath: id2LabelPath),
                tokenizerName: tokenizerName,
                tokenizerFolderURL: tokenizerFolderPath.map(URL.init(fileURLWithPath:))
            )
        default:
            throw BridgeError.unsupportedBackend(backendName)
        }
    }

    private static func spanDictionary(
        entity: EntityPrediction,
        text: String,
        options: BridgeOptions,
        action: String,
        replacement: String?
    ) -> [String: Any] {
        let canonicalLabel = Policy.canonicalLabel(for: entity.label)
        return spanDictionary(
            label: entity.label,
            canonicalLabel: canonicalLabel,
            start: entity.start,
            end: entity.end,
            confidence: entity.confidence,
            action: action,
            replacement: replacement,
            text: text,
            options: options
        )
    }

    private static func spanDictionary(
        action: DeidentifiedSpanAction,
        text: String,
        options: BridgeOptions
    ) -> [String: Any] {
        spanDictionary(
            label: action.label,
            canonicalLabel: action.canonicalLabel,
            start: action.start,
            end: action.end,
            confidence: action.confidence,
            action: action.action.rawValue,
            replacement: action.replacement,
            text: text,
            options: options
        )
    }

    private static func spanDictionary(
        label: String,
        canonicalLabel: String,
        start: Int,
        end: Int,
        confidence: Float,
        action: String,
        replacement: String?,
        text: String,
        options: BridgeOptions
    ) -> [String: Any] {
        let surface = substring(text, start: start, end: end)
        return [
            "schema_version": 1,
            "doc_id": options.docID,
            "start": start,
            "end": end,
            "text_hash": hmacTextHash(surface, secret: options.hashSecret),
            "entity_type": label,
            "canonical_label": canonicalLabel,
            "policy_label": policyLabel(for: canonicalLabel),
            "regulatory_tags": [],
            "score": confidence,
            "detector": options.detector ?? "openmedkit-ios",
            "evidence": [
                "bridge": "react-native",
                "runtime": "OpenMedKit",
            ],
            "action": action,
            "replacement": replacement ?? NSNull(),
            "reversible_id": NSNull(),
            "section": NSNull(),
            "metadata": options.metadata,
        ]
    }

    private static func substring(_ text: String, start: Int, end: Int) -> String {
        guard start >= 0, end > start, end <= text.count else {
            return ""
        }
        let lowerBound = text.index(text.startIndex, offsetBy: start)
        let upperBound = text.index(text.startIndex, offsetBy: end)
        return String(text[lowerBound..<upperBound])
    }

    private static func hmacTextHash(_ surface: String, secret: String) -> String {
        let key = SymmetricKey(data: Data(secret.utf8))
        let signature = HMAC<SHA256>.authenticationCode(
            for: Data(surface.utf8),
            using: key
        )
        return "hmac-sha256:" + signature.map { String(format: "%02x", $0) }.joined()
    }

    private static func policyLabel(for canonicalLabel: String) -> String {
        if clinicalConceptLabels.contains(canonicalLabel) {
            return "CLINICAL_CONCEPT"
        }
        if quasiIdentifierLabels.contains(canonicalLabel) {
            return "QUASI_IDENTIFIER"
        }
        return "DIRECT_IDENTIFIER"
    }

    private static func requiredString(
        _ options: NSDictionary,
        _ key: String
    ) throws -> String {
        guard let value = string(options, key), !value.trimmingCharacters(
            in: .whitespacesAndNewlines
        ).isEmpty else {
            throw BridgeError.missingOption(key)
        }
        return value
    }

    private static func string(_ options: NSDictionary, _ key: String) -> String? {
        options[key] as? String
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
        "DIET_TYPE",
        "NUTRITION_TARGET",
        "FEEDING_ROUTE",
        "NUTRITIONAL_STATUS",
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
}

private struct BridgeOptions {
    let confidenceThreshold: Float
    let useSmartMerging: Bool
    let docID: String
    let hashSecret: String
    let detector: String?
    let metadata: [String: Any]
    let policy: String

    init(_ options: NSDictionary?) {
        self.confidenceThreshold = options?["confidenceThreshold"] as? Float
            ?? (options?["confidenceThreshold"] as? NSNumber)?.floatValue
            ?? 0.5
        self.useSmartMerging = options?["useSmartMerging"] as? Bool ?? true
        self.docID = options?["docId"] as? String ?? "document"
        self.hashSecret = options?["hashSecret"] as? String ?? "openmedkit-react-native"
        self.detector = options?["detector"] as? String
        self.metadata = options?["metadata"] as? [String: Any] ?? [:]
        self.policy = options?["policy"] as? String ?? Policy.defaultName
    }
}

private enum BridgeError: LocalizedError {
    case missingOption(String)
    case unsupportedBackend(String)
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .missingOption(let key):
            return "missing required OpenMedKit bridge option: \(key)"
        case .unsupportedBackend(let backend):
            return "unsupported OpenMedKit iOS backend: \(backend)"
        case .modelNotLoaded:
            return "OpenMedKit model is not loaded"
        }
    }
}
