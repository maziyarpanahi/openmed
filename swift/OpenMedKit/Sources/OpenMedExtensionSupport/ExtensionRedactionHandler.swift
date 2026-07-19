import Foundation
import OpenMedKit

/// Errors raised before extension input or local model assets are processed.
public enum ExtensionRedactionError: Error, Equatable, LocalizedError, Sendable {
    case emptyInput
    case inputTooLarge(actual: Int, limit: Int)
    case missingPlainTextInput
    case nonLocalAsset(URL)
    case missingAsset(String)
    case unsupportedModelFormat(String)
    case modelAssetsTooLarge(actual: Int64, limit: Int64)

    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "The extension did not receive any text to redact."
        case .inputTooLarge(let actual, let limit):
            return "The selected text has \(actual) characters; the extension limit is \(limit)."
        case .missingPlainTextInput:
            return "The host app did not provide a plain-text extension item."
        case .nonLocalAsset(let url):
            return "Extension model assets must be local files, not \(url.scheme ?? "unknown") URLs."
        case .missingAsset(let path):
            return "A required extension model asset is missing at \(path)."
        case .unsupportedModelFormat(let fileName):
            return "The extension requires a precompiled .mlmodelc model; received \(fileName)."
        case .modelAssetsTooLarge(let actual, let limit):
            return "The Nano model assets use \(actual) bytes; the extension limit is \(limit)."
        }
    }
}

/// Static security properties enforced by the extension support layer.
public enum ExtensionSecurityPolicy {
    /// Extension inference is intentionally local-only.
    public static let allowsNetworkAccess = false

    /// The only URL scheme accepted for model, label, and tokenizer assets.
    public static let modelAssetURLScheme = "file"
}

/// Conservative memory limits for an iOS Share or Action extension model.
public struct NanoModelMemoryBudget: Equatable, Sendable {
    public static let extensionWorkingSetEnvelopeBytes: Int64 = 120 * 1_024 * 1_024
    public static let maximumEstimatedPeakBytes: Int64 = 96 * 1_024 * 1_024
    public static let maximumModelAssetBytes: Int64 = 40 * 1_024 * 1_024
    public static let runtimeHeadroomBytes: Int64 = 56 * 1_024 * 1_024

    public let modelAssetBytes: Int64

    public var estimatedPeakBytes: Int64 {
        modelAssetBytes + Self.runtimeHeadroomBytes
    }

    public init(modelAssetBytes: Int64) throws {
        guard modelAssetBytes >= 0,
            modelAssetBytes <= Self.maximumModelAssetBytes,
            modelAssetBytes + Self.runtimeHeadroomBytes <= Self.maximumEstimatedPeakBytes
        else {
            throw ExtensionRedactionError.modelAssetsTooLarge(
                actual: modelAssetBytes,
                limit: Self.maximumModelAssetBytes
            )
        }
        self.modelAssetBytes = modelAssetBytes
    }
}

/// Local Core ML assets accepted by the extension-safe model loader.
public struct NanoModelConfiguration: Sendable {
    public static let resourceDirectoryName = "OpenMedPIINano"
    public static let compiledModelName = "OpenMedPIINano"
    public static let maximumSequenceLength = 256

    public let modelURL: URL
    public let id2labelURL: URL
    public let tokenizerFolderURL: URL
    public let memoryBudget: NanoModelMemoryBudget

    public init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerFolderURL: URL
    ) throws {
        let assetURLs = [modelURL, id2labelURL, tokenizerFolderURL]
        for url in assetURLs {
            guard url.isFileURL else {
                throw ExtensionRedactionError.nonLocalAsset(url)
            }
        }

        guard modelURL.pathExtension == "mlmodelc" else {
            throw ExtensionRedactionError.unsupportedModelFormat(modelURL.lastPathComponent)
        }

        let fileManager = FileManager.default
        for url in assetURLs where !fileManager.fileExists(atPath: url.path) {
            throw ExtensionRedactionError.missingAsset(url.path)
        }

        for fileName in ["tokenizer.json", "tokenizer_config.json"] {
            let url = tokenizerFolderURL.appending(path: fileName)
            guard fileManager.fileExists(atPath: url.path) else {
                throw ExtensionRedactionError.missingAsset(url.path)
            }
        }

        let assetBytes = try assetURLs.reduce(Int64(0)) { total, url in
            try total + Self.logicalFileSize(at: url, fileManager: fileManager)
        }

        self.modelURL = modelURL.standardizedFileURL
        self.id2labelURL = id2labelURL.standardizedFileURL
        self.tokenizerFolderURL = tokenizerFolderURL.standardizedFileURL
        self.memoryBudget = try NanoModelMemoryBudget(modelAssetBytes: assetBytes)
    }

    /// Resolve the expected pre-bundled Nano Core ML model and tokenizer assets.
    public static func bundled(in bundle: Bundle = .main) throws -> Self {
        guard let resources = bundle.resourceURL else {
            throw ExtensionRedactionError.missingAsset(bundle.bundlePath)
        }
        let directory = resources.appending(
            path: resourceDirectoryName,
            directoryHint: .isDirectory
        )
        return try Self(
            modelURL: directory.appending(
                path: "\(compiledModelName).mlmodelc",
                directoryHint: .isDirectory
            ),
            id2labelURL: directory.appending(path: "id2label.json"),
            tokenizerFolderURL: directory.appending(
                path: "tokenizer",
                directoryHint: .isDirectory
            )
        )
    }

    fileprivate func makeRuntime() throws -> OpenMed {
        try OpenMed(
            backend: .coreML(
                modelURL: modelURL,
                id2labelURL: id2labelURL,
                tokenizerName: Self.compiledModelName,
                tokenizerFolderURL: tokenizerFolderURL
            ),
            maxSeqLength: Self.maximumSequenceLength,
            allowNetworkAccess: ExtensionSecurityPolicy.allowsNetworkAccess
        )
    }

    private static func logicalFileSize(
        at url: URL,
        fileManager: FileManager
    ) throws -> Int64 {
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .fileSizeKey])
        guard values.isDirectory == true else {
            return Int64(values.fileSize ?? 0)
        }

        let keys: [URLResourceKey] = [.isRegularFileKey, .fileSizeKey]
        guard
            let enumerator = fileManager.enumerator(
                at: url,
                includingPropertiesForKeys: keys,
                options: [.skipsHiddenFiles]
            )
        else {
            throw ExtensionRedactionError.missingAsset(url.path)
        }

        var total: Int64 = 0
        for case let childURL as URL in enumerator {
            let childValues = try childURL.resourceValues(forKeys: Set(keys))
            if childValues.isRegularFile == true {
                total += Int64(childValues.fileSize ?? 0)
            }
        }
        return total
    }
}

/// A redaction action returned to the host while preserving original offsets.
public struct ExtensionRedactedSpan: Equatable, Sendable {
    public let label: String
    public let canonicalLabel: String
    public let action: PolicyAction
    public let start: Int
    public let end: Int
    public let confidence: Float
    public let replacement: String?

    public init(
        label: String,
        canonicalLabel: String,
        action: PolicyAction,
        start: Int,
        end: Int,
        confidence: Float,
        replacement: String?
    ) {
        self.label = label
        self.canonicalLabel = canonicalLabel
        self.action = action
        self.start = start
        self.end = end
        self.confidence = confidence
        self.replacement = replacement
    }
}

/// Redacted extension output with action spans relative to the original input.
public struct ExtensionRedactionOutput: Equatable, Sendable {
    public let redactedText: String
    public let policyName: String
    public let spans: [ExtensionRedactedSpan]

    public init(
        redactedText: String,
        policyName: String,
        spans: [ExtensionRedactedSpan]
    ) {
        self.redactedText = redactedText
        self.policyName = policyName
        self.spans = spans
    }
}

/// Applies OpenMedKit policy redaction to plain text supplied by a host app.
public final class ExtensionRedactionHandler {
    public static let maximumInputCharacters = 16_384

    public typealias Redact = (String, Policy) throws -> PolicyDeidentificationResult

    private let redactWithPolicy: Redact

    /// Create a handler around a test or app-provided OpenMedKit redaction function.
    public init(redact: @escaping Redact) {
        self.redactWithPolicy = redact
    }

    /// Load one validated Nano runtime for the lifetime of this handler.
    public convenience init(configuration: NanoModelConfiguration) throws {
        let runtime = try configuration.makeRuntime()
        self.init { text, policy in
            try runtime.deidentify(text, policy: policy)
        }
    }

    /// Redact a selected text item with a bundled policy profile.
    public func redact(
        _ text: String,
        policyName: String = Policy.defaultName
    ) throws -> ExtensionRedactionOutput {
        guard !text.isEmpty else {
            throw ExtensionRedactionError.emptyInput
        }
        guard text.count <= Self.maximumInputCharacters else {
            throw ExtensionRedactionError.inputTooLarge(
                actual: text.count,
                limit: Self.maximumInputCharacters
            )
        }

        let policy = try Policy(named: policyName)
        let result = try redactWithPolicy(text, policy)
        let spans = result.actions.map { action in
            ExtensionRedactedSpan(
                label: action.label,
                canonicalLabel: action.canonicalLabel,
                action: action.action,
                start: action.start,
                end: action.end,
                confidence: action.confidence,
                replacement: action.replacement
            )
        }
        return ExtensionRedactionOutput(
            redactedText: result.redactedText,
            policyName: result.policyName,
            spans: spans
        )
    }
}
