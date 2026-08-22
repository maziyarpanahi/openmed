import Foundation
import OpenMedKit

/// Errors raised before extension input or local model assets are processed.
public enum ExtensionRedactionError: Error, Equatable, LocalizedError, Sendable {
    case emptyInput
    case inputTooLarge(actual: Int, limit: Int)
    case inputBytesTooLarge(actual: Int, limit: Int)
    case tooManyInputItems(actual: Int, limit: Int)
    case aggregateInputTooLarge(actual: Int, limit: Int)
    case missingPlainTextInput
    case nonLocalAsset(URL)
    case missingAsset(String)
    case invalidAsset(String)
    case tooManyAssetFiles(limit: Int)
    case unsupportedModelFormat(String)
    case modelAssetsTooLarge(actual: Int64, limit: Int64)
    case invalidRedactionOutput

    public var errorDescription: String? {
        switch self {
        case .emptyInput:
            return "The extension did not receive any text to redact."
        case .inputTooLarge(let actual, let limit):
            return "The selected text has \(actual) characters; the extension limit is \(limit)."
        case .inputBytesTooLarge(let actual, let limit):
            return "The selected text uses \(actual) UTF-8 bytes; the extension limit is \(limit)."
        case .tooManyInputItems(let actual, let limit):
            return "The host supplied \(actual) text items; the extension limit is \(limit)."
        case .aggregateInputTooLarge(let actual, let limit):
            return "The selected text items use \(actual) UTF-8 bytes; the extension limit is \(limit)."
        case .missingPlainTextInput:
            return "The host app did not provide a plain-text extension item."
        case .nonLocalAsset(let url):
            return "Extension model assets must be local files, not \(url.scheme ?? "unknown") URLs."
        case .missingAsset(let path):
            return "A required extension model asset is missing at \(path)."
        case .invalidAsset(let name):
            return "A required extension model asset has an invalid type: \(name)."
        case .tooManyAssetFiles(let limit):
            return "The extension model assets contain more than \(limit) files."
        case .unsupportedModelFormat(let fileName):
            return "The extension requires a precompiled .mlmodelc model; received \(fileName)."
        case .modelAssetsTooLarge(let actual, let limit):
            return "The Nano model assets use \(actual) bytes; the extension limit is \(limit)."
        case .invalidRedactionOutput:
            return "The local redaction runtime returned invalid output."
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
        guard modelAssetBytes >= 0, modelAssetBytes <= Self.maximumModelAssetBytes else {
            throw ExtensionRedactionError.modelAssetsTooLarge(
                actual: modelAssetBytes,
                limit: Self.maximumModelAssetBytes
            )
        }
        let (estimatedPeak, overflow) = modelAssetBytes.addingReportingOverflow(
            Self.runtimeHeadroomBytes
        )
        guard !overflow, estimatedPeak <= Self.maximumEstimatedPeakBytes else {
            throw ExtensionRedactionError.modelAssetsTooLarge(
                actual: overflow ? Int64.max : estimatedPeak,
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
    public static let maximumAssetFileCount = 4_096

    public let modelURL: URL
    public let id2labelURL: URL
    public let tokenizerFolderURL: URL
    public let memoryBudget: NanoModelMemoryBudget

    public init(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerFolderURL: URL
    ) throws {
        for url in [modelURL, id2labelURL, tokenizerFolderURL] {
            guard url.isFileURL else {
                throw ExtensionRedactionError.nonLocalAsset(url)
            }
        }
        let standardizedModelURL = modelURL.standardizedFileURL
        let standardizedID2LabelURL = id2labelURL.standardizedFileURL
        let standardizedTokenizerURL = tokenizerFolderURL.standardizedFileURL
        let assetURLs = [
            standardizedModelURL,
            standardizedID2LabelURL,
            standardizedTokenizerURL,
        ]

        guard standardizedModelURL.pathExtension == "mlmodelc" else {
            throw ExtensionRedactionError.unsupportedModelFormat(
                standardizedModelURL.lastPathComponent
            )
        }

        let fileManager = FileManager.default
        for url in assetURLs where !fileManager.fileExists(atPath: url.path) {
            throw ExtensionRedactionError.missingAsset(url.lastPathComponent)
        }
        try Self.validateAssetType(at: standardizedModelURL, directory: true)
        try Self.validateAssetType(at: standardizedID2LabelURL, directory: false)
        try Self.validateAssetType(at: standardizedTokenizerURL, directory: true)

        for fileName in ["tokenizer.json", "tokenizer_config.json"] {
            let url = standardizedTokenizerURL.appending(path: fileName)
            guard fileManager.fileExists(atPath: url.path) else {
                throw ExtensionRedactionError.missingAsset(url.lastPathComponent)
            }
            try Self.validateAssetType(at: url, directory: false)
        }

        var assetBytes: Int64 = 0
        for url in assetURLs {
            let size = try Self.logicalFileSize(at: url, fileManager: fileManager)
            guard size > 0 else {
                throw ExtensionRedactionError.invalidAsset(url.lastPathComponent)
            }
            let (next, overflow) = assetBytes.addingReportingOverflow(size)
            guard !overflow, next <= NanoModelMemoryBudget.maximumModelAssetBytes else {
                throw ExtensionRedactionError.modelAssetsTooLarge(
                    actual: overflow ? Int64.max : next,
                    limit: NanoModelMemoryBudget.maximumModelAssetBytes
                )
            }
            assetBytes = next
        }

        self.modelURL = standardizedModelURL
        self.id2labelURL = standardizedID2LabelURL
        self.tokenizerFolderURL = standardizedTokenizerURL
        self.memoryBudget = try NanoModelMemoryBudget(modelAssetBytes: assetBytes)
    }

    /// Resolve the expected pre-bundled Nano Core ML model and tokenizer assets.
    public static func bundled(in bundle: Bundle = .main) throws -> Self {
        guard let resources = bundle.resourceURL else {
            throw ExtensionRedactionError.missingAsset(bundle.bundleURL.lastPathComponent)
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
        let keys: Set<URLResourceKey> = [
            .fileSizeKey,
            .isDirectoryKey,
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ]
        let values = try url.resourceValues(forKeys: keys)
        guard values.isSymbolicLink != true else {
            throw ExtensionRedactionError.invalidAsset(url.lastPathComponent)
        }
        guard values.isDirectory == true else {
            return try regularFileSize(values, at: url)
        }

        var enumerationError: Error?
        guard
            let enumerator = fileManager.enumerator(
                at: url,
                includingPropertiesForKeys: Array(keys),
                options: [],
                errorHandler: { _, error in
                    enumerationError = error
                    return false
                }
            )
        else {
            throw ExtensionRedactionError.missingAsset(url.lastPathComponent)
        }

        var total: Int64 = 0
        var fileCount = 0
        for case let childURL as URL in enumerator {
            let childValues = try childURL.resourceValues(forKeys: keys)
            guard childValues.isSymbolicLink != true else {
                throw ExtensionRedactionError.invalidAsset(childURL.lastPathComponent)
            }
            if childValues.isRegularFile == true {
                fileCount += 1
                guard fileCount <= maximumAssetFileCount else {
                    throw ExtensionRedactionError.tooManyAssetFiles(
                        limit: maximumAssetFileCount
                    )
                }
                let size = try regularFileSize(childValues, at: childURL)
                let (next, overflow) = total.addingReportingOverflow(size)
                guard
                    !overflow,
                    next <= NanoModelMemoryBudget.maximumModelAssetBytes
                else {
                    throw ExtensionRedactionError.modelAssetsTooLarge(
                        actual: overflow ? Int64.max : next,
                        limit: NanoModelMemoryBudget.maximumModelAssetBytes
                    )
                }
                total = next
            } else if childValues.isDirectory != true {
                throw ExtensionRedactionError.invalidAsset(childURL.lastPathComponent)
            }
        }
        if enumerationError != nil {
            throw ExtensionRedactionError.invalidAsset(url.lastPathComponent)
        }
        return total
    }

    private static func validateAssetType(at url: URL, directory: Bool) throws {
        let values = try url.resourceValues(
            forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey]
        )
        guard values.isSymbolicLink != true,
            directory ? values.isDirectory == true : values.isRegularFile == true
        else {
            throw ExtensionRedactionError.invalidAsset(url.lastPathComponent)
        }
    }

    private static func regularFileSize(
        _ values: URLResourceValues,
        at url: URL
    ) throws -> Int64 {
        guard values.isRegularFile == true,
            let fileSize = values.fileSize,
            fileSize >= 0
        else {
            throw ExtensionRedactionError.invalidAsset(url.lastPathComponent)
        }
        return Int64(fileSize)
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
    public static let maximumInputUTF8Bytes = 64 * 1_024
    public static let maximumInputItems = 8
    public static let maximumAggregateInputUTF8Bytes = 128 * 1_024
    public static let maximumOutputUTF8Bytes = 256 * 1_024

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
        try Self.validateInput(text)

        let policy = try Policy(named: policyName)
        let result = try redactWithPolicy(text, policy)
        try Self.validateResult(result, for: text, policy: policy)
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

    /// Validate one host-provided text before model loading or inference.
    public static func validateInput(_ text: String) throws {
        guard !text.isEmpty else {
            throw ExtensionRedactionError.emptyInput
        }
        let characterCount = text.count
        guard characterCount <= maximumInputCharacters else {
            throw ExtensionRedactionError.inputTooLarge(
                actual: characterCount,
                limit: maximumInputCharacters
            )
        }
        let utf8Count = text.utf8.count
        guard utf8Count <= maximumInputUTF8Bytes else {
            throw ExtensionRedactionError.inputBytesTooLarge(
                actual: utf8Count,
                limit: maximumInputUTF8Bytes
            )
        }
    }

    /// Validate a bounded batch of host-provided text attachments.
    public static func validateInputs(_ texts: [String]) throws {
        guard !texts.isEmpty else {
            throw ExtensionRedactionError.missingPlainTextInput
        }
        guard texts.count <= maximumInputItems else {
            throw ExtensionRedactionError.tooManyInputItems(
                actual: texts.count,
                limit: maximumInputItems
            )
        }

        var aggregateBytes = 0
        for text in texts {
            try validateInput(text)
            let (nextBytes, overflow) = aggregateBytes.addingReportingOverflow(
                text.utf8.count
            )
            guard !overflow, nextBytes <= maximumAggregateInputUTF8Bytes else {
                throw ExtensionRedactionError.aggregateInputTooLarge(
                    actual: overflow ? Int.max : nextBytes,
                    limit: maximumAggregateInputUTF8Bytes
                )
            }
            aggregateBytes = nextBytes
        }
    }

    private static func validateResult(
        _ result: PolicyDeidentificationResult,
        for text: String,
        policy: Policy
    ) throws {
        let scalarCount = text.unicodeScalars.count
        guard result.policyName == policy.name,
            result.redactedText.utf8.count <= maximumOutputUTF8Bytes,
            result.actions.count <= scalarCount
        else {
            throw ExtensionRedactionError.invalidRedactionOutput
        }

        var previousEnd = 0
        for action in result.actions {
            guard !action.label.isEmpty,
                !action.canonicalLabel.isEmpty,
                action.label.utf8.count <= 256,
                action.canonicalLabel.utf8.count <= 256,
                action.start >= previousEnd,
                action.end > action.start,
                action.end <= scalarCount,
                action.confidence.isFinite,
                action.confidence >= 0,
                action.confidence <= 1,
                (action.replacement?.utf8.count ?? 0) <= maximumOutputUTF8Bytes
            else {
                throw ExtensionRedactionError.invalidRedactionOutput
            }
            previousEnd = action.end
        }
    }
}
