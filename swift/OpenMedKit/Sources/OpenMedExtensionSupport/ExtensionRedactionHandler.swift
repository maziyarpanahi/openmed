import Foundation
import OpenMedKit

/// Extension model assets referenced by role instead of by URL or file name.
public enum ExtensionAssetRole: String, CaseIterable, Sendable {
    case bundleResources = "bundle_resources"
    case model
    case labelMap = "label_map"
    case tokenizerFolder = "tokenizer_folder"
    case tokenizerFile = "tokenizer_file"
}

/// Errors raised before extension input or local model assets are processed.
///
/// Cases carry counts, limits or an ``ExtensionAssetRole``. They never hold
/// host text, an asset URL, path or file name.
public enum ExtensionRedactionError: Error, Equatable, LocalizedError, Sendable {
    case emptyInput
    case inputTooLarge(actual: Int, limit: Int)
    case inputBytesTooLarge(actual: Int, limit: Int)
    case tooManyInputItems(actual: Int, limit: Int)
    case aggregateInputTooLarge(actual: Int, limit: Int)
    case missingPlainTextInput
    case nonLocalAsset(ExtensionAssetRole)
    case missingAsset(ExtensionAssetRole)
    case invalidAsset(ExtensionAssetRole)
    case tooManyAssetFiles(limit: Int)
    case unsupportedModelFormat
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
        case .nonLocalAsset(let role):
            return "Extension model assets must be local files (\(role.rawValue))."
        case .missingAsset(let role):
            return "A required extension model asset is missing (\(role.rawValue))."
        case .invalidAsset(let role):
            return "A required extension model asset has an invalid type (\(role.rawValue))."
        case .tooManyAssetFiles(let limit):
            return "The extension model assets contain more than \(limit) files."
        case .unsupportedModelFormat:
            return "The extension requires a precompiled .mlmodelc model."
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

/// Aggregate limits applied while decoding host-provided extension items.
public enum ExtensionInputBudget {
    public static let maximumItems = 8
    public static let maximumAggregateUTF8Bytes = 128 * 1_024

    /// Validate a bounded collection before retaining or redacting it.
    public static func validate(_ texts: [String]) throws {
        guard !texts.isEmpty else {
            throw ExtensionRedactionError.missingPlainTextInput
        }
        guard texts.count <= maximumItems else {
            throw ExtensionRedactionError.tooManyInputItems(
                actual: texts.count,
                limit: maximumItems
            )
        }

        var aggregateBytes = 0
        for text in texts {
            try ExtensionRedactionHandler.validateInput(text)
            let (nextBytes, overflow) = aggregateBytes.addingReportingOverflow(
                text.utf8.count
            )
            guard !overflow, nextBytes <= maximumAggregateUTF8Bytes else {
                throw ExtensionRedactionError.aggregateInputTooLarge(
                    actual: overflow ? Int.max : nextBytes,
                    limit: maximumAggregateUTF8Bytes
                )
            }
            aggregateBytes = nextBytes
        }
    }
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
        let roles: [(URL, ExtensionAssetRole)] = [
            (modelURL, .model),
            (id2labelURL, .labelMap),
            (tokenizerFolderURL, .tokenizerFolder),
        ]
        for (url, role) in roles {
            guard url.isFileURL else {
                throw ExtensionRedactionError.nonLocalAsset(role)
            }
        }
        let standardizedModelURL = modelURL.standardizedFileURL
        let standardizedID2LabelURL = id2labelURL.standardizedFileURL
        let standardizedTokenizerURL = tokenizerFolderURL.standardizedFileURL
        let assetURLs: [(URL, ExtensionAssetRole)] = [
            (standardizedModelURL, .model),
            (standardizedID2LabelURL, .labelMap),
            (standardizedTokenizerURL, .tokenizerFolder),
        ]

        guard standardizedModelURL.pathExtension == "mlmodelc" else {
            throw ExtensionRedactionError.unsupportedModelFormat
        }

        let fileManager = FileManager.default
        for (url, role) in assetURLs where !fileManager.fileExists(atPath: url.path) {
            throw ExtensionRedactionError.missingAsset(role)
        }
        try Self.validateAssetType(at: standardizedModelURL, directory: true, role: .model)
        try Self.validateAssetType(at: standardizedID2LabelURL, directory: false, role: .labelMap)
        try Self.validateAssetType(
            at: standardizedTokenizerURL,
            directory: true,
            role: .tokenizerFolder
        )

        for fileName in ["tokenizer.json", "tokenizer_config.json"] {
            let url = standardizedTokenizerURL.appending(path: fileName)
            guard fileManager.fileExists(atPath: url.path) else {
                throw ExtensionRedactionError.missingAsset(.tokenizerFile)
            }
            try Self.validateAssetType(at: url, directory: false, role: .tokenizerFile)
        }

        var assetBytes: Int64 = 0
        for (url, role) in assetURLs {
            let size = try Self.logicalFileSize(at: url, role: role, fileManager: fileManager)
            guard size > 0 else {
                throw ExtensionRedactionError.invalidAsset(role)
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
            throw ExtensionRedactionError.missingAsset(.bundleResources)
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

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
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
    #endif

    private static func logicalFileSize(
        at url: URL,
        role: ExtensionAssetRole,
        fileManager: FileManager
    ) throws -> Int64 {
        let keys: Set<URLResourceKey> = [
            .fileSizeKey,
            .isDirectoryKey,
            .isRegularFileKey,
            .isSymbolicLinkKey,
        ]
        let values = try resourceValues(of: url, keys: keys, role: role)
        guard values.isSymbolicLink != true else {
            throw ExtensionRedactionError.invalidAsset(role)
        }
        guard values.isDirectory == true else {
            return try regularFileSize(values, role: role)
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
            throw ExtensionRedactionError.missingAsset(role)
        }

        var total: Int64 = 0
        var fileCount = 0
        for case let childURL as URL in enumerator {
            let childValues = try resourceValues(of: childURL, keys: keys, role: role)
            guard childValues.isSymbolicLink != true else {
                throw ExtensionRedactionError.invalidAsset(role)
            }
            if childValues.isRegularFile == true {
                fileCount += 1
                guard fileCount <= maximumAssetFileCount else {
                    throw ExtensionRedactionError.tooManyAssetFiles(
                        limit: maximumAssetFileCount
                    )
                }
                let size = try regularFileSize(childValues, role: role)
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
                throw ExtensionRedactionError.invalidAsset(role)
            }
        }
        if enumerationError != nil {
            throw ExtensionRedactionError.invalidAsset(role)
        }
        return total
    }

    private static func validateAssetType(
        at url: URL,
        directory: Bool,
        role: ExtensionAssetRole
    ) throws {
        let values = try resourceValues(
            of: url,
            keys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey],
            role: role
        )
        guard values.isSymbolicLink != true,
            directory ? values.isDirectory == true : values.isRegularFile == true
        else {
            throw ExtensionRedactionError.invalidAsset(role)
        }
    }

    /// Foundation resource errors carry the asset path, so report the role.
    private static func resourceValues(
        of url: URL,
        keys: Set<URLResourceKey>,
        role: ExtensionAssetRole
    ) throws -> URLResourceValues {
        do {
            return try url.resourceValues(forKeys: keys)
        } catch {
            throw ExtensionRedactionError.invalidAsset(role)
        }
    }

    private static func regularFileSize(
        _ values: URLResourceValues,
        role: ExtensionAssetRole
    ) throws -> Int64 {
        guard values.isRegularFile == true,
            let fileSize = values.fileSize,
            fileSize >= 0
        else {
            throw ExtensionRedactionError.invalidAsset(role)
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
    public static let maximumInputItems = ExtensionInputBudget.maximumItems
    public static let maximumAggregateInputUTF8Bytes =
        ExtensionInputBudget.maximumAggregateUTF8Bytes
    public static let maximumOutputUTF8Bytes = 256 * 1_024
    public static let defaultPolicyName = Policy.defaultName

    public typealias Redact = (String, Policy) throws -> PolicyDeidentificationResult

    private let redactWithPolicy: Redact

    /// Create a handler around a test or app-provided OpenMedKit redaction function.
    public init(redact: @escaping Redact) {
        self.redactWithPolicy = redact
    }

    #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
        /// Load one validated Nano runtime for the lifetime of this handler.
        public convenience init(configuration: NanoModelConfiguration) throws {
            let runtime = try configuration.makeRuntime()
            self.init { text, policy in
                try runtime.deidentify(text, policy: policy)
            }
        }

        /// Use one runtime and release its cache after either success or failure.
        package static func withRuntime<T>(
            configuration: NanoModelConfiguration,
            operation: (ExtensionRedactionHandler) throws -> T
        ) throws -> T {
            try withCleanup(
                makeHandler: { try ExtensionRedactionHandler(configuration: configuration) },
                cleanup: OpenMed.clearRuntimeMemoryCache,
                operation: operation
            )
        }
    #endif

    package static func withCleanup<T>(
        makeHandler: () throws -> ExtensionRedactionHandler,
        cleanup: () -> Void,
        operation: (ExtensionRedactionHandler) throws -> T
    ) rethrows -> T {
        do {
            let output: T
            do {
                let handler = try makeHandler()
                output = try operation(handler)
            }
            cleanup()
            return output
        } catch {
            cleanup()
            throw error
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

    /// Redact a bounded extension request one item at a time.
    public func redact(
        _ texts: [String],
        policyName: String = Policy.defaultName
    ) throws -> [ExtensionRedactionOutput] {
        try Self.validateInputs(texts)
        return try texts.map { try redact($0, policyName: policyName) }
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
        try ExtensionInputBudget.validate(texts)
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
