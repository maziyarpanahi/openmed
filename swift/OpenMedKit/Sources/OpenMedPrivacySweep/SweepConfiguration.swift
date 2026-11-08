import Foundation

/// Value-free class of an app-writable storage root.
public enum AppStorageClass: String, CaseIterable, Codable, Comparable, Sendable {
    case documents
    case applicationSupport = "application_support"
    case caches
    case preferences
    /// `Library` children other than Application Support, Caches and Preferences.
    case library
    case temporary = "tmp"
    case appGroup = "app_group"

    public static func < (lhs: AppStorageClass, rhs: AppStorageClass) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// One directory tree to sweep, labelled with its value-free storage class.
public struct SweepRoot: Sendable {
    public let url: URL
    public let storageClass: AppStorageClass
    /// Direct children of ``url`` to skip because another root covers them.
    public let excludedChildNames: Set<String>

    public init(url: URL, storageClass: AppStorageClass, excludedChildNames: Set<String> = []) {
        self.url = url
        self.storageClass = storageClass
        self.excludedChildNames = excludedChildNames
    }
}

/// The bound that stopped or shortened a sweep.
public enum SweepBound: String, CaseIterable, Codable, Comparable, Sendable {
    case maxFiles = "max_files"
    case maxDepth = "max_depth"
    case maxBytesPerFile = "max_bytes_per_file"
    case maxTotalBytes = "max_total_bytes"
    case maxDuration = "max_duration"
    case maxLogEntries = "max_log_entries"
    case maxLogBytes = "max_log_bytes"

    public static func < (lhs: SweepBound, rhs: SweepBound) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// File, byte, depth, time and log bounds for one sweep.
///
/// Reaching a bound never fails silently: it is listed in
/// ``PHISentinelSweepReport/boundsReached`` and makes the report non-exhaustive.
public struct SweepBounds: Equatable, Codable, Sendable {
    /// Regular files visited across all roots, including allow-listed files.
    public var maxFiles: Int
    /// Directory levels below a root that are entered; `0` scans only direct children.
    public var maxDepth: Int
    /// Content bytes read from one file; later bytes are not scanned.
    public var maxBytesPerFile: Int
    /// Content bytes read across all files.
    public var maxTotalBytes: Int
    /// Wall-clock seconds for files and logs together, measured by the injected clock.
    public var maxDuration: TimeInterval
    /// Log entries read across all log readers.
    public var maxLogEntries: Int
    /// UTF-8 log bytes read across all log readers.
    public var maxLogBytes: Int

    public init(
        maxFiles: Int,
        maxDepth: Int,
        maxBytesPerFile: Int,
        maxTotalBytes: Int,
        maxDuration: TimeInterval,
        maxLogEntries: Int,
        maxLogBytes: Int
    ) {
        self.maxFiles = maxFiles
        self.maxDepth = maxDepth
        self.maxBytesPerFile = maxBytesPerFile
        self.maxTotalBytes = maxTotalBytes
        self.maxDuration = maxDuration
        self.maxLogEntries = maxLogEntries
        self.maxLogBytes = maxLogBytes
    }

    /// Defaults for a test-host app container.
    public static let standard = SweepBounds(
        maxFiles: 5_000,
        maxDepth: 12,
        maxBytesPerFile: 8_388_608,
        maxTotalBytes: 134_217_728,
        maxDuration: 30,
        maxLogEntries: 20_000,
        maxLogBytes: 16_777_216
    )

    /// Hard ceilings no configuration may exceed.
    public static let ceiling = SweepBounds(
        maxFiles: 100_000,
        maxDepth: 64,
        maxBytesPerFile: 268_435_456,
        maxTotalBytes: 1_073_741_824,
        maxDuration: 600,
        maxLogEntries: 1_000_000,
        maxLogBytes: 268_435_456
    )

    func validate() throws {
        let ceiling = Self.ceiling
        guard (1...ceiling.maxFiles).contains(maxFiles),
            (0...ceiling.maxDepth).contains(maxDepth),
            (1...ceiling.maxBytesPerFile).contains(maxBytesPerFile),
            (1...ceiling.maxTotalBytes).contains(maxTotalBytes),
            maxDuration > 0, maxDuration <= ceiling.maxDuration,
            (1...ceiling.maxLogEntries).contains(maxLogEntries),
            (1...ceiling.maxLogBytes).contains(maxLogBytes)
        else {
            throw PHISentinelSweepError.boundsInvalid
        }
    }
}

/// Explicit model-artifact allow-list: matching files keep their names swept but
/// their contents skipped, because weights and vocabularies hold arbitrary bytes.
public struct ArtifactAllowList: Equatable, Sendable {
    /// A storage-class-relative path prefix, matched on whole `/` components.
    public struct PathPrefix: Hashable, Sendable {
        public let storageClass: AppStorageClass
        public let relativePath: String

        public init(storageClass: AppStorageClass, relativePath: String) {
            self.storageClass = storageClass
            self.relativePath = relativePath
        }
    }

    /// Lowercased extensions without a dot, matched on any path component.
    public let fileExtensions: Set<String>
    public let pathPrefixes: Set<PathPrefix>

    public init(fileExtensions: Set<String> = [], pathPrefixes: Set<PathPrefix> = []) {
        self.fileExtensions = Set(fileExtensions.map { $0.lowercased() })
        self.pathPrefixes = pathPrefixes
    }

    /// Nothing is allow-listed: every file's contents are swept.
    public static let none = ArtifactAllowList()

    /// Model bundles and OpenMedKit's own model and tokenizer caches.
    public static let modelArtifacts = ArtifactAllowList(
        fileExtensions: ["gguf", "mlmodel", "mlmodelc", "mlpackage", "onnx", "ort", "safetensors", "tflite"],
        pathPrefixes: [
            PathPrefix(storageClass: .caches, relativePath: "OpenMed/MLXModels"),
            PathPrefix(storageClass: .caches, relativePath: "OpenMed/TokenizerAssets"),
            PathPrefix(storageClass: .caches, relativePath: "OpenMed/PreparedTokenizerAssets"),
        ]
    )

    func allows(components: [String], storageClass: AppStorageClass) -> Bool {
        for component in components {
            let suffix = (component as NSString).pathExtension.lowercased()
            if !suffix.isEmpty && fileExtensions.contains(suffix) {
                return true
            }
        }
        let relative = components.joined(separator: "/")
        return pathPrefixes.contains { prefix in
            prefix.storageClass == storageClass
                && (relative == prefix.relativePath || relative.hasPrefix(prefix.relativePath + "/"))
        }
    }
}

/// Value-free class of the log stream a reader returns.
public enum ProcessLogSource: String, CaseIterable, Codable, Comparable, Sendable {
    /// The unified log for the current process (`OSLogStore`).
    case osLog = "os_log"
    /// Lines captured by an injected sink or fake reader.
    case captured

    public static func < (lhs: ProcessLogSource, rhs: ProcessLogSource) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Reads the current process's own log entries for a sweep.
///
/// Implementations must only read; the sweep never logs what it reads.
public protocol ProcessLogReader {
    var source: ProcessLogSource { get }

    /// Visit entries in order until `visit` returns `false` or entries run out.
    func readEntries(_ visit: (String) -> Bool) throws
}

/// A reader over an in-memory list of log lines, for tests and captured sinks.
public struct CapturedLogReader: ProcessLogReader {
    public let source: ProcessLogSource
    private let lines: [String]

    public init(lines: [String], source: ProcessLogSource = .captured) {
        self.lines = lines
        self.source = source
    }

    public func readEntries(_ visit: (String) -> Bool) throws {
        for line in lines where !visit(line) {
            return
        }
    }
}
