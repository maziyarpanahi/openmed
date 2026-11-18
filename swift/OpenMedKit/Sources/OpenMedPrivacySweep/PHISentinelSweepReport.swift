import Foundation

/// Value-free file category derived from a file name's extension.
public enum SweepFileKind: String, CaseIterable, Codable, Comparable, Sendable {
    case plist
    case json
    case xml
    case sqlite
    case text
    case other

    /// Classify a file by extension only; the name itself is never reported.
    public init(fileName: String) {
        switch (fileName as NSString).pathExtension.lowercased() {
        case "plist":
            self = .plist
        case "json", "jsonl", "ndjson":
            self = .json
        case "xml":
            self = .xml
        case "db", "db-shm", "db-wal", "sqlite", "sqlite-shm", "sqlite-wal", "sqlite3", "wal", "shm":
            self = .sqlite
        case "csv", "log", "md", "txt":
            self = .text
        default:
            self = .other
        }
    }

    public static func < (lhs: SweepFileKind, rhs: SweepFileKind) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Where a sentinel was found.
public enum SweepLocation: String, CaseIterable, Codable, Comparable, Sendable {
    case fileName = "file_name"
    case fileContent = "file_content"
    case logEntry = "log_entry"

    public static func < (lhs: SweepLocation, rhs: SweepLocation) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}

/// Aggregated, value-free evidence that a sentinel survived somewhere.
///
/// Carries the sentinel identifier, never its value, the matched bytes, the file
/// name or a path.
public struct SweepFinding: Codable, Hashable, Comparable, Sendable {
    public let sentinelIdentifier: String
    public let location: SweepLocation
    /// An ``AppStorageClass`` raw value for files, a ``ProcessLogSource`` raw value for logs.
    public let surface: String
    /// File category for file findings; `nil` for log entries.
    public let fileKind: SweepFileKind?
    public let encoding: SentinelEncoding
    /// Matching files, names or log entries.
    public let itemCount: Int
    /// Scanned bytes of the matching items (name bytes for name matches).
    public let itemBytes: Int

    public static func < (lhs: SweepFinding, rhs: SweepFinding) -> Bool {
        lhs.sortKey < rhs.sortKey
    }

    var sortKey: String {
        [sentinelIdentifier, location.rawValue, surface, fileKind?.rawValue ?? "", encoding.rawValue]
            .joined(separator: "\u{1F}")
    }
}

/// Per-root counters, labelled only by storage class.
public struct SweepRootSummary: Codable, Equatable, Sendable {
    public let storageClass: AppStorageClass
    public let present: Bool
    public let filesScanned: Int
    public let bytesScanned: Int
}

/// Whether a log reader was read to completion, failed or was skipped by a bound.
public enum SweepLogStatus: String, Codable, Sendable {
    case read
    case unreadable
    case skipped
}

/// Per-reader counters, labelled only by log source.
public struct SweepLogSummary: Codable, Equatable, Sendable {
    public let source: ProcessLogSource
    public let status: SweepLogStatus
    public let entriesScanned: Int
    public let bytesScanned: Int
}

/// Value-free result of one sentinel sweep.
///
/// Contains counts, storage classes, file kinds, log sources, encodings and
/// sentinel identifiers only. It never contains sentinel values, file contents,
/// file names or absolute paths, so it is safe to attach to CI artifacts.
public struct PHISentinelSweepReport: Codable, Equatable, Sendable, CustomStringConvertible {
    public static let schemaIdentifier = "openmed.phi_sentinel_sweep.v1"

    public let schema: String
    public let sentinelIdentifiers: [String]
    public let encodings: [SentinelEncoding]
    public let bounds: SweepBounds
    public let roots: [SweepRootSummary]
    public let logs: [SweepLogSummary]
    public let filesScanned: Int
    public let bytesScanned: Int
    public let allowListedFiles: Int
    public let truncatedFiles: Int
    public let skippedSymbolicLinks: Int
    public let skippedSpecialFiles: Int
    public let unreadableItems: Int
    public let logEntriesScanned: Int
    public let logBytesScanned: Int
    public let boundsReached: [SweepBound]
    public let findings: [SweepFinding]
    /// `true` when no sentinel was found in any scanned name, file or log entry.
    public let clean: Bool
    /// `true` when no bound was reached, nothing was unreadable and every log reader was read.
    public let exhaustive: Bool

    /// Sorted-key JSON with snake_case keys for CI artifacts.
    public func jsonData() throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return try encoder.encode(self)
    }

    public var description: String {
        let findingText = findings.map { finding in
            let kind = finding.fileKind.map { "/\($0.rawValue)" } ?? ""
            return "\(finding.sentinelIdentifier) \(finding.location.rawValue) "
                + "\(finding.surface)\(kind) \(finding.encoding.rawValue) x\(finding.itemCount)"
        }
        return "PHISentinelSweepReport(clean: \(clean), exhaustive: \(exhaustive), "
            + "findings: [\(findingText.joined(separator: ", "))], "
            + "bounds_reached: [\(boundsReached.map(\.rawValue).joined(separator: ", "))], "
            + "files: \(filesScanned), bytes: \(bytesScanned), "
            + "allow_listed: \(allowListedFiles), unreadable: \(unreadableItems), "
            + "log_entries: \(logEntriesScanned))"
    }
}
