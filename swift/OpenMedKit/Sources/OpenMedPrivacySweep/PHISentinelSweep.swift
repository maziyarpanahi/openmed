import Foundation

/// Sweeps app-writable directories and the process's own logs for planted sentinels.
///
/// Run a workload with ``PHISentinel`` values as inputs, then call ``run()``. The
/// sweep only reads: it walks each ``SweepRoot`` without following symbolic links,
/// matches file names and up to ``SweepBounds/maxBytesPerFile`` content bytes per
/// file against every ``SentinelEncoding`` form of every sentinel, and then reads
/// each ``ProcessLogReader``. The returned report is value-free.
///
/// This is test support. The `OpenMedPrivacySweep` target is not a package product,
/// so it is never vended to apps that depend on OpenMedKit.
public struct PHISentinelSweep {
    /// Bytes read per file read call; matches spanning chunks are still found.
    static let chunkSize = 1_048_576

    public let sentinels: [PHISentinel]
    public let roots: [SweepRoot]
    public let bounds: SweepBounds
    public let allowList: ArtifactAllowList
    private let logReaders: [any ProcessLogReader]
    private let clock: () -> TimeInterval
    let needles: [SentinelNeedle]

    /// Configure a sweep.
    ///
    /// - Parameters:
    ///   - sentinels: One to ``PHISentinel/maximumCount`` sentinels with unique identifiers.
    ///   - roots: Directory trees to sweep; missing roots are reported, not errors.
    ///   - logReaders: Readers for this process's own log streams.
    ///   - bounds: File, byte, depth, time and log bounds, within ``SweepBounds/ceiling``.
    ///   - allowList: Model artifacts whose contents are skipped (names are still swept).
    ///   - clock: Monotonic seconds used for ``SweepBounds/maxDuration``.
    /// - Throws: A value-free ``PHISentinelSweepError``.
    public init(
        sentinels: [PHISentinel],
        roots: [SweepRoot],
        logReaders: [any ProcessLogReader] = [],
        bounds: SweepBounds = .standard,
        allowList: ArtifactAllowList = .none,
        clock: @escaping () -> TimeInterval = { ProcessInfo.processInfo.systemUptime }
    ) throws {
        guard !sentinels.isEmpty else {
            throw PHISentinelSweepError.sentinelsEmpty
        }
        guard sentinels.count <= PHISentinel.maximumCount else {
            throw PHISentinelSweepError.sentinelsTooMany
        }
        guard Set(sentinels.map(\.identifier)).count == sentinels.count else {
            throw PHISentinelSweepError.sentinelIdentifierDuplicate
        }
        try bounds.validate()
        self.sentinels = sentinels
        self.roots = roots
        self.logReaders = logReaders
        self.bounds = bounds
        self.allowList = allowList
        self.clock = clock
        self.needles = sentinels.enumerated().flatMap { index, sentinel in
            SentinelNeedles.make(for: sentinel.plantedValue, sentinelIndex: index)
        }
    }

    /// Run the sweep and return its value-free report.
    public func run() -> PHISentinelSweepReport {
        let run = SweepRun(
            sentinels: sentinels,
            needles: needles,
            bounds: bounds,
            allowList: allowList,
            clock: clock
        )
        for root in roots {
            run.sweep(root)
        }
        for reader in logReaders {
            run.sweep(reader)
        }
        return run.report()
    }
}

private struct FindingKey: Hashable {
    let sentinelIndex: Int
    let encoding: SentinelEncoding
    let location: SweepLocation
    let surface: String
    let fileKind: SweepFileKind?
}

private final class SweepRun {
    private static let resourceKeys: [URLResourceKey] = [
        .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey,
    ]

    private let sentinels: [PHISentinel]
    private let needles: [SentinelNeedle]
    private let bounds: SweepBounds
    private let allowList: ArtifactAllowList
    private let clock: () -> TimeInterval
    private let deadline: TimeInterval
    private let carryLength: Int
    private let fileManager = FileManager()

    private var boundsReached = Set<SweepBound>()
    private var findings: [FindingKey: (count: Int, bytes: Int)] = [:]
    private var rootSummaries: [SweepRootSummary] = []
    private var logSummaries: [SweepLogSummary] = []
    private var filesVisited = 0
    private var filesScanned = 0
    private var bytesScanned = 0
    private var allowListedFiles = 0
    private var truncatedFiles = 0
    private var skippedSymbolicLinks = 0
    private var skippedSpecialFiles = 0
    private var unreadableItems = 0
    private var logEntriesScanned = 0
    private var logBytesScanned = 0
    private var filesStopped = false
    private var timeExpired = false

    init(
        sentinels: [PHISentinel],
        needles: [SentinelNeedle],
        bounds: SweepBounds,
        allowList: ArtifactAllowList,
        clock: @escaping () -> TimeInterval
    ) {
        self.sentinels = sentinels
        self.needles = needles
        self.bounds = bounds
        self.allowList = allowList
        self.clock = clock
        self.deadline = clock() + bounds.maxDuration
        self.carryLength = max(0, (needles.map(\.bytes.count).max() ?? 1) - 1)
    }

    private func outOfTime() -> Bool {
        if timeExpired || clock() >= deadline {
            timeExpired = true
            boundsReached.insert(.maxDuration)
        }
        return timeExpired
    }

    // MARK: Files

    func sweep(_ root: SweepRoot) {
        let filesBefore = filesScanned
        let bytesBefore = bytesScanned
        var isDirectory: ObjCBool = false
        let present =
            fileManager.fileExists(atPath: root.url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
        if present && !filesStopped {
            walk(root)
        }
        rootSummaries.append(
            SweepRootSummary(
                storageClass: root.storageClass,
                present: present,
                filesScanned: filesScanned - filesBefore,
                bytesScanned: bytesScanned - bytesBefore
            )
        )
    }

    private func walk(_ root: SweepRoot) {
        var stack: [(url: URL, depth: Int, components: [String])] = [(root.url, 0, [])]
        while let directory = stack.popLast() {
            if filesStopped || outOfTime() {
                filesStopped = true
                return
            }
            let children: [URL]
            do {
                children = try fileManager.contentsOfDirectory(
                    at: directory.url,
                    includingPropertiesForKeys: Self.resourceKeys,
                    options: []
                )
            } catch {
                unreadableItems += 1
                continue
            }
            var subdirectories: [(url: URL, depth: Int, components: [String])] = []
            for child in children.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
                let name = child.lastPathComponent
                if directory.depth == 0 && root.excludedChildNames.contains(name) {
                    continue
                }
                if outOfTime() {
                    filesStopped = true
                    return
                }
                let components = directory.components + [name]
                let kind = SweepFileKind(fileName: name)
                scanName(name, storageClass: root.storageClass, kind: kind)
                guard let values = try? child.resourceValues(forKeys: Set(Self.resourceKeys)) else {
                    unreadableItems += 1
                    continue
                }
                if values.isSymbolicLink == true {
                    skippedSymbolicLinks += 1
                } else if values.isDirectory == true {
                    if directory.depth + 1 > bounds.maxDepth {
                        boundsReached.insert(.maxDepth)
                    } else {
                        subdirectories.append((child, directory.depth + 1, components))
                    }
                } else if values.isRegularFile == true {
                    guard filesVisited < bounds.maxFiles else {
                        boundsReached.insert(.maxFiles)
                        filesStopped = true
                        return
                    }
                    filesVisited += 1
                    if allowList.allows(components: components, storageClass: root.storageClass) {
                        allowListedFiles += 1
                    } else {
                        scanContents(of: child, storageClass: root.storageClass, kind: kind)
                    }
                    if filesStopped {
                        return
                    }
                } else {
                    skippedSpecialFiles += 1
                }
            }
            stack.append(contentsOf: subdirectories.reversed())
        }
    }

    private func scanName(_ name: String, storageClass: AppStorageClass, kind: SweepFileKind) {
        let bytes = Data(name.utf8)
        record(
            matches(in: bytes, excluding: []),
            location: .fileName,
            surface: storageClass.rawValue,
            kind: kind,
            bytes: bytes.count
        )
    }

    private func scanContents(of url: URL, storageClass: AppStorageClass, kind: SweepFileKind) {
        let handle: FileHandle
        do {
            handle = try FileHandle(forReadingFrom: url)
        } catch {
            unreadableItems += 1
            return
        }
        defer { try? handle.close() }

        var matched = Set<Int>()
        var carry = Data()
        var fileBytes = 0
        while true {
            if outOfTime() {
                filesStopped = true
                break
            }
            let perFileRemaining = bounds.maxBytesPerFile - fileBytes
            let totalRemaining = bounds.maxTotalBytes - bytesScanned
            if perFileRemaining <= 0 || totalRemaining <= 0 {
                let hasMore = (try? handle.read(upToCount: 1))?.isEmpty == false
                if hasMore {
                    truncatedFiles += 1
                    if perFileRemaining <= 0 {
                        boundsReached.insert(.maxBytesPerFile)
                    } else {
                        boundsReached.insert(.maxTotalBytes)
                        filesStopped = true
                    }
                }
                break
            }
            let chunk: Data
            do {
                chunk =
                    try handle.read(
                        upToCount: min(PHISentinelSweep.chunkSize, perFileRemaining, totalRemaining)
                    ) ?? Data()
            } catch {
                unreadableItems += 1
                break
            }
            if chunk.isEmpty {
                break
            }
            fileBytes += chunk.count
            bytesScanned += chunk.count
            let window = carry + chunk
            matched.formUnion(matches(in: window, excluding: matched))
            carry = Data(window.suffix(carryLength))
        }
        filesScanned += 1
        record(
            matched,
            location: .fileContent,
            surface: storageClass.rawValue,
            kind: kind,
            bytes: fileBytes
        )
    }

    // MARK: Logs

    func sweep(_ reader: any ProcessLogReader) {
        guard !outOfTime() else {
            logSummaries.append(
                SweepLogSummary(source: reader.source, status: .skipped, entriesScanned: 0, bytesScanned: 0)
            )
            return
        }
        var entries = 0
        var bytes = 0
        let status: SweepLogStatus
        do {
            try reader.readEntries { message in
                if self.outOfTime() {
                    return false
                }
                guard self.logEntriesScanned < self.bounds.maxLogEntries else {
                    self.boundsReached.insert(.maxLogEntries)
                    return false
                }
                var data = Data(message.utf8)
                let remaining = self.bounds.maxLogBytes - self.logBytesScanned
                var exhausted = false
                if data.count > remaining {
                    data = Data(data.prefix(remaining))
                    exhausted = true
                    self.boundsReached.insert(.maxLogBytes)
                }
                entries += 1
                bytes += data.count
                self.logEntriesScanned += 1
                self.logBytesScanned += data.count
                self.record(
                    self.matches(in: data, excluding: []),
                    location: .logEntry,
                    surface: reader.source.rawValue,
                    kind: nil,
                    bytes: data.count
                )
                return !exhausted
            }
            status = .read
        } catch {
            status = .unreadable
        }
        logSummaries.append(
            SweepLogSummary(source: reader.source, status: status, entriesScanned: entries, bytesScanned: bytes)
        )
    }

    // MARK: Matching and reporting

    private func matches(in haystack: Data, excluding already: Set<Int>) -> Set<Int> {
        var found = Set<Int>()
        guard !haystack.isEmpty else { return found }
        for (index, needle) in needles.enumerated()
        where !already.contains(index) && needle.bytes.count <= haystack.count
            && haystack.range(of: needle.bytes) != nil
        {
            found.insert(index)
        }
        return found
    }

    private func record(
        _ matched: Set<Int>,
        location: SweepLocation,
        surface: String,
        kind: SweepFileKind?,
        bytes: Int
    ) {
        let keys = Set(
            matched.map { index in
                FindingKey(
                    sentinelIndex: needles[index].sentinelIndex,
                    encoding: needles[index].encoding,
                    location: location,
                    surface: surface,
                    fileKind: kind
                )
            }
        )
        for key in keys {
            let current = findings[key] ?? (0, 0)
            findings[key] = (current.count + 1, current.bytes + bytes)
        }
    }

    func report() -> PHISentinelSweepReport {
        let reportedFindings = findings.map { key, value in
            SweepFinding(
                sentinelIdentifier: sentinels[key.sentinelIndex].identifier,
                location: key.location,
                surface: key.surface,
                fileKind: key.fileKind,
                encoding: key.encoding,
                itemCount: value.count,
                itemBytes: value.bytes
            )
        }
        .sorted()
        let exhaustive =
            boundsReached.isEmpty && unreadableItems == 0
            && logSummaries.allSatisfy { $0.status == .read }
        return PHISentinelSweepReport(
            schema: PHISentinelSweepReport.schemaIdentifier,
            sentinelIdentifiers: sentinels.map(\.identifier),
            encodings: SentinelEncoding.allCases,
            bounds: bounds,
            roots: rootSummaries,
            logs: logSummaries,
            filesScanned: filesScanned,
            bytesScanned: bytesScanned,
            allowListedFiles: allowListedFiles,
            truncatedFiles: truncatedFiles,
            skippedSymbolicLinks: skippedSymbolicLinks,
            skippedSpecialFiles: skippedSpecialFiles,
            unreadableItems: unreadableItems,
            logEntriesScanned: logEntriesScanned,
            logBytesScanned: logBytesScanned,
            boundsReached: boundsReached.sorted(),
            findings: reportedFindings,
            clean: reportedFindings.isEmpty,
            exhaustive: exhaustive
        )
    }
}
