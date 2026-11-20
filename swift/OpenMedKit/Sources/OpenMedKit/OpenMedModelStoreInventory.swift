import Foundation

/// One snapshot directory in the MLX model cache.
public struct OpenMedCachedModel: Equatable, Sendable {
    /// Repository id decoded from the cache directory name.
    public let repoID: String
    /// Revision decoded from the snapshot directory name.
    public let revision: String
    /// The same state ``OpenMedModelStore/mlxModelCacheState(repoID:revision:cacheDirectory:)``
    /// reports. A snapshot whose files fail confinement or decoding is `partial`.
    public let state: OpenMedMLXModelCacheState
    /// Bytes of the regular files in the snapshot. Symlinks are never followed.
    public let sizeBytes: Int64
    /// Latest content modification date of the snapshot directory or its files.
    public let modificationDate: Date?

    /// Create an inventory row.
    public init(
        repoID: String,
        revision: String,
        state: OpenMedMLXModelCacheState,
        sizeBytes: Int64,
        modificationDate: Date?
    ) {
        self.repoID = repoID
        self.revision = revision
        self.state = state
        self.sizeBytes = sizeBytes
        self.modificationDate = modificationDate
    }
}

extension OpenMedModelStore {
    private static let deletionTombstonePrefix = ".openmed-deleting-"

    /// List every snapshot in the MLX model cache.
    ///
    /// The cache holds `<root>/<org>__<name>/<revision>/`. Every non-hidden
    /// directory one level below a repository directory is reported, including
    /// empty strays (`missing`). Hidden entries, files, symlinks, and top-level
    /// directories that are not repository directories are skipped, and no
    /// symlink is followed.
    ///
    /// - Parameter cacheDirectory: Cache root; defaults to `Caches/OpenMed/MLXModels`.
    /// - Returns: Rows sorted by repository id, then revision.
    /// - Throws: ``OpenMedModelStoreError/invalidManifestPath(_:)`` when the
    ///   cache root is not a file URL.
    public static func listCachedModels(cacheDirectory: URL? = nil) throws -> [OpenMedCachedModel] {
        let boundary = try OpenMedMLXPathBoundary(directoryURL: cacheDirectory ?? defaultCacheDirectory())
        var models: [OpenMedCachedModel] = []
        for repoName in visibleChildren(of: boundary.rootURL) {
            guard let repoID = decodedRepoID(repoName),
                let repoURL = try? boundary.fileURL(repoName, rejectSymlinks: true),
                isRealDirectory(repoURL)
            else { continue }
            for revisionName in visibleChildren(of: repoURL) {
                guard let snapshotURL = try? boundary.fileURL("\(repoName)/\(revisionName)", rejectSymlinks: true),
                    isRealDirectory(snapshotURL)
                else { continue }
                let state = (try? cacheState(at: snapshotURL)) ?? .partial
                let usage = diskUsage(of: snapshotURL)
                models.append(
                    OpenMedCachedModel(
                        repoID: repoID,
                        revision: revisionName.replacingOccurrences(of: "__", with: "/"),
                        state: state,
                        sizeBytes: usage.bytes,
                        modificationDate: usage.latestModification
                    )
                )
            }
        }
        return models.sorted { ($0.repoID, $0.revision) < ($1.repoID, $1.revision) }
    }

    /// Bytes of every regular file under the cache root, including partial
    /// snapshots and stray files. Symlinks are never followed.
    ///
    /// - Parameter cacheDirectory: Cache root; defaults to `Caches/OpenMed/MLXModels`.
    public static func totalCacheSizeBytes(cacheDirectory: URL? = nil) throws -> Int64 {
        let boundary = try OpenMedMLXPathBoundary(directoryURL: cacheDirectory ?? defaultCacheDirectory())
        return diskUsage(of: boundary.rootURL).bytes
    }

    /// Delete one cached snapshot. Deleting a missing snapshot is a no-op.
    ///
    /// The ready marker is removed first, then the snapshot directory is
    /// renamed to a hidden tombstone inside its repository directory and the
    /// tombstone is removed. An interrupted delete therefore leaves the
    /// snapshot `missing` or `partial`, never a ready marker over missing
    /// files. Symlinks inside the snapshot are removed without touching their
    /// targets. Other repositories and revisions are untouched.
    ///
    /// - Throws: ``OpenMedModelStoreError/invalidManifestPath(_:)`` with
    ///   `unsafe_symlink` when the repository or snapshot directory is a
    ///   symlink (its target is left unchanged), or
    ///   ``OpenMedModelStoreError/cacheDeletionFailed`` when the filesystem
    ///   refuses the rename or removal.
    public static func deleteModel(repoID: String, revision: String = "main", cacheDirectory: URL? = nil) throws {
        let modelDirectory = try cachedMLXModelDirectory(
            repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
        let repoDirectory = modelDirectory.deletingLastPathComponent()
        let repoBoundary = try OpenMedMLXPathBoundary(directoryURL: repoDirectory)
        guard (try? FileManager.default.attributesOfItem(atPath: modelDirectory.path)) != nil else {
            removeLeftoverTombstones(in: repoBoundary)
            return
        }

        if isRealDirectory(modelDirectory) {
            let boundary = try OpenMedMLXPathBoundary(directoryURL: modelDirectory)
            boundary.removeFileIfConfined(readyMarkerFileName)
            guard (try? FileManager.default.attributesOfItem(atPath: try boundary.fileURL(readyMarkerFileName).path)) == nil
            else {
                throw OpenMedModelStoreError.cacheDeletionFailed
            }
        }

        let tombstone = try repoBoundary.fileURL(deletionTombstonePrefix + UUID().uuidString, rejectSymlinks: true)
        do {
            try FileManager.default.moveItem(at: modelDirectory, to: tombstone)
        } catch {
            throw OpenMedModelStoreError.cacheDeletionFailed
        }
        do {
            try FileManager.default.removeItem(at: tombstone)
        } catch {
            throw OpenMedModelStoreError.cacheDeletionFailed
        }
        removeLeftoverTombstones(in: repoBoundary)
        if (try? FileManager.default.contentsOfDirectory(atPath: repoDirectory.path))?.isEmpty == true {
            try? FileManager.default.removeItem(at: repoDirectory)
        }
    }

    // MARK: - Internals

    /// `OpenMed__Name` -> `OpenMed/Name`; names without the separator are not
    /// repository directories.
    static func decodedRepoID(_ directoryName: String) -> String? {
        guard let separator = directoryName.range(of: "__"),
            separator.lowerBound != directoryName.startIndex,
            separator.upperBound != directoryName.endIndex
        else {
            return nil
        }
        return directoryName.replacingCharacters(in: separator, with: "/")
    }

    private static func visibleChildren(of directory: URL) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? [])
            .filter { !$0.hasPrefix(".") }
            .sorted()
    }

    private static func isRealDirectory(_ url: URL) -> Bool {
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return false }
        return attributes[.type] as? FileAttributeType == .typeDirectory
    }

    private static func diskUsage(of directory: URL) -> (bytes: Int64, latestModification: Date?) {
        let keys: [URLResourceKey] = [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey, .contentModificationDateKey]
        var bytes: Int64 = 0
        var latest = (try? directory.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory, includingPropertiesForKeys: keys, options: [], errorHandler: { _, _ in true })
        else {
            return (0, latest)
        }
        for case let url as URL in enumerator {
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { continue }
            if values.isSymbolicLink == true {
                enumerator.skipDescendants()
                continue
            }
            if let date = values.contentModificationDate, latest.map({ date > $0 }) ?? true {
                latest = date
            }
            if values.isRegularFile == true {
                bytes += Int64(values.fileSize ?? 0)
            }
        }
        return (bytes, latest)
    }

    private static func removeLeftoverTombstones(in repoBoundary: OpenMedMLXPathBoundary) {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: repoBoundary.rootURL.path)) ?? []
        for name in names where name.hasPrefix(deletionTombstonePrefix) {
            guard let url = try? repoBoundary.fileURL(name, rejectSymlinks: true) else {
                // A symlinked tombstone is removed as a link, never followed.
                repoBoundary.removeFileIfConfined(name)
                continue
            }
            try? FileManager.default.removeItem(at: url)
        }
    }
}
