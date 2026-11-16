import CryptoKit
import Foundation

/// Reproducibility hashes for published OpenMed model snapshots.
///
/// The hash binds a repository id, the commit sha, the release date and the
/// full sibling file list. It is the value `models.jsonl`, the bundled
/// ``OpenMedModelCatalog`` and Android's `ModelIntegrity` record as
/// `reproducibility_hash`.
public enum OpenMedModelIntegrity {
    /// Return `sha256:<hex>` over the canonical snapshot description.
    ///
    /// The canonical string is the JSON object
    /// `{"released":…,"repo_id":…,"sha":…,"siblings":[…]}` with keys in that
    /// (sorted) order, no whitespace, `null` for a missing sha or date,
    /// siblings sorted by Unicode code point, and strings escaped as Python's
    /// `json.dumps` does by default: `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`,
    /// and `\uXXXX` (UTF-16, lowercase hex) for every other character outside
    /// printable ASCII.
    ///
    /// - Parameters:
    ///   - repoID: Hugging Face repository id.
    ///   - revisionSHA: Commit sha of the snapshot, if known.
    ///   - released: Release timestamp or date; only the first ten characters
    ///     (`YYYY-MM-DD`) are used.
    ///   - siblings: Every file path in the snapshot, in any order.
    public static func reproducibilityHash(
        repoID: String,
        revisionSHA: String?,
        released: String?,
        siblings: [String]
    ) -> String {
        let payload = canonicalPayload(
            repoID: repoID,
            revisionSHA: revisionSHA,
            released: released,
            siblings: siblings
        )
        let digest = SHA256.hash(data: Data(payload.utf8))
        return "sha256:" + digest.map { String(format: "%02x", $0) }.joined()
    }

    static func canonicalPayload(
        repoID: String,
        revisionSHA: String?,
        released: String?,
        siblings: [String]
    ) -> String {
        let sorted = siblings.sorted { $0.unicodeScalars.lexicographicallyPrecedes($1.unicodeScalars) }
        return "{\"released\":" + jsonValue(normalizedReleaseDate(released))
            + ",\"repo_id\":" + jsonString(repoID)
            + ",\"sha\":" + jsonValue(revisionSHA)
            + ",\"siblings\":[" + sorted.map(jsonString).joined(separator: ",") + "]}"
    }

    /// `YYYY-MM-DD` prefix of a timestamp, or `nil` when absent or blank.
    static func normalizedReleaseDate(_ value: String?) -> String? {
        guard let value else { return nil }
        let date = String(value.prefix(10))
        return date.trimmingCharacters(in: .whitespaces).isEmpty ? nil : date
    }

    private static func jsonValue(_ value: String?) -> String {
        value.map(jsonString) ?? "null"
    }

    private static func jsonString(_ value: String) -> String {
        var output = "\""
        for unit in value.utf16 {
            switch unit {
            case 0x22: output += "\\\""
            case 0x5C: output += "\\\\"
            case 0x08: output += "\\b"
            case 0x0C: output += "\\f"
            case 0x0A: output += "\\n"
            case 0x0D: output += "\\r"
            case 0x09: output += "\\t"
            case 0x20...0x7E: output.unicodeScalars.append(Unicode.Scalar(UInt8(unit)))
            default: output += String(format: "\\u%04x", unit)
            }
        }
        return output + "\""
    }
}

/// The parts of a Hub model-info response that the reproducibility hash uses.
struct OpenMedHubModelInfo: Equatable, Sendable {
    let revisionSHA: String?
    let released: String?
    let siblings: [String]

    init(revisionSHA: String?, released: String?, siblings: [String]) {
        self.revisionSHA = revisionSHA
        self.released = released
        self.siblings = siblings
    }

    /// Parse `GET /api/models/<repo>/revision/<revision>` JSON.
    ///
    /// The release date is `lastModified`, else `createdAt`. Siblings without
    /// an `rfilename` are skipped, as Android does.
    init(jsonData: Data) throws {
        guard let object = (try? JSONSerialization.jsonObject(with: jsonData)) as? [String: Any] else {
            throw OpenMedModelStoreError.invalidModelInfo
        }
        func nonEmptyString(_ key: String) -> String? {
            (object[key] as? String).flatMap { $0.isEmpty ? nil : $0 }
        }
        revisionSHA = nonEmptyString("sha")
        released = OpenMedModelIntegrity.normalizedReleaseDate(nonEmptyString("lastModified") ?? nonEmptyString("createdAt"))
        switch object["siblings"] {
        case nil, is NSNull:
            siblings = []
        case let items as [Any]:
            siblings = try items.compactMap { item in
                guard let sibling = item as? [String: Any] else { throw OpenMedModelStoreError.invalidModelInfo }
                return sibling["rfilename"] as? String
            }
        default:
            throw OpenMedModelStoreError.invalidModelInfo
        }
    }
}

/// Ready-marker fields recorded for a snapshot downloaded from a catalog entry.
struct OpenMedMLXSnapshotPin: Equatable, Sendable {
    let revisionSHA: String
    let reproducibilityHash: String
    let requestedRevision: String
}

extension OpenMedModelStore {
    /// Download the snapshot a catalog entry describes, pinned to one commit.
    ///
    /// The store first looks for a ready snapshot already verified against
    /// `entry.reproducibilityHash` and returns it without any network request.
    /// Otherwise it fetches the model info for `revision`, recomputes the
    /// reproducibility hash from the repository id, commit sha, release date
    /// and sibling list, and throws
    /// ``OpenMedModelStoreError/reproducibilityHashMismatch`` before creating
    /// any file when it differs from the catalog. On a match, every file is
    /// fetched from the resolved commit sha rather than the moving `revision`
    /// name, the snapshot is cached under that sha, and the ready marker
    /// records the sha (see ``cachedMLXModelRevisionSHA(repoID:revision:cacheDirectory:)``).
    ///
    /// - Parameters:
    ///   - entry: The catalog entry to download, usually from
    ///     ``OpenMedModelCatalog/bundled()``.
    ///   - revision: Branch, tag or commit to resolve (default `main`).
    ///   - cacheDirectory: Cache root; defaults to `Caches/OpenMed/MLXModels`.
    /// - Returns: The ready snapshot directory, `<root>/<repo>/<commit sha>`.
    /// - Throws: ``OpenMedModelStoreError``.
    public static func downloadMLXModel(
        entry: OpenMedModelCatalogEntry,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) async throws -> URL {
        try await downloadMLXModel(entry: entry, revision: revision, cacheDirectory: cacheDirectory, session: .shared)
    }

    static func downloadMLXModel(
        entry: OpenMedModelCatalogEntry,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        session: URLSession
    ) async throws -> URL {
        if let cached = try cachedMLXModelDirectory(entry: entry, cacheDirectory: cacheDirectory) {
            return cached
        }
        let info = try await fetchModelInfo(repoID: entry.repoID, revision: revision, session: session)
        let actualHash = OpenMedModelIntegrity.reproducibilityHash(
            repoID: entry.repoID,
            revisionSHA: info.revisionSHA,
            released: info.released,
            siblings: info.siblings
        )
        guard actualHash == entry.reproducibilityHash else {
            throw OpenMedModelStoreError.reproducibilityHashMismatch
        }
        guard let commit = info.revisionSHA, isCommitSHA(commit) else {
            throw OpenMedModelStoreError.invalidModelInfo
        }
        let directory = try await downloadMLXModel(
            repoID: entry.repoID, revision: commit, cacheDirectory: cacheDirectory, session: session)
        guard try mlxModelCacheState(repoID: entry.repoID, revision: commit, cacheDirectory: cacheDirectory) == .ready
        else {
            return directory
        }
        try writePinnedReadyMarker(
            in: directory,
            pin: OpenMedMLXSnapshotPin(
                revisionSHA: commit,
                reproducibilityHash: entry.reproducibilityHash,
                requestedRevision: revision
            )
        )
        return directory
    }

    /// The ready snapshot previously downloaded and verified for `entry`, if
    /// any. Reads only the local cache.
    ///
    /// - Throws: ``OpenMedModelStoreError/invalidManifestPath(_:)`` when the
    ///   repository directory is a symlink or escapes the cache root.
    public static func cachedMLXModelDirectory(
        entry: OpenMedModelCatalogEntry,
        cacheDirectory: URL? = nil
    ) throws -> URL? {
        let cacheRoot = try cacheDirectory ?? defaultCacheDirectory()
        let boundary = try OpenMedMLXPathBoundary(directoryURL: cacheRoot)
        let repoDirectory = try boundary.fileURL(sanitizedPathComponent(entry.repoID), rejectSymlinks: true)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: repoDirectory.path) else {
            return nil
        }
        for name in names.filter(isCommitSHA).sorted() {
            guard
                (try? mlxModelCacheState(repoID: entry.repoID, revision: name, cacheDirectory: cacheDirectory)) == .ready,
                let directory = try? cachedMLXModelDirectory(
                    repoID: entry.repoID, revision: name, cacheDirectory: cacheDirectory),
                let pin = readPin(in: directory),
                pin.revisionSHA == name,
                pin.reproducibilityHash == entry.reproducibilityHash
            else { continue }
            return directory
        }
        return nil
    }

    /// The commit sha recorded when a pinned download made this snapshot
    /// ready, or `nil` when the snapshot is not ready or was downloaded by
    /// name without a catalog entry.
    public static func cachedMLXModelRevisionSHA(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> String? {
        guard try mlxModelCacheState(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory) == .ready
        else {
            return nil
        }
        let directory = try cachedMLXModelDirectory(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
        return readPin(in: directory)?.revisionSHA
    }

    // MARK: - Internals

    static func isCommitSHA(_ value: String) -> Bool {
        value.utf8.count == 40 && value.utf8.allSatisfy { (0x30...0x39).contains($0) || (0x61...0x66).contains($0) }
    }

    static func fetchModelInfo(repoID: String, revision: String, session: URLSession) async throws -> OpenMedHubModelInfo {
        var request = URLRequest(url: try modelInfoURL(repoID: repoID, revision: revision))
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenMedModelStoreError.invalidResponse
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw OpenMedModelStoreError.httpError(statusCode: httpResponse.statusCode)
        }
        return try OpenMedHubModelInfo(jsonData: data)
    }

    static func modelInfoURL(repoID: String, revision: String) throws -> URL {
        let allowed = CharacterSet.urlPathAllowed.subtracting(CharacterSet(charactersIn: "%?#/"))
        func encode(_ value: String) -> String {
            value.split(separator: "/", omittingEmptySubsequences: false)
                .map { String($0).addingPercentEncoding(withAllowedCharacters: allowed) ?? String($0) }
                .joined(separator: "/")
        }
        let revisionComponent = revision.addingPercentEncoding(withAllowedCharacters: allowed) ?? revision
        guard
            let url = URL(
                string: "https://huggingface.co/api/models/\(encode(repoID))/revision/\(revisionComponent)")
        else {
            throw OpenMedModelStoreError.invalidResponse
        }
        return url
    }

    static func readPin(in directory: URL) -> OpenMedMLXSnapshotPin? {
        guard
            let boundary = try? OpenMedMLXPathBoundary(directoryURL: directory),
            let markerURL = try? boundary.fileURL(readyMarkerFileName, rejectSymlinks: true),
            let data = try? Data(contentsOf: markerURL),
            let marker = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
            let sha = marker["revision_sha"] as? String, isCommitSHA(sha),
            let hash = marker["reproducibility_hash"] as? String,
            let requested = marker["requested_revision"] as? String
        else {
            return nil
        }
        return OpenMedMLXSnapshotPin(revisionSHA: sha, reproducibilityHash: hash, requestedRevision: requested)
    }

    private static func writePinnedReadyMarker(in directory: URL, pin: OpenMedMLXSnapshotPin) throws {
        let boundary = try OpenMedMLXPathBoundary(directoryURL: directory)
        let markerURL = try boundary.fileURL(readyMarkerFileName, rejectSymlinks: true)
        let marker = [
            "state": OpenMedMLXModelCacheState.ready.rawValue,
            "completed_at": ISO8601DateFormatter().string(from: Date()),
            "revision_sha": pin.revisionSHA,
            "reproducibility_hash": pin.reproducibilityHash,
            "requested_revision": pin.requestedRevision,
        ]
        let data = try JSONSerialization.data(withJSONObject: marker, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: markerURL, options: .atomic)
    }
}
