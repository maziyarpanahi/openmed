import Foundation

/// Value-free model-store failures. No case carries a URL, repository id,
/// revision, or filesystem path, so errors are safe to log and report.
public enum OpenMedModelStoreError: LocalizedError, Equatable, Sendable {
    case invalidResponse
    /// A cached model could not be moved aside or removed.
    case cacheDeletionFailed
    case httpError(statusCode: Int)
    case missingManifest
    /// The Core ML snapshot has no usable `openmed-coreml.json` or package
    /// inventory.
    case invalidCoreMLManifest
    /// Core ML could not compile the downloaded `.mlpackage`.
    case coreMLCompilationFailed
    case missingWeights
    /// A file transfer failed below HTTP (connection, TLS or local staging).
    case transferFailed
    /// Carries a controlled confinement reason code such as `outside_snapshot`.
    case invalidManifestPath(String)
    /// The Hub snapshot does not match the catalog entry's reproducibility hash.
    case reproducibilityHashMismatch
    /// The Hub model-info response has no valid commit sha or file list.
    case invalidModelInfo

    /// Stable reason code for logs and metrics.
    public var reason: String {
        switch self {
        case .invalidResponse: return "invalid_response"
        case .cacheDeletionFailed: return "cache_deletion_failed"
        case .httpError: return "http_status"
        case .missingManifest: return "missing_manifest"
        case .invalidCoreMLManifest: return "invalid_coreml_manifest"
        case .coreMLCompilationFailed: return "coreml_compilation_failed"
        case .missingWeights: return "missing_weights"
        case .transferFailed: return "transfer_failed"
        case .invalidManifestPath: return "invalid_manifest_path"
        case .reproducibilityHashMismatch: return "reproducibility_hash_mismatch"
        case .invalidModelInfo: return "invalid_model_info"
        }
    }

    public var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "Invalid response while downloading an MLX model file."
        case .cacheDeletionFailed:
            return "The cached model could not be deleted."
        case .httpError(let statusCode):
            return "HTTP \(statusCode) while downloading an MLX model file."
        case .missingManifest:
            return "Downloaded MLX model is missing openmed-mlx.json."
        case .invalidCoreMLManifest:
            return "The Core ML model snapshot has no usable openmed-coreml.json."
        case .coreMLCompilationFailed:
            return "The downloaded Core ML package could not be compiled."
        case .missingWeights:
            return "Downloaded MLX model does not contain any usable weight file."
        case .transferFailed:
            return "A model file transfer failed."
        case .invalidManifestPath:
            return "MLX artifact path rejected by snapshot confinement."
        case .reproducibilityHashMismatch:
            return "The model snapshot does not match the catalog reproducibility hash."
        case .invalidModelInfo:
            return "The model info response has no valid commit or file list."
        }
    }
}

public enum OpenMedMLXModelCacheState: String, Sendable {
    case missing
    case partial
    case ready
}

/// One boundary for manifest preflight, cache inspection and local artifact reads.
/// The caller supplies the trusted root; declared paths never become new roots.
struct OpenMedMLXPathBoundary {
    let rootURL: URL

    // AutoTokenizer and prepared-tokenizer copying read these without requiring
    // manifest declarations. Keep that implicit read set inside the same root.
    private static let tokenizerDiscoveryFiles = [
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.txt", "vocab.json", "merges.txt", "spm.model", "sentencepiece.bpe.model",
        "added_tokens.json", "chat_template.jinja", "chat_template.json",
    ]

    init(directoryURL: URL) throws {
        guard directoryURL.isFileURL else {
            throw OpenMedModelStoreError.invalidManifestPath("invalid_root")
        }
        rootURL = directoryURL.resolvingSymlinksInPath().standardizedFileURL
    }

    static func validateRelativePath(_ path: String) throws {
        let components = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty,
            !path.contains("\\"), !path.contains("\0"),
            components.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." })
        else {
            throw OpenMedModelStoreError.invalidManifestPath("invalid_relative_path")
        }
    }

    func fileURL(_ path: String, rejectSymlinks: Bool = false) throws -> URL {
        try Self.validateRelativePath(path)
        guard rootURL.resolvingSymlinksInPath().standardizedFileURL.path == rootURL.path else {
            throw OpenMedModelStoreError.invalidManifestPath("root_changed")
        }
        let prefix = rootURL.path.hasSuffix("/") ? rootURL.path : rootURL.path + "/"
        var current = rootURL
        for component in path.split(separator: "/") {
            current.append(path: String(component))
            if (try? FileManager.default.destinationOfSymbolicLink(atPath: current.path)) != nil {
                let resolved = current.resolvingSymlinksInPath().standardizedFileURL
                guard !rejectSymlinks,
                    resolved.path == rootURL.path || resolved.path.hasPrefix(prefix),
                    FileManager.default.fileExists(atPath: resolved.path)
                else {
                    throw OpenMedModelStoreError.invalidManifestPath("unsafe_symlink")
                }
            }
        }
        let resolved = current.resolvingSymlinksInPath().standardizedFileURL.path
        guard resolved == rootURL.path || resolved.hasPrefix(prefix) else {
            throw OpenMedModelStoreError.invalidManifestPath("outside_snapshot")
        }
        return current
    }

    func tokenizerPath(base: String, file: String) throws -> String {
        try Self.validateRelativePath(file)
        if base == "." { return file }
        _ = try fileURL(base)
        return "\(base)/\(file)"
    }

    func validate(_ manifest: OpenMedMLXManifest) throws {
        let paths =
            [manifest.configPath, manifest.preferredWeights]
            + [manifest.labelMapPath].compactMap { $0 }
            + manifest.availableWeights + manifest.fallbackWeights
            + (manifest.segmenter?.resourceFiles.map(\.path) ?? [])
        for path in paths { _ = try fileURL(path) }
        try validateTokenizerDirectory(manifest.tokenizer.path)
        for file in manifest.tokenizer.files {
            _ = try fileURL(tokenizerPath(base: manifest.tokenizer.path, file: file))
        }
    }

    func validateTokenizerDirectory(_ path: String) throws {
        if path != "." { _ = try fileURL(path) }
        for file in Self.tokenizerDiscoveryFiles {
            _ = try fileURL(tokenizerPath(base: path, file: file))
        }
    }

    func containsFile(_ path: String) throws -> Bool {
        let url = try fileURL(path).resolvingSymlinksInPath()
        return (try? FileManager.default.attributesOfItem(atPath: url.path)[.type])
            as? FileAttributeType == .typeRegular
    }

    /// Remove the leaf itself, including a replaced symlink, without following it.
    /// A changed parent is refused, so cleanup cannot delete an outside target.
    func removeFileIfConfined(_ path: String) {
        guard (try? Self.validateRelativePath(path)) != nil else { return }
        let components = path.split(separator: "/").map(String.init)
        let parent: URL
        if components.count == 1 {
            guard rootURL.resolvingSymlinksInPath().standardizedFileURL.path == rootURL.path else { return }
            parent = rootURL
        } else {
            guard let safeParent = try? fileURL(components.dropLast().joined(separator: "/")) else { return }
            parent = safeParent
        }
        try? FileManager.default.removeItem(at: parent.appending(path: components.last!))
    }
}

/// Download and cache OpenMed MLX model snapshots from the Hugging Face Hub.
public enum OpenMedModelStore {
    static let readyMarkerFileName = ".openmed-artifact-ready"

    private static let legacyTokenizerFiles = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    ]

    /// Crash-safe generation cache for signed portable bundles, stored next to
    /// the MLX snapshots (`Caches/OpenMed/Bundles` by default). Unlike the MLX
    /// snapshot path, it verifies every declared file against the signed
    /// bundle manifest before a generation becomes ready.
    public static func bundleCache(
        cacheDirectory: URL? = nil,
        options: OpenMedBundleCacheOptions = OpenMedBundleCacheOptions()
    ) throws -> OpenMedBundleCache {
        let root =
            try cacheDirectory
            ?? defaultCacheDirectory().deletingLastPathComponent().appending(path: "Bundles", directoryHint: .isDirectory)
        return try OpenMedBundleCache(rootDirectory: root, options: options)
    }

    public static func downloadMLXModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) async throws -> URL {
        try await downloadMLXModel(
            repoID: repoID, revision: revision, cacheDirectory: cacheDirectory, session: .shared)
    }

    // Internal transport injection keeps confinement tests fully offline.
    // Cancelling the calling task stops the transfer and removes the files
    // this call wrote; `progress` receives byte and file counts.
    static func downloadMLXModel(
        repoID: String, revision: String = "main", cacheDirectory: URL? = nil,
        session: URLSession, progress: OpenMedDownloadProgressReporter? = nil
    ) async throws -> URL {
        let modelDirectory = try cachedMLXModelDirectory(
            repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
        let boundary = try OpenMedMLXPathBoundary(directoryURL: modelDirectory)
        try FileManager.default.createDirectory(
            at: modelDirectory,
            withIntermediateDirectories: true
        )

        if try mlxModelCacheState(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheDirectory
        ) == .ready {
            return modelDirectory
        }

        let transfers = OpenMedFileTransferSession(configuration: session.configuration)
        defer { transfers.invalidate() }
        var writtenFiles: [String] = []
        func fetch(_ path: String, optional: Bool = false) async throws -> Bool {
            try Task.checkCancellation()
            let exists = try boundary.containsFile(path)
            progress?.fileStarted()
            let result: Bool
            do {
                if optional {
                    result = try await downloadOptionalFile(
                        repoID: repoID, revision: revision, relativePath: path,
                        boundary: boundary, transfers: transfers, progress: progress)
                } else {
                    try await downloadFile(
                        repoID: repoID, revision: revision, relativePath: path,
                        boundary: boundary, transfers: transfers, progress: progress)
                    result = true
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                progress?.fileFinished(path, sizeBytes: 0, transferredBytes: nil)
                throw error
            }
            if result && !exists { writtenFiles.append(path) }
            let size = result ? fileSize(try boundary.fileURL(path)) : 0
            progress?.fileFinished(path, sizeBytes: size, transferredBytes: result && !exists ? size : nil)
            return result
        }

        do {
            progress?.plan(files: 1)
            let hasManifest = try await fetch("openmed-mlx.json", optional: true)
            guard hasManifest else {
                progress?.plan(files: 4 + legacyTokenizerFiles.count)
                _ = try await fetch("config.json")
                _ = try await fetch("id2label.json", optional: true)
                var hasWeights = false
                for name in ["weights.safetensors", "weights.npz"] {
                    let exists = try await fetch(name, optional: true)
                    hasWeights = hasWeights || exists
                }
                guard hasWeights else { throw OpenMedModelStoreError.missingWeights }
                for name in legacyTokenizerFiles { _ = try await fetch(name, optional: true) }
                try Task.checkCancellation()
                try markArtifactReadyIfComplete(boundary: boundary)
                return modelDirectory
            }

            let manifestData = try Data(contentsOf: boundary.fileURL("openmed-mlx.json"))
            let manifest = try JSONDecoder().decode(OpenMedMLXManifest.self, from: manifestData)
            // Preflight every declaration, including candidates the loader may use later.
            try boundary.validate(manifest)
            progress?.plan(
                files: [manifest.configPath, manifest.labelMapPath].compactMap({ $0 }).count
                    + manifest.availableWeights.count + manifest.tokenizer.files.count
                    + (manifest.segmenter?.resourceFiles.count ?? 0))

            for path in [manifest.configPath, manifest.labelMapPath].compactMap({ $0 }) {
                _ = try await fetch(path)
            }

            var downloadedWeights = false
            for path in manifest.availableWeights {
                do {
                    _ = try await fetch(path)
                    downloadedWeights = true
                } catch OpenMedModelStoreError.invalidManifestPath(let reason) {
                    throw OpenMedModelStoreError.invalidManifestPath(reason)
                } catch is CancellationError {
                    throw CancellationError()
                } catch {
                    continue
                }
            }
            guard downloadedWeights else { throw OpenMedModelStoreError.missingWeights }

            for file in manifest.tokenizer.files {
                _ = try await fetch(boundary.tokenizerPath(base: manifest.tokenizer.path, file: file))
            }
            for resource in manifest.segmenter?.resourceFiles ?? [] { _ = try await fetch(resource.path) }

            try boundary.validate(manifest)
            try Task.checkCancellation()
            try markArtifactReadyIfComplete(boundary: boundary)
            return modelDirectory
        } catch OpenMedModelStoreError.invalidManifestPath(let reason) {
            boundary.removeFileIfConfined(readyMarkerFileName)
            for path in writtenFiles.reversed() { boundary.removeFileIfConfined(path) }
            throw OpenMedModelStoreError.invalidManifestPath(reason)
        } catch is CancellationError {
            boundary.removeFileIfConfined(readyMarkerFileName)
            for path in writtenFiles.reversed() { boundary.removeFileIfConfined(path) }
            throw CancellationError()
        }
    }

    public static func cachedMLXModelDirectory(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> URL {
        let cacheRoot = try cacheDirectory ?? defaultCacheDirectory()
        let boundary = try OpenMedMLXPathBoundary(directoryURL: cacheRoot)
        return try boundary.fileURL(
            "\(sanitizedPathComponent(repoID))/\(sanitizedPathComponent(revision))",
            rejectSymlinks: true)
    }

    public static func isMLXModelCached(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> Bool {
        try mlxModelCacheState(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheDirectory
        ) == .ready
    }

    public static func mlxModelCacheState(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> OpenMedMLXModelCacheState {
        let modelDirectory = try cachedMLXModelDirectory(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheDirectory
        )

        guard FileManager.default.fileExists(atPath: modelDirectory.path) else {
            return .missing
        }

        return try cacheState(at: modelDirectory)
    }

    static func cacheState(at modelDirectory: URL) throws -> OpenMedMLXModelCacheState {
        let fileManager = FileManager.default
        let boundary = try OpenMedMLXPathBoundary(directoryURL: modelDirectory)
        let isComplete: Bool
        do {
            _ = try boundary.fileURL(readyMarkerFileName)
            isComplete = try hasCompleteArtifact(boundary: boundary)
        } catch OpenMedModelStoreError.invalidManifestPath(let reason) {
            boundary.removeFileIfConfined(readyMarkerFileName)
            throw OpenMedModelStoreError.invalidManifestPath(reason)
        }
        let readyMarkerURL = try boundary.fileURL(readyMarkerFileName)

        if isComplete {
            if !fileManager.fileExists(atPath: readyMarkerURL.path) {
                try writeReadyMarker(to: readyMarkerURL)
            }
            return .ready
        }

        if fileManager.fileExists(atPath: readyMarkerURL.path) {
            try? fileManager.removeItem(at: readyMarkerURL)
        }

        let contents = try fileManager.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )
        return contents.isEmpty ? .missing : .partial
    }

    private static func hasCompleteArtifact(boundary: OpenMedMLXPathBoundary) throws -> Bool {
        let manifestURL = try boundary.fileURL("openmed-mlx.json")
        if !FileManager.default.fileExists(atPath: manifestURL.path) {
            try boundary.validateTokenizerDirectory(".")
            let hasLegacyConfig = try boundary.containsFile("config.json")
            let weightFiles = try ["weights.safetensors", "weights.npz"].map { try boundary.containsFile($0) }
            let hasLegacyWeights = weightFiles.contains(true)
            return hasLegacyConfig && hasLegacyWeights
        }

        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(OpenMedMLXManifest.self, from: data)
        try boundary.validate(manifest)

        let requiredFiles =
            [manifest.configPath] + [manifest.labelMapPath].compactMap { $0 }
            + (try manifest.tokenizer.files.map {
                try boundary.tokenizerPath(base: manifest.tokenizer.path, file: $0)
            })
            + (manifest.segmenter?.resourceFiles.map(\.path) ?? [])
        let hasWeights = try manifest.availableWeights.map { try boundary.containsFile($0) }.contains(true)

        let requiredPresence = try requiredFiles.map { try boundary.containsFile($0) }
        return hasWeights && requiredPresence.allSatisfy { $0 }
    }

    private static func markArtifactReadyIfComplete(boundary: OpenMedMLXPathBoundary) throws {
        guard try hasCompleteArtifact(boundary: boundary) else { return }
        try writeReadyMarker(to: boundary.fileURL(readyMarkerFileName))
    }

    static func defaultCacheDirectory() throws -> URL {
        let base =
            try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        return
            base
            .appending(path: "OpenMed", directoryHint: .isDirectory)
            .appending(path: "MLXModels", directoryHint: .isDirectory)
    }

    private static func downloadFile(
        repoID: String,
        revision: String,
        relativePath: String,
        boundary: OpenMedMLXPathBoundary,
        transfers: OpenMedFileTransferSession,
        progress: OpenMedDownloadProgressReporter?
    ) async throws {
        let destinationURL = try boundary.fileURL(relativePath)
        if try boundary.containsFile(relativePath) {
            return
        }

        let remoteURL = try resolveHubURL(repoID: repoID, revision: revision, relativePath: relativePath)
        var request = URLRequest(url: remoteURL)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")

        let (staged, httpResponse) = try await transfers.download(request) { bytesDone, bytesTotal in
            progress?.fileProgress(relativePath, bytesDone: bytesDone, bytesTotal: bytesTotal)
        }
        defer { try? FileManager.default.removeItem(at: staged) }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw OpenMedModelStoreError.httpError(statusCode: httpResponse.statusCode)
        }
        try Task.checkCancellation()

        // URLSession suspends: check again before creating directories or writing.
        _ = try boundary.fileURL(relativePath)
        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        _ = try boundary.fileURL(relativePath)
        // Replace an existing leaf as an atomic write would, without following it.
        if (try? FileManager.default.attributesOfItem(atPath: destinationURL.path)) != nil {
            boundary.removeFileIfConfined(relativePath)
        }
        do {
            try FileManager.default.moveItem(at: staged, to: destinationURL)
        } catch {
            throw OpenMedModelStoreError.transferFailed
        }
    }

    private static func fileSize(_ url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? NSNumber)?.int64Value ?? 0
    }

    @discardableResult
    private static func downloadOptionalFile(
        repoID: String,
        revision: String,
        relativePath: String,
        boundary: OpenMedMLXPathBoundary,
        transfers: OpenMedFileTransferSession,
        progress: OpenMedDownloadProgressReporter?
    ) async throws -> Bool {
        do {
            try await downloadFile(
                repoID: repoID,
                revision: revision,
                relativePath: relativePath,
                boundary: boundary, transfers: transfers, progress: progress
            )
            return try boundary.containsFile(relativePath)
        } catch OpenMedModelStoreError.httpError(statusCode: 404) {
            return try boundary.containsFile(relativePath)
        }
    }

    static func resolveHubURL(
        repoID: String,
        revision: String,
        relativePath: String
    ) throws -> URL {
        func encodePath(_ value: String) -> String {
            value
                .split(separator: "/")
                .map {
                    String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed.subtracting(CharacterSet(charactersIn: "%?#")))
                        ?? String($0)
                }
                .joined(separator: "/")
        }

        let encodedRepo = encodePath(repoID)
        let encodedRevision = encodePath(revision)
        let encodedPath = encodePath(relativePath)
        guard
            let url = URL(
                string: "https://huggingface.co/\(encodedRepo)/resolve/\(encodedRevision)/\(encodedPath)?download=1"
            )
        else {
            throw OpenMedModelStoreError.invalidResponse
        }
        return url
    }

    static func sanitizedPathComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "__")
            .replacingOccurrences(of: ":", with: "_")
    }

    static func writeReadyMarker(to url: URL) throws {
        let marker = [
            "state": OpenMedMLXModelCacheState.ready.rawValue,
            "completed_at": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(withJSONObject: marker, options: [.prettyPrinted])
        try data.write(to: url, options: .atomic)
    }
}
