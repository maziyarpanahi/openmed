import Foundation

public enum OpenMedModelStoreError: LocalizedError {
    case invalidResponse(URL)
    case httpError(URL, Int)
    case missingManifest(URL)
    case missingWeights(URL)

    public var errorDescription: String? {
        switch self {
        case .invalidResponse(let url):
            return "Invalid response while downloading \(url.absoluteString)"
        case .httpError(let url, let statusCode):
            return "HTTP \(statusCode) while downloading \(url.absoluteString)"
        case .missingManifest(let url):
            return "Downloaded MLX model is missing openmed-mlx.json in \(url.path)"
        case .missingWeights(let url):
            return "Downloaded MLX model does not contain any usable weight file in \(url.path)"
        }
    }
}

public enum OpenMedMLXModelCacheState: String, Sendable {
    case missing
    case partial
    case ready
}

/// Download and cache OpenMed MLX model snapshots from the Hugging Face Hub.
public enum OpenMedModelStore {
    private static let readyMarkerFileName = ".openmed-artifact-ready"

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

    public static func downloadMLXModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) async throws -> URL {
        let cacheRoot = try cacheDirectory ?? defaultCacheDirectory()
        let modelDirectory = cacheRoot
            .appending(path: sanitizedPathComponent(repoID), directoryHint: .isDirectory)
            .appending(path: sanitizedPathComponent(revision), directoryHint: .isDirectory)
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

        let manifestURL = modelDirectory.appending(path: "openmed-mlx.json")
        let hasManifest = try await downloadOptionalFile(
            repoID: repoID,
            revision: revision,
            relativePath: "openmed-mlx.json",
            destinationURL: manifestURL
        )

        guard hasManifest else {
            try await downloadLegacyArtifact(
                repoID: repoID,
                revision: revision,
                into: modelDirectory
            )
            try markArtifactReadyIfComplete(at: modelDirectory)
            return modelDirectory
        }

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(OpenMedMLXManifest.self, from: manifestData)

        let fixedFiles = [manifest.configPath, manifest.labelMapPath].compactMap { $0 }
        for relativePath in fixedFiles {
            try await downloadFile(
                repoID: repoID,
                revision: revision,
                relativePath: relativePath,
                destinationURL: modelDirectory.appending(path: relativePath)
            )
        }

        var downloadedWeights = false
        for weightPath in manifest.availableWeights {
            let destinationURL = modelDirectory.appending(path: weightPath)
            do {
                try await downloadFile(
                    repoID: repoID,
                    revision: revision,
                    relativePath: weightPath,
                    destinationURL: destinationURL
                )
                downloadedWeights = true
            } catch {
                continue
            }
        }
        if !downloadedWeights {
            throw OpenMedModelStoreError.missingWeights(modelDirectory)
        }

        for file in manifest.tokenizer.files {
            let relativePath = tokenizerRelativePath(
                basePath: manifest.tokenizer.path,
                fileName: file
            )
            try await downloadFile(
                repoID: repoID,
                revision: revision,
                relativePath: relativePath,
                destinationURL: modelDirectory.appending(path: relativePath)
            )
        }

        try markArtifactReadyIfComplete(at: modelDirectory)
        return modelDirectory
    }

    public static func cachedMLXModelDirectory(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> URL {
        let cacheRoot = try cacheDirectory ?? defaultCacheDirectory()
        return cacheRoot
            .appending(path: sanitizedPathComponent(repoID), directoryHint: .isDirectory)
            .appending(path: sanitizedPathComponent(revision), directoryHint: .isDirectory)
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

    private static func cacheState(at modelDirectory: URL) throws -> OpenMedMLXModelCacheState {
        let fileManager = FileManager.default
        let readyMarkerURL = readyMarkerURL(for: modelDirectory)
        let isComplete = try hasCompleteArtifact(at: modelDirectory)

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

    private static func hasCompleteArtifact(at modelDirectory: URL) throws -> Bool {
        let manifestURL = modelDirectory.appending(path: "openmed-mlx.json")
        if !FileManager.default.fileExists(atPath: manifestURL.path) {
            let hasLegacyConfig = FileManager.default.fileExists(
                atPath: modelDirectory.appending(path: "config.json").path
            )
            let hasLegacyWeights = ["weights.safetensors", "weights.npz"].contains {
                FileManager.default.fileExists(atPath: modelDirectory.appending(path: $0).path)
            }
            return hasLegacyConfig && hasLegacyWeights
        }

        let data = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(OpenMedMLXManifest.self, from: data)

        let requiredFiles = [manifest.configPath] + manifest.tokenizer.files.map {
            tokenizerRelativePath(basePath: manifest.tokenizer.path, fileName: $0)
        }
        let hasWeights = manifest.availableWeights.contains {
            FileManager.default.fileExists(atPath: modelDirectory.appending(path: $0).path)
        }

        return hasWeights && requiredFiles.allSatisfy {
            FileManager.default.fileExists(atPath: modelDirectory.appending(path: $0).path)
        }
    }

    private static func markArtifactReadyIfComplete(at modelDirectory: URL) throws {
        guard try hasCompleteArtifact(at: modelDirectory) else {
            return
        }
        try writeReadyMarker(to: readyMarkerURL(for: modelDirectory))
    }

    private static func downloadLegacyArtifact(
        repoID: String,
        revision: String,
        into modelDirectory: URL
    ) async throws {
        try await downloadFile(
            repoID: repoID,
            revision: revision,
            relativePath: "config.json",
            destinationURL: modelDirectory.appending(path: "config.json")
        )

        _ = try await downloadOptionalFile(
            repoID: repoID,
            revision: revision,
            relativePath: "id2label.json",
            destinationURL: modelDirectory.appending(path: "id2label.json")
        )

        var hasWeights = false
        for fileName in ["weights.safetensors", "weights.npz"] {
            let didDownload = try await downloadOptionalFile(
                repoID: repoID,
                revision: revision,
                relativePath: fileName,
                destinationURL: modelDirectory.appending(path: fileName)
            )
            hasWeights = hasWeights || didDownload || FileManager.default.fileExists(
                atPath: modelDirectory.appending(path: fileName).path
            )
        }
        guard hasWeights else {
            throw OpenMedModelStoreError.missingWeights(modelDirectory)
        }

        for fileName in legacyTokenizerFiles {
            _ = try await downloadOptionalFile(
                repoID: repoID,
                revision: revision,
                relativePath: fileName,
                destinationURL: modelDirectory.appending(path: fileName)
            )
        }
    }

    private static func defaultCacheDirectory() throws -> URL {
        let base =
            try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        return base
            .appending(path: "OpenMed", directoryHint: .isDirectory)
            .appending(path: "MLXModels", directoryHint: .isDirectory)
    }

    private static func downloadFile(
        repoID: String,
        revision: String,
        relativePath: String,
        destinationURL: URL
    ) async throws {
        if FileManager.default.fileExists(atPath: destinationURL.path) {
            return
        }

        let remoteURL = try resolveHubURL(repoID: repoID, revision: revision, relativePath: relativePath)
        var request = URLRequest(url: remoteURL)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenMedModelStoreError.invalidResponse(remoteURL)
        }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw OpenMedModelStoreError.httpError(remoteURL, httpResponse.statusCode)
        }

        try FileManager.default.createDirectory(
            at: destinationURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: destinationURL, options: .atomic)
    }

    @discardableResult
    private static func downloadOptionalFile(
        repoID: String,
        revision: String,
        relativePath: String,
        destinationURL: URL
    ) async throws -> Bool {
        do {
            try await downloadFile(
                repoID: repoID,
                revision: revision,
                relativePath: relativePath,
                destinationURL: destinationURL
            )
            return FileManager.default.fileExists(atPath: destinationURL.path)
        } catch OpenMedModelStoreError.httpError(_, 404) {
            return FileManager.default.fileExists(atPath: destinationURL.path)
        }
    }

    private static func resolveHubURL(
        repoID: String,
        revision: String,
        relativePath: String
    ) throws -> URL {
        func encodePath(_ value: String) -> String {
            value
                .split(separator: "/")
                .map {
                    String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
                        ?? String($0)
                }
                .joined(separator: "/")
        }

        let encodedRepo = encodePath(repoID)
        let encodedRevision = encodePath(revision)
        let encodedPath = encodePath(relativePath)
        guard let url = URL(
            string: "https://huggingface.co/\(encodedRepo)/resolve/\(encodedRevision)/\(encodedPath)?download=1"
        ) else {
            throw OpenMedModelStoreError.invalidResponse(
                URL(fileURLWithPath: "/\(repoID)/\(relativePath)")
            )
        }
        return url
    }

    private static func sanitizedPathComponent(_ value: String) -> String {
        value
            .replacingOccurrences(of: "/", with: "__")
            .replacingOccurrences(of: ":", with: "_")
    }

    private static func tokenizerRelativePath(basePath: String, fileName: String) -> String {
        if basePath == "." || basePath.isEmpty {
            return fileName
        }
        return "\(basePath)/\(fileName)"
    }

    private static func readyMarkerURL(for modelDirectory: URL) -> URL {
        modelDirectory.appending(path: readyMarkerFileName)
    }

    private static func writeReadyMarker(to url: URL) throws {
        let marker = [
            "state": OpenMedMLXModelCacheState.ready.rawValue,
            "completed_at": ISO8601DateFormatter().string(from: Date()),
        ]
        let data = try JSONSerialization.data(withJSONObject: marker, options: [.prettyPrinted])
        try data.write(to: url, options: .atomic)
    }
}
