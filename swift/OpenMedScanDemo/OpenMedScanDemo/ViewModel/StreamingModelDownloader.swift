import Foundation
import OpenMedKit
import os.log

/// Streaming replacement for `OpenMedModelStore.downloadMLXModel`. The stock
/// implementation uses `URLSession.data(for:)` which buffers a whole file in
/// memory before writing it — so a 180 MB artifact shows no progress for a
/// minute+ and then jumps to 100%. This version streams bytes to disk and
/// surfaces per-file progress through a delegate-backed `URLSession`.
///
/// The finished cache layout is identical to `OpenMedModelStore`'s — same
/// directory, same `.openmed-artifact-ready` marker file — so downstream
/// code (`OpenMed(backend:)`, `OpenMedModelStore.mlxModelCacheState`) keeps
/// working unchanged.
public actor StreamingModelDownloader {
    public static let shared = StreamingModelDownloader()

    private let log = Logger(subsystem: "com.openmed.scan", category: "streaming-download")
    private var currentTask: URLSessionDownloadTask?
    private var currentDelegate: ProgressDelegate?

    public init() {}

    /// Reports bytes written so far during a single file download, plus the
    /// aggregate bytes-so-far across the whole artifact.
    public typealias ProgressHandler = @Sendable (_ file: String, _ fileBytes: Int64, _ fileTotal: Int64?, _ aggregateBytes: Int64) -> Void

    /// Downloads the artifact referenced by `repoID` into OpenMedKit's
    /// standard cache location and marks it ready. Resumes cleanly from a
    /// partial cache by skipping files already on disk.
    @discardableResult
    public func prepare(
        repoID: String,
        revision: String = "main",
        progress: @escaping ProgressHandler
    ) async throws -> URL {
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(repoID: repoID, revision: revision)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        if (try? OpenMedModelStore.mlxModelCacheState(repoID: repoID, revision: revision)) == .ready {
            progress("ready", 0, 0, 0)
            return directory
        }

        // 1. Fetch the manifest (may not exist on legacy repos).
        let manifestPath = "openmed-mlx.json"
        let manifestURL = directory.appending(path: manifestPath)
        let manifestExists = try await downloadFileIfMissing(
            repoID: repoID,
            revision: revision,
            relativePath: manifestPath,
            destination: manifestURL,
            requiredStatusCodes: nil,    // tolerate 404
            progress: { _, _, _, _ in }
        )

        var aggregate: Int64 = sizeOf(manifestURL)
        let manifest = try await decodeManifestIfPresent(
            repoID: repoID,
            revision: revision,
            manifestURL: manifestURL,
            manifestExists: manifestExists
        )

        // 2. Either manifest-driven or legacy layout.
        let filesToDownload: [(path: String, optional: Bool)] = {
            if let manifest {
                var list: [(String, Bool)] = []
                list.append((manifest.configPath, false))
                if let labelMap = manifest.labelMapPath { list.append((labelMap, true)) }
                let weights = manifest.availableWeights.isEmpty ? [manifest.preferredWeights].compactMap { $0 } : manifest.availableWeights
                for w in weights { list.append((w, false)) }
                for tokenizerFile in manifest.tokenizer.files {
                    let basePath = manifest.tokenizer.path
                    let full = (basePath == "." || basePath.isEmpty) ? tokenizerFile : "\(basePath)/\(tokenizerFile)"
                    list.append((full, false))
                }
                return list
            } else {
                // Legacy fallback: download config.json, id2label.json (optional),
                // weights, then try a handful of common tokenizer files.
                return [
                    ("config.json",                    false),
                    ("id2label.json",                  true),
                    ("weights.safetensors",            true),
                    ("weights.npz",                    true),
                    ("tokenizer.json",                 true),
                    ("tokenizer_config.json",          true),
                    ("special_tokens_map.json",        true),
                    ("vocab.txt",                      true),
                    ("vocab.json",                     true),
                    ("merges.txt",                     true),
                    ("spm.model",                      true),
                    ("sentencepiece.bpe.model",        true),
                    ("added_tokens.json",              true),
                ]
            }
        }()

        // 3. Download each referenced file with streaming progress.
        for (path, optional) in filesToDownload {
            try Task.checkCancellation()
            let destination = directory.appending(path: path)
            let existing = sizeOf(destination)
            if existing > 0 {
                // Already downloaded on a previous run — count toward aggregate.
                aggregate += existing
                progress(path, existing, existing, aggregate)
                continue
            }
            let wroteSomething = try await downloadFileIfMissing(
                repoID: repoID,
                revision: revision,
                relativePath: path,
                destination: destination,
                requiredStatusCodes: optional ? nil : Set(200..<300),
                progress: { file, bytes, total, _ in
                    progress(file, bytes, total, aggregate + bytes)
                }
            )
            if wroteSomething {
                aggregate += sizeOf(destination)
            } else if !optional {
                throw DownloaderError.missingFile(path)
            }
        }

        // 4. Write the ready marker (OpenMedModelStore is the canonical writer,
        //    but its marker helper is private — we call the public state check
        //    which writes the marker for us when the directory is complete).
        _ = try? OpenMedModelStore.mlxModelCacheState(repoID: repoID, revision: revision)
        return directory
    }

    public func cancel() {
        currentTask?.cancel()
        currentTask = nil
        currentDelegate = nil
    }

    // MARK: - Internal plumbing

    private func downloadFileIfMissing(
        repoID: String,
        revision: String,
        relativePath: String,
        destination: URL,
        requiredStatusCodes: Set<Int>?,
        progress: @escaping ProgressHandler
    ) async throws -> Bool {
        if FileManager.default.fileExists(atPath: destination.path) { return true }

        let url = try buildHubURL(repoID: repoID, revision: revision, relativePath: relativePath)
        let delegate = ProgressDelegate(file: relativePath, progress: progress)
        let session = URLSession(configuration: .ephemeral, delegate: delegate, delegateQueue: nil)
        defer { session.finishTasksAndInvalidate() }

        let request = URLRequest(url: url)
        do {
            let (tempURL, response) = try await session.download(for: request, delegate: delegate)
            if let http = response as? HTTPURLResponse {
                let required = requiredStatusCodes ?? Set(200..<300)
                guard required.contains(http.statusCode) else {
                    try? FileManager.default.removeItem(at: tempURL)
                    if http.statusCode == 404 { return false }
                    throw DownloaderError.httpStatus(url, http.statusCode)
                }
            }
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: tempURL, to: destination)
            return true
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as NSError where error.code == NSURLErrorCancelled {
            throw CancellationError()
        }
    }

    private func decodeManifestIfPresent(
        repoID: String,
        revision: String,
        manifestURL: URL,
        manifestExists: Bool
    ) async throws -> MLXManifest? {
        guard manifestExists else { return nil }
        do {
            return try decodeManifest(at: manifestURL)
        } catch {
            // A previous private/gated attempt may have cached the 401 body at
            // openmed-mlx.json. Remove it once, re-fetch, then decode again.
            try? FileManager.default.removeItem(at: manifestURL)
            let refetched = try await downloadFileIfMissing(
                repoID: repoID,
                revision: revision,
                relativePath: manifestURL.lastPathComponent,
                destination: manifestURL,
                requiredStatusCodes: nil,
                progress: { _, _, _, _ in }
            )
            guard refetched else { return nil }
            do {
                return try decodeManifest(at: manifestURL)
            } catch {
                throw DownloaderError.invalidManifest(manifestURL.lastPathComponent, error.localizedDescription)
            }
        }
    }

    private func decodeManifest(at url: URL) throws -> MLXManifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(MLXManifest.self, from: data)
    }

    private func buildHubURL(repoID: String, revision: String, relativePath: String) throws -> URL {
        func encode(_ s: String) -> String {
            s.split(separator: "/")
                .map {
                    String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
                    ?? String($0)
                }
                .joined(separator: "/")
        }
        guard let url = URL(
            string: "https://huggingface.co/\(encode(repoID))/resolve/\(encode(revision))/\(encode(relativePath))?download=1"
        ) else {
            throw DownloaderError.invalidURL(repoID, relativePath)
        }
        return url
    }

    private func sizeOf(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }
}

// MARK: - Error type
public enum DownloaderError: LocalizedError {
    case httpStatus(URL, Int)
    case missingFile(String)
    case invalidURL(String, String)
    case invalidManifest(String, String)

    public var errorDescription: String? {
        switch self {
        case .httpStatus(let url, let code):
            return "HTTP \(code) while fetching \(url.absoluteString)"
        case .missingFile(let path):
            return "Required file missing: \(path)"
        case .invalidURL(let repo, let path):
            return "Invalid URL for \(repo)/\(path)"
        case .invalidManifest(let path, let reason):
            return "Invalid MLX manifest \(path): \(reason)"
        }
    }
}

// MARK: - Delegate

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let file: String
    private let progress: StreamingModelDownloader.ProgressHandler
    init(file: String, progress: @escaping StreamingModelDownloader.ProgressHandler) {
        self.file = file
        self.progress = progress
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didFinishDownloadingTo location: URL) {
        // The async-download API above handles the final move; nothing to do here.
    }

    func urlSession(_ session: URLSession,
                    downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64,
                    totalBytesExpectedToWrite: Int64) {
        let expected: Int64? = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : nil
        progress(file, totalBytesWritten, expected, totalBytesWritten)
    }
}

// MARK: - Minimal manifest decoder (matches OpenMedKit's schema)

private struct MLXManifest: Decodable {
    let configPath: String
    let labelMapPath: String?
    let preferredWeights: String?
    let availableWeights: [String]
    let tokenizer: Tokenizer

    struct Tokenizer: Decodable {
        let path: String
        let files: [String]
    }

    enum CodingKeys: String, CodingKey {
        case configPath = "config_path"
        case labelMapPath = "label_map_path"
        case preferredWeights = "preferred_weights"
        case availableWeights = "available_weights"
        case tokenizer
    }
}
