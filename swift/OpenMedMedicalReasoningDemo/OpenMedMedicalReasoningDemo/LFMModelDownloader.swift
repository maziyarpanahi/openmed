import Foundation
import OpenMedKit

/// Downloads only the exact OpenMedKit-pinned LFM2.5 repository into the
/// package's shared MLX cache. Files are moved into place atomically.
actor LFMModelDownloader {
    static let shared = LFMModelDownloader()
    static let exactWeightBytes: Int64 = 1_583_152_892

    typealias ProgressHandler =
        @Sendable (
            _ file: String,
            _ aggregateBytes: Int64,
            _ expectedBytes: Int64
        ) -> Void

    func prepare(progress: @escaping ProgressHandler) async throws -> URL {
        let directory = try Self.cachedDirectory()
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )

        if Self.isCachedArtifactValid(at: directory) {
            progress(
                "ready",
                OpenMedLFM.estimatedDownloadBytes,
                OpenMedLFM.estimatedDownloadBytes
            )
            return directory
        }

        var aggregate = try prepareExistingFiles(in: directory)
        progress("Preparing download", aggregate, OpenMedLFM.estimatedDownloadBytes)

        for file in OpenMedLFM.requiredRepositoryFiles {
            try Task.checkCancellation()
            let destination = directory.appending(path: file)
            if Self.isExistingFileValid(destination, named: file) {
                continue
            }

            let completedBytes = aggregate
            try await download(
                file: file,
                destination: destination,
                progress: { fileName, fileBytes in
                    progress(
                        fileName,
                        completedBytes + fileBytes,
                        OpenMedLFM.estimatedDownloadBytes
                    )
                }
            )
            aggregate = completedBytes + Self.fileSize(destination)
            progress(file, aggregate, OpenMedLFM.estimatedDownloadBytes)
        }

        guard Self.isCachedArtifactValid(at: directory) else {
            try? Self.clearRequiredFiles(in: directory)
            throw LFMDownloadError.invalidArtifact(
                "The downloaded files did not match the pinned official LFM2.5 4-bit artifact. The invalid cache was cleared; retry the download."
            )
        }
        return directory
    }

    nonisolated static func cachedDirectory() throws -> URL {
        #if DEBUG && os(macOS)
            // Explicit local artifact override for native development and integration
            // tests. Never changes repository URLs or enables a network fallback.
            if let path = ProcessInfo.processInfo.environment["OPENMED_LFM_MODEL_DIRECTORY"],
                !path.isEmpty
            {
                return URL(filePath: path, directoryHint: .isDirectory)
            }
        #endif
        return try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: OpenMedLFM.repositoryID,
            revision: OpenMedLFM.pinnedRevision
        )
    }

    nonisolated static func cachedBytes() -> Int64 {
        guard let directory = try? cachedDirectory() else { return 0 }
        return OpenMedLFM.requiredRepositoryFiles.reduce(into: 0) { result, file in
            result += fileSize(directory.appending(path: file))
        }
    }

    nonisolated static func isCachedArtifactValid(at directory: URL) -> Bool {
        guard OpenMedLFM.isModelDirectoryReady(directory) else { return false }

        let weights = directory.appending(path: "model.safetensors")
        guard fileSize(weights) == exactWeightBytes else { return false }

        let total = OpenMedLFM.requiredRepositoryFiles.reduce(into: Int64(0)) {
            result,
            file in
            result += fileSize(directory.appending(path: file))
        }
        guard total == OpenMedLFM.estimatedDownloadBytes else { return false }

        struct Header: Decodable {
            let modelType: String

            enum CodingKeys: String, CodingKey {
                case modelType = "model_type"
            }
        }

        let config = directory.appending(path: "config.json")
        guard
            let data = try? Data(contentsOf: config),
            let header = try? JSONDecoder().decode(Header.self, from: data)
        else { return false }
        return header.modelType == "lfm2"
    }

    private func prepareExistingFiles(in directory: URL) throws -> Int64 {
        var bytes: Int64 = 0
        for file in OpenMedLFM.requiredRepositoryFiles {
            let url = directory.appending(path: file)
            if Self.isExistingFileValid(url, named: file) {
                bytes += Self.fileSize(url)
            } else if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
        }
        return bytes
    }

    private nonisolated static func isExistingFileValid(_ url: URL, named file: String) -> Bool {
        let size = fileSize(url)
        if file == "model.safetensors" {
            return size == exactWeightBytes
        }
        return size > 0
    }

    private nonisolated static func clearRequiredFiles(in directory: URL) throws {
        for file in OpenMedLFM.requiredRepositoryFiles {
            let url = directory.appending(path: file)
            if FileManager.default.fileExists(atPath: url.path) {
                try FileManager.default.removeItem(at: url)
            }
        }
    }

    private func download(
        file: String,
        destination: URL,
        progress: @escaping @Sendable (_ file: String, _ fileBytes: Int64) -> Void
    ) async throws {
        let url = try hubURL(for: file)
        let delegate = LFMDownloadProgressDelegate(file: file, progress: progress)
        let session = URLSession(
            configuration: .ephemeral,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        do {
            let (temporaryURL, response) = try await session.download(
                for: URLRequest(url: url),
                delegate: delegate
            )
            guard
                let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else {
                let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
                throw LFMDownloadError.httpStatus(file, statusCode)
            }

            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            if FileManager.default.fileExists(atPath: destination.path) {
                try FileManager.default.removeItem(at: destination)
            }
            try FileManager.default.moveItem(at: temporaryURL, to: destination)
        } catch is CancellationError {
            throw CancellationError()
        } catch let error as NSError where error.code == NSURLErrorCancelled {
            throw CancellationError()
        }
    }

    private func hubURL(for file: String) throws -> URL {
        func encodePath(_ value: String) -> String {
            value.split(separator: "/")
                .map {
                    String($0).addingPercentEncoding(withAllowedCharacters: .urlPathAllowed)
                        ?? String($0)
                }
                .joined(separator: "/")
        }

        guard
            let url = URL(
                string: "https://huggingface.co/\(encodePath(OpenMedLFM.repositoryID))/resolve/\(encodePath(OpenMedLFM.pinnedRevision))/\(encodePath(file))?download=1"
            )
        else { throw LFMDownloadError.invalidURL(file) }
        return url
    }

    private nonisolated static func fileSize(_ url: URL) -> Int64 {
        let values = try? url.resourceValues(forKeys: [.fileSizeKey])
        return Int64(values?.fileSize ?? 0)
    }
}

enum LFMDownloadError: LocalizedError {
    case httpStatus(String, Int)
    case invalidURL(String)
    case invalidArtifact(String)

    var errorDescription: String? {
        switch self {
        case .httpStatus(let file, let code):
            return "HTTP \(code) while downloading \(file)."
        case .invalidURL(let file):
            return "Could not construct the pinned model URL for \(file)."
        case .invalidArtifact(let detail):
            return detail
        }
    }
}

private final class LFMDownloadProgressDelegate: NSObject, URLSessionDownloadDelegate,
    @unchecked Sendable
{
    private let file: String
    private let progress: @Sendable (String, Int64) -> Void

    init(
        file: String,
        progress: @escaping @Sendable (String, Int64) -> Void
    ) {
        self.file = file
        self.progress = progress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {}

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        progress(file, totalBytesWritten)
    }
}
