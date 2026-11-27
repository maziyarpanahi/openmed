import Foundation

/// Value-free progress for one model download.
///
/// Carries a repository-relative file name and counts only: never a URL,
/// token, revision, repository id or absolute path.
public struct OpenMedModelDownloadProgress: Equatable, Sendable {
    /// Repository-relative path of the file this event is about, for example
    /// `tokenizer/tokenizer.json`.
    public let fileName: String
    /// Bytes received (or already present) for `fileName`.
    public let bytesDone: Int64
    /// Expected size of `fileName`, or `nil` while the server has not
    /// declared one. Equals `bytesDone` on the event that completes the file.
    public let bytesTotal: Int64?
    /// Files finished so far: downloaded, already cached, or optional files
    /// the repository does not have.
    public let filesDone: Int
    /// Files the download plans to fetch. Starts at 1 (`openmed-mlx.json`)
    /// and grows once, when the file list is known.
    public let filesTotal: Int
    /// Bytes received by this download across every file. Never decreases.
    public let totalBytesDone: Int64

    /// Create a progress value.
    public init(
        fileName: String,
        bytesDone: Int64,
        bytesTotal: Int64?,
        filesDone: Int,
        filesTotal: Int,
        totalBytesDone: Int64
    ) {
        self.fileName = fileName
        self.bytesDone = bytesDone
        self.bytesTotal = bytesTotal
        self.filesDone = filesDone
        self.filesTotal = filesTotal
        self.totalBytesDone = totalBytesDone
    }
}

/// A running, cancellable MLX model download with byte-level progress.
///
/// ```swift
/// let download = OpenMedModelStore.startMLXModelDownload(repoID: "OpenMed/<name>-mlx")
/// Task {
///     for await event in download.progress {
///         print(event.fileName, event.bytesDone, event.bytesTotal ?? -1)
///     }
/// }
/// let directory = try await download.value
/// ```
///
/// The download starts immediately. `cancel()` stops the current transfer,
/// removes the files this download wrote, never writes the ready marker, and
/// makes ``value`` throw `CancellationError`. The snapshot is then `missing`
/// or `partial`. `progress` finishes when the download ends either way.
public final class OpenMedModelDownloadTask: Sendable {
    /// Progress events in order. Only the newest events are buffered when the
    /// consumer falls behind, so the last event is never dropped.
    public let progress: AsyncStream<OpenMedModelDownloadProgress>
    private let task: Task<URL, any Error>

    init(operation: @escaping @Sendable (OpenMedDownloadProgressReporter) async throws -> URL) {
        let (stream, continuation) = AsyncStream.makeStream(
            of: OpenMedModelDownloadProgress.self,
            bufferingPolicy: .bufferingNewest(1_024)
        )
        progress = stream
        let reporter = OpenMedDownloadProgressReporter(continuation: continuation)
        task = Task {
            defer { continuation.finish() }
            return try await operation(reporter)
        }
    }

    /// The ready snapshot directory once the download finishes.
    ///
    /// - Throws: ``OpenMedModelStoreError``, or `CancellationError` after
    ///   ``cancel()``.
    public var value: URL {
        get async throws { try await task.value }
    }

    /// Whether ``cancel()`` was called.
    public var isCancelled: Bool { task.isCancelled }

    /// Stop the download and remove the partial files it wrote.
    public func cancel() {
        task.cancel()
    }
}

extension OpenMedModelStore {
    /// Start downloading an MLX model snapshot and return a handle that
    /// reports byte progress and can be cancelled.
    ///
    /// Same files, cache layout and confinement rules as
    /// ``downloadMLXModel(repoID:revision:cacheDirectory:)``, which remains
    /// the simple `async` form of this call.
    public static func startMLXModelDownload(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) -> OpenMedModelDownloadTask {
        startMLXModelDownload(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory, session: .shared)
    }

    static func startMLXModelDownload(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        session: URLSession
    ) -> OpenMedModelDownloadTask {
        OpenMedModelDownloadTask { reporter in
            try await downloadMLXModel(
                repoID: repoID, revision: revision, cacheDirectory: cacheDirectory,
                session: session, progress: reporter)
        }
    }
}

/// Turns per-file transfer callbacks into ordered progress events.
final class OpenMedDownloadProgressReporter: @unchecked Sendable {
    // Every mutable field is read and written under `lock`, and events are
    // yielded under it, so the stream sees them in a single total order.
    private let lock = NSLock()
    private let continuation: AsyncStream<OpenMedModelDownloadProgress>.Continuation?
    private var filesDone = 0
    private var filesTotal = 0
    private var totalBytesDone: Int64 = 0
    private var currentBytes: Int64 = 0

    init(continuation: AsyncStream<OpenMedModelDownloadProgress>.Continuation?) {
        self.continuation = continuation
    }

    func plan(files count: Int) {
        lock.lock()
        defer { lock.unlock() }
        filesTotal += count
    }

    func fileStarted() {
        lock.lock()
        defer { lock.unlock() }
        currentBytes = 0
    }

    func fileProgress(_ fileName: String, bytesDone: Int64, bytesTotal: Int64?) {
        lock.lock()
        defer { lock.unlock() }
        guard bytesDone > currentBytes else { return }
        totalBytesDone += bytesDone - currentBytes
        currentBytes = bytesDone
        emit(fileName, bytesDone: bytesDone, bytesTotal: bytesTotal)
    }

    /// Complete a file. `transferredBytes` is the size written by this
    /// download, or `nil` when nothing was transferred.
    func fileFinished(_ fileName: String, sizeBytes: Int64, transferredBytes: Int64?) {
        lock.lock()
        defer { lock.unlock() }
        if let transferredBytes, transferredBytes > currentBytes {
            totalBytesDone += transferredBytes - currentBytes
        }
        currentBytes = 0
        filesDone = min(filesDone + 1, filesTotal)
        emit(fileName, bytesDone: sizeBytes, bytesTotal: sizeBytes)
    }

    private func emit(_ fileName: String, bytesDone: Int64, bytesTotal: Int64?) {
        continuation?.yield(
            OpenMedModelDownloadProgress(
                fileName: fileName,
                bytesDone: bytesDone,
                bytesTotal: bytesTotal,
                filesDone: filesDone,
                filesTotal: filesTotal,
                totalBytesDone: totalBytesDone
            ))
    }
}

/// Delegate-based per-file downloads.
///
/// The async `URLSession.download(for:delegate:)` API never calls
/// `urlSession(_:downloadTask:didWriteData:totalBytesWritten:totalBytesExpectedToWrite:)`,
/// so byte progress needs a session that owns this delegate. The session
/// copies the caller's configuration, which keeps injected `URLProtocol`
/// stubs working in tests.
final class OpenMedFileTransferSession: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    typealias ProgressHandler = @Sendable (_ bytesDone: Int64, _ bytesTotal: Int64?) -> Void

    private struct Transfer {
        let continuation: CheckedContinuation<(URL, HTTPURLResponse), any Error>
        let progress: ProgressHandler
        var staged: URL?
    }

    // `transfers` is only touched under `lock`; `session` is set once in init.
    private let lock = NSLock()
    private var transfers: [Int: Transfer] = [:]
    private var session: URLSession!

    init(configuration: URLSessionConfiguration) {
        super.init()
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: configuration, delegate: self, delegateQueue: queue)
    }

    /// Let running transfers finish, then release the session and its delegate.
    func invalidate() {
        session.finishTasksAndInvalidate()
    }

    /// Download `request` to a private staging file and return it with the
    /// response. The caller moves or removes the staged file.
    ///
    /// - Throws: `CancellationError` when the calling task is cancelled, or
    ///   ``OpenMedModelStoreError/transferFailed`` for transport failures.
    func download(_ request: URLRequest, progress: @escaping ProgressHandler) async throws -> (URL, HTTPURLResponse) {
        try Task.checkCancellation()
        let task = session.downloadTask(with: request)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                lock.lock()
                transfers[task.taskIdentifier] = Transfer(continuation: continuation, progress: progress)
                lock.unlock()
                task.resume()
                if Task.isCancelled { task.cancel() }
            }
        } onCancel: {
            task.cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let response = downloadTask.response as? HTTPURLResponse,
            (200..<300).contains(response.statusCode)
        else { return }
        lock.lock()
        let progress = transfers[downloadTask.taskIdentifier]?.progress
        lock.unlock()
        progress?(totalBytesWritten, totalBytesExpectedToWrite >= 0 ? totalBytesExpectedToWrite : nil)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // The system deletes `location` when this method returns.
        let staged = FileManager.default.temporaryDirectory.appending(
            path: "openmed-transfer-\(UUID().uuidString)")
        guard (try? FileManager.default.moveItem(at: location, to: staged)) != nil else { return }
        lock.lock()
        transfers[downloadTask.taskIdentifier]?.staged = staged
        lock.unlock()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        lock.lock()
        let transfer = transfers.removeValue(forKey: task.taskIdentifier)
        lock.unlock()
        guard let transfer else { return }
        if let error {
            transfer.staged.map { try? FileManager.default.removeItem(at: $0) }
            let cancelled = (error as? URLError)?.code == .cancelled
            transfer.continuation.resume(
                throwing: cancelled ? CancellationError() : OpenMedModelStoreError.transferFailed)
            return
        }
        guard let staged = transfer.staged, let response = task.response as? HTTPURLResponse else {
            transfer.staged.map { try? FileManager.default.removeItem(at: $0) }
            transfer.continuation.resume(throwing: OpenMedModelStoreError.invalidResponse)
            return
        }
        transfer.continuation.resume(returning: (staged, response))
    }
}
