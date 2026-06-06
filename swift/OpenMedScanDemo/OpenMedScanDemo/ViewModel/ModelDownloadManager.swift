import Foundation
import Combine
import OpenMedKit
import os.log

/// One source of truth for model-download state, consumed by every stage
/// that gates on a particular model. Wraps `StreamingModelDownloader` so
/// the UI gets real byte-level progress, and still uses `OpenMedModelStore`
/// to probe cache state (keeping the inference-side API untouched).
@MainActor
public final class ModelDownloadManager: ObservableObject {
    public static let shared = ModelDownloadManager()

    public struct Entry: Identifiable, Equatable {
        public let id: ScanModelID
        public var state: ModelDownloadState
        public var bytesOnDisk: Int64
        public var bytesEstimatedTotal: Int64?

        public var fraction: Double? {
            guard let total = bytesEstimatedTotal, total > 0 else { return nil }
            return max(0, min(1, Double(bytesOnDisk) / Double(total)))
        }
    }

    @Published public private(set) var entries: [ScanModelID: Entry] = [:]
    @Published public private(set) var activeDownloadID: ScanModelID?

    private var tasks: [ScanModelID: Task<Void, Never>] = [:]
    private let log = Logger(subsystem: "com.openmed.scan", category: "downloads")

    public init() {
        for id in ScanModelID.allCases {
            entries[id] = Entry(
                id: id,
                state: .missing,
                bytesOnDisk: 0,
                bytesEstimatedTotal: id.conservativeTotalBytes
            )
        }
        refreshAll()
    }

    // MARK: - Public API

    public func refreshAll() {
        for id in ScanModelID.allCases {
            refresh(id)
        }
    }

    public func refresh(_ id: ScanModelID) {
        let cacheState = currentCacheState(for: id)
        let bytes = directoryBytes(for: id)
        var entry = entries[id] ?? Entry(id: id, state: .missing, bytesOnDisk: 0, bytesEstimatedTotal: id.conservativeTotalBytes)
        switch cacheState {
        case .ready:
            entry.state = .ready
        case .partial:
            if case .downloading = entry.state {
                // Don't clobber an in-flight state.
            } else {
                entry.state = .partial(bytesOnDisk: bytes, bytesExpected: entry.bytesEstimatedTotal)
            }
        case .missing:
            if case .downloading = entry.state { /* keep */ }
            else { entry.state = .missing }
        }
        entry.bytesOnDisk = bytes
        entries[id] = entry
    }

    /// Starts downloading a model if it isn't already ready or in-flight.
    public func prepare(_ id: ScanModelID) {
        guard tasks[id] == nil else { return }
        if case .ready = entries[id]?.state { return }

        let startingBytes = entries[id]?.bytesOnDisk ?? 0
        let totalEstimate = entries[id]?.bytesEstimatedTotal
        markState(id, .downloading(
            bytesDownloaded: startingBytes,
            bytesExpected: totalEstimate,
            bytesPerSecond: nil
        ))
        activeDownloadID = id
        HapticsCenter.impact(.soft)

        let task = Task { [weak self] in
            guard let self else { return }
            let repoID = id.artifactRepoID
            var lastAggregate: Int64 = startingBytes
            var lastTimestamp = Date()

            do {
                try await StreamingModelDownloader.shared.prepare(
                    repoID: repoID,
                    revision: "main",
                    progress: { [weak self] _, _, fileTotal, aggregate in
                        let now = Date()
                        let dt = now.timeIntervalSince(lastTimestamp)
                        let rate: Double? = dt > 0.2 ? Double(aggregate - lastAggregate) / dt : nil
                        let committed = aggregate
                        let fileTotalCaptured = fileTotal
                        Task { @MainActor in
                            guard let self else { return }
                            guard var entry = self.entries[id], case .downloading = entry.state else { return }
                            let total = entry.bytesEstimatedTotal
                            entry.bytesOnDisk = max(entry.bytesOnDisk, committed)
                            entry.state = .downloading(
                                bytesDownloaded: committed,
                                bytesExpected: total ?? fileTotalCaptured,
                                bytesPerSecond: rate
                            )
                            self.entries[id] = entry
                        }
                        lastAggregate = committed
                        lastTimestamp = now
                    }
                )
                await MainActor.run {
                    self.finish(id, success: true)
                    HapticsCenter.notify(.success)
                }
            } catch is CancellationError {
                await MainActor.run {
                    self.finish(id, success: false, cancelled: true)
                    HapticsCenter.impact(.rigid)
                }
            } catch {
                self.log.error("Download failed for \(id.rawValue, privacy: .public): \(error.localizedDescription, privacy: .public)")
                await MainActor.run {
                    self.finish(id, success: false, error: error.localizedDescription)
                    HapticsCenter.notify(.warning)
                }
            }
        }
        tasks[id] = task
    }

    public func cancel(_ id: ScanModelID) {
        tasks[id]?.cancel()
        tasks[id] = nil
        markState(id, .cancelled)
        if activeDownloadID == id { activeDownloadID = nil }
        Task {
            await StreamingModelDownloader.shared.cancel()
        }
    }

    public func state(for id: ScanModelID) -> ModelDownloadState {
        entries[id]?.state ?? .missing
    }

    public var anyActiveDownload: Entry? {
        if let active = activeDownloadID { return entries[active] }
        return entries.values.first(where: { $0.state.isActive })
    }

    // MARK: - Internals

    private func finish(_ id: ScanModelID, success: Bool, cancelled: Bool = false, error: String? = nil) {
        tasks[id] = nil
        if activeDownloadID == id { activeDownloadID = nil }

        if success {
            markState(id, .ready)
            entries[id]?.bytesOnDisk = directoryBytes(for: id)
        } else if cancelled {
            markState(id, .cancelled)
            refresh(id)
        } else {
            markState(id, .failed(message: error ?? "Unknown error"))
        }
    }

    private func markState(_ id: ScanModelID, _ state: ModelDownloadState) {
        if entries[id] == nil {
            entries[id] = Entry(id: id, state: state, bytesOnDisk: 0, bytesEstimatedTotal: id.conservativeTotalBytes)
        } else {
            entries[id]?.state = state
        }
    }

    private func currentCacheState(for id: ScanModelID) -> OpenMedMLXModelCacheState {
        (try? OpenMedModelStore.mlxModelCacheState(
            repoID: id.artifactRepoID,
            revision: "main"
        )) ?? .missing
    }

    private func directoryBytes(for id: ScanModelID) -> Int64 {
        guard let dir = try? OpenMedModelStore.cachedMLXModelDirectory(
            repoID: id.artifactRepoID,
            revision: "main"
        ) else { return 0 }
        guard FileManager.default.fileExists(atPath: dir.path) else { return 0 }
        let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey]
        )
        var total: Int64 = 0
        while let url = enumerator?.nextObject() as? URL {
            let values = try? url.resourceValues(forKeys: [.totalFileAllocatedSizeKey, .fileAllocatedSizeKey])
            if let size = values?.totalFileAllocatedSize ?? values?.fileAllocatedSize {
                total += Int64(size)
            }
        }
        return total
    }
}

public extension ScanModelID {
    /// Hugging Face repo-id for this model's MLX artifact.
    var artifactRepoID: String {
        switch self {
        case .piiLiteClinical:     return "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx"
        case .openaiPrivacyFilter: return "OpenMed/privacy-filter-nemotron-mlx-8bit"
        case .multilingualPrivacyFilter: return "OpenMed/privacy-filter-multilingual-mlx-8bit"
        case .glinerRelex:         return "OpenMed/gliner-relex-base-v1.0-mlx"
        }
    }

    /// Conservative upper bound for the progress bar. The real total comes
    /// in from the HTTP headers during download and overrides this estimate.
    var conservativeTotalBytes: Int64 {
        switch self {
        case .piiLiteClinical:     return 278 * 1024 * 1024
        case .openaiPrivacyFilter: return 1_550 * 1024 * 1024
        case .multilingualPrivacyFilter: return 1_550 * 1024 * 1024
        case .glinerRelex:         return 230 * 1024 * 1024
        }
    }
}
