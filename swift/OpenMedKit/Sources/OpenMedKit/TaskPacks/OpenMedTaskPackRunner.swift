#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation

    /// Loads a candidate's local snapshot directory into an `OpenMed` runtime.
    ///
    /// Production packs load the MLX artifact offline. Tests inject a loader
    /// that returns `OpenMed(recognizer:)`, so no model or MLX kernel runs.
    typealias OpenMedTaskPackLoader = (URL) async throws -> OpenMed

    /// Shared resolve and prepare logic behind every task pack.
    ///
    /// Not `Sendable`: like `OpenMed`, use one instance from one isolation
    /// domain.
    final class OpenMedTaskPackRunner {
        let pack: OpenMedTaskPackDefinition
        private let loader: OpenMedTaskPackLoader
        private var prepared: (candidate: OpenMedTaskPackCandidate, runtime: OpenMed)?

        /// The offline MLX loader every pack uses by default.
        static func loadMLX(_ directory: URL) async throws -> OpenMed {
            try await OpenMed.load(backend: .mlx(modelDirectoryURL: directory))
        }

        init(packID: String, manifest: OpenMedTaskPackManifest, loader: @escaping OpenMedTaskPackLoader) throws {
            self.pack = try manifest.pack(packID)
            self.loader = loader
        }

        /// The candidate the pack runs, once prepared.
        var preparedCandidate: OpenMedTaskPackCandidate? { prepared?.candidate }

        /// Usable candidates on this device; reads the manifest only.
        func resolve() -> OpenMedTaskPackResolution {
            OpenMedTaskPackResolution.resolve(
                pack,
                mlxSupported: OpenMedMLXRuntimeSupport.isSupported,
                isSimulator: OpenMedMLXRuntimeSupport.isSimulator
            )
        }

        /// Load the best usable candidate.
        ///
        /// Cached candidates are tried first, in rank order. Only when none of
        /// them loads and `allowDownload` is `true` are candidates downloaded,
        /// again in rank order, falling back to the next one on failure.
        /// Without consent no request is made.
        @discardableResult
        func prepare(store: OpenMedTaskPackStore, allowDownload: Bool) async throws -> OpenMedTaskPackCandidate {
            let resolution = resolve()
            if let refusal = resolution.refusal {
                throw refusal
            }
            var sawLocalCandidate = false
            var tried: Set<String> = []
            for candidate in resolution.candidates {
                let directory: URL?
                do {
                    directory = try store.readyDirectory(for: candidate)
                } catch {
                    directory = nil
                }
                guard let directory else { continue }
                sawLocalCandidate = true
                tried.insert(candidate.repoID)
                if let runtime = try await load(directory) {
                    prepared = (candidate, runtime)
                    return candidate
                }
            }
            guard allowDownload else {
                throw sawLocalCandidate ? OpenMedTaskPackError.loadFailed : OpenMedTaskPackError.notCached
            }
            var downloaded = false
            for candidate in resolution.candidates where !tried.contains(candidate.repoID) {
                let directory: URL
                do {
                    directory = try await store.download(candidate)
                } catch {
                    try Task.checkCancellation()
                    continue
                }
                downloaded = true
                if let runtime = try await load(directory) {
                    prepared = (candidate, runtime)
                    return candidate
                }
            }
            throw downloaded || sawLocalCandidate ? OpenMedTaskPackError.loadFailed : OpenMedTaskPackError.downloadFailed
        }

        /// The prepared runtime and the candidate it runs.
        func runtime() throws -> (candidate: OpenMedTaskPackCandidate, runtime: OpenMed) {
            guard let prepared else {
                throw OpenMedTaskPackError.notPrepared
            }
            return prepared
        }

        /// Load a directory, returning `nil` when the candidate cannot load
        /// so the caller can fall back. Cancellation is rethrown.
        private func load(_ directory: URL) async throws -> OpenMed? {
            do {
                return try await loader(directory)
            } catch {
                try Task.checkCancellation()
                return nil
            }
        }
    }
#endif
