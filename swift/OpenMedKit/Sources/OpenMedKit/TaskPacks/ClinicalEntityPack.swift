#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation

    /// One clinical mention whose model label maps to a pack category.
    ///
    /// An entity carries no text from the input. Slice the source yourself with
    /// ``range(in:)`` when you need the matched characters.
    public struct ClinicalEntity: Equatable, Sendable {
        /// The pack's canonical label for this mention, one of
        /// ``OpenMedTaskPackDefinition/canonicalLabels`` (for example
        /// `DISEASE`).
        public let category: String
        /// Label as emitted by the model.
        public let label: String
        /// Inclusive Unicode scalar offset in the input text.
        public let start: Int
        /// Exclusive Unicode scalar offset in the input text.
        public let end: Int
        /// Model confidence (0.0 to 1.0).
        public let confidence: Float
        /// Repository id of the model that produced the mention.
        public let sourceModelID: String

        /// The Swift range these scalar offsets cover in `source`.
        public func range(in source: String) -> Range<String.Index>? {
            PostProcessing.scalarRange(in: source, start: start, end: end)
        }
    }

    /// Mentions found by ``ClinicalEntityPack/extract(_:confidenceThreshold:)``.
    public struct ClinicalEntityExtraction: Equatable, Sendable {
        /// Mapped mentions ordered by offset.
        public let entities: [ClinicalEntity]
        /// Repository id of the model that produced the mentions.
        public let sourceModelID: String
        /// The pack's extraction-only notice; show it with the output.
        public let notice: OpenMedTaskPackNotice
    }

    /// The portable `clinical_entity` task pack: picks, prepares and runs an
    /// on-device clinical NER model and maps its labels to the pack's
    /// canonical categories.
    ///
    /// Candidates come from the bundled `task_packs.json`, in the manifest's
    /// rank order, and ``prepare(store:allowDownload:)`` follows the same
    /// cache-first, consent-gated rules as ``PIIPack``. ``extract(_:confidenceThreshold:)``
    /// runs `OpenMed.analyzeText`, normalizes each model label with the
    /// portable OpenMed taxonomy, and keeps a mention only when that label is
    /// one of the pack's canonical labels. Other labels are dropped rather
    /// than guessed. The output reports mentions and offsets only; it does
    /// not diagnose, rank or recommend care.
    ///
    /// Not `Sendable`: like `OpenMed`, use one instance from one isolation
    /// domain.
    public final class ClinicalEntityPack {
        /// Manifest identifier of this pack.
        public static let packID = "clinical_entity"

        private let runner: OpenMedTaskPackRunner
        private let categories: Set<String>

        /// Create the pack from the bundled manifest.
        ///
        /// - Throws: ``OpenMedTaskPackError`` when the manifest is missing,
        ///   invalid or does not declare the pack.
        public convenience init() throws {
            try self.init(manifest: OpenMedTaskPackManifest.bundled())
        }

        /// Create the pack from an already decoded manifest.
        public convenience init(manifest: OpenMedTaskPackManifest) throws {
            try self.init(manifest: manifest, loader: OpenMedTaskPackRunner.loadMLX)
        }

        /// Internal seam: tests inject a loader that returns a fake pipeline.
        init(manifest: OpenMedTaskPackManifest, loader: @escaping OpenMedTaskPackLoader) throws {
            runner = try OpenMedTaskPackRunner(packID: Self.packID, manifest: manifest, loader: loader)
            categories = Set(runner.pack.canonicalLabels)
        }

        /// The pack declaration from the manifest.
        public var definition: OpenMedTaskPackDefinition { runner.pack }

        /// The pack's extraction-only notice.
        public var notice: OpenMedTaskPackNotice { runner.pack.notice }

        /// The candidate in use after a successful prepare.
        public var preparedCandidate: OpenMedTaskPackCandidate? { runner.preparedCandidate }

        /// Ordered candidates this device can run, and the ones it skips.
        ///
        /// Reads the bundled manifest only; makes no network request.
        public func resolve() -> OpenMedTaskPackResolution {
            runner.resolve()
        }

        /// Load the best available candidate.
        ///
        /// - Parameters:
        ///   - store: Model cache to read and, with consent, fill.
        ///   - allowDownload: Explicit consent to download a candidate when
        ///     none is cached. Defaults to `false`, which fails closed with
        ///     ``OpenMedTaskPackError/notCached`` and makes no request.
        /// - Returns: The candidate now in use.
        /// - Throws: ``OpenMedTaskPackError`` with a value-free reason.
        @discardableResult
        public func prepare(
            store: OpenMedTaskPackStore = .standard,
            allowDownload: Bool = false
        ) async throws -> OpenMedTaskPackCandidate {
            try await runner.prepare(store: store, allowDownload: allowDownload)
        }

        /// The pack category for a model label, or `nil` when the label is
        /// outside the pack's canonical labels.
        public func category(forModelLabel label: String) -> String? {
            let canonical = Policy.canonicalLabel(for: label)
            return categories.contains(canonical) ? canonical : nil
        }

        /// Extract clinical mentions through `OpenMed.analyzeText`.
        ///
        /// - Throws: ``OpenMedTaskPackError/notPrepared`` before a successful
        ///   prepare, or the runtime's inference error.
        public func extract(_ text: String, confidenceThreshold: Float = 0.5) throws -> ClinicalEntityExtraction {
            let (candidate, runtime) = try runner.runtime()
            let predictions = try runtime.analyzeText(text, confidenceThreshold: confidenceThreshold)
            let entities =
                predictions
                .compactMap { prediction -> ClinicalEntity? in
                    guard let category = category(forModelLabel: prediction.label) else { return nil }
                    return ClinicalEntity(
                        category: category,
                        label: prediction.label,
                        start: prediction.start,
                        end: prediction.end,
                        confidence: prediction.confidence,
                        sourceModelID: candidate.repoID
                    )
                }
                .sorted { ($0.start, $0.end) < ($1.start, $1.end) }
            return ClinicalEntityExtraction(
                entities: entities,
                sourceModelID: candidate.repoID,
                notice: runner.pack.notice
            )
        }
    }
#endif
