#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation

    /// One medication mention found in the input text.
    ///
    /// A mention is extracted text data only: offsets, a label, a score and
    /// provenance. It carries no dose, interaction or recommended action, and
    /// it carries the pack's extraction-only ``notice`` so the app can show it
    /// wherever the mention is shown. It carries no text from the input; slice
    /// the source with ``range(in:)``.
    public struct MedicationMention: Equatable, Sendable {
        /// The pack's canonical label for the mention (`CHEMICAL`).
        public let normalizedLabel: String
        /// Label as emitted by the model, for example `CHEM`.
        public let label: String
        /// Inclusive Unicode scalar offset in the input text.
        public let start: Int
        /// Exclusive Unicode scalar offset in the input text.
        public let end: Int
        /// Model confidence (0.0 to 1.0).
        public let confidence: Float
        /// Repository id of the model that produced the mention.
        public let sourceModelID: String
        /// The pack's extraction-only notice.
        public let notice: OpenMedTaskPackNotice

        /// The Swift range these scalar offsets cover in `source`.
        public func range(in source: String) -> Range<String.Index>? {
            PostProcessing.scalarRange(in: source, start: start, end: end)
        }
    }

    /// Mentions found by ``MedicationPack/extract(_:confidenceThreshold:)``.
    public struct MedicationExtraction: Equatable, Sendable {
        /// Mentions ordered by offset.
        public let mentions: [MedicationMention]
        /// Repository id of the model that produced the mentions.
        public let sourceModelID: String
        /// The pack's extraction-only notice; show it with the output.
        public let notice: OpenMedTaskPackNotice
    }

    /// The portable `medication` task pack: picks, prepares and runs an
    /// on-device medication NER model and returns medication mentions.
    ///
    /// Candidates come from the bundled `task_packs.json`, in the manifest's
    /// rank order, and ``prepare(store:allowDownload:)`` follows the same
    /// cache-first, consent-gated rules as ``PIIPack``. The pack extracts text
    /// mentions only. It gives no dosing or interaction advice, links no drug
    /// terminology and triggers no follow-up action: results are data for the
    /// app to show, always with the pack's extraction-only notice. Creating
    /// the pack fails closed when the manifest does not attach that notice.
    ///
    /// Not `Sendable`: like `OpenMed`, use one instance from one isolation
    /// domain.
    public final class MedicationPack {
        /// Manifest identifier of this pack.
        public static let packID = "medication"
        /// The notice the medication pack must carry.
        public static let noticeKey = "openmed.notice.medication.extraction_only"

        private let runner: OpenMedTaskPackRunner
        private let labels: Set<String>

        /// Create the pack from the bundled manifest.
        ///
        /// - Throws: ``OpenMedTaskPackError`` when the manifest is missing,
        ///   invalid, does not declare the pack, or does not attach the
        ///   extraction-only notice.
        public convenience init() throws {
            try self.init(manifest: OpenMedTaskPackManifest.bundled())
        }

        /// Create the pack from an already decoded manifest.
        public convenience init(manifest: OpenMedTaskPackManifest) throws {
            try self.init(manifest: manifest, loader: OpenMedTaskPackRunner.loadMLX)
        }

        /// Internal seam: tests inject a loader that returns a fake pipeline.
        init(manifest: OpenMedTaskPackManifest, loader: @escaping OpenMedTaskPackLoader) throws {
            let runner = try OpenMedTaskPackRunner(packID: Self.packID, manifest: manifest, loader: loader)
            guard runner.pack.notice.key == Self.noticeKey else {
                throw OpenMedTaskPackError.manifestInvalid
            }
            self.runner = runner
            labels = Set(runner.pack.canonicalLabels)
        }

        /// The pack declaration from the manifest.
        public var definition: OpenMedTaskPackDefinition { runner.pack }

        /// The pack's extraction-only notice, for display.
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

        /// The pack label for a model label, or `nil` when the label is
        /// outside the pack's canonical labels.
        public func normalizedLabel(forModelLabel label: String) -> String? {
            let canonical = Policy.canonicalLabel(for: label)
            return labels.contains(canonical) ? canonical : nil
        }

        /// Extract medication mentions through `OpenMed.analyzeText`.
        ///
        /// Mentions whose model label does not normalize to one of the pack's
        /// canonical labels are dropped.
        ///
        /// - Throws: ``OpenMedTaskPackError/notPrepared`` before a successful
        ///   prepare, or the runtime's inference error.
        public func extract(_ text: String, confidenceThreshold: Float = 0.5) throws -> MedicationExtraction {
            let (candidate, runtime) = try runner.runtime()
            let notice = runner.pack.notice
            let predictions = try runtime.analyzeText(text, confidenceThreshold: confidenceThreshold)
            let mentions =
                predictions
                .compactMap { prediction -> MedicationMention? in
                    guard let normalized = normalizedLabel(forModelLabel: prediction.label) else { return nil }
                    return MedicationMention(
                        normalizedLabel: normalized,
                        label: prediction.label,
                        start: prediction.start,
                        end: prediction.end,
                        confidence: prediction.confidence,
                        sourceModelID: candidate.repoID,
                        notice: notice
                    )
                }
                .sorted { ($0.start, $0.end) < ($1.start, $1.end) }
            return MedicationExtraction(mentions: mentions, sourceModelID: candidate.repoID, notice: notice)
        }
    }
#endif
