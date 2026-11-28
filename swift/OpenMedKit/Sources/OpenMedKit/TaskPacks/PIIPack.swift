#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation

    /// One detected identifier, by offsets only.
    ///
    /// A span carries no text from the input. Slice the source yourself with
    /// ``range(in:)`` when you need the matched characters.
    public struct PIIPackSpan: Equatable, Sendable {
        /// Label as emitted by the model, for example `first_name`.
        public let label: String
        /// The label in the portable OpenMed taxonomy, for example `FIRST_NAME`.
        public let canonicalLabel: String
        /// Inclusive Unicode scalar offset in the input text.
        public let start: Int
        /// Exclusive Unicode scalar offset in the input text.
        public let end: Int
        /// Model confidence (0.0 to 1.0).
        public let confidence: Float

        /// The Swift range these scalar offsets cover in `source`.
        public func range(in source: String) -> Range<String.Index>? {
            PostProcessing.scalarRange(in: source, start: start, end: end)
        }
    }

    /// Identifiers detected by ``PIIPack/extract(_:confidenceThreshold:)``.
    public struct PIIPackExtraction: Equatable, Sendable {
        /// Detected spans in input order.
        public let spans: [PIIPackSpan]
        /// Repository id of the model that produced the spans.
        public let sourceModelID: String
        /// The pack's review notice; show it with the output.
        public let notice: OpenMedTaskPackNotice
    }

    /// Text de-identified by ``PIIPack/deidentify(_:policy:confidenceThreshold:)``.
    public struct PIIPackDeidentification: Equatable, Sendable {
        /// The policy result: redacted text, policy name and offset-only actions.
        public let result: PolicyDeidentificationResult
        /// Repository id of the model that detected the spans.
        public let sourceModelID: String
        /// The pack's review notice; show it with the output.
        public let notice: OpenMedTaskPackNotice

        /// The de-identified text.
        public var redactedText: String { result.redactedText }
    }

    /// The portable `pii` task pack: picks, prepares and runs an on-device PII
    /// model without the app naming a repository.
    ///
    /// Candidates come from the bundled `task_packs.json`, in the manifest's
    /// rank order. ``prepare(store:allowDownload:)`` loads the best cached
    /// candidate and downloads one only with explicit consent. On the iOS
    /// Simulator every MLX candidate is skipped and preparation fails with
    /// ``OpenMedTaskPackError/unsupportedOnSimulator``. Inference runs on
    /// device with no network fallback, and results carry offsets, labels and
    /// scores rather than input text.
    ///
    /// ```swift
    /// let pack = try PIIPack()
    /// try await pack.prepare(allowDownload: userAgreedToDownload)
    /// let found = try pack.extract(note)
    /// let redacted = try pack.deidentify(note).redactedText
    /// ```
    ///
    /// Not `Sendable`: like `OpenMed`, use one instance from one isolation
    /// domain.
    public final class PIIPack {
        /// Manifest identifier of this pack.
        public static let packID = "pii"

        private let runner: OpenMedTaskPackRunner

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
        }

        /// The pack declaration from the manifest.
        public var definition: OpenMedTaskPackDefinition { runner.pack }

        /// The pack's review notice.
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

        /// Detect identifiers through `OpenMed.extractPII`.
        ///
        /// - Throws: ``OpenMedTaskPackError/notPrepared`` before a successful
        ///   prepare, or the runtime's inference error.
        public func extract(_ text: String, confidenceThreshold: Float = 0.5) throws -> PIIPackExtraction {
            let (candidate, runtime) = try runner.runtime()
            let entities = try runtime.extractPII(text, confidenceThreshold: confidenceThreshold)
            return PIIPackExtraction(
                spans: entities.map {
                    PIIPackSpan(
                        label: $0.label,
                        canonicalLabel: Policy.canonicalLabel(for: $0.label),
                        start: $0.start,
                        end: $0.end,
                        confidence: $0.confidence
                    )
                },
                sourceModelID: candidate.repoID,
                notice: runner.pack.notice
            )
        }

        /// De-identify text under a bundled policy profile through
        /// `OpenMed.deidentify(_:policy:)`.
        ///
        /// - Parameters:
        ///   - text: Input text.
        ///   - policy: Bundled profile name or alias (default
        ///     `hipaa_safe_harbor`).
        ///   - confidenceThreshold: Minimum span confidence.
        /// - Throws: ``OpenMedTaskPackError/notPrepared``, a `PolicyError`, or
        ///   the runtime's inference error.
        public func deidentify(
            _ text: String,
            policy: String = Policy.defaultName,
            confidenceThreshold: Float = 0.5
        ) throws -> PIIPackDeidentification {
            try deidentify(text, policy: Policy(named: policy), confidenceThreshold: confidenceThreshold)
        }

        /// De-identify text under an already loaded policy profile.
        public func deidentify(
            _ text: String,
            policy: Policy,
            confidenceThreshold: Float = 0.5
        ) throws -> PIIPackDeidentification {
            let (candidate, runtime) = try runner.runtime()
            let result = try runtime.deidentify(text, policy: policy, confidenceThreshold: confidenceThreshold)
            return PIIPackDeidentification(
                result: result,
                sourceModelID: candidate.repoID,
                notice: runner.pack.notice
            )
        }
    }
#endif
