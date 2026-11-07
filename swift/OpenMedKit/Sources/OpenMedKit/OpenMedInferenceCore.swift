import CryptoKit
import Foundation

// MARK: - Stages and checkpoints

/// Cooperative-cancellation stages of a text inference.
///
/// Every stage boundary is a checkpoint. Cancellation requested while a stage
/// runs is observed at the next checkpoint, so the latency to a terminal
/// cancelled state is bounded by the longest single step between checkpoints:
/// one tokenizer pass, one model window, or one merge or serialization pass.
public enum OpenMedInferenceStage: String, Sendable, CaseIterable {
    case tokenization
    case windowing
    case modelPrediction = "model_prediction"
    case merging
    case serialization
}

/// A cooperative cancellation checkpoint handed to each inference stage.
///
/// A checkpoint created with `init()` observes only the calling task's
/// cancellation. Checkpoints issued by `OpenMedRuntime` additionally observe
/// runtime cancellation caused by memory pressure, background expiration or
/// shutdown.
public struct OpenMedInferenceCheckpoint: Sendable {
    let lease: OpenMedInferenceLease?

    /// Create a checkpoint that observes only the calling task's cancellation.
    public init() {
        self.lease = nil
    }

    init(lease: OpenMedInferenceLease) {
        self.lease = lease
    }

    /// Throw if the inference must stop before or during `stage`.
    ///
    /// - Throws: `OpenMedRuntimeError.cancelled` for task cancellation, or the
    ///   runtime reason (`memoryPressure`, `backgroundExpired`, `released`).
    public func check(_ stage: OpenMedInferenceStage) throws {
        if let lease {
            try lease.check(stage)
        } else if Task.isCancelled {
            throw OpenMedRuntimeError.cancelled
        }
    }
}

/// Lock-protected cancellation state for one admitted inference.
///
/// It holds only lifecycle facts: a reason code, a monotonic timestamp and a
/// stage name. It never stores input text, spans or model paths.
final class OpenMedInferenceLease: @unchecked Sendable {
    struct Cancellation: Equatable {
        let reason: OpenMedRuntimeError
        let requestedAtMilliseconds: UInt64
        let observedStage: OpenMedInferenceStage?
    }

    let id: UInt64
    private let clock: any OpenMedRuntimeClock
    private let probe: (@Sendable (OpenMedInferenceStage) -> Void)?
    private let lock = NSLock()
    private var reason: OpenMedRuntimeError?
    private var requestedAt: UInt64 = 0
    private var observedStage: OpenMedInferenceStage?
    private var interrupt: (@Sendable () -> Void)?

    init(
        id: UInt64,
        clock: any OpenMedRuntimeClock,
        probe: (@Sendable (OpenMedInferenceStage) -> Void)?
    ) {
        self.id = id
        self.clock = clock
        self.probe = probe
    }

    /// Request cancellation. The first reason wins; later calls are ignored.
    func cancel(_ reason: OpenMedRuntimeError) {
        let now = clock.nowMilliseconds()
        lock.lock()
        guard self.reason == nil else {
            lock.unlock()
            return
        }
        self.reason = reason
        requestedAt = now
        let interrupt = self.interrupt
        lock.unlock()
        interrupt?()
    }

    /// Install the action that interrupts suspended model work.
    func attachInterrupt(_ action: @escaping @Sendable () -> Void) {
        lock.lock()
        let alreadyCancelled = reason != nil
        if !alreadyCancelled { interrupt = action }
        lock.unlock()
        if alreadyCancelled { action() }
    }

    var cancellation: Cancellation? {
        lock.lock()
        defer { lock.unlock() }
        guard let reason else { return nil }
        return Cancellation(
            reason: reason,
            requestedAtMilliseconds: requestedAt,
            observedStage: observedStage
        )
    }

    func check(_ stage: OpenMedInferenceStage) throws {
        probe?(stage)
        if Task.isCancelled { cancel(.cancelled) }
        lock.lock()
        defer { lock.unlock() }
        guard let reason else { return }
        if observedStage == nil { observedStage = stage }
        throw reason
    }
}

/// Lock-protected set of in-flight leases, so platform signal handlers can
/// cancel work synchronously before the runtime actor processes the event.
final class OpenMedLeaseRegistry: @unchecked Sendable {
    private let lock = NSLock()
    private var leases: [UInt64: OpenMedInferenceLease] = [:]

    var count: Int {
        lock.lock()
        defer { lock.unlock() }
        return leases.count
    }

    func insert(_ lease: OpenMedInferenceLease) {
        lock.lock()
        leases[lease.id] = lease
        lock.unlock()
    }

    func remove(_ lease: OpenMedInferenceLease) {
        lock.lock()
        leases[lease.id] = nil
        lock.unlock()
    }

    func cancelAll(_ reason: OpenMedRuntimeError) {
        lock.lock()
        let active = Array(leases.values)
        lock.unlock()
        for lease in active { lease.cancel(reason) }
    }
}

// MARK: - Model seam

/// How detected spans are merged into PII entities.
public enum OpenMedPIIMergeProfile: String, Sendable, CaseIterable {
    /// BIO token-classification models: semantic patterns may add or widen spans.
    case tokenClassification = "token_classification"
    /// Span-decoding models such as the Privacy Filter: keep model spans and
    /// never add semantic-only matches or widen labels.
    case modelSpansOnly = "model_spans_only"
}

/// A loaded, verified text model session owned by an `OpenMedRuntime`.
///
/// The runtime calls `predictEntities(in:checkpoint:)` only while it holds an
/// active lease on the session, never after it has released the session, and
/// by default never concurrently. Implementations should call
/// `checkpoint.check(_:)` at least at the tokenization, windowing and
/// model-prediction stages, and between model windows.
public protocol OpenMedRuntimeModel: AnyObject, Sendable {
    /// The merge behaviour applied to this model's detections.
    var mergeProfile: OpenMedPIIMergeProfile { get }

    /// Tokenize, window and run the model, returning raw entity predictions
    /// before confidence filtering, with Unicode scalar offsets into `text`.
    func predictEntities(
        in text: String,
        checkpoint: OpenMedInferenceCheckpoint
    ) async throws -> [EntityPrediction]

    /// Release process-wide caches after the runtime dropped its final
    /// reference to a session of this type. The default does nothing.
    static func releaseUnreferencedCaches()
}

extension OpenMedRuntimeModel {
    /// Default: token-classification merging.
    public var mergeProfile: OpenMedPIIMergeProfile { .tokenClassification }

    /// Default: nothing to release.
    public static func releaseUnreferencedCaches() {}
}

// MARK: - Shared staged core

/// Post-model stages shared by the synchronous `OpenMed` APIs and the
/// lifecycle-managed `OpenMedRuntime`, so both produce identical offsets,
/// labels, actions and result bytes for the same model detections.
enum OpenMedInferenceCore {
    static func applyConfidenceThreshold(
        _ entities: [EntityPrediction],
        _ confidenceThreshold: Float
    ) -> [EntityPrediction] {
        entities.filter { $0.confidence >= confidenceThreshold }
    }

    static func mergePII(
        _ entities: [EntityPrediction],
        text: String,
        useSmartMerging: Bool,
        profile: OpenMedPIIMergeProfile,
        checkpoint: OpenMedInferenceCheckpoint
    ) throws -> [EntityPrediction] {
        try checkpoint.check(.merging)
        let repairedEntities = PostProcessing.repairEntitySpans(entities, text: text)
        guard useSmartMerging else {
            return repairedEntities
        }
        try checkpoint.check(.merging)
        switch profile {
        case .modelSpansOnly:
            return PostProcessing.mergePIIEntities(
                repairedEntities,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true,
                allowSemanticOnlyMatches: false,
                allowSemanticLabelExpansion: false
            )
        case .tokenClassification:
            return PostProcessing.mergePIIEntities(
                repairedEntities,
                text: text,
                useSemanticPatterns: true,
                preferModelLabels: true
            )
        }
    }

    static func policyResult(
        _ text: String,
        entities: [EntityPrediction],
        policy: Policy,
        checkpoint: OpenMedInferenceCheckpoint
    ) throws -> PolicyDeidentificationResult {
        try checkpoint.check(.serialization)
        let result = OpenMedRedaction.deidentify(text, entities: entities, policy: policy)
        try checkpoint.check(.serialization)
        return result
    }

    static func methodResult(
        _ text: String,
        entities: [EntityPrediction],
        method: DeidentificationMethod,
        checkpoint: OpenMedInferenceCheckpoint
    ) throws -> DeidentificationResult {
        try checkpoint.check(.serialization)
        let deidentifiedText = OpenMedRedaction.deidentifiedText(
            text,
            entities: entities,
            method: method
        )
        let result = DeidentificationResult(
            originalText: text,
            deidentifiedText: deidentifiedText,
            entities: entities,
            method: method.rawValue
        )
        try checkpoint.check(.serialization)
        return result
    }
}

// MARK: - Redaction

/// Policy and method redaction shared by every OpenMedKit text surface.
enum OpenMedRedaction {
    static func deidentify(
        _ text: String,
        entities: [EntityPrediction],
        policy: Policy
    ) -> PolicyDeidentificationResult {
        let candidates = deidentificationCandidates(entities, in: text)
        var redactedText = text
        var actionRecords: [DeidentifiedSpanAction] = []

        for entity in candidates {
            let canonicalLabel = Policy.canonicalLabel(for: entity.label)
            let action = policy.action(for: entity.label)
            let original = PostProcessing.scalarSubstring(text, start: entity.start, end: entity.end)
            let replacement = replacementText(
                for: action,
                canonicalLabel: canonicalLabel,
                original: original
            )

            actionRecords.append(
                DeidentifiedSpanAction(
                    label: entity.label,
                    canonicalLabel: canonicalLabel,
                    action: action,
                    start: entity.start,
                    end: entity.end,
                    confidence: entity.confidence,
                    replacement: replacement
                )
            )
        }

        for record in actionRecords.reversed() {
            guard let replacement = record.replacement else {
                continue
            }
            guard
                let range = PostProcessing.scalarRange(
                    in: redactedText,
                    start: record.start,
                    end: record.end
                )
            else {
                continue
            }
            redactedText.replaceSubrange(range, with: replacement)
        }

        return PolicyDeidentificationResult(
            redactedText: redactedText,
            policyName: policy.name,
            actions: actionRecords
        )
    }

    static func deidentifiedText(
        _ text: String,
        entities: [EntityPrediction],
        method: DeidentificationMethod
    ) -> String {
        var redacted = text
        for entity in entities.sorted(by: { $0.start > $1.start }) {
            guard
                let range = PostProcessing.scalarRange(
                    in: redacted,
                    start: entity.start,
                    end: entity.end
                )
            else {
                continue
            }
            redacted.replaceSubrange(
                range,
                with: replacementText(for: entity, method: method)
            )
        }
        return redacted
    }

    private static func deidentificationCandidates(
        _ entities: [EntityPrediction],
        in text: String
    ) -> [EntityPrediction] {
        let textLength = PostProcessing.unicodeScalarCount(in: text)
        let sorted =
            entities
            .compactMap { $0.snappedToGraphemeBoundaries(in: text) }
            .filter { $0.start >= 0 && $0.end > $0.start && $0.end <= textLength }
            .sorted {
                if $0.start == $1.start {
                    let lhsLength = entityLength($0)
                    let rhsLength = entityLength($1)
                    if lhsLength == rhsLength {
                        return $0.confidence > $1.confidence
                    }
                    return lhsLength > rhsLength
                }
                return $0.start < $1.start
            }

        var selected: [EntityPrediction] = []
        for entity in sorted {
            let overlapsSelected = selected.contains { existing in
                entity.start < existing.end && entity.end > existing.start
            }
            if !overlapsSelected {
                selected.append(entity)
            }
        }

        return selected.sorted {
            if $0.start == $1.start {
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }
    }

    private static func replacementText(
        for action: PolicyAction,
        canonicalLabel: String,
        original: String
    ) -> String? {
        switch action.redactionEquivalent {
        case .keep:
            return nil
        case .mask:
            return "[\(canonicalLabel)]"
        case .replace:
            return "[\(canonicalLabel)_REPLACED]"
        case .remove:
            return ""
        case .hash:
            return "\(canonicalLabel)_\(stableHash(original))"
        case .redact:
            return "[\(canonicalLabel)]"
        }
    }

    private static func replacementText(
        for entity: EntityPrediction,
        method: DeidentificationMethod
    ) -> String {
        switch method {
        case .mask:
            return "[\(entity.entityType.uppercased())]"
        case .remove:
            return ""
        }
    }

    private static func stableHash(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.prefix(12).map { String(format: "%02x", $0) }.joined()
    }

    private static func entityLength(_ entity: EntityPrediction) -> Int {
        max(0, entity.end - entity.start)
    }
}
