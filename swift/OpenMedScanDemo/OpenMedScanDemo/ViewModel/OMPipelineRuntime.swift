import Foundation
import OpenMedKit
#if canImport(UIKit)
import UIKit
import Vision
#endif

/// Thin, self-contained wrapper around OpenMedKit used by `ScanFlowViewModel`.
/// Caches PII + clinical runtimes per artifact so returning to the same stage
/// is instant. Loading work runs on a background queue to keep the main actor free.
public actor OMPipelineRuntime {
    public static let shared = OMPipelineRuntime()

    private var piiRuntimes: [String: OpenMed] = [:]
    private var relationRuntimes: [String: OpenMedRelationExtractor] = [:]
    private let blockingQueue = DispatchQueue(label: "com.openmed.scan.pipeline", qos: .userInitiated)

    public init() {}

    // MARK: - Public

    public func runPII(
        text: String,
        modelID: ScanModelID,
        confidenceThreshold: Float = 0.5
    ) async throws -> PIIOutput {
        await unloadRelationRuntimes()
        let runtime = try await loadPIIRuntime(for: modelID)
        let predictions = try await runBlocking {
            try runtime.extractPIIChunked(
                text,
                confidenceThreshold: confidenceThreshold,
                chunkTokenLimit: 256,
                tokenOverlap: 32
            )
        }
        let entities = predictions.map(DetectedEntity.init(openMedKit:))
        let masked = Self.mask(text: text, entities: entities)
        return PIIOutput(entities: entities, maskedText: masked)
    }

    public func runClinical(
        maskedText: String,
        labels: [String],
        threshold: Float
    ) async throws -> ClinicalOutput {
        await unloadPIIRuntimes()
        let extractor = try await loadRelationRuntime(for: .glinerRelex)
        let entities = try await runBlocking {
            try extractor.extract(
                maskedText,
                entityLabels: labels,
                relationLabels: Self.defaultRelationLabels,
                threshold: threshold,
                relationThreshold: 0.9,
                flatNER: true
            ).entities
        }
        let converted = entities.map(DetectedEntity.init(zeroShot:))
        return ClinicalOutput(entities: converted)
    }

    #if canImport(UIKit)
    public func recognizeText(in images: [UIImage]) async throws -> TextRecognitionResult {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    var combined: [String] = []
                    for image in images {
                        guard let cgImage = image.cgImage else {
                            throw PipelineError.invalidImage
                        }
                        let request = VNRecognizeTextRequest()
                        request.recognitionLevel = .accurate
                        request.usesLanguageCorrection = true
                        request.automaticallyDetectsLanguage = true
                        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                        try handler.perform([request])
                        let observations = (request.results ?? []).sorted(by: Self.recognitionSort)
                        let lines = observations.compactMap { $0.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines) }
                        let page = lines.filter { !$0.isEmpty }.joined(separator: "\n")
                        if !page.isEmpty { combined.append(page) }
                    }
                    continuation.resume(returning: TextRecognitionResult(
                        text: combined.joined(separator: "\n\n"),
                        pageCount: images.count
                    ))
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    #endif

    // MARK: - Cached runtimes

    public func unloadPIIRuntimes() async {
        guard !piiRuntimes.isEmpty else { return }
        piiRuntimes.removeAll(keepingCapacity: false)
        await clearRuntimeMemoryCacheOnPipelineQueue()
    }

    public func unloadRelationRuntimes() async {
        guard !relationRuntimes.isEmpty else { return }
        relationRuntimes.removeAll(keepingCapacity: false)
        await clearRuntimeMemoryCacheOnPipelineQueue()
    }

    #if canImport(UIKit)
    private static func recognitionSort(
        lhs: VNRecognizedTextObservation,
        rhs: VNRecognizedTextObservation
    ) -> Bool {
        let lhsBox = lhs.boundingBox
        let rhsBox = rhs.boundingBox
        let verticalOverlap = min(lhsBox.maxY, rhsBox.maxY) - max(lhsBox.minY, rhsBox.minY)
        let centerDelta = abs(lhsBox.midY - rhsBox.midY)
        let sameLineThreshold = max(0.004, max(lhsBox.height, rhsBox.height) * 0.65)
        let sameLine = verticalOverlap > min(lhsBox.height, rhsBox.height) * 0.2
            || centerDelta <= sameLineThreshold

        if sameLine { return lhsBox.minX < rhsBox.minX }
        return lhsBox.midY > rhsBox.midY
    }
    #endif

    private func loadPIIRuntime(for modelID: ScanModelID) async throws -> OpenMed {
        if let cached = piiRuntimes[modelID.artifactRepoID] { return cached }
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: modelID.artifactRepoID,
            revision: "main"
        )
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw PipelineError.modelNotReady(modelID)
        }
        let runtime = try await runBlocking {
            try OpenMed(backend: .mlx(modelDirectoryURL: directory))
        }
        piiRuntimes[modelID.artifactRepoID] = runtime
        return runtime
    }

    private func loadRelationRuntime(for modelID: ScanModelID) async throws -> OpenMedRelationExtractor {
        if let cached = relationRuntimes[modelID.artifactRepoID] { return cached }
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: modelID.artifactRepoID,
            revision: "main"
        )
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw PipelineError.modelNotReady(modelID)
        }
        let runtime = try await runBlocking {
            try OpenMedRelationExtractor(modelDirectoryURL: directory)
        }
        relationRuntimes[modelID.artifactRepoID] = runtime
        return runtime
    }

    private func runBlocking<T>(_ work: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            blockingQueue.async {
                do {
                    continuation.resume(returning: try work())
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    private func clearRuntimeMemoryCacheOnPipelineQueue() async {
        await withCheckedContinuation { continuation in
            blockingQueue.async {
                autoreleasepool {
                    OpenMed.clearRuntimeMemoryCache()
                }
                continuation.resume()
            }
        }
    }

    // MARK: - Masking helper

    /// Replaces each detected-entity span with a bracketed uppercase token
    /// (e.g. `[NAME]`). Overlapping spans are resolved by preferring the
    /// span that starts first and extends longest.
    public static func mask(text: String, entities: [DetectedEntity]) -> String {
        let sorted = entities.sorted { lhs, rhs in
            if lhs.start == rhs.start { return lhs.end > rhs.end }
            return lhs.start < rhs.start
        }
        let scalars = Array(text.unicodeScalars)
        var output = ""
        var cursor = 0
        for entity in sorted {
            let safeStart = min(max(entity.start, 0), scalars.count)
            let safeEnd = min(max(entity.end, safeStart), scalars.count)
            guard safeStart >= cursor, safeStart < safeEnd else { continue }
            output.append(String(String.UnicodeScalarView(scalars[cursor..<safeStart])))
            output.append(" [\(entity.category.shortToken)] ")
            cursor = safeEnd
        }
        if cursor < scalars.count {
            output.append(String(String.UnicodeScalarView(scalars[cursor..<scalars.count])))
        }
        return output
    }

    // MARK: - Default relation labels (used by clinical extractor)
    fileprivate static let defaultRelationLabels: [String] = [
        "has symptom",
        "diagnosed with",
        "treated with",
        "takes medication",
        "allergic to",
        "requires test",
        "follow-up for",
        "care plan includes",
    ]
}

// MARK: - Supporting types

public struct PIIOutput: Sendable {
    public let entities: [DetectedEntity]
    public let maskedText: String
    public init(entities: [DetectedEntity], maskedText: String) {
        self.entities = entities
        self.maskedText = maskedText
    }
}

public struct ClinicalOutput: Sendable {
    public let entities: [DetectedEntity]
    public init(entities: [DetectedEntity]) {
        self.entities = entities
    }
}

public struct TextRecognitionResult: Sendable {
    public let text: String
    public let pageCount: Int
}

public enum PipelineError: LocalizedError {
    case modelNotReady(ScanModelID)
    case invalidImage

    public var errorDescription: String? {
        switch self {
        case .modelNotReady(let id):
            return "The \(id.displayName) model is not yet prepared. Tap download first."
        case .invalidImage:
            return "Could not read the scanned image."
        }
    }
}

// MARK: - Conversions

private extension DetectedEntity {
    init(openMedKit prediction: EntityPrediction) {
        self.init(
            label: prediction.label,
            text: prediction.text,
            confidence: Double(prediction.confidence),
            start: prediction.start,
            end: prediction.end
        )
    }

    init(zeroShot entity: OpenMedZeroShotEntity) {
        self.init(
            label: entity.label,
            text: entity.text,
            confidence: Double(entity.score),
            start: entity.start,
            end: entity.end
        )
    }
}

private extension EntityCategory {
    /// Compact token used inside the masked paragraph, e.g. `[NAME]`.
    var shortToken: String {
        switch self {
        case .person:       return "NAME"
        case .date:         return "DATE"
        case .identifier:   return "ID"
        case .contact:      return "CONTACT"
        case .location:     return "ADDRESS"
        case .organization: return "ORG"
        case .condition:    return "CONDITION"
        case .symptom:      return "SYMPTOM"
        case .medication:   return "MED"
        case .dosage:       return "DOSE"
        case .procedure:    return "PROCEDURE"
        case .test:         return "TEST"
        case .allergy:      return "ALLERGY"
        case .followUp:     return "FOLLOW-UP"
        case .carePlan:     return "CARE PLAN"
        case .other:        return "REDACTED"
        }
    }
}
