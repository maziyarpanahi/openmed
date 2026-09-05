import Foundation
import OpenMedKit

#if canImport(UIKit)
    import UIKit
    import Vision
#endif

/// Thin, self-contained wrapper around OpenMedKit used by `ScanFlowViewModel`.
/// Loads model work on a background queue to keep the main actor free. At most
/// one PII runtime is retained: selecting another engine does not load it, and
/// the next explicit run unloads the prior engine before loading the new one.
public actor OMPipelineRuntime {
    public static let shared = OMPipelineRuntime()

    private var piiRuntime: (repositoryID: String, runtime: OpenMed)?
    private let blockingQueue = DispatchQueue(label: "com.openmed.scan.pipeline", qos: .userInitiated)

    public init() {}

    // MARK: - Public

    public func runPII(
        text: String,
        modelID: ScanModelID,
        confidenceThreshold: Float = 0.5
    ) async throws -> PIIOutput {
        guard Self.piiModelIDs.contains(modelID) else {
            throw PipelineError.unsupportedPIIModel(modelID)
        }

        let predictions = try await extractPII(
            text: text,
            modelID: modelID,
            confidenceThreshold: confidenceThreshold
        )
        let entities = predictions.map(DetectedEntity.init(openMedKit:))
        let masked = Self.mask(text: text, entities: entities)
        return PIIOutput(entities: entities, maskedText: masked)
    }

    public func runNER(
        text: String,
        modelID: ScanModelID
    ) async throws -> NEROutput {
        guard Self.nerModelIDs.contains(modelID) else {
            throw PipelineError.unsupportedNERModel(modelID)
        }

        await unloadPIIRuntime()
        let predictions = try await extractNERAndUnload(
            text: text,
            modelID: modelID
        )
        return NEROutput(entities: predictions.map(DetectedEntity.init(openMedKit:)))
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
                        continuation.resume(
                            returning: TextRecognitionResult(
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

    public func unloadPIIRuntime() async {
        guard piiRuntime != nil else { return }
        piiRuntime = nil
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
            let sameLine =
                verticalOverlap > min(lhsBox.height, rhsBox.height) * 0.2
                || centerDelta <= sameLineThreshold

            if sameLine { return lhsBox.minX < rhsBox.minX }
            return lhsBox.midY > rhsBox.midY
        }
    #endif

    private func loadPIIRuntime(for modelID: ScanModelID) async throws -> OpenMed {
        if let cached = piiRuntime, cached.repositoryID == modelID.artifactRepoID {
            return cached.runtime
        }
        await unloadPIIRuntime()
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: modelID.artifactRepoID,
            revision: modelID.revision
        )
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw PipelineError.modelNotReady(modelID)
        }
        let runtime = try await runBlocking {
            try OpenMed(backend: .mlx(modelDirectoryURL: directory))
        }
        piiRuntime = (modelID.artifactRepoID, runtime)
        return runtime
    }

    /// Keeps the strong local runtime reference inside this helper. Once it
    /// returns, `unloadPIIRuntime()` can drop the actor-owned reference before
    /// clearing MLX's buffer cache.
    private func extractPII(
        text: String,
        modelID: ScanModelID,
        confidenceThreshold: Float
    ) async throws -> [EntityPrediction] {
        let runtime = try await loadPIIRuntime(for: modelID)
        return try await runBlocking {
            try runtime.extractPIIChunked(
                text,
                confidenceThreshold: confidenceThreshold,
                chunkTokenLimit: 256,
                tokenOverlap: 32
            )
        }
    }

    /// NER models are deliberately transient. Each explicit run creates one
    /// runtime, completes inference, drops the final strong reference, and
    /// clears MLX's buffer cache before returning to the UI.
    private func extractNERAndUnload(
        text: String,
        modelID: ScanModelID
    ) async throws -> [EntityPrediction] {
        do {
            let predictions = try await runTransientNER(text: text, modelID: modelID)
            await clearRuntimeMemoryCacheOnPipelineQueue()
            return predictions
        } catch {
            await clearRuntimeMemoryCacheOnPipelineQueue()
            throw error
        }
    }

    /// The runtime is owned only by this function. Returning guarantees its
    /// final strong reference has left scope before the caller clears MLX.
    private func runTransientNER(
        text: String,
        modelID: ScanModelID
    ) async throws -> [EntityPrediction] {
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: modelID.artifactRepoID,
            revision: modelID.revision
        )
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw PipelineError.modelNotReady(modelID)
        }
        let runtime = try await runBlocking {
            try OpenMed(backend: .mlx(modelDirectoryURL: directory))
        }
        return try await runBlocking {
            try runtime.analyzeTextChunked(
                text,
                confidenceThreshold: 0.5,
                chunkTokenLimit: 256,
                tokenOverlap: 32
            )
        }
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

    private static let piiModelIDs: Set<ScanModelID> = [
        .piiLiteClinical,
        .openaiPrivacyFilter,
        .multilingualPrivacyFilter,
    ]

    private static let nerModelIDs: Set<ScanModelID> = [
        .nerDisease,
        .nerMedication,
        .nerAnatomy,
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

public struct NEROutput: Sendable {
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
    case unsupportedPIIModel(ScanModelID)
    case unsupportedNERModel(ScanModelID)
    case invalidImage

    public var errorDescription: String? {
        switch self {
        case .modelNotReady(let id):
            return "The \(id.displayName) model is not yet prepared. Tap download first."
        case .unsupportedPIIModel(let id):
            return "The \(id.displayName) model is not a PII redaction engine."
        case .unsupportedNERModel(let id):
            return "The \(id.displayName) model is not a supported clinical NER model."
        case .invalidImage:
            return "Could not read the scanned image."
        }
    }
}

// MARK: - Conversions

extension DetectedEntity {
    fileprivate init(openMedKit prediction: EntityPrediction) {
        self.init(
            label: prediction.label,
            text: prediction.text,
            confidence: Double(prediction.confidence),
            start: prediction.start,
            end: prediction.end
        )
    }

}

extension EntityCategory {
    /// Compact token used inside the masked paragraph, e.g. `[NAME]`.
    fileprivate var shortToken: String {
        switch self {
        case .person: return "NAME"
        case .date: return "DATE"
        case .identifier: return "ID"
        case .contact: return "CONTACT"
        case .location: return "ADDRESS"
        case .organization: return "ORG"
        case .condition: return "CONDITION"
        case .symptom: return "SYMPTOM"
        case .medication: return "MED"
        case .dosage: return "DOSE"
        case .procedure: return "PROCEDURE"
        case .test: return "TEST"
        case .allergy: return "ALLERGY"
        case .followUp: return "FOLLOW-UP"
        case .carePlan: return "CARE PLAN"
        case .other: return "REDACTED"
        }
    }
}
