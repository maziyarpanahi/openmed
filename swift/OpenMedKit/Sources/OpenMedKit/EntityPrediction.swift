import CryptoKit
import Foundation

/// A single entity predicted by the NER pipeline.
public struct EntityPrediction: Codable, Equatable, Sendable {
    /// The entity type label (e.g., "first_name", "date_of_birth", "ssn").
    public let label: String

    /// The text span matched by this entity.
    public let text: String

    /// Model confidence score (0.0 – 1.0).
    public let confidence: Float

    /// Start character offset in the original text.
    public let start: Int

    /// End character offset in the original text (exclusive).
    public let end: Int

    /// Python-compatible entity type used by de-identification exports.
    public var entityType: String {
        label
    }

    public init(label: String, text: String, confidence: Float, start: Int, end: Int) {
        self.label = label
        self.text = text
        self.confidence = confidence
        self.start = start
        self.end = end
    }

    /// Stable SHA-256 digest of the matched text span for PHI-safe diagnostics.
    public var textHash: String {
        let digest = SHA256.hash(data: Data(text.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return "sha256:\(hex)"
    }
}

extension EntityPrediction: CustomStringConvertible {
    public var description: String {
        let formattedConfidence = String(format: "%.2f", confidence)
        return
            "[\(label)] span=(\(start):\(end)) text_hash=\(textHash) conf=\(formattedConfidence)"
    }
}
