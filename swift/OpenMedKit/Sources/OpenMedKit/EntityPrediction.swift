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

    public init(label: String, text: String, confidence: Float, start: Int, end: Int) {
        self.label = label
        self.text = text
        self.confidence = confidence
        self.start = start
        self.end = end
    }
}

extension EntityPrediction: CustomStringConvertible {
    public var description: String {
        "[\(label)] \"\(text)\" (\(start):\(end)) conf=\(String(format: "%.2f", confidence))"
    }
}
