import Foundation

/// A single entity predicted by the NER pipeline.
public struct EntityPrediction: Codable, Equatable, Sendable {
    /// The entity type label (e.g., "first_name", "date_of_birth", "ssn").
    public let label: String

    /// The text span matched by this entity.
    public let text: String

    /// Model confidence score (0.0 – 1.0).
    public let confidence: Float

    /// Inclusive Unicode scalar (code point) offset in the original text.
    public let start: Int

    /// Exclusive Unicode scalar (code point) offset in the original text.
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

    /// Return the Swift range represented by this entity's scalar offsets.
    public func range(in source: String) -> Range<String.Index>? {
        PostProcessing.scalarRange(in: source, start: start, end: end)
    }

    /// Return a copy whose offsets enclose complete grapheme clusters.
    ///
    /// The copied `text` is sliced from `source` so it always agrees with the
    /// adjusted offsets.
    public func snappedToGraphemeBoundaries(
        in source: String
    ) -> EntityPrediction? {
        let snapped = PostProcessing.snapScalarSpanToGraphemeBoundaries(
            start: start,
            end: end,
            in: source
        )
        guard
            PostProcessing.scalarRange(
                in: source,
                start: snapped.start,
                end: snapped.end
            ) != nil
        else {
            return nil
        }
        return EntityPrediction(
            label: label,
            text: PostProcessing.scalarSubstring(
                source,
                start: snapped.start,
                end: snapped.end
            ),
            confidence: confidence,
            start: snapped.start,
            end: snapped.end
        )
    }
}

extension EntityPrediction: CustomStringConvertible {
    public var description: String {
        "[\(label)] \"\(text)\" (\(start):\(end)) conf=\(String(format: "%.2f", confidence))"
    }
}
