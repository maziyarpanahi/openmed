import Foundation

/// Decodes BIO-tagged token predictions into grouped entity spans.
public enum PostProcessing {

    /// A single per-token prediction before grouping.
    public struct TokenPrediction {
        public let labelId: Int
        public let label: String
        public let score: Float
        public let startOffset: Int
        public let endOffset: Int
    }

    /// Aggregation strategy for multi-token entity scores.
    public enum AggregationStrategy {
        case first
        case average
        case max
    }

    /// Decode a sequence of per-token predictions into grouped entities.
    ///
    /// - Parameters:
    ///   - tokens: Per-token predictions (excluding special tokens like [CLS], [SEP]).
    ///   - text: The original input text.
    ///   - strategy: How to combine scores across tokens within an entity.
    /// - Returns: An array of `EntityPrediction` instances.
    public static func decodeEntities(
        tokens: [TokenPrediction],
        text: String,
        strategy: AggregationStrategy = .average
    ) -> [EntityPrediction] {
        var entities: [EntityPrediction] = []

        var currentLabel: String? = nil
        var currentStart: Int = 0
        var currentEnd: Int = 0
        var currentScores: [Float] = []

        for token in tokens {
            let label = token.label
            guard label != "O" else {
                if let curLabel = currentLabel {
                    entities.append(makeEntity(
                        label: curLabel, start: currentStart, end: currentEnd,
                        scores: currentScores, text: text, strategy: strategy
                    ))
                    currentLabel = nil
                    currentScores = []
                }
                continue
            }

            // Parse BIO prefix
            let entityType: String
            let isBeginning: Bool
            if label.hasPrefix("B-") {
                entityType = String(label.dropFirst(2))
                isBeginning = true
            } else if label.hasPrefix("I-") {
                entityType = String(label.dropFirst(2))
                isBeginning = false
            } else {
                entityType = label
                isBeginning = true
            }

            if isBeginning || currentLabel == nil || currentLabel != entityType {
                // Flush previous entity
                if let curLabel = currentLabel {
                    entities.append(makeEntity(
                        label: curLabel, start: currentStart, end: currentEnd,
                        scores: currentScores, text: text, strategy: strategy
                    ))
                }
                // Start new entity
                currentLabel = entityType
                currentStart = token.startOffset
                currentEnd = token.endOffset
                currentScores = [token.score]
            } else {
                // Continue current entity
                currentEnd = token.endOffset
                currentScores.append(token.score)
            }
        }

        // Flush last entity
        if let curLabel = currentLabel {
            entities.append(makeEntity(
                label: curLabel, start: currentStart, end: currentEnd,
                scores: currentScores, text: text, strategy: strategy
            ))
        }

        return entities
    }

    private static func makeEntity(
        label: String,
        start: Int,
        end: Int,
        scores: [Float],
        text: String,
        strategy: AggregationStrategy
    ) -> EntityPrediction {
        let confidence: Float
        switch strategy {
        case .first:
            confidence = scores.first ?? 0.0
        case .max:
            confidence = scores.max() ?? 0.0
        case .average:
            confidence = scores.reduce(0, +) / Float(max(scores.count, 1))
        }

        let startIdx = text.index(text.startIndex, offsetBy: min(start, text.count))
        let endIdx = text.index(text.startIndex, offsetBy: min(end, text.count))
        let span = String(text[startIdx..<endIdx])

        return EntityPrediction(
            label: label,
            text: span,
            confidence: confidence,
            start: start,
            end: end
        )
    }
}
