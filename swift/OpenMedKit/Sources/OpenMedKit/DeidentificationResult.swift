import Foundation

/// Redaction methods currently supported by OpenMedKit de-identification.
public enum DeidentificationMethod: String, Codable, Equatable, Sendable {
    case mask
    case remove
}

/// De-identification output encoded with the Python `DeidentificationResult.to_dict()`
/// schema: snake_case top-level keys and `pii_entities` records containing
/// `text`, `label`, `entity_type`, `start`, `end`, `confidence`, and `action`.
/// Entity offsets use half-open Unicode scalar coordinates and are normalized
/// to extended-grapheme boundaries when the result is initialized.
public struct DeidentificationResult: Codable, Equatable, Sendable {
    /// A single PII entity record in the Python-compatible export schema.
    public struct PIIEntityRecord: Codable, Equatable, Sendable {
        public let text: String
        public let label: String
        public let entityType: String
        public let start: Int
        public let end: Int
        public let confidence: Float
        public let action: String

        public init(
            text: String,
            label: String,
            entityType: String? = nil,
            start: Int,
            end: Int,
            confidence: Float,
            action: String
        ) {
            self.text = text
            self.label = label
            self.entityType = entityType ?? label
            self.start = start
            self.end = end
            self.confidence = confidence
            self.action = action
        }

        public init(entity: EntityPrediction, action: String) {
            self.init(
                text: entity.text,
                label: entity.label,
                entityType: entity.entityType,
                start: entity.start,
                end: entity.end,
                confidence: entity.confidence,
                action: action
            )
        }

        fileprivate func snappedToGraphemeBoundaries(
            in source: String
        ) -> PIIEntityRecord {
            let snapped = PostProcessing.snapScalarSpanToGraphemeBoundaries(
                start: start,
                end: end,
                in: source
            )
            return PIIEntityRecord(
                text: PostProcessing.scalarSubstring(
                    source,
                    start: snapped.start,
                    end: snapped.end
                ),
                label: label,
                entityType: entityType,
                start: snapped.start,
                end: snapped.end,
                confidence: confidence,
                action: action
            )
        }

        private enum CodingKeys: String, CodingKey {
            case text
            case label
            case entityType = "entity_type"
            case start
            case end
            case confidence
            case action
        }
    }

    public let originalText: String
    public let deidentifiedText: String
    public let piiEntities: [PIIEntityRecord]
    public let method: String
    public let timestamp: Date
    public let numEntitiesRedacted: Int
    public let metadata: [String: String]

    public init(
        originalText: String,
        deidentifiedText: String,
        piiEntities: [PIIEntityRecord],
        method: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:]
    ) {
        self.originalText = originalText
        self.deidentifiedText = deidentifiedText
        let normalizedEntities = piiEntities.map {
            $0.snappedToGraphemeBoundaries(in: originalText)
        }
        self.piiEntities = normalizedEntities
        self.method = method
        self.timestamp = timestamp
        self.numEntitiesRedacted = normalizedEntities.count
        self.metadata = metadata
    }

    public init(
        originalText: String,
        deidentifiedText: String,
        entities: [EntityPrediction],
        method: String,
        timestamp: Date = Date(),
        metadata: [String: String] = [:],
        action: String? = nil
    ) {
        let entityAction = action ?? method
        self.init(
            originalText: originalText,
            deidentifiedText: deidentifiedText,
            piiEntities: entities.map {
                PIIEntityRecord(entity: $0, action: entityAction)
            },
            method: method,
            timestamp: timestamp,
            metadata: metadata
        )
    }

    public func encode(prettyPrinted: Bool = false) throws -> Data {
        let encoder = JSONEncoder()
        var formatting: JSONEncoder.OutputFormatting = [.sortedKeys]
        if prettyPrinted {
            formatting.insert(.prettyPrinted)
        }
        encoder.outputFormatting = formatting
        encoder.dateEncodingStrategy = .iso8601
        return try encoder.encode(self)
    }

    public func toJSON(prettyPrinted: Bool = false) throws -> String {
        String(decoding: try encode(prettyPrinted: prettyPrinted), as: UTF8.self)
    }

    private enum CodingKeys: String, CodingKey {
        case originalText = "original_text"
        case deidentifiedText = "deidentified_text"
        case piiEntities = "pii_entities"
        case method
        case timestamp
        case numEntitiesRedacted = "num_entities_redacted"
        case metadata
    }
}
