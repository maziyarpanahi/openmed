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

    struct SemanticUnitPattern {
        let regex: NSRegularExpression
        let entityType: String
        let priority: Int
        let captureGroup: Int?
        let baseScore: Float
        let contextBoost: Float
        let contextWords: [String]
        let validator: ((String) -> Bool)?

        init(
            _ pattern: String,
            entityType: String,
            priority: Int = 0,
            captureGroup: Int? = nil,
            options: NSRegularExpression.Options = [.caseInsensitive],
            baseScore: Float = 0.5,
            contextBoost: Float = 0.35,
            contextWords: [String] = [],
            validator: ((String) -> Bool)? = nil
        ) {
            do {
                self.regex = try NSRegularExpression(pattern: pattern, options: options)
            } catch {
                fatalError("Invalid PII regex pattern: \(pattern)")
            }
            self.entityType = entityType
            self.priority = priority
            self.captureGroup = captureGroup
            self.baseScore = baseScore
            self.contextBoost = contextBoost
            self.contextWords = contextWords
            self.validator = validator
        }
    }

    struct SemanticUnitMatch {
        let start: Int
        let end: Int
        let entityType: String
        let score: Float
        let validated: Bool
    }

    static let defaultPIIPatterns: [SemanticUnitPattern] = [
        .init(
            #"\bPatient:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b"#,
            entityType: "full_name",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["patient", "name"]
        ),
        .init(
            #"\bEmergency Contact:\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)\b"#,
            entityType: "full_name",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["emergency contact", "contact"]
        ),
        .init(
            #"^PATIENT NAME\s*$\n([A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*(?:\s+[A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*)+)(?=\n|$)"#,
            entityType: "full_name",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["patient", "name"]
        ),
        .init(
            #"^EMERGENCY CONTACT\s*$\n([A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*(?:\s+[A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*)+)(?:,\s*\(\d{3}\)\s*\d{3}[-.\s]?\d{4})?(?=\n|$)"#,
            entityType: "full_name",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["emergency contact", "contact"]
        ),
        .init(
            #"^PRIMARY CLINIC(?:IAN|AN)\s*$\n((?:Dr\.\s+)?[A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*(?:\s+[A-Z][A-Za-z]+(?:[ '-][A-Za-z]+)*)+(?:,\s*[A-Z.]+)?)(?=\n|$)"#,
            entityType: "full_name",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.94,
            contextBoost: 0.03,
            contextWords: ["clinician", "provider", "doctor", "physician"]
        ),
        .init(
            #"\bDOB:\s*(\d{1,2}/\d{1,2}/\d{2,4})\b"#,
            entityType: "date_of_birth",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["dob", "date of birth", "birthdate"]
        ),
        .init(
            #"^DOB\s*$\n(\d{1,2}/\d{1,2}/\d{2,4})(?=\n|$)"#,
            entityType: "date_of_birth",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["dob", "date of birth", "birthdate"]
        ),
        .init(
            #"\bSSN:\s*(\d{3}-\d{2}-\d{4})\b"#,
            entityType: "ssn",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["ssn", "social security", "social security number"]
        ),
        .init(
            #"\bMRN:\s*([A-Z0-9][A-Z0-9-]{4,})\b"#,
            entityType: "medical_record_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["mrn", "medical record", "record number"]
        ),
        .init(
            #"^MRN\s*$\n([A-Z0-9][A-Z0-9-]{4,})(?=\n|$)"#,
            entityType: "medical_record_number",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["mrn", "medical record", "record number"]
        ),
        .init(
            #"\bAddress:\s*([^,\n]+(?:,\s*[^,\n]+)*?)(?=,\s*(?:Phone|Email|Insurance ID|Driver License|Passport|Emergency Contact|Employer|Employee ID|Bank Account|Routing):|$)"#,
            entityType: "street_address",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["address", "street", "residence", "lives at"]
        ),
        .init(
            #"^ADDRESS\s*$\n([^\n]+)(?=\n|$)"#,
            entityType: "street_address",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["address", "street", "residence", "lives at"]
        ),
        .init(
            #"\bPhone:\s*(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b"#,
            entityType: "phone_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["phone", "telephone", "contact"]
        ),
        .init(
            #"^PHONE\s*$\n(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})(?=\n|$)"#,
            entityType: "phone_number",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["phone", "telephone", "contact"]
        ),
        .init(
            #"\bEmergency Contact:\s*[A-Za-z][A-Za-z\s'-]+,\s*(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})\b"#,
            entityType: "phone_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["emergency contact", "contact", "phone"]
        ),
        .init(
            #"^EMERGENCY CONTACT\s*$\n[A-Za-z][A-Za-z\s'-]+,\s*(\(\d{3}\)\s*\d{3}[-.\s]?\d{4})(?=\n|$)"#,
            entityType: "phone_number",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["emergency contact", "contact", "phone"]
        ),
        .init(
            #"\bEmail:\s*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"#,
            entityType: "email",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["email", "e-mail", "mail"]
        ),
        .init(
            #"^EMAIL\s*$\n([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?=\n|$)"#,
            entityType: "email",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["email", "e-mail", "mail"]
        ),
        .init(
            #"^([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\s*$\nEMAIL(?:\n|$)"#,
            entityType: "email",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.97,
            contextBoost: 0.02,
            contextWords: ["email", "e-mail", "mail"]
        ),
        .init(
            #"\bInsurance ID:\s*([A-Z0-9][A-Z0-9-]{4,})\b"#,
            entityType: "insurance_id",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["insurance", "policy", "coverage", "payer"]
        ),
        .init(
            #"^INSURANCE ID\s*$\n([A-Z0-9][A-Z0-9-]{4,})(?=\n|$)"#,
            entityType: "insurance_id",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["insurance", "policy", "coverage", "payer"]
        ),
        .init(
            #"^([A-Z0-9][A-Z0-9-]{4,})\s*$\nINSURANCE ID(?:\n|$)"#,
            entityType: "insurance_id",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.96,
            contextBoost: 0.02,
            contextWords: ["insurance", "policy", "coverage", "payer"]
        ),
        .init(
            #"\bDriver License:\s*([A-Z0-9][A-Z0-9-]{4,})\b"#,
            entityType: "driver_license",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["driver license", "license", "dl"]
        ),
        .init(
            #"\bPassport:\s*([A-Z0-9][A-Z0-9-]{4,})\b"#,
            entityType: "passport_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["passport", "travel document"]
        ),
        .init(
            #"\bEmployer:\s*([^,\n]+)"#,
            entityType: "organization",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.93,
            contextBoost: 0.03,
            contextWords: ["employer", "company", "organization", "work"]
        ),
        .init(
            #"^EMPLOYER\s*$\n([^\n]+)(?=\n|$)"#,
            entityType: "organization",
            priority: 15,
            captureGroup: 1,
            options: [.caseInsensitive, .anchorsMatchLines],
            baseScore: 0.94,
            contextBoost: 0.03,
            contextWords: ["employer", "company", "organization", "work"]
        ),
        .init(
            #"\bEmployee ID:\s*([A-Z0-9][A-Z0-9-]{3,})\b"#,
            entityType: "employee_id",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["employee id", "employee", "staff id"]
        ),
        .init(
            #"\bBank Account:\s*([A-Z0-9][A-Z0-9-]{5,})\b"#,
            entityType: "account_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.95,
            contextBoost: 0.03,
            contextWords: ["bank account", "account", "banking"]
        ),
        .init(
            #"\bRouting:\s*(\d{9})\b"#,
            entityType: "routing_number",
            priority: 14,
            captureGroup: 1,
            baseScore: 0.94,
            contextBoost: 0.03,
            contextWords: ["routing", "aba", "bank"]
        ),
        .init(
            #"\b\d{4}-\d{2}-\d{2}\b"#,
            entityType: "date",
            priority: 10,
            baseScore: 0.6,
            contextBoost: 0.3,
            contextWords: ["dob", "birth", "born", "date of birth", "birthdate", "deceased", "died", "admitted", "discharged"]
        ),
        .init(
            #"\b\d{1,2}/\d{1,2}/\d{2,4}\b"#,
            entityType: "date",
            priority: 9,
            baseScore: 0.6,
            contextBoost: 0.3,
            contextWords: ["dob", "birth", "born", "date of birth", "birthdate", "deceased", "died", "admitted", "discharged"]
        ),
        .init(
            #"\b\d{1,2}-\d{1,2}-\d{2,4}\b"#,
            entityType: "date",
            priority: 9,
            baseScore: 0.6,
            contextBoost: 0.3,
            contextWords: ["dob", "birth", "born", "date of birth", "birthdate", "deceased", "died", "admitted", "discharged"]
        ),
        .init(
            #"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b"#,
            entityType: "date",
            priority: 8,
            baseScore: 0.7,
            contextBoost: 0.25,
            contextWords: ["dob", "birth", "born", "date of birth", "birthdate", "deceased", "died", "admitted", "discharged"]
        ),
        .init(
            #"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b"#,
            entityType: "date",
            priority: 8,
            baseScore: 0.7,
            contextBoost: 0.25,
            contextWords: ["dob", "birth", "born", "date of birth", "birthdate", "deceased", "died", "admitted", "discharged"]
        ),
        .init(
            #"\b\d{3}-\d{2}-\d{4}\b"#,
            entityType: "ssn",
            priority: 10,
            baseScore: 0.3,
            contextBoost: 0.55,
            contextWords: ["ssn", "social security", "social security number", "ss#", "ss number"],
            validator: validateSSN
        ),
        .init(
            #"\b\d{3}\s\d{2}\s\d{4}\b"#,
            entityType: "ssn",
            priority: 9,
            baseScore: 0.3,
            contextBoost: 0.55,
            contextWords: ["ssn", "social security", "social security number", "ss#", "ss number"],
            validator: validateSSN
        ),
        .init(
            #"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b"#,
            entityType: "phone_number",
            priority: 9,
            baseScore: 0.6,
            contextBoost: 0.3,
            contextWords: ["phone", "tel", "telephone", "cell", "mobile", "fax", "call", "contact"],
            validator: validatePhoneUS
        ),
        .init(
            #"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"#,
            entityType: "phone_number",
            priority: 8,
            baseScore: 0.5,
            contextBoost: 0.35,
            contextWords: ["phone", "tel", "telephone", "cell", "mobile", "fax", "call", "contact"],
            validator: validatePhoneUS
        ),
        .init(
            #"\b\d{10}\b"#,
            entityType: "npi",
            priority: 6,
            baseScore: 0.15,
            contextBoost: 0.65,
            contextWords: ["npi", "national provider", "provider number", "provider id", "provider identifier"],
            validator: validateNPI
        ),
        .init(
            #"\b\d{10}\b"#,
            entityType: "phone_number",
            priority: 5,
            baseScore: 0.2,
            contextBoost: 0.5,
            contextWords: ["phone", "tel", "telephone", "cell", "mobile", "fax", "call", "contact"],
            validator: validatePhoneUS
        ),
        .init(
            #"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"#,
            entityType: "email",
            priority: 10,
            baseScore: 0.9,
            contextBoost: 0.1,
            contextWords: ["email", "e-mail", "contact", "mail"]
        ),
        .init(
            #"\b\d{5}(?:-\d{4})?\b"#,
            entityType: "postcode",
            priority: 7,
            baseScore: 0.4,
            contextBoost: 0.45,
            contextWords: ["zip", "zipcode", "zip code", "postal", "postal code"]
        ),
        .init(
            #"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"#,
            entityType: "credit_debit_card",
            priority: 8,
            baseScore: 0.4,
            contextBoost: 0.4,
            contextWords: ["card", "credit", "debit", "visa", "mastercard", "amex", "discover", "payment"],
            validator: validateLuhn
        ),
        .init(
            #"\b(?:MRN|mrn)[:\s#]*\d{6,10}\b"#,
            entityType: "medical_record_number",
            priority: 9,
            baseScore: 0.8,
            contextBoost: 0.15,
            contextWords: ["medical record", "patient id", "patient number", "record number"]
        ),
        .init(
            #"\b[A-Z]{2,3}\d{6,9}\b"#,
            entityType: "medical_record_number",
            priority: 5,
            baseScore: 0.3,
            contextBoost: 0.5,
            contextWords: ["mrn", "medical record", "patient id", "patient number", "record number"]
        ),
        .init(
            #"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b"#,
            entityType: "street_address",
            priority: 7,
            baseScore: 0.7,
            contextBoost: 0.2,
            contextWords: ["address", "street", "resides", "residence", "lives at", "located at"]
        ),
        .init(
            #"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b"#,
            entityType: "url",
            priority: 8,
            baseScore: 0.8,
            contextBoost: 0.15,
            contextWords: ["url", "website", "link", "webpage"]
        ),
        .init(
            #"\b(?:\d{1,3}\.){3}\d{1,3}\b"#,
            entityType: "ipv4",
            priority: 7,
            baseScore: 0.6,
            contextBoost: 0.3,
            contextWords: ["ip", "ip address", "address", "server", "host"]
        ),
        .init(
            #"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"#,
            entityType: "ipv6",
            priority: 8,
            baseScore: 0.85,
            contextBoost: 0.15,
            contextWords: ["ip", "ipv6", "ip address", "address", "server", "host"]
        ),
        .init(
            #"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"#,
            entityType: "mac_address",
            priority: 8,
            baseScore: 0.75,
            contextBoost: 0.2,
            contextWords: ["mac", "mac address", "hardware address"]
        ),
    ]

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
                        label: curLabel,
                        start: currentStart,
                        end: currentEnd,
                        scores: currentScores,
                        text: text,
                        strategy: strategy
                    ))
                    currentLabel = nil
                    currentScores = []
                }
                continue
            }

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
                if let curLabel = currentLabel {
                    entities.append(makeEntity(
                        label: curLabel,
                        start: currentStart,
                        end: currentEnd,
                        scores: currentScores,
                        text: text,
                        strategy: strategy
                    ))
                }

                currentLabel = entityType
                currentStart = token.startOffset
                currentEnd = token.endOffset
                currentScores = [token.score]
            } else {
                currentEnd = token.endOffset
                currentScores.append(token.score)
            }
        }

        if let curLabel = currentLabel {
            entities.append(makeEntity(
                label: curLabel,
                start: currentStart,
                end: currentEnd,
                scores: currentScores,
                text: text,
                strategy: strategy
            ))
        }

        return repairEntitySpans(entities, text: text)
    }

    static func repairEntitySpans(
        _ entities: [EntityPrediction],
        text: String
    ) -> [EntityPrediction] {
        guard !text.isEmpty else {
            return entities
        }

        let textLength = text.count

        return entities.map { entity in
            var start = max(0, min(entity.start, textLength))
            var end = max(start, min(entity.end, textLength))

            var extended = 0
            while end < textLength,
                  extended < 10,
                  let character = character(at: end, in: text),
                  isWordLike(character)
            {
                end += 1
                extended += 1
            }

            while start < end,
                  let character = character(at: start, in: text),
                  character.isWhitespace
            {
                start += 1
            }

            while end > start,
                  let character = character(at: end - 1, in: text),
                  character.isWhitespace
            {
                end -= 1
            }

            guard start < end else {
                return entity
            }

            let span = substring(text, start: start, end: end)
            guard !span.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                return entity
            }

            return EntityPrediction(
                label: entity.label,
                text: span,
                confidence: entity.confidence,
                start: start,
                end: end
            )
        }
    }

    static func mergePIIEntities(
        _ entities: [EntityPrediction],
        text: String,
        useSemanticPatterns: Bool = true,
        preferModelLabels: Bool = true,
        patterns: [SemanticUnitPattern] = defaultPIIPatterns
    ) -> [EntityPrediction] {
        let sortedEntities = entities.sorted {
            if $0.start == $1.start {
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }

        guard useSemanticPatterns else {
            return sortedEntities
        }

        let semanticUnits = findSemanticUnits(in: text, patterns: patterns)
        guard !semanticUnits.isEmpty else {
            return sortedEntities
        }

        var merged: [EntityPrediction] = []
        var usedEntityIndices = Set<Int>()

        for unit in semanticUnits {
            let overlapping = sortedEntities.enumerated().filter { _, entity in
                entity.start < unit.end && entity.end > unit.start
            }

            guard !overlapping.isEmpty else {
                if unit.score >= 0.55 {
                    merged.append(
                        EntityPrediction(
                            label: unit.entityType,
                            text: substring(text, start: unit.start, end: unit.end),
                            confidence: unit.score,
                            start: unit.start,
                            end: unit.end
                        )
                    )
                }
                continue
            }

            let overlappingEntities = overlapping.map(\.element)
            let (dominantLabel, modelAverageConfidence) = calculateDominantLabel(in: overlappingEntities)

            let finalLabel: String
            if preferModelLabels {
                if isMoreSpecific(label: unit.entityType, than: dominantLabel) {
                    finalLabel = unit.entityType
                } else if normalizeLabel(dominantLabel) == normalizeLabel(unit.entityType)
                    || isMoreSpecific(label: dominantLabel, than: unit.entityType)
                {
                    finalLabel = dominantLabel
                } else {
                    finalLabel = unit.entityType
                }
            } else {
                let matchingModelLabel = overlappingEntities.contains {
                    normalizeLabel($0.label) == normalizeLabel(unit.entityType)
                }
                if matchingModelLabel {
                    finalLabel = dominantLabel
                } else {
                    finalLabel = isMoreSpecific(label: dominantLabel, than: unit.entityType)
                        ? dominantLabel
                        : unit.entityType
                }
            }

            let finalConfidence: Float
            if unit.validated {
                finalConfidence = (0.6 * modelAverageConfidence) + (0.4 * unit.score)
            } else {
                finalConfidence = (0.9 * modelAverageConfidence) + (0.1 * unit.score)
            }

            merged.append(
                EntityPrediction(
                    label: finalLabel,
                    text: substring(text, start: unit.start, end: unit.end),
                    confidence: finalConfidence,
                    start: unit.start,
                    end: unit.end
                )
            )

            for index in overlapping.map(\.offset) {
                usedEntityIndices.insert(index)
            }
        }

        for (index, entity) in sortedEntities.enumerated() where !usedEntityIndices.contains(index) {
            merged.append(entity)
        }

        return merged.sorted {
            if $0.start == $1.start {
                return $0.end < $1.end
            }
            return $0.start < $1.start
        }
    }

    static func findSemanticUnits(
        in text: String,
        patterns: [SemanticUnitPattern] = defaultPIIPatterns
    ) -> [SemanticUnitMatch] {
        guard !text.isEmpty else {
            return []
        }

        let fullRange = NSRange(text.startIndex..<text.endIndex, in: text)
        var units: [SemanticUnitMatch] = []

        for pattern in patterns.sorted(by: { $0.priority > $1.priority }) {
            let matches = pattern.regex.matches(in: text, range: fullRange)

            for match in matches {
                let matchedRange: NSRange
                if let captureGroup = pattern.captureGroup,
                   captureGroup > 0,
                   captureGroup < match.numberOfRanges
                {
                    matchedRange = match.range(at: captureGroup)
                } else {
                    matchedRange = match.range
                }

                guard matchedRange.location != NSNotFound,
                      let range = Range(matchedRange, in: text) else {
                    continue
                }

                let start = text.distance(from: text.startIndex, to: range.lowerBound)
                let end = text.distance(from: text.startIndex, to: range.upperBound)

                let overlaps = units.contains { existing in
                    start < existing.end && end > existing.start
                }
                if overlaps {
                    continue
                }

                let matchedText = String(text[range])
                var score = pattern.baseScore

                if !pattern.contextWords.isEmpty,
                   findContextWords(
                    in: text,
                    start: start,
                    end: end,
                    contextWords: pattern.contextWords
                   )
                {
                    score = min(1.0, score + pattern.contextBoost)
                }

                var validated = true
                if let validator = pattern.validator, !validator(matchedText) {
                    score *= 0.3
                    validated = false
                }

                units.append(
                    SemanticUnitMatch(
                        start: start,
                        end: end,
                        entityType: pattern.entityType,
                        score: score,
                        validated: validated
                    )
                )
            }
        }

        return units.sorted { $0.start < $1.start }
    }

    static func calculateDominantLabel(
        in entities: [EntityPrediction]
    ) -> (label: String, averageConfidence: Float) {
        precondition(!entities.isEmpty, "Cannot calculate dominant label from empty entity list")

        var labelCounts: [String: Int] = [:]
        var labelConfidences: [String: [Float]] = [:]

        for entity in entities {
            labelCounts[entity.label, default: 0] += 1
            labelConfidences[entity.label, default: []].append(entity.confidence)
        }

        let maxCount = labelCounts.values.max() ?? 0
        let candidates = labelCounts.keys.filter { labelCounts[$0] == maxCount }

        let dominantLabel: String
        if candidates.count == 1 {
            dominantLabel = candidates[0]
        } else {
            var bestLabel = candidates[0]
            var bestConfidence = average(labelConfidences[bestLabel] ?? [])

            for candidate in candidates.dropFirst() {
                let candidateConfidence = average(labelConfidences[candidate] ?? [])
                if candidateConfidence > bestConfidence {
                    bestLabel = candidate
                    bestConfidence = candidateConfidence
                }
            }

            dominantLabel = bestLabel
        }

        return (dominantLabel, average(entities.map(\.confidence)))
    }

    static func normalizeLabel(_ label: String) -> String {
        let normalized = label.lowercased()

        if normalized.contains("date") {
            return "date"
        }
        if normalized.contains("phone") || normalized.contains("fax") {
            return "phone"
        }
        if normalized.contains("address") {
            return "address"
        }
        if ["ssn", "social_security", "social_security_number"].contains(normalized) {
            return "ssn"
        }
        if ["national_id", "nir", "insee", "steuer_id", "steuernummer", "codice_fiscale", "bsn", "dni", "nie", "aadhaar"].contains(normalized) {
            return "national_id"
        }
        if ["postcode", "zipcode", "zip", "postal_code"].contains(normalized) {
            return "postcode"
        }
        if ["medical_record_number", "mrn", "medical_record"].contains(normalized) {
            return "medical_record"
        }
        if ["account_number", "account"].contains(normalized) {
            return "account"
        }
        if ["credit_debit_card", "credit_card", "debit_card", "payment_card"].contains(normalized) {
            return "payment_card"
        }

        return normalized
    }

    static func isMoreSpecific(label: String, than otherLabel: String) -> Bool {
        let lhs = label.lowercased()
        let rhs = otherLabel.lowercased()

        if rhs != lhs && lhs.contains(rhs) {
            return true
        }

        let specificityHierarchy: [String: [String]] = [
            "date": ["date_of_birth", "date_time"],
            "name": ["first_name", "last_name", "full_name"],
            "phone": ["phone_number", "fax_number", "mobile_number"],
            "address": ["street_address", "home_address", "billing_address"],
            "id": ["ssn", "medical_record_number", "account_number", "employee_id"],
            "national_id": ["nir", "insee", "steuer_id", "steuernummer", "codice_fiscale"],
        ]

        for (general, specificLabels) in specificityHierarchy {
            if normalizeLabel(otherLabel) == general && specificLabels.contains(lhs) {
                return true
            }
        }

        return false
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
            confidence = average(scores)
        }

        return EntityPrediction(
            label: label,
            text: substring(text, start: start, end: end),
            confidence: confidence,
            start: start,
            end: end
        )
    }

    private static func findContextWords(
        in text: String,
        start: Int,
        end: Int,
        contextWords: [String],
        contextWindow: Int = 100
    ) -> Bool {
        guard !contextWords.isEmpty else {
            return false
        }

        let windowStart = max(0, start - contextWindow)
        let windowEnd = min(text.count, end + contextWindow)
        let contextText = substring(text, start: windowStart, end: windowEnd).lowercased()

        for contextWord in contextWords {
            let normalizedWord = contextWord.lowercased()

            if contextText.contains(normalizedWord) {
                return true
            }

            let pattern = "\\b\(NSRegularExpression.escapedPattern(for: normalizedWord))\\b"
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }

            let range = NSRange(contextText.startIndex..<contextText.endIndex, in: contextText)
            if regex.firstMatch(in: contextText, range: range) != nil {
                return true
            }
        }

        return false
    }

    private static func validateSSN(_ text: String) -> Bool {
        let digits = digitsOnly(in: text)
        guard digits.count == 9 else {
            return false
        }

        let area = String(digits.prefix(3))
        let group = String(digits.dropFirst(3).prefix(2))
        let serial = String(digits.suffix(4))

        if area == "000" || area == "666" || area.hasPrefix("9") {
            return false
        }
        if group == "00" {
            return false
        }
        if serial == "0000" {
            return false
        }

        return true
    }

    private static func validateLuhn(_ text: String) -> Bool {
        let digits = digitsOnly(in: text)
        guard digits.count >= 13 else {
            return false
        }
        return luhnChecksum(of: digits) == 0
    }

    private static func validateNPI(_ text: String) -> Bool {
        let digits = digitsOnly(in: text)
        guard digits.count == 10 else {
            return false
        }
        return luhnChecksum(of: "80840" + digits) == 0
    }

    private static func validatePhoneUS(_ text: String) -> Bool {
        let digits = digitsOnly(in: text)

        if digits.count == 10 {
            let areaCode = Array(digits.prefix(3))
            let exchange = Array(digits.dropFirst(3).prefix(3))

            guard let firstArea = areaCode.first, let firstExchange = exchange.first else {
                return false
            }

            if firstArea == "0" || firstArea == "1" {
                return false
            }
            if firstExchange == "0" {
                return false
            }

            return true
        }

        if digits.count == 11, digits.first == "1" {
            return validatePhoneUS(String(digits.dropFirst()))
        }

        return false
    }

    private static func luhnChecksum(of digits: String) -> Int {
        let values = digits.compactMap(\.wholeNumberValue)
        var checksum = 0

        for (index, value) in values.reversed().enumerated() {
            if index.isMultiple(of: 2) {
                checksum += value
            } else {
                let doubled = value * 2
                checksum += doubled > 9 ? doubled - 9 : doubled
            }
        }

        return checksum % 10
    }

    private static func digitsOnly(in text: String) -> String {
        String(text.filter(\.isNumber))
    }

    private static func average(_ values: [Float]) -> Float {
        guard !values.isEmpty else {
            return 0.0
        }
        return values.reduce(0, +) / Float(values.count)
    }

    private static func substring(_ text: String, start: Int, end: Int) -> String {
        guard let range = characterRange(in: text, start: start, end: end) else {
            return ""
        }
        return String(text[range])
    }

    private static func characterRange(
        in text: String,
        start: Int,
        end: Int
    ) -> Range<String.Index>? {
        guard start >= 0, end >= start, end <= text.count else {
            return nil
        }

        let lowerBound = text.index(text.startIndex, offsetBy: start)
        let upperBound = text.index(lowerBound, offsetBy: end - start)
        return lowerBound..<upperBound
    }

    private static func character(
        at offset: Int,
        in text: String
    ) -> Character? {
        guard offset >= 0, offset < text.count else {
            return nil
        }

        let index = text.index(text.startIndex, offsetBy: offset)
        return text[index]
    }

    private static func isWordLike(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            switch scalar.properties.generalCategory {
            case .uppercaseLetter,
                .lowercaseLetter,
                .titlecaseLetter,
                .modifierLetter,
                .otherLetter,
                .nonspacingMark,
                .spacingMark,
                .enclosingMark,
                .decimalNumber,
                .letterNumber,
                .otherNumber:
                return true
            default:
                return false
            }
        }
    }
}
