import Foundation

/// Operational review tiers carried by portable capability declarations.
public enum OpenMedRegulatoryTier: String, Codable, Sendable {
    case nonClinical = "non_clinical"
    case informational
    case assistiveReview = "assistive_review"
    case restricted
    case unknown

    public init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: value) ?? .unknown
    }

    /// Required safeguards, including the strictest set for future tier codes.
    public var obligations: Set<String> {
        switch self {
        case .nonClinical: []
        case .informational: ["notice_required"]
        case .assistiveReview: ["notice_required", "acknowledgement_required"]
        case .restricted, .unknown:
            ["notice_required", "acknowledgement_required", "clinician_mode_required", "crisis_routing_required"]
        }
    }
}

/// A versioned capability declaration from the bundled OpenMed registry.
public struct OpenMedIntendedUse: Codable, Equatable, Sendable {
    public let schemaVersion: String
    public let regulatoryTier: OpenMedRegulatoryTier
    public let purpose: String
    public let intendedUsers: [String]
    public let excludedUses: [String]
    public let inputModalities: [String]
    public let languages: [String]
    public let humanOversight: String
    public let autonomousClinicalAction: Bool
    public let summary: String?

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case regulatoryTier = "regulatory_tier"
        case purpose
        case intendedUsers = "intended_users"
        case excludedUses = "excluded_uses"
        case inputModalities = "input_modalities"
        case languages
        case humanOversight = "human_oversight"
        case autonomousClinicalAction = "autonomous_clinical_action"
        case summary
    }

    private static let bundled = Result {
        guard let url = Bundle.module.url(forResource: "intended_use", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode([String: Self].self, from: Data(contentsOf: url))
    }

    /// Load every reviewed declaration from the installed kit.
    public static func declarations() throws -> [String: Self] {
        try bundled.get()
    }

    /// Return the declaration for an exact capability code, or nil for an unknown code.
    public static func lookup(_ capability: String) throws -> Self? {
        try declarations()[capability]
    }
}
