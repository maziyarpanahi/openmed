import Foundation

/// A named set of GLiNER labels. Built-ins mirror the original `ClinicalTaskPreset`
/// cases from `ContentView.swift`; user presets are persisted in UserDefaults.
public struct ClinicalPreset: Identifiable, Codable, Hashable, Sendable {
    public var id: UUID
    public var name: String
    public var summary: String
    public var labels: [String]
    public var isBuiltIn: Bool
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        summary: String = "",
        labels: [String],
        isBuiltIn: Bool = false,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.summary = summary
        self.labels = labels
        self.isBuiltIn = isBuiltIn
        self.createdAt = createdAt
    }

    // MARK: Built-in presets
    public static let builtInSummary = ClinicalPreset(
        id: UUID(uuidString: "00000000-0000-0000-0000-000000000001")!,
        name: "Clinical Summary",
        summary: "Broad concepts for a quick overview of the note.",
        labels: ["condition", "symptom", "medication", "dosage", "procedure", "test", "allergy", "follow-up", "care plan"],
        isBuiltIn: true
    )

    public static let builtInMedication = ClinicalPreset(
        id: UUID(uuidString: "00000000-0000-0000-0000-000000000002")!,
        name: "Medication Review",
        summary: "Medication names, doses, allergies, and treatment context.",
        labels: ["medication", "dosage", "frequency", "allergy", "adverse reaction", "treatment", "pharmacy instruction"],
        isBuiltIn: true
    )

    public static let builtInED = ClinicalPreset(
        id: UUID(uuidString: "00000000-0000-0000-0000-000000000003")!,
        name: "ED Follow-up",
        summary: "Symptoms, diagnoses, return precautions, and follow-up needs.",
        labels: ["chief concern", "symptom", "diagnosis", "test", "return precaution", "follow-up", "care setting"],
        isBuiltIn: true
    )

    public static let builtInCarePlan = ClinicalPreset(
        id: UUID(uuidString: "00000000-0000-0000-0000-000000000004")!,
        name: "Care Plan",
        summary: "Care instructions, planned tests, referrals, and next steps.",
        labels: ["care plan", "procedure", "test", "referral", "follow-up", "work status", "patient instruction"],
        isBuiltIn: true
    )

    public static let builtIns: [ClinicalPreset] = [
        .builtInSummary,
        .builtInMedication,
        .builtInED,
        .builtInCarePlan,
    ]
}

public extension ClinicalPreset {
    /// Normalises a proposed label list: trim, lowercase, dedupe, drop empties.
    static func normalize(_ raw: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for element in raw {
            let cleaned = element.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if cleaned.isEmpty { continue }
            if seen.insert(cleaned).inserted {
                out.append(cleaned)
            }
        }
        return out
    }
}
