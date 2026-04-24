import Foundation

/// A span of text the pipeline flagged as sensitive or clinically relevant.
/// Mirrors the private `DetectedEntity` in `ContentView.swift` but with
/// a category enum so the summary view can group/filter.
public struct DetectedEntity: Identifiable, Hashable, Sendable {
    public let id: UUID
    public let label: String
    public let text: String
    public let confidence: Double
    public let start: Int
    public let end: Int
    public let category: EntityCategory

    public init(
        id: UUID = UUID(),
        label: String,
        text: String,
        confidence: Double,
        start: Int,
        end: Int,
        category: EntityCategory? = nil
    ) {
        self.id = id
        self.label = label
        self.text = text
        self.confidence = confidence
        self.start = start
        self.end = end
        self.category = category ?? EntityCategory.classify(label: label)
    }
}

/// Semantic grouping of labels. The pipeline returns arbitrary label strings
/// ("name", "patient name", "DOB", "diagnosis"); this enum folds them into
/// a small fixed set used for colour tone and summary grouping.
public enum EntityCategory: String, CaseIterable, Hashable, Sendable, Identifiable {
    case person              // NAME, patient name, provider name
    case date                // DOB, visit date, admission
    case identifier          // MRN, DNI, passport, case id
    case contact             // phone, email
    case location            // address, city
    case organization        // hospital, clinic
    case condition           // diagnosis, condition, problem
    case symptom             // symptom, chief concern
    case medication          // medication, drug, prescription
    case dosage              // dosage, frequency, strength
    case procedure           // procedure, surgery
    case test                // test, imaging, lab
    case allergy             // allergy, adverse reaction
    case followUp            // follow-up, return precaution
    case carePlan            // care plan, referral, patient instruction
    case other

    public var id: String { rawValue }

    /// Human-readable name for the chip bar and summary section headers.
    public var displayName: String {
        switch self {
        case .person:       return "Person"
        case .date:         return "Date"
        case .identifier:   return "Identifier"
        case .contact:      return "Contact"
        case .location:     return "Location"
        case .organization: return "Organization"
        case .condition:    return "Condition"
        case .symptom:      return "Symptom"
        case .medication:   return "Medication"
        case .dosage:       return "Dosage"
        case .procedure:    return "Procedure"
        case .test:         return "Test"
        case .allergy:      return "Allergy"
        case .followUp:     return "Follow-up"
        case .carePlan:     return "Care plan"
        case .other:        return "Other"
        }
    }

    public var tone: OMEntityTone {
        switch self {
        case .person:       return .name
        case .date:         return .date
        case .identifier:   return .identifier
        case .contact:      return .contact
        case .location:     return .location
        case .organization: return .organization
        case .condition:    return .condition
        case .symptom:      return .symptom
        case .medication:   return .medication
        case .dosage:       return .dosage
        case .procedure:    return .procedure
        case .test:         return .test
        case .allergy:      return .allergy
        case .followUp:     return .followUp
        case .carePlan:     return .carePlan
        case .other:        return .generic
        }
    }

    /// Best-effort classification by substring match on the label.
    public static func classify(label: String) -> EntityCategory {
        let normalized = label.lowercased()
        if normalized.contains("name") || normalized == "person" { return .person }
        if normalized == "dob" || normalized.contains("date") { return .date }
        if normalized.contains("mrn") || normalized.contains("id") || normalized.contains("dni") ||
           normalized.contains("passport") || normalized.contains("ssn") || normalized.contains("case") { return .identifier }
        if normalized.contains("phone") || normalized.contains("email") || normalized.contains("fax") { return .contact }
        if normalized.contains("address") || normalized.contains("location") || normalized.contains("city") ||
           normalized.contains("zip") || normalized.contains("postal") { return .location }
        if normalized.contains("hospital") || normalized.contains("clinic") || normalized.contains("org") ||
           normalized.contains("organization") { return .organization }
        if normalized.contains("diagnos") || normalized.contains("condition") || normalized.contains("disease") ||
           normalized.contains("problem") || normalized.contains("medical history") { return .condition }
        if normalized.contains("symptom") || normalized.contains("chief concern") || normalized.contains("complaint") { return .symptom }
        if normalized.contains("medic") || normalized.contains("drug") || normalized.contains("prescription") ||
           normalized.contains("pharmacy") { return .medication }
        if normalized.contains("dos") || normalized.contains("frequency") || normalized.contains("strength") { return .dosage }
        if normalized.contains("procedure") || normalized.contains("surg") || normalized.contains("operation") { return .procedure }
        if normalized.contains("test") || normalized.contains("lab") || normalized.contains("imaging") ||
           normalized.contains("exam") { return .test }
        if normalized.contains("allerg") || normalized.contains("adverse") { return .allergy }
        if normalized.contains("follow") || normalized.contains("return") { return .followUp }
        if normalized.contains("care") || normalized.contains("plan") || normalized.contains("referral") ||
           normalized.contains("instruction") || normalized.contains("treatment") { return .carePlan }
        return .other
    }
}
