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
        let normalized = label
            .lowercased()
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")

        func containsAny(_ terms: [String]) -> Bool {
            terms.contains { normalized.contains($0) }
        }

        if normalized == "dob" || containsAny([
            "date", "birth", "born"
        ]) { return .date }

        if containsAny([
            "phone", "telephone", "mobile", "cell", "email", "e mail", "fax",
            "contact"
        ]) { return .contact }

        if normalized == "id" || containsAny([
            "mrn", "medical record", "record number", "record id", "identifier",
            " id", "id ", "id number", "id num", "national id", "ssn",
            "social security", "dni", "passport", "driver license", "license",
            "account", "encounter", "case", "document", "npi", "member",
            "insurance", "policy", "group", "employee id", "routing", "card",
            "provider id", "provider identifier"
        ]) { return .identifier }

        if containsAny([
            "person", "name", "patient", "provider", "doctor", "physician",
            "clinician", "pcp"
        ]) { return .person }

        if containsAny([
            "address", "location", "city", "state", "street", "zip", "zipcode",
            "postal", "postcode", "country"
        ]) { return .location }

        if containsAny([
            "hospital", "clinic", "facility", "employer", "company", "org",
            "organization", "payer"
        ]) { return .organization }

        if containsAny([
            "diagnos", "condition", "disease", "problem", "medical history"
        ]) { return .condition }
        if containsAny(["symptom", "chief concern", "complaint"]) { return .symptom }
        if containsAny([
            "medication", "medicine", "drug", "prescription", "pharmacy"
        ]) { return .medication }
        if containsAny(["dos", "frequency", "strength"]) { return .dosage }
        if containsAny(["procedure", "surg", "operation"]) { return .procedure }
        if containsAny(["test", "lab", "imaging", "exam"]) { return .test }
        if containsAny(["allerg", "adverse"]) { return .allergy }
        if containsAny(["follow", "return"]) { return .followUp }
        if containsAny(["care", "plan", "referral", "instruction", "treatment"]) { return .carePlan }
        return .other
    }
}
