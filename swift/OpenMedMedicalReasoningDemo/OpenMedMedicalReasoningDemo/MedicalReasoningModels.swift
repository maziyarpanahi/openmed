import Foundation

enum MedicalReasoningRoute: Hashable, Sendable {
    case modelSetup
    case clinicalContext
    case conversation
}

enum LFMDownloadState: Equatable, Sendable {
    case missing
    case partial(bytesOnDisk: Int64)
    case downloading(bytesDownloaded: Int64, bytesExpected: Int64)
    case ready
    case failed(message: String)
    case cancelled

    var isDownloading: Bool {
        if case .downloading = self { return true }
        return false
    }

    var fraction: Double? {
        guard case .downloading(let downloaded, let expected) = self, expected > 0 else {
            return nil
        }
        return max(0, min(1, Double(downloaded) / Double(expected)))
    }
}

struct MedicalConversationMessage: Identifiable, Hashable, Sendable {
    enum Role: String, Hashable, Sendable {
        case user
        case assistant
    }

    enum Activity: String, Hashable, Sendable {
        case complete
        case reasoning
        case answering
        case failed
        case stopped
    }

    let id: UUID
    let role: Role
    var content: String
    var reasoning: String
    var activity: Activity

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        reasoning: String = "",
        activity: Activity = .complete
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.reasoning = reasoning
        self.activity = activity
    }

    var isGenerating: Bool {
        activity == .reasoning || activity == .answering
    }
}

enum SyntheticClinicalCase {
    static let text = """
        DE-IDENTIFIED OUTPATIENT FOLLOW-UP NOTE

        Adult patient returns after an emergency department visit for episodic chest discomfort. Troponin testing was negative on two measurements. ECG showed sinus rhythm without acute ischemic changes. Chest radiograph showed no acute cardiopulmonary abnormality.

        The patient reports no recurrent chest pain, dyspnea, syncope, or fever since discharge. Current medications documented in the note are atorvastatin 20 mg nightly and lisinopril 10 mg daily. No medication allergies are documented.

        Assessment: symptoms have resolved; the etiology remains uncertain. The note documents primary-care follow-up within 48 hours and outpatient cardiology review if symptoms recur. Return precautions include recurrent chest pain, shortness of breath, fainting, or new neurologic symptoms.

        Missing from the supplied record: family cardiac history, smoking status, lipid results, blood-pressure measurements, and the final emergency-department discharge summary.
        """
}
