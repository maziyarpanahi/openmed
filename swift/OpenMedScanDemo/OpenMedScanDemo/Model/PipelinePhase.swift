import Foundation

/// A stage inside the live pipeline run. Reports to the PhaseIndicator strip.
public enum PipelinePhase: Int, CaseIterable, Identifiable, Sendable, Hashable {
    case recognizing
    case downloading
    case loading
    case inferencing

    public var id: Int { rawValue }

    public var title: String {
        switch self {
        case .recognizing: return "OCR"
        case .downloading: return "Download"
        case .loading:     return "Load"
        case .inferencing: return "Analyze"
        }
    }

    public var headline: String {
        switch self {
        case .recognizing: return "Extracting text from the scan"
        case .downloading: return "Downloading model artifacts"
        case .loading:     return "Loading MLX runtime"
        case .inferencing: return "Running on-device inference"
        }
    }
}

public struct PipelineProgress: Sendable, Hashable {
    public let phase: PipelinePhase
    public let detail: String
    public let bytesCompleted: Int64?
    public let bytesExpected: Int64?

    public init(
        phase: PipelinePhase,
        detail: String = "",
        bytesCompleted: Int64? = nil,
        bytesExpected: Int64? = nil
    ) {
        self.phase = phase
        self.detail = detail
        self.bytesCompleted = bytesCompleted
        self.bytesExpected = bytesExpected
    }

    public var fraction: Double? {
        guard let total = bytesExpected, total > 0, let done = bytesCompleted else { return nil }
        return max(0, min(1, Double(done) / Double(total)))
    }
}
