import Foundation

/// Identity of the three on-device models the demo uses.
public enum ScanModelID: String, CaseIterable, Hashable, Sendable, Identifiable {
    case piiLiteClinical
    case openaiPrivacyFilter
    case glinerRelex

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .piiLiteClinical:     return "OpenMed PII"
        case .openaiPrivacyFilter: return "OpenAI Privacy"
        case .glinerRelex:         return "GLiNER Clinical"
        }
    }

    /// Human-readable size estimate for gating UI before the first download.
    /// Values are conservative — real numbers come from the manifest once
    /// the first HEAD request lands.
    public var estimatedSizeLabel: String {
        switch self {
        case .piiLiteClinical:     return "~265 MB"
        case .openaiPrivacyFilter: return "~180 MB"
        case .glinerRelex:         return "~220 MB"
        }
    }

    public var shortCode: String {
        switch self {
        case .piiLiteClinical:     return "PII"
        case .openaiPrivacyFilter: return "PRIVACY"
        case .glinerRelex:         return "CLINICAL"
        }
    }
}

/// UI-facing cache/transfer state.
public enum ModelDownloadState: Hashable, Sendable {
    case missing
    case partial(bytesOnDisk: Int64, bytesExpected: Int64?)
    case queued
    case downloading(bytesDownloaded: Int64, bytesExpected: Int64?, bytesPerSecond: Double?)
    case installing
    case ready
    case failed(message: String)
    case cancelled

    public var isTerminal: Bool {
        switch self {
        case .ready, .failed, .cancelled: return true
        default: return false
        }
    }

    public var isActive: Bool {
        switch self {
        case .queued, .downloading, .installing: return true
        default: return false
        }
    }
}

/// Discrete event emitted through the manager's AsyncStream.
public enum ModelDownloadEvent: Sendable {
    case stateChanged(ModelDownloadState)
}
