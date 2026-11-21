import os

/// Closed codes shared with the local Python recorder.
public enum OpenMedUseErrorCode: String, CaseIterable, Sendable {
    case acknowledgementSkipped = "acknowledgement_skipped"
    case restrictedCapabilityRefused = "restricted_capability_refused"
    case exportBeforeReview = "export_before_review"
}

/// Host-owned, in-memory recorder. Collection is opt-in through an injected
/// recorder or a task-local binding and contains no individual event records.
public final class OpenMedUseErrorRecorder: Sendable {
    @TaskLocal public static var current: OpenMedUseErrorRecorder?
    private let counts = OSAllocatedUnfairLock(initialState: [String: Int]())

    public init() {}

    public func record(_ code: OpenMedUseErrorCode) {
        counts.withLock { counts in
            let count = counts[code.rawValue, default: 0]
            if count < Int.max { counts[code.rawValue] = count + 1 }
        }
    }

    public func snapshot() -> [String: Int] {
        counts.withLock { $0 }
    }
}
