import Foundation

/// The shared session lifecycle vocabulary, `openmed.session.lifecycle.v1`.
///
/// Python's `openmed.multimodal.session_lifecycle` defines it; every runtime
/// projects its native session states onto it, so a session report can be
/// compared state by state across runtimes. `finalized`, `cancelled`, `failed`
/// and `closed` are terminal and allow no further transition.
public enum OpenMedSessionLifecycleState: String, Sendable, CaseIterable {
    /// The session accepts work.
    case open
    /// Cancellation or close was requested and active work is still unwinding.
    case cancelling
    /// The session completed its stream and accepts no input.
    case finalized
    /// The session was cancelled and accepts no input.
    case cancelled
    /// The session stopped on an error and accepts no input.
    case failed
    /// The session was closed and released its resources.
    case closed

    /// The vocabulary version, `openmed.session.lifecycle.v1`.
    public static let version = "openmed.session.lifecycle.v1"

    /// Whether the state allows no further transition.
    public var isTerminal: Bool {
        switch self {
        case .open, .cancelling: return false
        case .finalized, .cancelled, .failed, .closed: return true
        }
    }

    /// The states this state may move to, in vocabulary order.
    public var allowedTargets: [OpenMedSessionLifecycleState] {
        switch self {
        case .open: return [.cancelling, .finalized, .cancelled, .failed, .closed]
        case .cancelling: return [.cancelled, .failed, .closed]
        case .finalized, .cancelled, .failed, .closed: return []
        }
    }

    /// Why a transition from this state to `target` is rejected.
    ///
    /// A terminal source gives its terminal code (`session_closed` for
    /// `finalized` and `closed`, `session_cancelled` for `cancelled`,
    /// `session_failed` for `failed`); any other pair outside
    /// ``allowedTargets``, including staying in the same state, gives
    /// `transition_not_allowed`.
    ///
    /// - Parameter target: The requested state.
    /// - Returns: The reason code, or `nil` when the transition is allowed.
    public func transitionReason(to target: OpenMedSessionLifecycleState) -> OpenMedSessionLifecycleError? {
        switch self {
        case .finalized, .closed: return .sessionClosed
        case .cancelled: return .sessionCancelled
        case .failed: return .sessionFailed
        case .open, .cancelling: return allowedTargets.contains(target) ? nil : .transitionNotAllowed
        }
    }

    /// Why a transition between two state names is rejected, exactly as
    /// Python's `transition_reason()` evaluates it.
    ///
    /// A name outside the vocabulary, such as the native Swift name
    /// `finished`, gives `unknown_state` before any other check. Names are
    /// case-sensitive.
    ///
    /// - Parameters:
    ///   - source: The current state name.
    ///   - target: The requested state name.
    /// - Returns: The reason code, or `nil` when the transition is allowed.
    public static func transitionReason(from source: String, to target: String) -> OpenMedSessionLifecycleError? {
        guard let start = OpenMedSessionLifecycleState(rawValue: source),
            let end = OpenMedSessionLifecycleState(rawValue: target)
        else { return .unknownState }
        return start.transitionReason(to: end)
    }
}

/// A rejected lifecycle transition. The raw value is the shared reason code;
/// it never carries session content.
public enum OpenMedSessionLifecycleError: String, Error, LocalizedError, Sendable, CaseIterable {
    /// The source state is `cancelled`.
    case sessionCancelled = "session_cancelled"
    /// The source state is `finalized` or `closed`.
    case sessionClosed = "session_closed"
    /// The source state is `failed`.
    case sessionFailed = "session_failed"
    /// The pair is outside the transition table, including staying in place.
    case transitionNotAllowed = "transition_not_allowed"
    /// A state name is outside the vocabulary.
    case unknownState = "unknown_state"

    public var errorDescription: String? { rawValue }
}
