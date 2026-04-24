import SwiftUI

/// Five dots + rules progress — completed, current, upcoming states.
/// Tapping a completed dot jumps back; tapping an upcoming dot is a no-op
/// with a rigid haptic so the user understands it's gated.
public struct OMWorkflowIndicator: View {
    public let currentStage: ScanStage
    public let furthestReached: ScanStage
    public let onJump: (ScanStage) -> Void

    public init(
        currentStage: ScanStage,
        furthestReached: ScanStage,
        onJump: @escaping (ScanStage) -> Void
    ) {
        self.currentStage = currentStage
        self.furthestReached = furthestReached
        self.onJump = onJump
    }

    public var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(ScanStage.allCases.enumerated()), id: \.element.id) { index, stage in
                dot(for: stage)
                if index < ScanStage.count - 1 {
                    rule(between: stage, next: ScanStage(rawValue: stage.rawValue + 1)!)
                }
            }
        }
        .frame(height: 38)
    }

    // MARK: Subviews

    @ViewBuilder
    private func dot(for stage: ScanStage) -> some View {
        let state = state(for: stage)
        let canTap = state != .upcoming
        Button {
            if canTap {
                onJump(stage)
            } else {
                HapticsCenter.impact(.rigid)
            }
        } label: {
            VStack(spacing: 4) {
                ZStack {
                    Circle()
                        .fill(fill(for: state))
                        .frame(width: 16, height: 16)
                    Circle()
                        .strokeBorder(stroke(for: state), lineWidth: OM.Stroke.hairline)
                        .frame(width: 16, height: 16)
                    if state == .completed {
                        Image(systemName: "checkmark")
                            .font(.system(size: 9, weight: .heavy))
                            .foregroundStyle(Color.omPaper)
                    } else if state == .current {
                        Circle()
                            .fill(Color.omPaper)
                            .frame(width: 5, height: 5)
                    }
                }
                Text(stage.shortTitle)
                    .font(.om.mono(9, weight: .medium))
                    .textCase(.uppercase)
                    .kerning(1.0)
                    .foregroundStyle(foreground(for: state))
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(!canTap)
        .accessibilityLabel("\(stage.shortTitle) stage, \(state.accessibilityHint)")
    }

    private func rule(between: ScanStage, next: ScanStage) -> some View {
        let ruleState = ruleState(from: between, to: next)
        return Rectangle()
            .fill(ruleState == .filled ? Color.omInk : Color.omBorderStrong)
            .frame(height: OM.Stroke.hairline)
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 2)
            .offset(y: -8)
    }

    // MARK: State derivation

    private enum DotState { case completed, current, upcoming
        var accessibilityHint: String {
            switch self {
            case .completed: return "completed"
            case .current:   return "current"
            case .upcoming:  return "not started"
            }
        }
    }

    private enum RuleState { case filled, empty }

    private func state(for stage: ScanStage) -> DotState {
        if stage == currentStage { return .current }
        if stage.rawValue < currentStage.rawValue { return .completed }
        return .upcoming
    }

    private func ruleState(from: ScanStage, to: ScanStage) -> RuleState {
        (from.rawValue < currentStage.rawValue) ? .filled : .empty
    }

    private func fill(for state: DotState) -> Color {
        switch state {
        case .completed: return .omTealAccent
        case .current:   return .omInk
        case .upcoming:  return .omBgElevated
        }
    }

    private func stroke(for state: DotState) -> Color {
        switch state {
        case .completed: return .omTealAccent
        case .current:   return .omInk
        case .upcoming:  return .omBorderStrong
        }
    }

    private func foreground(for state: DotState) -> Color {
        switch state {
        case .completed: return .omTealAccent
        case .current:   return .omInk
        case .upcoming:  return .omFgSubtle
        }
    }
}
