import SwiftUI

/// Ghost / link button — teal text with an underline on press.
public struct OMGhostButtonStyle: ButtonStyle {
    public var size: OMPrimaryButtonStyle.Size
    public var tone: Tone
    public init(size: OMPrimaryButtonStyle.Size = .md, tone: Tone = .accent) {
        self.size = size
        self.tone = tone
    }

    public enum Tone {
        case accent, ink, signal
        fileprivate var color: Color {
            switch self {
            case .accent: return .omTealAccent
            case .ink:    return .omInk
            case .signal: return .omSignal
            }
        }
    }

    public func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        return configuration.label
            .font(.om.body(size == .sm ? 13 : 15, weight: .semibold))
            .foregroundStyle(tone.color)
            .padding(.vertical, size == .sm ? 6 : 8)
            .padding(.horizontal, size == .sm ? 8 : 10)
            .underline(pressed, color: tone.color)
            .opacity(pressed ? 0.8 : 1)
    }
}

public extension ButtonStyle where Self == OMGhostButtonStyle {
    static var omGhost: OMGhostButtonStyle { OMGhostButtonStyle() }
    static func omGhost(_ tone: OMGhostButtonStyle.Tone) -> OMGhostButtonStyle { OMGhostButtonStyle(tone: tone) }
}
