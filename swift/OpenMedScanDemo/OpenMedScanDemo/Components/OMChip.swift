import SwiftUI

/// Chip / tag — monospace label inside a pill. Three variants mirror the
/// CSS `.tag` / `.tag--accent` / `.tag--ink` rules.
public struct OMChip: View {
    public enum Tone {
        case neutral, accent, ink, signal, highlight, soft

        fileprivate var bg: Color {
            switch self {
            case .neutral:   return .omBgElevated
            case .accent:    return .omTealSoft
            case .ink:       return .omInk
            case .signal:    return .omSignalSoft
            case .highlight: return .omHighlight
            case .soft:      return .omPaper2
            }
        }

        fileprivate var fg: Color {
            switch self {
            case .neutral:   return .omInk
            case .accent:    return .omTealHover
            case .ink:       return .omPaper
            case .signal:    return .omSignal
            case .highlight: return .omInk
            case .soft:      return .omFgMuted
            }
        }

        fileprivate var borderColor: Color? {
            switch self {
            case .neutral: return .omBorderStrong
            case .soft:    return .omBorder
            default:       return nil
            }
        }
    }

    private let text: String
    private let tone: Tone
    private let leadingSystemImage: String?
    private let trailingSystemImage: String?
    private let action: (() -> Void)?
    private let onRemove: (() -> Void)?

    public init(
        _ text: String,
        tone: Tone = .neutral,
        leadingSystemImage: String? = nil,
        trailingSystemImage: String? = nil,
        action: (() -> Void)? = nil,
        onRemove: (() -> Void)? = nil
    ) {
        self.text = text
        self.tone = tone
        self.leadingSystemImage = leadingSystemImage
        self.trailingSystemImage = trailingSystemImage
        self.action = action
        self.onRemove = onRemove
    }

    public var body: some View {
        let content = HStack(spacing: 6) {
            if let leadingSystemImage {
                Image(systemName: leadingSystemImage)
                    .font(.system(size: 10, weight: .semibold))
            }
            Text(text)
                .font(.om.mono(12, weight: .medium))
                .textCase(.uppercase)
                .kerning(1.2)
                .lineLimit(1)
            if let trailingSystemImage {
                Image(systemName: trailingSystemImage)
                    .font(.system(size: 10, weight: .semibold))
            }
            if let onRemove {
                Button {
                    onRemove()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 9, weight: .bold))
                        .padding(3)
                        .background(Circle().fill(tone.fg.opacity(0.12)))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Remove \(text)")
            }
        }
        .foregroundStyle(tone.fg)
        .padding(.vertical, 6)
        .padding(.horizontal, 10)
        .background(
            Capsule(style: .continuous).fill(tone.bg)
        )
        .overlay(alignment: .center) {
            if let borderColor = tone.borderColor {
                Capsule(style: .continuous).strokeBorder(borderColor, lineWidth: OM.Stroke.hairline)
            }
        }

        if let action {
            Button(action: action) { content }
                .buttonStyle(.plain)
                .contentShape(Capsule(style: .continuous))
        } else {
            content
        }
    }
}
