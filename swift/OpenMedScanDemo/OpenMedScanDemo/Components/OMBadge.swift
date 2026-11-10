import SwiftUI

/// Small status pill — shorter than a chip, always static.
/// Examples: `DOWNLOADING`, `READY`, `262 MB`.
public struct OMBadge: View {
    public enum Tone { case neutral, accent, signal, highlight, ink, positive }

    public let text: String
    public let tone: Tone
    public let systemImage: String?

    public init(_ text: String, tone: Tone = .neutral, systemImage: String? = nil) {
        self.text = text
        self.tone = tone
        self.systemImage = systemImage
    }

    public var body: some View {
        HStack(spacing: 4) {
            if let systemImage {
                Image(systemName: systemImage)
                    .font(.system(size: 9, weight: .bold))
            }
            Text(text)
                .font(.om.mono(10, weight: .semibold))
                .textCase(.uppercase)
                .kerning(1.1)
                .lineLimit(1)
        }
        .foregroundStyle(fg)
        .padding(.vertical, 3)
        .padding(.horizontal, 7)
        .background(bg, in: Capsule(style: .continuous))
        .overlay {
            if let borderColor {
                Capsule(style: .continuous).strokeBorder(borderColor, lineWidth: OM.Stroke.hairline)
            }
        }
    }

    private var bg: Color {
        switch tone {
        case .neutral:   return .omBgElevated
        case .accent:    return .omTealSoft
        case .signal:    return .omSignalSoft
        case .highlight: return .omHighlight
        case .ink:       return .omInk
        case .positive:  return .omTealSoft
        }
    }

    private var fg: Color {
        switch tone {
        case .neutral:   return .omFgMuted
        case .accent:    return .omTealHover
        case .signal:    return .omSignal
        case .highlight: return .omInk
        case .ink:       return .omPaper
        case .positive:  return .omTealHover
        }
    }

    private var borderColor: Color? {
        tone == .neutral ? .omBorder : nil
    }
}
