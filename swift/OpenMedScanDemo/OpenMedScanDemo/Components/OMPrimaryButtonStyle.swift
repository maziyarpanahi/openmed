import SwiftUI

/// Primary button — ink fill, paper text. Matches `.btn--primary` in the
/// design system: pressed translates 1pt without transition; hovered darkens.
public struct OMPrimaryButtonStyle: ButtonStyle {
    public var size: Size
    public init(size: Size = .md) { self.size = size }

    public enum Size {
        case sm, md, lg
        fileprivate var fontSize: CGFloat { switch self { case .sm: 13; case .md: 15; case .lg: 16 } }
        fileprivate var vertical: CGFloat { switch self { case .sm: 8; case .md: 12; case .lg: 14 } }
        fileprivate var horizontal: CGFloat { switch self { case .sm: 14; case .md: 20; case .lg: 22 } }
    }

    public func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        return configuration.label
            .font(.om.body(size.fontSize, weight: .semibold))
            .foregroundStyle(Color.omPaper)
            .padding(.vertical, size.vertical)
            .padding(.horizontal, size.horizontal)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                    .fill(pressed ? Color.omStone600 : Color.omInk)
            )
            .offset(y: pressed ? 1 : 0)
            .opacity(configuration.isPressed ? 0.96 : 1)
    }
}

public extension ButtonStyle where Self == OMPrimaryButtonStyle {
    static var omPrimary: OMPrimaryButtonStyle { OMPrimaryButtonStyle() }
    static func omPrimary(_ size: OMPrimaryButtonStyle.Size) -> OMPrimaryButtonStyle { OMPrimaryButtonStyle(size: size) }
}
