import SwiftUI

/// Outline button — hairline border, paper bg, ink text. Fills on press.
public struct OMSecondaryButtonStyle: ButtonStyle {
    public var size: OMPrimaryButtonStyle.Size
    public init(size: OMPrimaryButtonStyle.Size = .md) { self.size = size }

    public func makeBody(configuration: Configuration) -> some View {
        let pressed = configuration.isPressed
        return configuration.label
            .font(.om.body(size == .sm ? 13 : 15, weight: .semibold))
            .foregroundStyle(Color.omInk)
            .padding(.vertical, size == .sm ? 8 : 11)
            .padding(.horizontal, size == .sm ? 14 : 18)
            .frame(maxWidth: .infinity)
            .background(
                RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                    .fill(pressed ? Color.omPaper2 : Color.omBgElevated)
            )
            .overlay(
                RoundedRectangle(cornerRadius: OM.Radius.md, style: .continuous)
                    .strokeBorder(pressed ? Color.omInk : Color.omBorderStrong, lineWidth: OM.Stroke.hairline)
            )
            .offset(y: pressed ? 1 : 0)
    }
}

public extension ButtonStyle where Self == OMSecondaryButtonStyle {
    static var omSecondary: OMSecondaryButtonStyle { OMSecondaryButtonStyle() }
    static func omSecondary(_ size: OMPrimaryButtonStyle.Size) -> OMSecondaryButtonStyle { OMSecondaryButtonStyle(size: size) }
}
