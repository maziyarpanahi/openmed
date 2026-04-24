import SwiftUI

/// Card elevation tokens. `raised` adds `--shadow-2` over the hairline border.
public enum OMCardElevation { case flat, raised }

/// Editorial card — white (or dark-elevated) surface with a hairline border
/// and optional `--shadow-2`. No rounded-corner bitmap, always vector continuous.
public struct OMCard<Content: View>: View {
    private let elevation: OMCardElevation
    private let padding: CGFloat
    private let cornerRadius: CGFloat
    private let content: () -> Content

    public init(
        elevation: OMCardElevation = .flat,
        padding: CGFloat = OM.Space.s5,
        cornerRadius: CGFloat = OM.Radius.lg,
        @ViewBuilder content: @escaping () -> Content
    ) {
        self.elevation = elevation
        self.padding = padding
        self.cornerRadius = cornerRadius
        self.content = content
    }

    public var body: some View {
        let shape = RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
        let base = content()
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color.omBgElevated, in: shape)
            .overlay(shape.strokeBorder(Color.omBorder, lineWidth: OM.Stroke.hairline))
        switch elevation {
        case .flat:
            base
        case .raised:
            base.omShadowElevated()
        }
    }
}
