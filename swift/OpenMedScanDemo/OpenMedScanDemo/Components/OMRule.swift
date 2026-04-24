import SwiftUI

/// 1pt hairline rule — the structural separator that the design system
/// leans on instead of gradients or heavy shadows.
public struct OMRule: View {
    public enum Axis { case horizontal, vertical }
    public var axis: Axis = .horizontal
    public var color: Color = .omBorder

    public init(axis: Axis = .horizontal, color: Color = .omBorder) {
        self.axis = axis
        self.color = color
    }

    public var body: some View {
        Rectangle()
            .fill(color)
            .frame(
                width: axis == .vertical ? OM.Stroke.hairline : nil,
                height: axis == .horizontal ? OM.Stroke.hairline : nil
            )
    }
}
