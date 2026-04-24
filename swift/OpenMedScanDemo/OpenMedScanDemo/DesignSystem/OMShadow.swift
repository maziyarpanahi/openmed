import SwiftUI

/// Shadow recipes mirroring `--shadow-1/2/ink` in the design system.
/// The design brief forbids colored glows — these are the only three allowed.
public extension View {
    /// `--shadow-1` — rendered as a 1pt hairline border instead of a CSS box-shadow.
    /// Use on flat cards that need structural separation from the paper substrate.
    func omHairline(
        color: Color = .omBorder,
        radius: CGFloat = OM.Radius.lg
    ) -> some View {
        self.overlay {
            RoundedRectangle(cornerRadius: radius, style: .continuous)
                .strokeBorder(color, lineWidth: OM.Stroke.hairline)
        }
    }

    /// `--shadow-2` — soft elevated shadow for floating cards and popovers.
    /// Uses heavier opacity in dark mode, mirroring the CSS dark-mode override.
    func omShadowElevated() -> some View {
        modifier(OMShadowElevatedModifier())
    }

    /// `--shadow-ink` — deeper shadow for dark-surface sections.
    func omShadowInkDeep() -> some View {
        self
            .shadow(color: Color.black.opacity(0.35), radius: 8, x: 0, y: 8)
    }
}

private struct OMShadowElevatedModifier: ViewModifier {
    @Environment(\.colorScheme) private var colorScheme

    func body(content: Content) -> some View {
        let primaryOpacity: Double = colorScheme == .dark ? 0.50 : 0.12
        let secondaryOpacity: Double = colorScheme == .dark ? 0.25 : 0.04
        return content
            .shadow(color: Color.black.opacity(primaryOpacity), radius: 32, x: 0, y: 12)
            .shadow(color: Color.black.opacity(secondaryOpacity), radius: 4, x: 0, y: 2)
    }
}
