import SwiftUI

/// The OpenMed mark — ink rounded square, paper cross, teal detection-pin dot.
/// Drawn with Canvas so it stays crisp at any scale.
public struct OMLogoMark: View {
    public var size: CGFloat
    public init(size: CGFloat = 28) { self.size = size }

    public var body: some View {
        Canvas { context, canvasSize in
            let width = canvasSize.width
            let height = canvasSize.height
            let cornerRadius = width * 0.22
            let square = Path(roundedRect: CGRect(origin: .zero, size: CGSize(width: width, height: height)),
                              cornerSize: CGSize(width: cornerRadius, height: cornerRadius))
            context.fill(square, with: .color(.omInk))

            let barThickness = width * 0.12
            let barLength = width * 0.50
            let horizontal = Path(roundedRect: CGRect(
                x: (width - barLength) / 2,
                y: (height - barThickness) / 2,
                width: barLength,
                height: barThickness
            ), cornerSize: CGSize(width: barThickness / 2, height: barThickness / 2))
            let vertical = Path(roundedRect: CGRect(
                x: (width - barThickness) / 2,
                y: (height - barLength) / 2,
                width: barThickness,
                height: barLength
            ), cornerSize: CGSize(width: barThickness / 2, height: barThickness / 2))
            context.fill(horizontal, with: .color(.omPaper))
            context.fill(vertical, with: .color(.omPaper))

            let dotSize = width * 0.09
            let dotRect = CGRect(
                x: (width - dotSize) / 2,
                y: (height - dotSize) / 2,
                width: dotSize,
                height: dotSize
            )
            context.fill(Path(ellipseIn: dotRect), with: .color(.omTealAccent))
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

/// Logo plus wordmark — used in the top bar of each screen.
public struct OMBrandLockup: View {
    public var compact: Bool
    public init(compact: Bool = false) { self.compact = compact }

    public var body: some View {
        HStack(spacing: compact ? 8 : 10) {
            OMLogoMark(size: compact ? 22 : 26)
            VStack(alignment: .leading, spacing: -2) {
                HStack(spacing: 2) {
                    Text("open")
                        .font(.om.display(compact ? 17 : 19, weight: .medium))
                        .foregroundStyle(Color.omInk)
                    Text("med")
                        .font(.om.display(compact ? 17 : 19, weight: .medium, italic: true))
                        .foregroundStyle(Color.omTealAccent)
                    Text(".")
                        .font(.om.display(compact ? 17 : 19, weight: .medium))
                        .foregroundStyle(Color.omTealAccent)
                }
                Text("SCAN · LOCAL · ON-DEVICE")
                    .font(.om.mono(9, weight: .medium))
                    .textCase(.uppercase)
                    .kerning(1.4)
                    .foregroundStyle(Color.omFgMuted)
            }
        }
    }
}
