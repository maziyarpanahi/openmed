import SwiftUI

/// Determinate progress bar — ink fill on a stone-100 track.
/// Indeterminate mode runs a shimmering band at `omBase` speed.
public struct OMProgressBar: View {
    public enum Mode: Equatable {
        case determinate(progress: Double)
        case indeterminate
    }

    public var mode: Mode
    public var height: CGFloat = 6
    public var fill: Color = .omInk
    public var track: Color = .omStone100

    public init(mode: Mode, height: CGFloat = 6, fill: Color = .omInk, track: Color = .omStone100) {
        self.mode = mode
        self.height = height
        self.fill = fill
        self.track = track
    }

    public var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule(style: .continuous).fill(track)
                switch mode {
                case .determinate(let value):
                    let clamped = max(0, min(1, value))
                    Capsule(style: .continuous)
                        .fill(fill)
                        .frame(width: max(0, geo.size.width * clamped))
                        .animation(.omBase, value: clamped)
                case .indeterminate:
                    IndeterminateBand(width: geo.size.width, fill: fill)
                }
            }
            .frame(height: height)
        }
        .frame(height: height)
    }
}

private struct IndeterminateBand: View {
    let width: CGFloat
    let fill: Color
    @State private var phase: CGFloat = -0.35

    var body: some View {
        let bandWidth = max(width * 0.35, 80)
        Capsule(style: .continuous)
            .fill(fill)
            .frame(width: bandWidth)
            .offset(x: phase * width)
            .onAppear {
                withAnimation(.linear(duration: 1.1).repeatForever(autoreverses: false)) {
                    phase = 1.1
                }
            }
            .mask(Capsule(style: .continuous))
    }
}
