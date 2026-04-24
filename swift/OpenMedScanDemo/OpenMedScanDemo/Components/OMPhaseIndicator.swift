import SwiftUI

/// Indeterminate phase label with a rotating caret: `LOADING ▸ MLX runtime`.
public struct OMPhaseIndicator: View {
    public let phase: String
    public let detail: String?
    @State private var spin = false

    public init(phase: String, detail: String? = nil) {
        self.phase = phase
        self.detail = detail
    }

    public var body: some View {
        HStack(spacing: OM.Space.s2) {
            Image(systemName: "circle.dotted")
                .rotationEffect(.degrees(spin ? 360 : 0))
                .animation(.linear(duration: 1.6).repeatForever(autoreverses: false), value: spin)
                .foregroundStyle(Color.omTealAccent)
            Text(phase)
                .omMonoTag(size: 11)
                .foregroundStyle(Color.omTealAccent)
            if let detail {
                Text("·")
                    .font(.om.mono(11))
                    .foregroundStyle(Color.omFgSubtle)
                Text(detail)
                    .font(.om.mono(11))
                    .foregroundStyle(Color.omFgMuted)
                    .lineLimit(1)
            }
        }
        .onAppear { spin = true }
    }
}
