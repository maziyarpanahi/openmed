import SwiftUI

enum MedicalReasoningTheme {
    static let canvas = Color(red: 0.965, green: 0.957, blue: 0.925)
    static let surface = Color.white.opacity(0.78)
    static let surfaceStrong = Color.white.opacity(0.94)
    static let ink = Color(red: 0.035, green: 0.055, blue: 0.07)
    static let secondaryInk = Color(red: 0.30, green: 0.32, blue: 0.30)
    static let teal = Color(red: 0.02, green: 0.43, blue: 0.44)
    static let tealSoft = Color(red: 0.82, green: 0.92, blue: 0.90)
    static let coral = Color(red: 0.75, green: 0.25, blue: 0.20)
    static let border = Color.black.opacity(0.10)

    static let pagePadding: CGFloat = 20
    static let cardRadius: CGFloat = 22
}

extension View {
    func medicalCard(padding: CGFloat = 18) -> some View {
        self
            .padding(padding)
            .background(
                MedicalReasoningTheme.surface,
                in: RoundedRectangle(
                    cornerRadius: MedicalReasoningTheme.cardRadius,
                    style: .continuous
                )
            )
            .overlay {
                RoundedRectangle(
                    cornerRadius: MedicalReasoningTheme.cardRadius,
                    style: .continuous
                )
                .strokeBorder(MedicalReasoningTheme.border, lineWidth: 1)
            }
    }
}

struct MedicalReasoningBrand: View {
    var compact = false

    var body: some View {
        HStack(spacing: 11) {
            ZStack {
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(MedicalReasoningTheme.ink)
                    .frame(width: compact ? 36 : 44, height: compact ? 36 : 44)
                Image(systemName: "cross.case.fill")
                    .font(.system(size: compact ? 15 : 18, weight: .semibold))
                    .foregroundStyle(MedicalReasoningTheme.canvas)
            }

            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 4) {
                    Text("open")
                        .font(.system(size: compact ? 17 : 21, weight: .semibold, design: .serif))
                    Text("med.")
                        .font(.system(size: compact ? 17 : 21, weight: .semibold, design: .serif))
                        .italic()
                        .foregroundStyle(MedicalReasoningTheme.teal)
                }
                Text("MEDICAL REASONING · LOCAL")
                    .font(.system(size: compact ? 8 : 9, weight: .semibold, design: .monospaced))
                    .tracking(1.8)
                    .foregroundStyle(MedicalReasoningTheme.secondaryInk)
            }
            .foregroundStyle(MedicalReasoningTheme.ink)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("OpenMed Medical Reasoning, local on device")
    }
}

struct MedicalStatusPill: View {
    let title: String
    let systemImage: String
    var accent = MedicalReasoningTheme.teal

    var body: some View {
        Label(title, systemImage: systemImage)
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .foregroundStyle(accent)
            .padding(.horizontal, 11)
            .padding(.vertical, 7)
            .background(accent.opacity(0.11), in: Capsule())
    }
}

#Preview("Brand") {
    MedicalReasoningBrand()
        .padding()
        .background(MedicalReasoningTheme.canvas)
}
