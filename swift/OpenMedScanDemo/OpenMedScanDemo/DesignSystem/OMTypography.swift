import SwiftUI
import os.log
#if canImport(UIKit)
import UIKit
#endif

/// Font registration + semantic type scale for OpenMed Scan.
///
/// The three families are SIL Open Font License variable fonts (bundled in
/// `Fonts/`). SwiftUI's `.weight()` modifier drives the `wght` axis at render
/// time, so six files cover every style. Every helper falls back to
/// `Font.system` with the right `design` when a PostScript lookup fails —
/// the app stays legible even if registration silently breaks.
public enum OMTypography {
    public enum Family: String {
        case newsreaderRegular = "Newsreader16pt-Regular"
        case newsreaderItalic  = "Newsreader16pt-Italic"
        case interTight        = "InterTight-Regular"
        case interTightItalic  = "InterTight-Italic"
        case jetbrainsMono     = "JetBrainsMono-Regular"
        case jetbrainsMonoItalic = "JetBrainsMono-Italic"
    }

    /// One-shot check the app can call from `init()` to confirm the variable
    /// fonts landed in the bundle and registered with Core Text.
    public static func verifyRegistration() {
        #if DEBUG && canImport(UIKit)
        let log = Logger(subsystem: "com.openmed.scan", category: "typography")
        let expected: [Family] = [
            .newsreaderRegular, .newsreaderItalic,
            .interTight, .interTightItalic,
            .jetbrainsMono, .jetbrainsMonoItalic,
        ]
        var missing: [String] = []
        for family in expected where UIFont(name: family.rawValue, size: 12) == nil {
            missing.append(family.rawValue)
        }
        if missing.isEmpty {
            log.debug("Custom fonts registered OK (\(expected.count, privacy: .public) faces).")
        } else {
            log.error("Missing font PostScript names: \(missing.joined(separator: ", "), privacy: .public)")
        }
        #endif
    }

    // MARK: - Internal helpers

    fileprivate static func resolved(
        _ family: Family,
        size: CGFloat,
        fallbackDesign: Font.Design
    ) -> Font {
        #if canImport(UIKit)
        if UIFont(name: family.rawValue, size: size) != nil {
            return .custom(family.rawValue, size: size)
        }
        #endif
        return .system(size: size, design: fallbackDesign)
    }
}

// MARK: - Font.om convenience namespace

public extension Font {
    enum OM {
        // Display (Newsreader serif, variable wght axis)
        public static func display(
            _ size: CGFloat,
            weight: Font.Weight = .medium,
            italic: Bool = false
        ) -> Font {
            let family: OMTypography.Family = italic ? .newsreaderItalic : .newsreaderRegular
            return OMTypography.resolved(family, size: size, fallbackDesign: .serif)
                .weight(weight)
        }

        // Heading / body (Inter Tight sans)
        public static func heading(
            _ size: CGFloat,
            weight: Font.Weight = .semibold,
            italic: Bool = false
        ) -> Font {
            let family: OMTypography.Family = italic ? .interTightItalic : .interTight
            return OMTypography.resolved(family, size: size, fallbackDesign: .default)
                .weight(weight)
        }

        public static func body(
            _ size: CGFloat = 16,
            weight: Font.Weight = .regular,
            italic: Bool = false
        ) -> Font {
            let family: OMTypography.Family = italic ? .interTightItalic : .interTight
            return OMTypography.resolved(family, size: size, fallbackDesign: .default)
                .weight(weight)
        }

        // Monospace (JetBrains Mono)
        public static func mono(
            _ size: CGFloat = 14,
            weight: Font.Weight = .regular,
            italic: Bool = false
        ) -> Font {
            let family: OMTypography.Family = italic ? .jetbrainsMonoItalic : .jetbrainsMono
            return OMTypography.resolved(family, size: size, fallbackDesign: .monospaced)
                .weight(weight)
        }

        // Eyebrow kicker — JetBrains Mono 12, UPPERCASE, tracked.
        // Callers still apply `.textCase(.uppercase)` + `.kerning(1.68)` on the Text.
        public static func eyebrow() -> Font {
            OMTypography.resolved(.jetbrainsMono, size: 12, fallbackDesign: .monospaced)
                .weight(.medium)
        }
    }

    /// Typed namespace: `Font.om.display(40)`, `Font.om.body(16)`, etc.
    static let om = OM.self
}

// MARK: - Text modifier sugar

public extension Text {
    /// Applies the standard eyebrow look: teal, mono 12, UPPERCASE, tracked +14%.
    func omEyebrow() -> some View {
        self
            .font(.om.eyebrow())
            .textCase(.uppercase)
            .kerning(1.68)
            .foregroundStyle(Color.omTealAccent)
    }

    /// Monospace tag used on entity labels / stat numerals.
    func omMonoTag(size: CGFloat = 11) -> some View {
        self
            .font(.om.mono(size, weight: .medium))
            .textCase(.uppercase)
            .kerning(0.88)
    }
}
