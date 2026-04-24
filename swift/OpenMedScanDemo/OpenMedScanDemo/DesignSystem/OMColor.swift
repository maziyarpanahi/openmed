import SwiftUI

/// Asset-backed semantic colors. Every token has matching light and dark
/// entries in `Assets.xcassets/Colors/*.colorset`, so `@Environment(\.colorScheme)`
/// swaps them atomically.
public extension Color {
    // MARK: Surfaces
    static let omPaper        = Color("Paper",       bundle: .main)
    static let omPaper2       = Color("Paper2",      bundle: .main)
    static let omBgElevated   = Color("BgElevated",  bundle: .main)

    // MARK: Foreground / text
    static let omInk          = Color("Ink",         bundle: .main)
    static let omInk2         = Color("Ink2",        bundle: .main)
    static let omFgMuted      = Color("FgMuted",     bundle: .main)
    static let omFgSubtle     = Color("FgSubtle",    bundle: .main)

    // MARK: Borders / rules
    static let omBorder       = Color("Border",      bundle: .main)
    static let omBorderStrong = Color("BorderStrong", bundle: .main)

    // MARK: Accent (teal)
    static let omTealAccent   = Color("TealAccent",  bundle: .main)
    static let omTealHover    = Color("TealHover",   bundle: .main)
    static let omTealSoft     = Color("TealSoft",    bundle: .main)

    // MARK: Signal (coral)
    static let omSignal       = Color("Signal",      bundle: .main)
    static let omSignalSoft   = Color("SignalSoft",  bundle: .main)

    // MARK: Highlight (paper-marker yellow)
    static let omHighlight    = Color("Highlight",   bundle: .main)

    // MARK: Stone scale
    static let omStone50      = Color("Stone50",     bundle: .main)
    static let omStone100     = Color("Stone100",    bundle: .main)
    static let omStone200     = Color("Stone200",    bundle: .main)
    static let omStone300     = Color("Stone300",    bundle: .main)
    static let omStone400     = Color("Stone400",    bundle: .main)
    static let omStone500     = Color("Stone500",    bundle: .main)
    static let omStone600     = Color("Stone600",    bundle: .main)
    static let omStone700     = Color("Stone700",    bundle: .main)
    static let omStone800     = Color("Stone800",    bundle: .main)
    static let omStone900     = Color("Stone900",    bundle: .main)
}

/// Category palette for detected-entity highlights. Values stay constant
/// across modes because the highlight fill is always yellow paper-marker;
/// only the ink underline + bracket tag change color.
public enum OMEntityTone: String, CaseIterable, Hashable {
    case name, date, identifier, contact, location, organization
    case condition, symptom, medication, dosage, procedure, test
    case allergy, followUp, carePlan, generic

    public var accent: Color {
        switch self {
        case .name, .identifier:       return .omTealAccent
        case .date, .contact:          return .omStone600
        case .location, .organization: return .omInk2
        case .condition, .symptom:     return .omSignal
        case .medication, .dosage:     return .omTealHover
        case .procedure, .test:        return .omInk
        case .allergy:                 return .omSignal
        case .followUp, .carePlan:     return .omTealAccent
        case .generic:                 return .omStone500
        }
    }
}
