import Foundation

/// The five pipeline stages the user walks through in order.
/// Name kept distinct from ContentView's private enum so both can coexist
/// during the migration; the private one is removed once all screens ship.
public enum ScanStage: Int, CaseIterable, Identifiable, Sendable, Hashable {
    case input
    case review
    case deidentify
    case clinical
    case summary

    public var id: Int { rawValue }

    /// Short name used in the workflow indicator and the action bar hint.
    public var shortTitle: String {
        switch self {
        case .input:      return "Input"
        case .review:     return "Review"
        case .deidentify: return "De-ID"
        case .clinical:   return "Clinical"
        case .summary:    return "Summary"
        }
    }

    /// `STAGE 01 · INPUT` eyebrow for the screen header.
    public var eyebrow: String {
        String(format: "STAGE %02d · %@", rawValue + 1, shortTitle.uppercased())
    }

    public var next: ScanStage? { ScanStage(rawValue: rawValue + 1) }
    public var previous: ScanStage? { ScanStage(rawValue: rawValue - 1) }

    /// Total stage count, handy for progress math.
    public static let count: Int = ScanStage.allCases.count
}
