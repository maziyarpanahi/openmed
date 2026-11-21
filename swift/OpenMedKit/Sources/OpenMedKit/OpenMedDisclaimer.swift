import Foundation

#if canImport(SwiftUI)
    import SwiftUI
#endif

/// Value-free evidence of an acknowledgement made at an explicit instant.
public struct OpenMedNoticeAcknowledgement: Codable, Equatable, Sendable {
    public let noticeID: String
    public let versionDigest: String
    public let acknowledgedAt: String

    private enum CodingKeys: String, CodingKey {
        case noticeID = "notice_id"
        case versionDigest = "version_digest"
        case acknowledgedAt = "acknowledged_at"
    }
}

/// A reviewed notice shipped with the kit's offline catalog.
public struct OpenMedDisclaimer: Decodable, Equatable, Sendable {
    public let id: String
    public let version: String
    public let versionDigest: String
    public let locale: String
    public let text: String
    public let tier: OpenMedRegulatoryTier
    public var requiresAcknowledgement: Bool { tier.obligations.contains("acknowledgement_required") }

    private enum CodingKeys: String, CodingKey {
        case id, version, locale, text, tier
        case versionDigest = "version_digest"
    }

    private struct Catalog: Decodable, Sendable {
        let notices: [String: OpenMedDisclaimer]
        let tiers: [String: String]
        let capabilities: [String: String]
        let overrides: [String: String]
        let clinicalBriefLimitation: String
        enum CodingKeys: String, CodingKey {
            case notices, tiers, capabilities, overrides
            case clinicalBriefLimitation = "clinical_brief_limitation"
        }
    }

    private static let catalog = Result {
        guard let url = Bundle.module.url(forResource: "safety_notices", withExtension: "json") else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
    }

    public static func forTier(_ tier: OpenMedRegulatoryTier) throws -> Self {
        let data = try catalog.get()
        let key = tier == .unknown ? OpenMedRegulatoryTier.restricted.rawValue : tier.rawValue
        guard let id = data.tiers[key], let notice = data.notices[id] else {
            throw CocoaError(.coderReadCorrupt)
        }
        return notice
    }

    public static func forCapability(_ code: String) throws -> Self? {
        let data = try catalog.get()
        return (data.overrides[code] ?? data.capabilities[code]).flatMap { data.notices[$0] }
    }

    public static func notice(id: String) throws -> Self? {
        try catalog.get().notices[id]
    }

    static var clinicalBriefLimitation: String {
        get throws { try catalog.get().clinicalBriefLimitation }
    }

    static let mapleReasoningLimitation = "For clinician review — not a diagnosis or treatment recommendation."

    public func acknowledge(at instant: Date) -> OpenMedNoticeAcknowledgement? {
        guard requiresAcknowledgement, (-62_135_596_800..<253_402_300_800).contains(instant.timeIntervalSince1970) else { return nil }
        // Python's aware UTC datetime uses six fractional digits only when nonzero.
        let micros = (instant.timeIntervalSince1970 * 1_000_000).rounded()
        let seconds = floor(micros / 1_000_000)
        let fraction = Int(micros - seconds * 1_000_000)
        let whole = ISO8601DateFormatter().string(from: Date(timeIntervalSince1970: seconds)).dropLast()
        let suffix = fraction == 0 ? "" : String(format: ".%06d", fraction)
        return OpenMedNoticeAcknowledgement(
            noticeID: id, versionDigest: versionDigest, acknowledgedAt: "\(whole)\(suffix)+00:00"
        )
    }

    /// Accept only an acknowledgement bound to this exact notice and version.
    public func permitsResults(with receipt: OpenMedNoticeAcknowledgement?) -> Bool {
        guard requiresAcknowledgement else { return true }
        guard let receipt, receipt.noticeID == id, receipt.versionDigest == versionDigest else { return false }
        let formatter = ISO8601DateFormatter()
        if formatter.date(from: receipt.acknowledgedAt) != nil { return true }
        formatter.formatOptions.insert(.withFractionalSeconds)
        return formatter.date(from: receipt.acknowledgedAt) != nil
    }
}

#if canImport(SwiftUI)
    /// Hosts restricted results behind a version-bound acknowledgement.
    public struct OpenMedDisclaimerView<Content: View>: View {
        private let notice: OpenMedDisclaimer
        private let content: () -> Content
        private let onAcknowledged: (OpenMedNoticeAcknowledgement) -> Void
        @State private var receipt: OpenMedNoticeAcknowledgement?

        public init(
            notice: OpenMedDisclaimer,
            onAcknowledged: @escaping (OpenMedNoticeAcknowledgement) -> Void = { _ in },
            @ViewBuilder content: @escaping () -> Content
        ) {
            self.notice = notice
            self.onAcknowledged = onAcknowledged
            self.content = content
        }

        public var body: some View {
            VStack(alignment: .leading) {
                Text(notice.text)
                if notice.permitsResults(with: receipt) {
                    content()
                } else {
                    Button("Acknowledge") {
                        if let accepted = notice.acknowledge(at: Date()) {
                            receipt = accepted
                            onAcknowledged(accepted)
                        }
                    }
                }
            }
            .interactiveDismissDisabled(!notice.permitsResults(with: receipt))
        }
    }
#endif
