import Foundation

/// Controlled, patient-value-free notice failures.
public enum MultimodalNoticeError: Error {
    case invalidNotice
    case invalidResult
    case reviewerConfirmationRequired
}

/// Clinical meaning of a review-only multimodal output; shared with Python.
public enum MultimodalNoticeKind: String, CaseIterable, Codable, Sendable {
    case measurement = "measurement_for_review"
    case visualDescription = "visual_description"
    case draft = "draft_for_review"

    /// Immutable published notice. Changed wording requires a new versioned ID.
    public var notice: MultimodalNotice {
        switch self {
        case .measurement:
            return MultimodalNotice(
                catalogIdentifier: "openmed.multimodal.measurement_for_review.v1",
                text: "Measurement candidate for clinician review only. This output is not a "
                    + "diagnosis. A qualified clinician must independently review the source "
                    + "and explicitly confirm before any consequential use. This output must "
                    + "never automatically trigger a clinical decision."
            )
        case .visualDescription:
            return MultimodalNotice(
                catalogIdentifier: "openmed.multimodal.visual_description.v1",
                text: "Visual description for clinician review only. This output is not a "
                    + "diagnosis. A qualified clinician must independently review the source "
                    + "and explicitly confirm before any consequential use. This output must "
                    + "never automatically trigger a clinical decision."
            )
        case .draft:
            return MultimodalNotice(
                catalogIdentifier: "openmed.multimodal.draft_for_review.v1",
                text: "Draft for clinician review only. This output is not a diagnosis. A "
                    + "qualified clinician must independently review the source and explicitly "
                    + "confirm before any consequential use. This output must never "
                    + "automatically trigger a clinical decision."
            )
        }
    }
}

/// An exact identifier/text pair from the published, versioned notice catalog.
public struct MultimodalNotice: Codable, Equatable, Sendable {
    public let identifier: String
    public let text: String

    fileprivate init(catalogIdentifier: String, text: String) {
        self.identifier = catalogIdentifier
        self.text = text
    }

    /// Reject caller interpolation or any mismatch between identifier and text.
    public init(identifier: String, text: String) throws {
        guard
            MultimodalNoticeKind.allCases.contains(where: {
                $0.notice.identifier == identifier && $0.notice.text == text
            })
        else { throw MultimodalNoticeError.invalidNotice }
        self.identifier = identifier
        self.text = text
    }

    private enum CodingKeys: String, CodingKey { case identifier, text }

    public init(from decoder: Decoder) throws {
        try rejectUnknownNoticeFields(decoder, allowed: ["identifier", "text"])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            identifier: container.decode(String.self, forKey: .identifier),
            text: container.decode(String.self, forKey: .text)
        )
    }
}

/// Digest-bound review wrapper; protected measurement/draft schemas stay with producers.
public struct MultimodalReviewResult: Codable, Equatable, Sendable, CustomStringConvertible {
    public static let schemaVersion = "openmed.multimodal.review_result.v1"
    public let kind: MultimodalNoticeKind
    public let outputDigest: String
    public let notice: MultimodalNotice
    public var requiresReviewerConfirmation: Bool { true }
    public var isDiagnostic: Bool { false }

    /// Require the matching notice and an opaque lowercase SHA-256 output reference.
    public init(kind: MultimodalNoticeKind, outputDigest: String, notice: MultimodalNotice) throws {
        guard notice == kind.notice else { throw MultimodalNoticeError.invalidNotice }
        guard outputDigest.utf8.count == 64,
            outputDigest.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { throw MultimodalNoticeError.invalidResult }
        self.kind = kind
        self.outputDigest = outputDigest
        self.notice = notice
    }

    /// Require explicit review before host-controlled consequential use; performs no action.
    public func requireReviewerConfirmation(reviewerConfirmed: Bool) throws {
        guard reviewerConfirmed else { throw MultimodalNoticeError.reviewerConfirmationRequired }
    }

    public var description: String {
        "[\(notice.identifier)] \(notice.text)\nOutput digest: \(outputDigest)"
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case outputDigest = "output_digest"
        case notice
        case requiresReviewerConfirmation = "requires_reviewer_confirmation"
        case isDiagnostic = "is_diagnostic"
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownNoticeFields(
            decoder,
            allowed: [
                "schema_version", "output_digest", "notice", "requires_reviewer_confirmation", "is_diagnostic",
            ])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard try container.decode(String.self, forKey: .schemaVersion) == Self.schemaVersion,
            try container.decode(Bool.self, forKey: .requiresReviewerConfirmation),
            try !container.decode(Bool.self, forKey: .isDiagnostic)
        else { throw MultimodalNoticeError.invalidResult }
        let notice = try container.decode(MultimodalNotice.self, forKey: .notice)
        guard let kind = MultimodalNoticeKind.allCases.first(where: { $0.notice == notice }) else {
            throw MultimodalNoticeError.invalidNotice
        }
        try self.init(
            kind: kind, outputDigest: container.decode(String.self, forKey: .outputDigest), notice: notice
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Self.schemaVersion, forKey: .schemaVersion)
        try container.encode(outputDigest, forKey: .outputDigest)
        try container.encode(notice, forKey: .notice)
        try container.encode(true, forKey: .requiresReviewerConfirmation)
        try container.encode(false, forKey: .isDiagnostic)
    }
}

private struct NoticeFieldKey: CodingKey {
    let stringValue: String
    var intValue: Int? { nil }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { return nil }
}

func rejectUnknownNoticeFields(_ decoder: Decoder, allowed: Set<String>) throws {
    let container = try decoder.container(keyedBy: NoticeFieldKey.self)
    guard Set(container.allKeys.map(\.stringValue)) == allowed else {
        throw MultimodalNoticeError.invalidResult
    }
}
