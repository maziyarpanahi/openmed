import Foundation

/// Protected visual text and generation metadata with a mandatory review notice.
public struct OpenMedVisionLanguageGeneration: Codable, Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let schemaVersion = "openmed.multimodal.vision_generation.v1"
    public static let noticeKind = MultimodalNoticeKind.visualDescription
    public let text: String
    public let tokenIDs: [Int]
    public let promptTokenCount: Int
    public let generationTokenCount: Int
    public let promptTime: TimeInterval
    public let generationTime: TimeInterval
    public let peakMemoryGB: Double
    public let notice: MultimodalNotice
    public var requiresReviewerConfirmation: Bool { true }
    public var isDiagnostic: Bool { false }

    /// Construct a generation only with the matching catalog notice.
    public init(
        text: String, tokenIDs: [Int] = [], promptTokenCount: Int,
        generationTokenCount: Int, promptTime: TimeInterval, generationTime: TimeInterval,
        peakMemoryGB: Double = 0, notice: MultimodalNotice
    ) throws {
        guard notice == Self.noticeKind.notice else { throw MultimodalNoticeError.invalidNotice }
        guard tokenIDs.allSatisfy({ $0 >= 0 && $0 < 2_147_483_648 }),
            promptTokenCount >= 0, generationTokenCount >= 0,
            [promptTime, generationTime, peakMemoryGB].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1e12 })
        else { throw MultimodalNoticeError.invalidResult }
        self.text = text
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generationTime = generationTime
        self.peakMemoryGB = peakMemoryGB
        self.notice = notice
    }

    /// Protected display text including the mandatory notice; never log this string.
    public var description: String { "[\(notice.identifier)] \(notice.text)\n\n\(text)" }

    /// Content-free diagnostic rendering excludes generated text and token values.
    public var debugDescription: String {
        "OpenMedVisionLanguageGeneration(notice: \(notice.identifier), promptTokens: \(promptTokenCount), generationTokens: \(generationTokenCount))"
    }

    /// Explicit review guard; caller enforces reviewer identity and source validation.
    public func requireReviewerConfirmation(reviewerConfirmed: Bool) throws {
        guard reviewerConfirmed else { throw MultimodalNoticeError.reviewerConfirmationRequired }
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case text
        case tokenIDs = "token_ids"
        case promptTokenCount = "prompt_tokens"
        case generationTokenCount = "generation_tokens"
        case promptTime = "prompt_seconds"
        case generationTime = "generation_seconds"
        case peakMemoryGB = "peak_memory_gb"
        case notice
        case requiresReviewerConfirmation = "requires_reviewer_confirmation"
        case isDiagnostic = "is_diagnostic"
    }

    public init(from decoder: Decoder) throws {
        try rejectUnknownNoticeFields(
            decoder,
            allowed: [
                "schema_version", "text", "token_ids", "prompt_tokens", "generation_tokens",
                "prompt_seconds", "generation_seconds", "peak_memory_gb", "notice",
                "requires_reviewer_confirmation", "is_diagnostic",
            ])
        let container = try decoder.container(keyedBy: CodingKeys.self)
        guard try container.decode(String.self, forKey: .schemaVersion) == Self.schemaVersion,
            try container.decode(Bool.self, forKey: .requiresReviewerConfirmation),
            try !container.decode(Bool.self, forKey: .isDiagnostic)
        else { throw MultimodalNoticeError.invalidResult }
        try self.init(
            text: container.decode(String.self, forKey: .text),
            tokenIDs: container.decode([Int].self, forKey: .tokenIDs),
            promptTokenCount: container.decode(Int.self, forKey: .promptTokenCount),
            generationTokenCount: container.decode(Int.self, forKey: .generationTokenCount),
            promptTime: container.decode(Double.self, forKey: .promptTime),
            generationTime: container.decode(Double.self, forKey: .generationTime),
            peakMemoryGB: container.decode(Double.self, forKey: .peakMemoryGB),
            notice: container.decode(MultimodalNotice.self, forKey: .notice)
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(Self.schemaVersion, forKey: .schemaVersion)
        try container.encode(text, forKey: .text)
        try container.encode(tokenIDs, forKey: .tokenIDs)
        try container.encode(promptTokenCount, forKey: .promptTokenCount)
        try container.encode(generationTokenCount, forKey: .generationTokenCount)
        try container.encode(promptTime, forKey: .promptTime)
        try container.encode(generationTime, forKey: .generationTime)
        try container.encode(peakMemoryGB, forKey: .peakMemoryGB)
        try container.encode(notice, forKey: .notice)
        try container.encode(true, forKey: .requiresReviewerConfirmation)
        try container.encode(false, forKey: .isDiagnostic)
    }
}
