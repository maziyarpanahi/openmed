import AppIntents
import Foundation
import OpenMedKit

/// What a PII summary returns: redacted text and per-label counts.
///
/// Shortcuts can hand an intent's output to any other app or web action, so
/// the summary never carries the detected values, their offsets, confidences
/// or replacements. Only canonical label names and counts leave the device
/// alongside text that is already redacted.
public struct OpenMedPIISummary: TransientAppEntity, Sendable {
    public static var typeDisplayRepresentation: TypeDisplayRepresentation { "PII Summary" }

    /// The input with every detected identifier handled by the policy.
    @Property(title: "Redacted Text")
    public var redactedText: String

    /// Number of detected identifiers across all labels.
    @Property(title: "Identifier Count")
    public var identifierCount: Int

    /// One `label: count` entry per canonical label, sorted by label.
    @Property(title: "Counts by Label")
    public var labelCounts: [String]

    public var displayRepresentation: DisplayRepresentation {
        DisplayRepresentation(title: "\(identifierCount) identifiers", subtitle: "\(labelCounts.joined(separator: ", "))")
    }

    public init() {
        redactedText = ""
        identifierCount = 0
        labelCounts = []
    }

    /// Summarize a policy result, dropping every value, offset and replacement.
    public init(_ result: PolicyDeidentificationResult) {
        self.init()
        let counts = Self.counts(result)
        redactedText = result.redactedText
        identifierCount = result.actions.count
        labelCounts = counts.map { "\($0.label): \($0.count)" }
    }

    /// Canonical label counts of a policy result, sorted by label.
    public static func counts(_ result: PolicyDeidentificationResult) -> [(label: String, count: Int)] {
        Dictionary(grouping: result.actions, by: \.canonicalLabel)
            .map { (label: $0.key, count: $0.value.count) }
            .sorted { $0.label < $1.label }
    }
}

/// "Summarize PII with OpenMed": counts the identifiers in text on the device
/// and returns those counts with the redacted text, never the values.
///
/// Same on-device boundary as ``RedactTextIntent``: it uses the model the
/// app already cached, never downloads one, never opens the app, and logs
/// nothing about its input.
public struct PIISummaryIntent: AppIntent {
    public static var title: LocalizedStringResource { "Summarize PII with OpenMed" }

    public static var description: IntentDescription? {
        IntentDescription(
            "Counts the personal health information in text on this device and returns the counts by label with the redacted text. Detected values never leave the intent."
        )
    }

    public static var openAppWhenRun: Bool { false }

    /// The text to summarize.
    @Parameter(title: "Text", inputOptions: String.IntentInputOptions(multiline: true))
    public var text: String

    /// A bundled OpenMed policy name, such as `hipaa_safe_harbor`.
    @Parameter(title: "Policy", default: "hipaa_safe_harbor")
    public var policyName: String

    public static var parameterSummary: some ParameterSummary {
        Summary("Summarize PII in \(\.$text) with \(\.$policyName)")
    }

    public init() {}

    /// Create an intent for programmatic use.
    public init(text: String, policyName: String = Policy.defaultName) {
        self.text = text
        self.policyName = policyName
    }

    public func perform() async throws -> some IntentResult & ReturnsValue<OpenMedPIISummary> {
        let result = try await OpenMedIntentsConfiguration.redactor.deidentify(text, policyName: policyName)
        return .result(value: OpenMedPIISummary(result))
    }
}
