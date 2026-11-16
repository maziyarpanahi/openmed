import AppIntents
import Foundation
import OpenMedKit

/// "Redact Text with OpenMed": replaces protected health information in text
/// on the device and returns only the redacted text.
///
/// The intent runs entirely on device with the model the app already cached
/// (see ``OpenMedIntentsConfiguration/redactor``). It never downloads a model,
/// has no cloud fallback, never opens the app, and logs nothing about its
/// input. The result is the redacted string alone: no spans, labels, offsets
/// or original values leave the intent.
public struct RedactTextIntent: AppIntent {
    public static var title: LocalizedStringResource { "Redact Text with OpenMed" }

    public static var description: IntentDescription? {
        IntentDescription(
            "Replaces personal health information in text on this device and returns only the redacted text. Uses a model the app already downloaded; it never downloads one or sends text off the device."
        )
    }

    public static var openAppWhenRun: Bool { false }

    /// The text to redact.
    @Parameter(title: "Text", inputOptions: String.IntentInputOptions(multiline: true))
    public var text: String

    /// A bundled OpenMed policy name, such as `hipaa_safe_harbor`.
    @Parameter(title: "Policy", default: "hipaa_safe_harbor")
    public var policyName: String

    public static var parameterSummary: some ParameterSummary {
        Summary("Redact \(\.$text) with \(\.$policyName)")
    }

    public init() {}

    /// Create an intent for programmatic use.
    public init(text: String, policyName: String = Policy.defaultName) {
        self.text = text
        self.policyName = policyName
    }

    public func perform() async throws -> some IntentResult & ReturnsValue<String> {
        let result = try await OpenMedIntentsConfiguration.redactor.deidentify(text, policyName: policyName)
        return .result(value: result.redactedText)
    }
}

/// Exposes OpenMed's App Intents to an app or extension that links
/// `OpenMedIntents`. List it in your own package's `includedPackages`.
public struct OpenMedIntentsPackage: AppIntentsPackage {}
