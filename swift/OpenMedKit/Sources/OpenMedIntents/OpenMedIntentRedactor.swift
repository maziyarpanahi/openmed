import AppIntents
import Foundation
import OpenMedKit
import os

/// Value-free App Intent failures shown to the person running a shortcut.
///
/// No case carries input text, a span, a path, a URL or a repository id.
public enum OpenMedIntentError: Error, Equatable, Sendable, CaseIterable, CustomLocalizedStringResourceConvertible,
    LocalizedError
{
    /// The shortcut passed empty text.
    case emptyInput
    /// The text exceeds ``OpenMedIntentsConfiguration/maximumInputUTF8Bytes``.
    case inputTooLarge
    /// The policy name is not a bundled OpenMed policy.
    case unknownPolicy
    /// No model is cached on the device. Intents never download one.
    case noCachedModel
    /// The cached model cannot run here, for example MLX on the iOS Simulator.
    case runtimeUnavailable
    /// Loading the model or redacting failed.
    case redactionFailed

    /// Stable reason code for logs and metrics.
    public var reason: String {
        switch self {
        case .emptyInput: return "empty_input"
        case .inputTooLarge: return "input_too_large"
        case .unknownPolicy: return "unknown_policy"
        case .noCachedModel: return "no_cached_model"
        case .runtimeUnavailable: return "runtime_unavailable"
        case .redactionFailed: return "redaction_failed"
        }
    }

    /// The localized message, from the package's `Localizable` catalog.
    public var localizedStringResource: LocalizedStringResource {
        switch self {
        case .emptyInput:
            return Self.resource("intent.error.empty_input", "There is no text to redact.")
        case .inputTooLarge:
            return Self.resource("intent.error.input_too_large", "The text is too long to redact in a shortcut.")
        case .unknownPolicy:
            return Self.resource(
                "intent.error.unknown_policy", "The policy is not one of the bundled OpenMed policies.")
        case .noCachedModel:
            return Self.resource(
                "intent.error.no_cached_model",
                "No OpenMed model is downloaded on this device. Open the app and download a model first; shortcuts never download models."
            )
        case .runtimeUnavailable:
            return Self.resource(
                "intent.error.runtime_unavailable", "The downloaded OpenMed model cannot run on this device.")
        case .redactionFailed:
            return Self.resource("intent.error.redaction_failed", "OpenMed could not redact the text on this device.")
        }
    }

    public var errorDescription: String? {
        String(localized: localizedStringResource)
    }

    private static func resource(_ key: StaticString, _ value: String.LocalizationValue) -> LocalizedStringResource {
        LocalizedStringResource(key, defaultValue: value, table: "Localizable", bundle: .atURL(Bundle.module.bundleURL))
    }
}

/// The on-device de-identification step behind OpenMed's App Intents.
///
/// Apps choose the model at launch through
/// ``OpenMedIntentsConfiguration/redactor``. The default uses
/// ``OpenMedIntentsConfiguration/defaultModelRepoID`` if it is already in the
/// ``OpenMedModelStore`` cache. No redactor downloads a model or sends text off
/// the device, and none logs its input.
public struct OpenMedIntentRedactor: Sendable {
    /// Run de-identification for one text with a resolved policy.
    public typealias Redact = @Sendable (_ text: String, _ policy: Policy) async throws -> PolicyDeidentificationResult

    private let body: Redact

    /// Wrap a custom local pipeline, for example one that loads a bundled
    /// Core ML model or a test double.
    public init(_ redact: @escaping Redact) {
        body = redact
    }

    /// Validate the input and policy, then de-identify on device.
    ///
    /// Every failure is mapped to ``OpenMedIntentError`` so Shortcuts never
    /// shows an underlying error that could carry a path or text.
    public func deidentify(_ text: String, policyName: String) async throws -> PolicyDeidentificationResult {
        guard !text.isEmpty else { throw OpenMedIntentError.emptyInput }
        guard text.utf8.count <= OpenMedIntentsConfiguration.maximumInputUTF8Bytes else {
            throw OpenMedIntentError.inputTooLarge
        }
        let policy: Policy
        do {
            policy = try Policy(named: policyName)
        } catch {
            throw OpenMedIntentError.unknownPolicy
        }
        do {
            return try await body(text, policy)
        } catch let error as OpenMedIntentError {
            throw error
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            throw OpenMedIntentError.redactionFailed
        }
    }

    /// Redact with an MLX model that is already in the ``OpenMedModelStore``
    /// cache. Throws ``OpenMedIntentError/noCachedModel`` without any network
    /// request when the snapshot is not ready.
    public static func cachedMLXModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) -> OpenMedIntentRedactor {
        OpenMedIntentRedactor { text, policy in
            let isCached = try? OpenMedModelStore.isMLXModelCached(
                repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
            guard isCached == true else { throw OpenMedIntentError.noCachedModel }
            #if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
                let directory = try OpenMedModelStore.cachedMLXModelDirectory(
                    repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
                let runtime: OpenMed
                do {
                    runtime = try await OpenMed.load(backend: .mlx(modelDirectoryURL: directory), allowNetworkAccess: false)
                } catch OpenMedMLXRuntimeError.unsupportedPlatform {
                    throw OpenMedIntentError.runtimeUnavailable
                }
                defer { OpenMed.clearRuntimeMemoryCache() }
                return try runtime.deidentify(text, policy: policy)
            #else
                throw OpenMedIntentError.runtimeUnavailable
            #endif
        }
    }
}

/// Process-wide settings for OpenMed's App Intents.
public enum OpenMedIntentsConfiguration {
    /// The MLX model the default redactor looks for in the cache.
    public static let defaultModelRepoID = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx"
    /// Largest input, in UTF-8 bytes, an intent accepts.
    public static let maximumInputUTF8Bytes = 64 * 1_024

    private static let storage = OSAllocatedUnfairLock(
        initialState: OpenMedIntentRedactor.cachedMLXModel(repoID: defaultModelRepoID))

    /// The redactor every OpenMed intent uses. Set it once at app launch,
    /// before the system runs an intent.
    public static var redactor: OpenMedIntentRedactor {
        get { storage.withLock { $0 } }
        set { storage.withLock { $0 = newValue } }
    }
}
