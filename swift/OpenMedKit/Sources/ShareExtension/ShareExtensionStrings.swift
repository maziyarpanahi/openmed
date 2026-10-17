import Foundation
import OpenMedExtensionSupport

/// Localized presentation text owned by the share extension's resource bundle.
enum ShareExtensionStrings {
    enum Key: String, CaseIterable {
        case loading = "share.loading"
        case ready = "share.ready"
        case redacting = "share.redacting"
        case redact = "share.redact"
        case cancel = "share.cancel"
        case textLabel = "accessibility.text.label"
        case textHint = "accessibility.text.hint"
        case policyLabel = "accessibility.policy.label"
        case policyHint = "accessibility.policy.hint"
        case statusLabel = "accessibility.status.label"
        case redactHint = "accessibility.redact.hint"
        case cancelHint = "accessibility.cancel.hint"
        case errorNoText = "error.no_text"
        case errorInputTooLarge = "error.input_too_large"
        case errorMissingModel = "error.missing_model"
        case errorInvalidModel = "error.invalid_model"
        case errorModelTooLarge = "error.model_too_large"
        case errorInvalidOutput = "error.invalid_output"
        case errorRedactionFailed = "error.redaction_failed"
    }

    static let resourceBundle = Bundle.module

    static func text(_ key: Key, language: String? = nil) -> String {
        localized(key.rawValue, language: language)
    }

    static func policyDisplayName(_ profileName: String, language: String? = nil) -> String {
        localized("policy.\(profileName)", language: language)
    }

    static func localized(_ key: String, language: String? = nil) -> String {
        if let language,
            let url = resourceBundle.url(forResource: language, withExtension: "lproj"),
            let bundle = Bundle(url: url)
        {
            return bundle.localizedString(forKey: key, value: nil, table: "Localizable")
        }
        if language == nil {
            let value = resourceBundle.localizedString(forKey: key, value: nil, table: "Localizable")
            if value != key { return value }
        }

        // Older SwiftPM builds copy catalogs instead of compiling .strings resources.
        // Read that same bundled catalog; never duplicate translations in Swift code.
        let preferredLanguage =
            language
            ?? Bundle.preferredLocalizations(from: catalogLanguages).first
            ?? "en"
        return catalog?.strings[key]?.localizations[preferredLanguage]?.stringUnit.value ?? key
    }

    /// Present controlled messages without exposing associated paths or source text.
    static func errorMessage(_ error: Error, language: String? = nil) -> String {
        let key: Key
        switch error as? ExtensionRedactionError {
        case .emptyInput, .missingPlainTextInput:
            key = .errorNoText
        case .inputTooLarge, .inputBytesTooLarge, .tooManyInputItems, .aggregateInputTooLarge:
            key = .errorInputTooLarge
        case .missingAsset:
            key = .errorMissingModel
        case .nonLocalAsset, .invalidAsset, .unsupportedModelFormat:
            key = .errorInvalidModel
        case .tooManyAssetFiles, .modelAssetsTooLarge:
            key = .errorModelTooLarge
        case .invalidRedactionOutput:
            key = .errorInvalidOutput
        case nil:
            key = .errorRedactionFailed
        }
        return text(key, language: language)
    }

    private struct Catalog: Decodable {
        let strings: [String: Entry]

        struct Entry: Decodable {
            let localizations: [String: Translation]
        }

        struct Translation: Decodable {
            let stringUnit: Unit
        }

        struct Unit: Decodable {
            let value: String
        }
    }

    private static let catalog: Catalog? = {
        guard let url = resourceBundle.url(forResource: "Localizable", withExtension: "xcstrings"),
            let data = try? Data(contentsOf: url)
        else { return nil }
        return try? JSONDecoder().decode(Catalog.self, from: data)
    }()

    private static var catalogLanguages: [String] {
        Array(Set(catalog?.strings.values.flatMap { $0.localizations.keys } ?? [])).sorted()
    }
}
