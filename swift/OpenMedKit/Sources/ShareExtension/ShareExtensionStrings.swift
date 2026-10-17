import Foundation

/// Localized presentation text owned by the share extension's resource bundle.
enum ShareExtensionStrings {
    enum Key: String, CaseIterable {
        case loading = "share.loading"
        case ready = "share.ready"
        case redacting = "share.redacting"
        case redact = "share.redact"
        case cancel = "share.cancel"
    }

    static func text(_ key: Key) -> String {
        localized(key.rawValue)
    }

    static func policyDisplayName(_ profileName: String) -> String {
        localized("policy.\(profileName)")
    }

    static func localized(_ key: String) -> String {
        Bundle.module.localizedString(forKey: key, value: nil, table: "Localizable")
    }
}
