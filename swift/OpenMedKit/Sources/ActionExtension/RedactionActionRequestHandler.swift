#if canImport(UIKit)
    import OpenMedExtensionSupport
    import OpenMedKit
    import UIKit

    /// Non-UI Action extension handler that returns locally redacted text to its host.
    open class RedactionActionRequestHandler: NSObject, NSExtensionRequestHandling {
        public static let policyProfileUserInfoKey = "OpenMedPolicyProfile"

        public final func beginRequest(with context: NSExtensionContext) {
            Task {
                do {
                    let texts = try await ExtensionItemCodec.plainText(from: context.inputItems)
                    let policyName = Self.policyName(from: context.inputItems)
                    let configuration = try NanoModelConfiguration.bundled()
                    let redactedTexts = try await Task.detached(priority: .userInitiated) {
                        let results: [String]
                        do {
                            let handler = try ExtensionRedactionHandler(configuration: configuration)
                            results = try texts.map {
                                try handler.redact($0, policyName: policyName).redactedText
                            }
                        }
                        OpenMed.clearRuntimeMemoryCache()
                        return results
                    }.value
                    context.completeRequest(
                        returningItems: ExtensionItemCodec.extensionItems(for: redactedTexts)
                    )
                } catch {
                    context.cancelRequest(withError: error)
                }
            }
        }

        private static func policyName(from inputItems: [Any]) -> String {
            for item in inputItems.compactMap({ $0 as? NSExtensionItem }) {
                if let value = item.userInfo?[policyProfileUserInfoKey] as? String {
                    return value
                }
            }
            return Policy.defaultName
        }
    }
#endif
