#if os(iOS) && canImport(UIKit) && canImport(UniformTypeIdentifiers)
    import Foundation
    import UIKit
    import UniformTypeIdentifiers

    /// Converts between iOS extension items and OpenMed's plain-text contract.
    public enum ExtensionItemCodec {
        /// Load every plain-text attachment supplied by the host app.
        public static func plainText(from inputItems: [Any]) async throws -> [String] {
            var providers: [NSItemProvider] = []
            for item in inputItems.compactMap({ $0 as? NSExtensionItem }) {
                for provider in item.attachments ?? []
                where provider.hasItemConformingToTypeIdentifier(UTType.plainText.identifier) {
                    providers.append(provider)
                    guard providers.count <= ExtensionInputBudget.maximumItems else {
                        throw ExtensionRedactionError.tooManyInputItems(
                            actual: providers.count,
                            limit: ExtensionInputBudget.maximumItems
                        )
                    }
                }
            }

            var texts: [String] = []
            for provider in providers {
                let text = try await plainText(from: provider)
                texts.append(text)
                try ExtensionInputBudget.validate(texts)
            }

            try ExtensionInputBudget.validate(texts)
            return texts
        }

        /// Create host-returnable text items after local redaction.
        public static func extensionItems(for texts: [String]) -> [NSExtensionItem] {
            texts.map { text in
                let item = NSExtensionItem()
                item.attachments = [NSItemProvider(object: text as NSString)]
                return item
            }
        }

        private static func plainText(from provider: NSItemProvider) async throws -> String {
            try await withCheckedThrowingContinuation { continuation in
                provider.loadItem(
                    forTypeIdentifier: UTType.plainText.identifier,
                    options: nil
                ) { item, error in
                    if let error {
                        continuation.resume(throwing: error)
                        return
                    }
                    if let text = item as? String {
                        continuation.resume(returning: text)
                        return
                    }
                    if let text = item as? NSString {
                        continuation.resume(returning: text as String)
                        return
                    }
                    if let data = item as? Data {
                        guard data.count <= ExtensionRedactionHandler.maximumInputUTF8Bytes else {
                            continuation.resume(
                                throwing: ExtensionRedactionError.inputBytesTooLarge(
                                    actual: data.count,
                                    limit: ExtensionRedactionHandler.maximumInputUTF8Bytes
                                )
                            )
                            return
                        }
                        if let text = String(data: data, encoding: .utf8) {
                            continuation.resume(returning: text)
                            return
                        }
                    }
                    continuation.resume(throwing: ExtensionRedactionError.missingPlainTextInput)
                }
            }
        }
    }
#endif
