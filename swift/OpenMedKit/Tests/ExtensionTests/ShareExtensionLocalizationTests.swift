import Foundation
import OpenMedExtensionSupport
import OpenMedKit
import XCTest

@testable import OpenMedShareExtension

#if os(iOS) && canImport(UIKit)
    import UIKit
#endif

final class ShareExtensionLocalizationTests: XCTestCase {
    private let languages = ["en", "zh", "hi"]

    func testEveryInterfaceKeyResolvesFromTheResourceBundle() {
        for key in ShareExtensionStrings.Key.allCases {
            let value = ShareExtensionStrings.text(key)
            XCTAssertFalse(value.isEmpty, key.rawValue)
            XCTAssertNotEqual(value, key.rawValue, "Missing catalog entry: \(key.rawValue)")
        }
    }

    func testEveryBundledPolicyHasADisplayName() throws {
        for profile in Policy.bundledProfileNames {
            let value = ShareExtensionStrings.policyDisplayName(profile)
            XCTAssertFalse(value.isEmpty, profile)
            XCTAssertNotEqual(value, "policy.\(profile)", "Missing policy display name: \(profile)")
            XCTAssertFalse(value.contains("_"), profile)
        }
        XCTAssertEqual(
            ShareExtensionStrings.policyDisplayName("hipaa_safe_harbor", language: "en"),
            "HIPAA Safe Harbor"
        )
    }

    func testMissingKeyDoesNotSilentlyResolveToAnEnglishFallback() {
        let missingKey = "share.synthetic_missing_key"
        XCTAssertEqual(ShareExtensionStrings.localized(missingKey), missingKey)
    }
    func testEveryLocaleResolvesEveryKeyAndPolicy() throws {
        for language in languages {
            for key in ShareExtensionStrings.Key.allCases {
                let value = ShareExtensionStrings.text(key, language: language)
                XCTAssertFalse(value.isEmpty, "\(language):\(key.rawValue)")
                XCTAssertNotEqual(value, key.rawValue, "\(language):\(key.rawValue)")
            }
            for profile in Policy.bundledProfileNames {
                let value = ShareExtensionStrings.policyDisplayName(profile, language: language)
                XCTAssertFalse(value.isEmpty, "\(language):\(profile)")
                XCTAssertNotEqual(value, "policy.\(profile)")
                XCTAssertFalse(value.contains("_"))
            }
        }
        XCTAssertEqual(ShareExtensionStrings.text(.cancel, language: "zh"), "取消")
        XCTAssertEqual(ShareExtensionStrings.text(.cancel, language: "hi"), "रद्द करें")
    }

    func testCatalogContainsExplicitTranslationsWithoutFallback() throws {
        let catalog = try readCatalog()
        XCTAssertEqual(catalog.sourceLanguage, "en")
        XCTAssertEqual(Set(catalog.strings.keys), Set(requiredKeys))
        XCTAssertEqual(missingEntries(in: catalog), [])
    }

    func testCoverageDetectsMissingKeysAndTranslations() throws {
        var catalog = try readCatalog()
        catalog.strings.removeValue(forKey: "share.cancel")
        catalog.strings["share.ready"]?.localizations.removeValue(forKey: "zh")
        let missing = missingEntries(in: catalog)
        for language in languages {
            XCTAssertTrue(missing.contains("\(language):share.cancel"))
        }
        XCTAssertTrue(missing.contains("zh:share.ready"))
    }

    func testErrorsUseControlledLocalizedMessagesWithoutPrivatePayloads() throws {
        let privateText = "SYNTHETIC_PRIVATE_PAYLOAD"
        let cases: [(Error, ShareExtensionStrings.Key)] = [
            (ExtensionRedactionError.emptyInput, .errorNoText),
            (ExtensionRedactionError.missingPlainTextInput, .errorNoText),
            (ExtensionRedactionError.inputTooLarge(actual: 2, limit: 1), .errorInputTooLarge),
            (ExtensionRedactionError.inputBytesTooLarge(actual: 2, limit: 1), .errorInputTooLarge),
            (ExtensionRedactionError.tooManyInputItems(actual: 2, limit: 1), .errorInputTooLarge),
            (ExtensionRedactionError.aggregateInputTooLarge(actual: 2, limit: 1), .errorInputTooLarge),
            (ExtensionRedactionError.missingAsset(privateText), .errorMissingModel),
            (ExtensionRedactionError.invalidAsset(privateText), .errorInvalidModel),
            (ExtensionRedactionError.unsupportedModelFormat(privateText), .errorInvalidModel),
            (ExtensionRedactionError.nonLocalAsset(URL(string: "https://example.invalid/\(privateText)")!), .errorInvalidModel),
            (ExtensionRedactionError.tooManyAssetFiles(limit: 1), .errorModelTooLarge),
            (ExtensionRedactionError.modelAssetsTooLarge(actual: 2, limit: 1), .errorModelTooLarge),
            (ExtensionRedactionError.invalidRedactionOutput, .errorInvalidOutput),
            (NSError(domain: "synthetic", code: 1, userInfo: [NSLocalizedDescriptionKey: privateText]), .errorRedactionFailed),
        ]
        for language in languages {
            for (error, key) in cases {
                let message = ShareExtensionStrings.errorMessage(error, language: language)
                XCTAssertEqual(message, ShareExtensionStrings.text(key, language: language))
                XCTAssertFalse(message.contains(privateText))
            }
        }
    }

    private var requiredKeys: [String] {
        ShareExtensionStrings.Key.allCases.map(\.rawValue)
            + Policy.bundledProfileNames.map { "policy.\($0)" }
    }

    private struct Catalog: Decodable {
        var sourceLanguage: String
        var strings: [String: Entry]

        struct Entry: Decodable {
            var localizations: [String: Translation]
        }

        struct Translation: Decodable {
            var stringUnit: Unit
        }

        struct Unit: Decodable {
            var state: String
            var value: String
        }
    }

    private func readCatalog() throws -> Catalog {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let url = root.appending(path: "Sources/ShareExtension/Resources/Localizable.xcstrings")
        return try JSONDecoder().decode(Catalog.self, from: Data(contentsOf: url))
    }

    private func missingEntries(in catalog: Catalog) -> [String] {
        languages.flatMap { language in
            requiredKeys.compactMap { key in
                guard let unit = catalog.strings[key]?.localizations[language]?.stringUnit,
                    unit.state == "translated", !unit.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else { return "\(language):\(key)" }
                return nil
            }
        }
    }

}

#if os(iOS) && canImport(UIKit)
    extension ShareExtensionLocalizationTests {
        @MainActor
        func testLiveContentSizeChangeScalesControlsAndPreservesPolicy() throws {
            let controller = RedactionShareViewController()
            controller.traitOverrides.preferredContentSizeCategory = .large
            let window = UIWindow(frame: CGRect(x: 0, y: 0, width: 320, height: 568))
            window.rootViewController = controller
            window.makeKeyAndVisible()
            defer { window.isHidden = true }
            controller.view.layoutIfNeeded()

            let views = descendants(of: controller.view)
            let text = try XCTUnwrap(views.compactMap { $0 as? UITextView }.first)
            let picker = try XCTUnwrap(views.compactMap { $0 as? UIPickerView }.first)
            let status = try XCTUnwrap(
                views.compactMap { $0 as? UILabel }.first {
                    $0.accessibilityLabel == ShareExtensionStrings.text(.statusLabel)
                })
            let buttons = views.compactMap { $0 as? UIButton }.filter {
                $0.accessibilityHint == ShareExtensionStrings.text(.redactHint)
                    || $0.accessibilityHint == ShareExtensionStrings.text(.cancelHint)
            }
            XCTAssertEqual(buttons.count, 2)
            let bodySize = try XCTUnwrap(text.font).pointSize
            let statusSize = try XCTUnwrap(status.font).pointSize
            let buttonSizes = buttons.map { $0.titleLabel?.font.pointSize ?? 0 }
            let rowHeight = controller.pickerView(picker, rowHeightForComponent: 0)
            picker.selectRow(3, inComponent: 0, animated: false)
            controller.pickerView(picker, didSelectRow: 3, inComponent: 0)

            controller.traitOverrides.preferredContentSizeCategory = .accessibilityExtraExtraExtraLarge
            controller.view.setNeedsLayout()
            controller.view.layoutIfNeeded()

            XCTAssertGreaterThan(try XCTUnwrap(text.font).pointSize, bodySize)
            XCTAssertGreaterThan(try XCTUnwrap(status.font).pointSize, statusSize)
            for (button, oldSize) in zip(buttons, buttonSizes) {
                XCTAssertTrue(try XCTUnwrap(button.titleLabel).adjustsFontForContentSizeCategory)
                XCTAssertGreaterThan(try XCTUnwrap(button.titleLabel?.font).pointSize, oldSize)
                XCTAssertGreaterThanOrEqual(button.bounds.height, 44)
                XCTAssertEqual(button.accessibilityLabel, button.title(for: .normal))
            }
            XCTAssertTrue(text.adjustsFontForContentSizeCategory)
            XCTAssertTrue(status.adjustsFontForContentSizeCategory)
            XCTAssertEqual(picker.selectedRow(inComponent: 0), 3)
            XCTAssertEqual(picker.accessibilityValue, ShareExtensionStrings.policyDisplayName(Policy.bundledProfileNames[3]))
            XCTAssertGreaterThan(controller.pickerView(picker, rowHeightForComponent: 0), rowHeight)
            XCTAssertEqual((buttons.first?.superview as? UIStackView)?.axis, .vertical)
            let scrollView = try XCTUnwrap(controller.view.subviews.compactMap { $0 as? UIScrollView }.first)
            XCTAssertGreaterThan(scrollView.contentSize.height, scrollView.bounds.height)
            for button in buttons {
                scrollView.scrollRectToVisible(button.convert(button.bounds, to: scrollView), animated: false)
                // Layout uses fractional points; allow one physical pixel at the viewport edge.
                let pixel = 1 / controller.traitCollection.displayScale
                XCTAssertTrue(
                    scrollView.bounds.insetBy(dx: -pixel, dy: -pixel).contains(button.convert(button.bounds, to: scrollView)),
                    "Viewport: \(scrollView.bounds); button: \(button.convert(button.bounds, to: scrollView)); content: \(scrollView.contentSize)"
                )
            }
            XCTAssertFalse(controller.view.hasAmbiguousLayout)
        }

        @MainActor
        func testAccessibilityHintsDoNotIncludeSelectedText() throws {
            let controller = RedactionShareViewController()
            controller.loadViewIfNeeded()
            let views = descendants(of: controller.view)
            let text = try XCTUnwrap(views.compactMap { $0 as? UITextView }.first)
            let privateText = "SYNTHETIC_SELECTED_TEXT_DO_NOT_ANNOUNCE_IN_HINTS"
            text.text = privateText
            XCTAssertEqual(text.accessibilityLabel, ShareExtensionStrings.text(.textLabel))
            let picker = try XCTUnwrap(views.compactMap { $0 as? UIPickerView }.first)
            XCTAssertEqual(picker.accessibilityLabel, ShareExtensionStrings.text(.policyLabel))
            XCTAssertEqual(picker.accessibilityHint, ShareExtensionStrings.text(.policyHint))
            for view in views {
                XCTAssertFalse(view.accessibilityHint?.contains(privateText) ?? false)
            }
            let label = try XCTUnwrap(controller.pickerView(picker, viewForRow: 0, forComponent: 0, reusing: nil) as? UILabel)
            XCTAssertTrue(label.adjustsFontForContentSizeCategory)
            XCTAssertEqual(label.accessibilityLabel, ShareExtensionStrings.policyDisplayName(Policy.defaultName))
        }

        @MainActor
        private func descendants(of view: UIView) -> [UIView] {
            [view] + view.subviews.flatMap { descendants(of: $0) }
        }
    }
#endif
