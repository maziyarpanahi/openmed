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

    func testPolicyOptionsPromptIncludesTheAvailableCountInEveryLanguage() {
        for language in languages {
            for count in [1, Policy.bundledProfileNames.count, 12] {
                let prompt = ShareExtensionStrings.policyOptionsText(count: count, language: language)
                XCTAssertTrue(prompt.contains(String(count)))
                XCTAssertFalse(prompt.contains("%ld"))
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
            let selector = try XCTUnwrap(
                views.compactMap { $0 as? UIButton }.first {
                    $0.accessibilityLabel == ShareExtensionStrings.text(.policyLabel)
                })
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
            let policyFontSize = try XCTUnwrap(selector.titleLabel?.font).pointSize
            let actions = try XCTUnwrap(selector.menu).children.compactMap { $0 as? UIAction }
            selector.sendAction(actions[3])

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
            XCTAssertEqual(selector.accessibilityValue, ShareExtensionStrings.policyDisplayName(Policy.bundledProfileNames[3]))
            XCTAssertEqual(selector.configuration?.title, selector.accessibilityValue)
            XCTAssertGreaterThan(try XCTUnwrap(selector.titleLabel?.font).pointSize, policyFontSize)
            XCTAssertEqual(selector.frame.minY, text.frame.maxY, accuracy: 0.5)
            XCTAssertGreaterThanOrEqual(selector.bounds.height, 52)
            XCTAssertTrue(text.isScrollEnabled)
            XCTAssertGreaterThanOrEqual(text.bounds.height, 160)
            let previewHeight = text.bounds.height
            text.text = Array(repeating: "Synthetic shared text for scrolling.", count: 80).joined(separator: "\n")
            text.layoutIfNeeded()
            XCTAssertGreaterThan(text.contentSize.height, text.bounds.height)
            text.setContentOffset(CGPoint(x: 0, y: text.contentSize.height - text.bounds.height), animated: false)
            XCTAssertGreaterThan(text.contentOffset.y, 0)
            XCTAssertEqual(text.bounds.height, previewHeight, accuracy: 0.5)
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
        func testActionsHaveDistinctStylesAndWrapInEveryLanguage() throws {
            for appearance in [UIUserInterfaceStyle.light, .dark] {
                for size in [UIContentSizeCategory.large, .accessibilityExtraExtraExtraLarge] {
                    let controller = RedactionShareViewController()
                    controller.traitOverrides.userInterfaceStyle = appearance
                    controller.traitOverrides.preferredContentSizeCategory = size
                    let window = UIWindow(frame: CGRect(x: 0, y: 0, width: 320, height: 568))
                    window.rootViewController = controller
                    window.makeKeyAndVisible()
                    defer { window.isHidden = true }
                    let buttons = descendants(of: controller.view).compactMap { $0 as? UIButton }
                    let primary = try XCTUnwrap(buttons.first { $0.accessibilityHint == ShareExtensionStrings.text(.redactHint) })
                    let cancel = try XCTUnwrap(buttons.first { $0.accessibilityHint == ShareExtensionStrings.text(.cancelHint) })
                    primary.isEnabled = true
                    let row = try XCTUnwrap(primary.superview as? UIStackView)
                    XCTAssertEqual(row.axis, .vertical)
                    XCTAssertTrue(row.arrangedSubviews.first === primary)
                    XCTAssertTrue(row.arrangedSubviews.last === cancel)
                    let primaryStyle = try XCTUnwrap(primary.configuration)
                    let cancelStyle = try XCTUnwrap(cancel.configuration)
                    let fill = try XCTUnwrap(primaryStyle.baseBackgroundColor).resolvedColor(with: controller.traitCollection)
                    let foreground = try XCTUnwrap(primaryStyle.baseForegroundColor).resolvedColor(with: controller.traitCollection)
                    let cancelFill = try XCTUnwrap(cancelStyle.background.backgroundColor).resolvedColor(with: controller.traitCollection)
                    XCTAssertFalse(fill.isEqual(foreground))
                    XCTAssertFalse(fill.isEqual(cancelFill))
                    XCTAssertGreaterThan(cancelStyle.background.strokeWidth, 0)
                    for language in languages {
                        for (button, key) in [(primary, ShareExtensionStrings.Key.redact), (cancel, .cancel)] {
                            var configuration = try XCTUnwrap(button.configuration)
                            configuration.title = ShareExtensionStrings.text(key, language: language)
                            button.configuration = configuration
                        }
                        controller.view.setNeedsLayout()
                        controller.view.layoutIfNeeded()
                        XCTAssertGreaterThanOrEqual(cancel.frame.minY - primary.frame.maxY, 12 - 0.5)
                        for button in [primary, cancel] {
                            XCTAssertEqual(button.bounds.width, row.bounds.width, accuracy: 0.5)
                            XCTAssertGreaterThanOrEqual(button.bounds.height, 52)
                            let label = try XCTUnwrap(button.titleLabel)
                            XCTAssertGreaterThan(label.bounds.width, 0)
                            let required = label.sizeThatFits(CGSize(width: label.bounds.width, height: .greatestFiniteMagnitude))
                            XCTAssertLessThanOrEqual(required.height, label.bounds.height + 0.5)
                            XCTAssertGreaterThanOrEqual(button.bounds.height, label.bounds.height + 28 - 0.5)
                            XCTAssertTrue(button.accessibilityTraits.contains(.button))
                        }
                        if appearance == .light, language == "en" {
                            let scrollView = try XCTUnwrap(controller.view.subviews.compactMap { $0 as? UIScrollView }.first)
                            scrollView.scrollRectToVisible(row.convert(row.bounds, to: scrollView), animated: false)
                            let renderer = UIGraphicsImageRenderer(bounds: controller.view.bounds)
                            let image = renderer.image { context in
                                controller.view.layer.render(in: context.cgContext)
                            }
                            let attachment = XCTAttachment(image: image)
                            attachment.name = size == .large ? "actions-default" : "actions-largest"
                            attachment.lifetime = .keepAlways
                            add(attachment)
                        }
                    }
                }
            }
        }

        @MainActor
        func testPolicyMenuExposesEveryOptionAndUpdatesItsCheckmark() throws {
            let controller = RedactionShareViewController()
            controller.loadViewIfNeeded()
            let selector = try XCTUnwrap(
                descendants(of: controller.view).compactMap { $0 as? UIButton }.first {
                    $0.accessibilityLabel == ShareExtensionStrings.text(.policyLabel)
                })
            XCTAssertTrue(selector.showsMenuAsPrimaryAction)
            XCTAssertNotNil(selector.configuration?.image)
            XCTAssertEqual(selector.configuration?.subtitle, ShareExtensionStrings.policyOptionsText(count: Policy.bundledProfileNames.count))
            XCTAssertEqual(selector.accessibilityValue, ShareExtensionStrings.policyDisplayName(Policy.defaultName))
            for profile in Policy.bundledProfileNames {
                let actions = try XCTUnwrap(selector.menu).children.compactMap { $0 as? UIAction }
                XCTAssertEqual(actions.count, Policy.bundledProfileNames.count)
                XCTAssertEqual(actions.map(\.identifier.rawValue), Policy.bundledProfileNames)
                let action = try XCTUnwrap(actions.first { $0.identifier.rawValue == profile })
                selector.sendAction(action)
                XCTAssertEqual(selector.configuration?.title, ShareExtensionStrings.policyDisplayName(profile))
                XCTAssertEqual(selector.accessibilityValue, ShareExtensionStrings.policyDisplayName(profile))
                let checked = try XCTUnwrap(selector.menu).children.compactMap { $0 as? UIAction }.filter { $0.state == .on }
                XCTAssertEqual(checked.map(\.identifier.rawValue), [profile])
            }
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
            let selector = try XCTUnwrap(
                views.compactMap { $0 as? UIButton }.first {
                    $0.accessibilityLabel == ShareExtensionStrings.text(.policyLabel)
                })
            XCTAssertEqual(selector.accessibilityLabel, ShareExtensionStrings.text(.policyLabel))
            XCTAssertEqual(selector.accessibilityHint, ShareExtensionStrings.text(.policyHint))
            for view in views {
                XCTAssertFalse(view.accessibilityHint?.contains(privateText) ?? false)
            }
            let label = try XCTUnwrap(controller.pickerView(UIPickerView(), viewForRow: 0, forComponent: 0, reusing: nil) as? UILabel)
            XCTAssertTrue(label.adjustsFontForContentSizeCategory)
            XCTAssertEqual(label.accessibilityLabel, ShareExtensionStrings.policyDisplayName(Policy.defaultName))
        }

        @MainActor
        private func descendants(of view: UIView) -> [UIView] {
            [view] + view.subviews.flatMap { descendants(of: $0) }
        }
    }
#endif
