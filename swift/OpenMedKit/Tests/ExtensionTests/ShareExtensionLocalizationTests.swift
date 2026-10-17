import Foundation
import OpenMedKit
import XCTest

@testable import OpenMedShareExtension

final class ShareExtensionLocalizationTests: XCTestCase {
    func testEveryInterfaceKeyResolvesFromTheResourceBundle() {
        for key in ShareExtensionStrings.Key.allCases {
            let value = ShareExtensionStrings.text(key)
            XCTAssertFalse(value.isEmpty, key.rawValue)
            XCTAssertNotEqual(value, key.rawValue, "Missing catalog entry: \(key.rawValue)")
        }
    }

    func testEveryBundledPolicyHasADisplayName() {
        for profile in Policy.bundledProfileNames {
            let value = ShareExtensionStrings.policyDisplayName(profile)
            XCTAssertFalse(value.isEmpty, profile)
            XCTAssertNotEqual(value, "policy.\(profile)", "Missing policy display name: \(profile)")
            XCTAssertFalse(value.contains("_"), profile)
        }
        XCTAssertEqual(
            ShareExtensionStrings.policyDisplayName("hipaa_safe_harbor"),
            "HIPAA Safe Harbor"
        )
    }

    func testMissingKeyDoesNotSilentlyResolveToAnEnglishFallback() {
        let missingKey = "share.synthetic_missing_key"
        XCTAssertEqual(ShareExtensionStrings.localized(missingKey), missingKey)
    }
}
