import Foundation
import OpenMedKit
import XCTest

final class OpenMedIntendedUseTests: XCTestCase {
    func testEveryDeclarationLoadsFromTheBundle() throws {
        let declarations = try OpenMedIntendedUse.declarations()
        XCTAssertFalse(declarations.isEmpty)
        for (code, declaration) in declarations {
            XCTAssertEqual(try OpenMedIntendedUse.lookup(code), declaration)
            XCTAssertEqual(declaration.schemaVersion, "1.0.0")
            XCTAssertFalse(declaration.autonomousClinicalAction)
            XCTAssertFalse(declaration.excludedUses.isEmpty)
            XCTAssertFalse(declaration.intendedUsers.isEmpty)
        }
        XCTAssertEqual(try OpenMedIntendedUse.lookup("deidentify")?.regulatoryTier, .nonClinical)
        XCTAssertEqual(try OpenMedIntendedUse.lookup("ecg")?.regulatoryTier, .restricted)
        XCTAssertNil(try OpenMedIntendedUse.lookup("not_in_registry"))
    }

    func testUnknownTierRequiresAllRestrictedObligations() throws {
        let decoder = JSONDecoder()
        let unknown = try decoder.decode(OpenMedRegulatoryTier.self, from: Data("\"future-tropical\"".utf8))
        XCTAssertEqual(unknown, .unknown)
        XCTAssertEqual(unknown.obligations, OpenMedRegulatoryTier.restricted.obligations)
        XCTAssertEqual(unknown.obligations.count, 4)
    }
}
