import Foundation
import OpenMedKit
import XCTest

final class OpenMedCrisisRoutingTests: XCTestCase {
    struct Vectors: Decodable {
        struct Case: Decodable {
            let locale: String
            let matched: String?
            let ids: [String]
        }
        enum CodingKeys: String, CodingKey {
            case genericNotice = "generic_notice"
            case cases
        }
        let genericNotice: String
        let cases: [Case]
    }

    func testSharedOfflineVectorsAndUserInitiatedLinks() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let vectors = try JSONDecoder().decode(Vectors.self, from: Data(contentsOf: root.appending(path: "tests/fixtures/parity/crisis_routing_v1.json")))
        let router = try OpenMedCrisisRouting()
        for row in vectors.cases {
            let notice = try XCTUnwrap(router.route(capability: "mental_health", locale: row.locale, signal: true))
            XCTAssertEqual(notice.text, vectors.genericNotice)
            XCTAssertEqual(notice.matchedLocale, row.matched)
            XCTAssertEqual(notice.resources.map(\.id), row.ids)
            _ = OpenMedCrisisBanner(notice: notice)
        }
    }

    func testRefusedSuppressionAndAbsentSignal() throws {
        let router = try OpenMedCrisisRouting()
        for code in ["mental_health", "unknown"] {
            XCTAssertThrowsError(try router.route(capability: code, locale: "en-GB", signal: true, suppressNotice: true)) {
                XCTAssertEqual($0 as? OpenMedCrisisRoutingError, .noticeRequired)
            }
        }
        XCTAssertNil(try router.route(capability: "deidentify", locale: "en", signal: true, suppressNotice: true))
        XCTAssertNil(try router.route(capability: "mental_health", locale: "en-GB", signal: false))
    }

    func testOverrideReplacesOneLocaleAndRejectsUnverifiedLinks() throws {
        let row = #"{"id":"local","label":"Site resource","url":"https://example.org/info","source_url":"https://example.org/provenance","verified_on":"2025-04-02"}"#
        let table = #"{"schema_version":1,"locales":{"en-GB":[\#(row)],"fr":[\#(row)]}}"#
        let router = try OpenMedCrisisRouting(overrideData: Data(table.utf8))
        func ids(_ locale: String) throws -> [String]? {
            try router.route(capability: "mental_health", locale: locale, signal: true)?.resources.map(\.id)
        }
        XCTAssertEqual(try ids("en-GB"), ["local"])
        XCTAssertEqual(try ids("fr-CA"), ["local"])
        XCTAssertEqual(try ids("en-AU"), ["healthdirect_emergency_departments"])
        for invalid in [table.replacingOccurrences(of: "https:", with: "tel:"), table.replacingOccurrences(of: "2025-04-02", with: "2025-02-30")] {
            XCTAssertThrowsError(try OpenMedCrisisRouting(overrideData: Data(invalid.utf8)))
        }
    }
}
