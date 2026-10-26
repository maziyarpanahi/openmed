import Foundation
import XCTest

@testable import OpenMedKit

final class OpenMedAECGTests: XCTestCase {
    private func fixture() throws -> Data {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return try Data(contentsOf: root.appendingPathComponent("tests/fixtures/multimodal/aecg.xml"))
    }

    private func mutated(_ old: String, _ new: String) throws -> Data {
        let text = String(decoding: try fixture(), as: UTF8.self)
        let range = try XCTUnwrap(text.range(of: old))
        var changed = text
        changed.replaceSubrange(range, with: new)
        return Data(changed.utf8)
    }

    private func rejected(_ data: Data, _ reason: String, limits: OpenMedAECGLimits = .init(), file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try OpenMedAECG.read(data, limits: limits), file: file, line: line) { error in
            XCTAssertEqual((error as? OpenMedAECGError)?.reasonCode, reason, file: file, line: line)
            XCTAssertEqual(String(describing: error), reason, file: file, line: line)
        }
    }

    func testSyntheticHandCheckedSignalsAndReviewGate() throws {
        let record = try OpenMedAECG.read(fixture())
        XCTAssertEqual(record.rhythm.count, 1)
        XCTAssertEqual(record.derived.count, 1)
        XCTAssertEqual(record.rhythm[0].kind, "rhythm")
        XCTAssertEqual(record.derived[0].kind, "derived_representative_beat")
        XCTAssertEqual(record.rhythm[0].leads[0].samples, [-4, 1, 11])
        XCTAssertEqual(record.rhythm[0].leads[1].samples, [-3, -2, 0])
        XCTAssertEqual(record.derived[0].leads[0].samples, [1, 6])
        XCTAssertEqual(record.derived[0].leads[1].samples, [-2, -1])
        XCTAssertEqual(record.rhythm[0].sampleIntervalSeconds, 0.002)
        XCTAssertEqual(record.demographicElementCount, 4)
        XCTAssertEqual(record.annotationCount, 2)
        let report = try JSONSerialization.data(withJSONObject: record.report(), options: [.sortedKeys])
        let publicText = String(decoding: report, as: UTF8.self) + String(describing: record)
        for sentinel in ["SYNTHETIC-ID-3839", "架空", "Patient 3839", "19000101", "20000101000000", "SYNTHETIC-NOTE-3839", "patient@example.invalid"] {
            XCTAssertFalse(publicText.contains(sentinel))
        }
        XCTAssertFalse(String(decoding: report, as: UTF8.self).contains("samples"))
        XCTAssertEqual(record.notice, OpenMedAECG.notice)
        XCTAssertThrowsError(try record.confirmReview(reviewerConfirmed: false)) { error in
            XCTAssertEqual((error as? OpenMedAECGError)?.reasonCode, "review_required")
        }
        XCTAssertEqual(try record.confirmReview(reviewerConfirmed: true).notice, record.notice)
        XCTAssertEqual(try OpenMedAECG.read(fixture()).sourceDigest, record.sourceDigest)
    }

    func testEntitiesAndExternalReferencesFailClosed() throws {
        for declaration in [
            "<!DOCTYPE AnnotatedECG [<!ENTITY phi 'SYNTHETIC-ID-3839'>]>",
            "<!DOCTYPE AnnotatedECG SYSTEM 'file:///private/SYNTHETIC-ID-3839'>",
            "<!DOCTYPE AnnotatedECG SYSTEM 'https://example.invalid/external.dtd'>",
            "<!DOCTYPE AnnotatedECG [<!ENTITY a 'ha'><!ENTITY b '&a;&a;'><!ENTITY c '&b;&b;'>]>",
        ] {
            rejected(try mutated("<AnnotatedECG ", declaration + "<AnnotatedECG "), "unsafe_xml")
        }
        rejected(try mutated("<sequenceSet>", "<sequenceSet><reference value='file:///private/SYNTHETIC-ID-3839'/>"), "external_reference")
        rejected(try mutated("<sequenceSet>", "<sequenceSet><xi:include xmlns:xi='http://www.w3.org/2001/XInclude' href='https://example.invalid'/>"), "external_reference")
    }

    func testEveryBoundAndInclusiveSampleLimit() throws {
        let data = try fixture()
        rejected(data, "byte_limit", limits: .init(maxBytes: 100))
        rejected(data, "element_limit", limits: .init(maxElements: 10))
        rejected(data, "depth_limit", limits: .init(maxDepth: 3))
        rejected(data, "series_limit", limits: .init(maxSeries: 1))
        rejected(data, "lead_limit", limits: .init(maxLeadsPerSeries: 1))
        rejected(data, "sample_limit", limits: .init(maxSamplesPerLead: 2))
        rejected(data, "sample_limit", limits: .init(maxTotalSamples: 9))
        rejected(data, "invalid_limits", limits: .init(maxElements: 0))
        let record = try OpenMedAECG.read(data, limits: .init(maxBytes: data.count, maxSeries: 2, maxLeadsPerSeries: 2, maxSamplesPerLead: 3, maxTotalSamples: 10))
        XCTAssertEqual(record.derived.count, 1)
    }

    func testMalformedUnsupportedAndInconsistentInputs() throws {
        let cases = [
            ("<digits>-2 0 4</digits>", "<digits>-2 0</digits>", "inconsistent_lengths"),
            ("<digits>-2 0 4</digits>", "<digits>NaN SYNTHETIC-ID-3839</digits>", "invalid_numeric"),
            ("<digits>-2 0 4</digits>", "<digits>2147483648 0 4</digits>", "invalid_numeric"),
            ("<digits>-2 0 4</digits>", "<digits><text>1</text></digits>", "unsupported_encoding"),
            ("xsi:type=\"SLIST_PQ\"", "xsi:type=\"B64\"", "unsupported_encoding"),
            ("<digits>-2 0 4</digits>", "<digits encoding='base64'>123</digits>", "unsupported_encoding"),
            ("xsi:type=\"SLIST_PQ\"", "xsi:type=\"SLIST_PQ\" nullFlavor='UNK'", "unsupported_encoding"),
            ("<sequenceSet>", "<sequenceSet><component><unknown/></component>", "unsupported_layout"),
            ("scale value=\"2.5\"", "scale value=\"1e308\"", "invalid_numeric"),
            ("scale value=\"2.5\"", "scale value=\"0\"", "invalid_numeric"),
            ("increment value=\"0.002\"", "increment value=\"-1\"", "invalid_numeric"),
            ("code=\"REPRESENTATIVE_BEAT\"", "code=\"RHYTHM\"", "unsupported_layout"),
            ("code=\"MDC_ECG_LEAD_II\"", "code=\"MDC_ECG_LEAD_I\"", "unsupported_layout"),
            ("code=\"MDC_ECG_LEAD_I\"", "code=\"SYNTHETIC-ID-3839\"", "unsupported_lead"),
            ("<sequenceSet>", "<sequenceSet><sequence/>", "unsupported_layout"),
            ("encoding=\"UTF-8\"", "encoding=\"iso-8859-1\"", "unsupported_encoding"),
        ]
        for (old, new, reason) in cases { rejected(try mutated(old, new), reason) }
        rejected(Data("<AnnotatedECG>".utf8), "malformed_xml")
        rejected(Data("<AnnotatedECG xmlns='urn:wrong'/>".utf8), "unsupported_layout")
        rejected(Data([0xff]), "unsupported_encoding")
        rejected(try XCTUnwrap(String(decoding: fixture(), as: UTF8.self).data(using: .utf16)), "unsupported_encoding")
    }

    func testAlternateTypePrefixRelativeTimeAndNegativeScale() throws {
        var text = String(decoding: try fixture(), as: UTF8.self)
        text = text.replacingOccurrences(of: "xmlns:xsi=", with: "xmlns:types=").replacingOccurrences(of: "xsi:type", with: "types:type")
        text = text.replacingOccurrences(of: "TIME_ABSOLUTE", with: "TIME_RELATIVE").replacingOccurrences(of: "GLIST_TS", with: "GLIST_PQ")
        text = text.replacingOccurrences(of: "scale value=\"2.5\"", with: "scale value=\"-2.5\"")
        XCTAssertEqual(try OpenMedAECG.read(Data(text.utf8)).rhythm[0].leads[0].samples, [6, 1, -9])
    }
}
