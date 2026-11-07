import Foundation
import XCTest

@testable import OpenMedKit

final class DocumentIntakeParityTests: XCTestCase {
    private func corpus() throws -> [String: Any] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/parity/document_intake_v1.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func bits(_ value: Double) -> String { String(format: "%016llx", value.bitPattern) }

    private func number(_ value: Any?) throws -> Double { try XCTUnwrap(value as? NSNumber).doubleValue }

    private func box(_ value: Any?) throws -> DocumentIntakeBox {
        let values = try XCTUnwrap(value as? [Any]).map(number)
        return DocumentIntakeBox(left: values[0], top: values[1], right: values[2], bottom: values[3])
    }

    private func input(_ row: [String: Any]) throws -> DocumentPageInput {
        if let reason = row["unavailable"] as? String {
            return .unavailable(try XCTUnwrap(DocumentIntakeReason(rawValue: reason)))
        }
        let geometry = try XCTUnwrap(row["geometry"] as? [String: Any])
        let ocr = try XCTUnwrap(row["ocr"] as? [String: Any])
        let regions = try XCTUnwrap(ocr["regions"] as? [[String: Any]]).map { region in
            let span = try XCTUnwrap(region["span"] as? [Int])
            return DocumentRecognizedRegion(scalarStart: span[0], scalarEnd: span[1], box: try box(region["box"]))
        }
        return .rendered(
            DocumentPageGeometry(
                width: try number(geometry["width"]), height: try number(geometry["height"]),
                unit: try XCTUnwrap(MultimodalSpatialUnit(rawValue: try XCTUnwrap(geometry["unit"] as? String))),
                rotation: try XCTUnwrap(geometry["rotation"] as? Int),
                crop: geometry["crop"] is NSNull ? nil : try box(geometry["crop"])),
            DocumentRecognition(
                width: try XCTUnwrap(ocr["width"] as? Int), height: try XCTUnwrap(ocr["height"] as? Int),
                text: try XCTUnwrap(ocr["text"] as? String), regions: regions))
    }

    private func run(_ row: [String: Any]) throws -> DocumentIntakeReport {
        if let refusal = row["refusal"] as? String {
            return DocumentIntakeMapper.refused(try XCTUnwrap(DocumentIntakeReason(rawValue: refusal)))
        }
        var limits = DocumentIntakeLimits()
        if let overrides = row["limits"] as? [String: Int] {
            limits = DocumentIntakeLimits(
                maxPages: overrides["max_pages"] ?? limits.maxPages,
                maxRegionsPerPage: overrides["max_regions_per_page"] ?? limits.maxRegionsPerPage,
                maxPageScalars: overrides["max_page_scalars"] ?? limits.maxPageScalars)
        }
        let pages = try XCTUnwrap(row["pages"] as? [[String: Any]]).map(input)
        return DocumentIntakeMapper.map(sourceDigest: try XCTUnwrap(row["source_digest"] as? String), pages: pages, limits: limits)
    }

    private func digest(_ packet: MultimodalPacket?) -> Any { packet.map { $0.evidenceDigest as Any } ?? NSNull() }

    func testVocabulariesAndDefaultsMatchPython() throws {
        let data = try corpus()
        XCTAssertEqual(data["schema"] as? String, DocumentIntakeMapper.schemaVersion)
        XCTAssertEqual(data["synthetic_only"] as? Bool, true)
        XCTAssertEqual(data["page_separator"] as? String, DocumentIntakeMapper.pageSeparator)
        XCTAssertEqual(data["box_tolerance"] as? Double, DocumentIntakeMapper.boxTolerance)
        XCTAssertEqual(data["outcomes"] as? [String], DocumentIntakeOutcome.allCases.map(\.rawValue))
        XCTAssertEqual(data["reasons"] as? [String], DocumentIntakeReason.allCases.map(\.rawValue))
        XCTAssertEqual(data["page_statuses"] as? [String], DocumentIntakePageStatus.allCases.map(\.rawValue))
        let defaults = try XCTUnwrap(data["default_limits"] as? [String: Int])
        let limits = DocumentIntakeLimits()
        XCTAssertEqual(defaults["max_pages"], limits.maxPages)
        XCTAssertEqual(defaults["max_regions_per_page"], limits.maxRegionsPerPage)
        XCTAssertEqual(defaults["max_page_scalars"], limits.maxPageScalars)
    }

    func testSharedVectorsProduceIdenticalRecordsAndEvidence() throws {
        for row in try XCTUnwrap(corpus()["cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(row["id"] as? String)
            let expected = try XCTUnwrap(row["expected"] as? [String: Any])
            let report = try run(row)
            XCTAssertEqual(report.outcome.rawValue, expected["outcome"] as? String, id)
            XCTAssertEqual(report.reasons.map(\.rawValue), expected["reasons"] as? [String], id)
            XCTAssertEqual(report.text, expected["text"] as? String, id)
            let pages = try XCTUnwrap(expected["pages"] as? [[String: Any]])
            XCTAssertEqual(report.pages.count, pages.count, id)
            for (page, want) in zip(report.pages, pages) {
                XCTAssertEqual(page.page, want["page"] as? Int, id)
                XCTAssertEqual(page.status.rawValue, want["status"] as? String, id)
                XCTAssertEqual(page.reason?.rawValue, want["reason"] as? String, id)
                XCTAssertEqual(page.textSpan.map { [$0.lowerBound, $0.upperBound] }, want["text_span"] as? [Int], id)
                XCTAssertEqual(page.nodeRef, want["node_ref"] as? String, id)
                XCTAssertEqual(digest(page.evidence) as? String, want["evidence_digest"] as? String, id)
                XCTAssertEqual(digest(page.cropEvidence) as? String, want["crop_evidence_digest"] as? String, id)
                if case .visual(let visual)? = page.evidence?.projection {
                    XCTAssertEqual(visual.kind, .page, id)
                    XCTAssertEqual(bits(visual.displayWidth), want["display_width_f64"] as? String, id)
                    XCTAssertEqual(bits(visual.displayHeight), want["display_height_f64"] as? String, id)
                }
                if case .visual(let crop)? = page.cropEvidence?.projection {
                    XCTAssertEqual(crop.nodeRef, want["crop_node_ref"] as? String, id)
                    let display = try XCTUnwrap(crop.displayBox, id)
                    XCTAssertEqual([display.left, display.top, display.right, display.bottom].map(bits), want["crop_display_box_f64"] as? [String], id)
                }
            }
            let regions = try XCTUnwrap(expected["regions"] as? [[String: Any]])
            XCTAssertEqual(report.regions.count, regions.count, id)
            for (region, want) in zip(report.regions, regions) {
                XCTAssertEqual(region.page, want["page"] as? Int, id)
                XCTAssertEqual([region.scalarSpan.lowerBound, region.scalarSpan.upperBound], want["span"] as? [Int], id)
                XCTAssertEqual(region.nodeRef, want["node_ref"] as? String, id)
                let source = region.sourceBox
                XCTAssertEqual([source.left, source.top, source.right, source.bottom].map(bits), want["box_f64"] as? [String], id)
                XCTAssertEqual(region.evidence.evidenceDigest, want["evidence_digest"] as? String, id)
                let visual = try XCTUnwrap(region.visual, id)
                let display = try XCTUnwrap(visual.displayBox, id)
                XCTAssertEqual([display.left, display.top, display.right, display.bottom].map(bits), want["display_box_f64"] as? [String], id)
                XCTAssertEqual(visual.scalarSpan?.start, Int64(region.scalarSpan.lowerBound), id)
                XCTAssertNotNil(report.text(scalarSpan: region.scalarSpan), id)
            }
            for probe in try XCTUnwrap(expected["probes"] as? [[String: Any]]) {
                let span = try XCTUnwrap(probe["span"] as? [Int])
                XCTAssertEqual(report.regions(overlapping: span[0]..<span[1]).map(\.nodeRef), probe["node_refs"] as? [String], id)
            }
        }
    }

    func testMappedRegionsRecoverTheAuthoredOriginalBoxes() throws {
        for row in try XCTUnwrap(corpus()["cases"] as? [[String: Any]]) where row["refusal"] is NSNull {
            let id = try XCTUnwrap(row["id"] as? String)
            let tolerance = try number(row["tolerance"])
            let report = try run(row)
            var truths: [DocumentIntakeBox] = []
            for (page, source) in zip(report.pages, try XCTUnwrap(row["pages"] as? [[String: Any]])) where page.status == .mapped {
                let ocr = try XCTUnwrap(source["ocr"] as? [String: Any])
                truths += try XCTUnwrap(ocr["regions"] as? [[String: Any]]).map { try box($0["truth"]) }
            }
            XCTAssertEqual(truths.count, report.regions.count, id)
            for (truth, region) in zip(truths, report.regions) {
                let mapped = region.sourceBox
                let error =
                    zip([mapped.left, mapped.top, mapped.right, mapped.bottom], [truth.left, truth.top, truth.right, truth.bottom])
                    .map { abs($0 - $1) }.max() ?? 0
                XCTAssertLessThanOrEqual(error, tolerance, id)
            }
        }
    }

    func testMixedScriptSpansUseUnicodeScalars() throws {
        let rows = try XCTUnwrap(corpus()["cases"] as? [[String: Any]])
        let row = try XCTUnwrap(rows.first { $0["id"] as? String == "mixed_script_image" })
        let report = try run(row)
        XCTAssertEqual(report.text.unicodeScalars.count, 22)
        XCTAssertNotEqual(report.text.count, 22)
        XCTAssertEqual(report.regions(overlapping: 15..<16).compactMap { report.text(scalarSpan: $0.scalarSpan) }, ["😀"])
        XCTAssertEqual(report.regions(overlapping: 17..<22).compactMap { report.text(scalarSpan: $0.scalarSpan) }, ["Jose\u{301}"])
        XCTAssertFalse(String(describing: report).contains("Patient"))
        XCTAssertFalse(String(reflecting: report).contains("Patient"))
    }
}
