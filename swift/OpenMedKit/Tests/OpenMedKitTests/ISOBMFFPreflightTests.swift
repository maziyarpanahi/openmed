import Foundation
import XCTest

@testable import OpenMedKit

final class ISOBMFFPreflightTests: XCTestCase {
    private func cases() throws -> [[String: Any]] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/iso_bmff.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    private func source(_ row: [String: Any]) throws -> Data {
        let chars = Array(try XCTUnwrap(row["source_hex"] as? String))
        var data = Data()
        for i in stride(from: 0, to: chars.count, by: 2) {
            data.append(try XCTUnwrap(UInt8(String(chars[i...i + 1]), radix: 16)))
        }
        return data
    }

    func testSharedHandCheckedReportsAndStableRefusals() throws {
        for row in try cases() {
            let data = try source(row)
            let name = try XCTUnwrap(row["name"] as? String)
            XCTAssertEqual(ISOBMFFPreflight.detectMediaType(data), row["media_type"] as? String, name)
            let limits = row["limits"] as? [String: Int] ?? [:]
            do {
                let result = try ISOBMFFPreflight.read(
                    data, maxHeaderBytes: limits["max_header_bytes"] ?? 1_048_576,
                    maxDepth: limits["max_depth"] ?? 16, maxBoxes: limits["max_boxes"] ?? 4096,
                    maxDeclaredBoxBytes: UInt64(limits["max_declared_box_bytes"] ?? 4_294_967_296),
                    maxPixels: limits["max_pixels"] ?? 100_000_000)
                XCTAssertNil(row["error"], name)
                let expected = try XCTUnwrap(row["expected"] as? [String: Any])
                switch result {
                case .heif(let report):
                    XCTAssertEqual(report.width, expected["width"] as? Int, name)
                    XCTAssertEqual(report.height, expected["height"] as? Int, name)
                    XCTAssertEqual(report.itemCount, expected["item_count"] as? Int, name)
                    XCTAssertEqual(report.thumbnailCount, expected["thumbnail_count"] as? Int, name)
                    XCTAssertEqual(report.auxiliaryCount, expected["auxiliary_count"] as? Int, name)
                    XCTAssertEqual(report.burstCount, expected["burst_count"] as? Int, name)
                    XCTAssertEqual(report.hasEXIF, expected["has_exif"] as? Bool, name)
                    XCTAssertEqual(report.hasXMP, expected["has_xmp"] as? Bool, name)
                    XCTAssertEqual(report.status, "decoder_required", name)
                case .m4a(let report):
                    XCTAssertEqual(report.codecCategory, expected["codec_category"] as? String, name)
                    XCTAssertEqual(report.channels, expected["channels"] as? Int, name)
                    XCTAssertEqual(report.sampleRateHz, expected["sample_rate_hz"] as? Double, name)
                    XCTAssertEqual(report.durationSeconds, expected["duration_seconds"] as? Double, name)
                    XCTAssertEqual(report.status, "decoder_required", name)
                    XCTAssertEqual(report.notice, "non_diagnostic", name)
                    XCTAssertTrue(report.reviewerConfirmationRequired, name)
                }
                XCTAssertFalse(String(describing: result).contains("Synthetic"), name)
            } catch let error as BMFFPreflightError {
                XCTAssertEqual(error.rawValue, row["error"] as? String, name)
            }
        }
    }

    func testBoundedDetectionAndInvalidLimits() throws {
        let row = try XCTUnwrap(cases().first)
        let data = try source(row)
        for end in 0..<20 { XCTAssertNil(ISOBMFFPreflight.detectMediaType(data.prefix(end))) }
        XCTAssertThrowsError(try ISOBMFFPreflight.read(data, maxDepth: 65)) {
            XCTAssertEqual($0 as? BMFFPreflightError, .limitsInvalid)
        }
    }

    func testMediaPayloadMayExtendBeyondHeaderBound() throws {
        let data = try source(XCTUnwrap(cases().first))
        let marker = try XCTUnwrap(data.range(of: Data("mdat".utf8)))
        XCTAssertNoThrow(try ISOBMFFPreflight.read(data, maxHeaderBytes: marker.upperBound))
        XCTAssertThrowsError(try ISOBMFFPreflight.read(data, maxHeaderBytes: marker.upperBound - 1)) {
            XCTAssertEqual($0 as? BMFFPreflightError, .outOfBounds)
        }
    }
}
