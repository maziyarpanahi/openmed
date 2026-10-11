import Foundation
import XCTest

@testable import OpenMedKit

final class BigTiffPreflightTests: XCTestCase {
    private func fixtures() throws -> [[String: Any]] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appending(path: "tests/fixtures/multimodal/bigtiff_pyramids.json"))
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [[String: Any]])
    }

    private func bytes(_ row: [String: Any]) throws -> Data {
        let hex = Array(try XCTUnwrap(row["hex"] as? String))
        return Data(stride(from: 0, to: hex.count, by: 2).map { UInt8(String(hex[$0...($0 + 1)]), radix: 16)! })
    }

    private func put(_ data: inout Data, _ offset: Int, _ value: UInt64, size: Int = 8) {
        for i in 0..<size { data[offset + i] = UInt8(truncatingIfNeeded: value >> (i * 8)) }
    }

    private func entry(_ data: Data, _ tag: UInt16, level: Int = 0) -> Int {
        let base = 16 + 256 * level
        let count = Int(data[base])
        for i in 0..<count {
            let start = base + 8 + i * 20
            if UInt16(data[start]) | (UInt16(data[start + 1]) << 8) == tag { return start }
        }
        preconditionFailure("synthetic tag missing")
    }

    private func rejected(_ data: Data, _ code: String, profile: BigTiffProfile = .ifdChain, limits: BigTiffLimits = BigTiffLimits()) {
        XCTAssertThrowsError(try BigTiffPreflight.read(data, profile: profile, limits: limits)) { error in
            XCTAssertEqual((error as? BigTiffPreflightError)?.category, "bigtiff_" + code)
            XCTAssertEqual(String(describing: error), "bigtiff_" + code)
        }
    }

    func testSharedPythonManifestsWithoutPixels() throws {
        for row in try fixtures() {
            let data = try bytes(row)
            let profile = try XCTUnwrap(BigTiffProfile(rawValue: try XCTUnwrap(row["profile"] as? String)))
            var limits = BigTiffLimits()
            limits.maxHeaderBytes = try XCTUnwrap(row["header_bytes"] as? Int)
            let result = try BigTiffPreflight.read(prefix: data.prefix(limits.maxHeaderBytes), sourceSize: UInt64(data.count), profile: profile, limits: limits)
            let expectedData = try JSONSerialization.data(withJSONObject: XCTUnwrap(row["expected"]))
            let expected = try JSONDecoder().decode(BigTiffPyramid.self, from: expectedData)
            XCTAssertEqual(result, expected)
            XCTAssertEqual(result, try BigTiffPreflight.read(data, profile: profile, limits: limits))
            XCTAssertEqual(result.totalTiles, 5)
        }
    }

    func testEveryTruncatedMetadataPrefixFails() throws {
        let row = try XCTUnwrap(fixtures().first)
        let data = try bytes(row)
        let end = try XCTUnwrap(row["header_bytes"] as? Int)
        for count in 0..<end {
            XCTAssertThrowsError(try BigTiffPreflight.read(prefix: data.prefix(count), sourceSize: UInt64(data.count), profile: .ifdChain))
        }
    }

    func testResourceCeilings() throws {
        let data = try bytes(XCTUnwrap(fixtures().first))
        var limits = BigTiffLimits()
        limits.maxHeaderBytes = 16
        rejected(data, "header_limit_or_truncated", limits: limits)
        limits = BigTiffLimits()
        limits.maxIFDEntries = 2
        rejected(data, "entry_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxDirectories = 1
        rejected(data, "directory_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxLevels = 1
        rejected(data, "level_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxTiles = 4
        rejected(data, "tile_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxPixels = 63
        rejected(data, "pixel_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxOffset = 16
        rejected(data, "offset_limit_exceeded", limits: limits)
        limits = BigTiffLimits()
        limits.maxHeaderBytes = 0
        rejected(data, "limit_invalid", limits: limits)
    }

    func testMalformedEntriesAndUnsupportedLayouts() throws {
        let source = try bytes(XCTUnwrap(fixtures().first))
        let cases: [(UInt16, Int, UInt64, Int, String)] = [
            (256, 12, 0, 8, "geometry_invalid"),
            (259, 12, 999, 8, "compression_unsupported"),
            (284, 12, 2, 8, "layout_unsupported"),
            (277, 12, 65, 8, "layout_unsupported"),
            (324, 4, 3, 8, "tile_table_inconsistent"),
            (325, 4, 5, 8, "tile_table_inconsistent"),
            (324, 4, UInt64.max, 8, "offset_overflow"),
            (324, 12, UInt64.max - 7, 8, "offset_overflow"),
            (324, 12, 17, 8, "value_offset_invalid"),
            (324, 2, 2, 2, "value_type_unsupported"),
            (324, 2, 18, 2, "value_type_unsupported"),
            (256, 4, 2, 8, "value_count_limit_exceeded"),
            (325, 4, 0, 8, "value_count_invalid"),
            (277, 0, 65000, 2, "layout_unsupported"),
            (277, 0, 273, 2, "layout_unsupported"),
        ]
        for (tag, relative, value, size, code) in cases {
            var data = source
            put(&data, entry(data, tag) + relative, value, size: size)
            rejected(data, code)
        }
        var data = source
        put(&data, entry(data, 256, level: 1) + 12, 9)
        rejected(data, "pyramid_geometry_inconsistent")
        data = source
        put(&data, entry(data, 254, level: 1) + 12, 0)
        rejected(data, "layout_unsupported")
    }

    func testCyclesOverlapsAndTileOffsetOverflow() throws {
        let source = try bytes(XCTUnwrap(fixtures().first))
        var data = source
        put(&data, 16 + 8 + 10 * 20, 16)
        rejected(data, "directory_cycle")
        for (offset, code): (UInt64, String) in [(UInt64.max, "offset_overflow"), (16, "range_overlap"), (100_000, "range_out_of_source"), (0, "tile_range_invalid")] {
            data = source
            put(&data, entry(data, 324, level: 1) + 12, offset)
            rejected(data, code)
        }
        rejected(source, "layout_unsupported", profile: .subIFDs)
        let rows = try fixtures()
        rejected(try bytes(rows[1]), "layout_unsupported")
    }

    func testIdentifyingTagPayloadIsIgnored() throws {
        var data = try bytes(XCTUnwrap(fixtures().first))
        let start = entry(data, 277)
        put(&data, start, 270, size: 2)
        put(&data, start + 2, 2, size: 2)
        put(&data, start + 4, UInt64.max)
        put(&data, start + 12, UInt64.max)
        data.append(Data("SYNTHETIC-PRIVATE-NAME /synthetic/private/slide.tif".utf8))
        let result = try BigTiffPreflight.read(data, profile: .ifdChain)
        let report = String(decoding: try JSONEncoder().encode(result), as: UTF8.self)
        XCTAssertFalse(report.contains("SYNTHETIC"))
        XCTAssertFalse(report.contains("private"))
        XCTAssertFalse(String(describing: result).contains("SYNTHETIC"))
    }
}
