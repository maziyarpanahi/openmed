import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

final class SlideDeidentificationTests: XCTestCase {
    private func vectors() throws -> [[String: Any]] {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(
                    contentsOf:
                        root.appending(path: "tests/fixtures/multimodal/slide_deid_vectors.json"))) as? [[String: Any]])
    }
    private func decode(_ text: String) -> Data {
        var data = Data()
        var index = text.startIndex
        while index < text.endIndex {
            let next = text.index(index, offsetBy: 2)
            data.append(UInt8(text[index..<next], radix: 16) ?? 0)
            index = next
        }
        return data
    }
    func testSharedClassicAndBigTIFFVectorsPreserveTilesAndRemoveSentinels() throws {
        for row in try vectors() {
            let source = decode(try XCTUnwrap(row["source_hex"] as? String))
            let result = try SlideDeidentification.deidentifyTIFF(source)
            XCTAssertEqual(result.outputSHA256, row["output_sha256"] as? String)
            let actual = try XCTUnwrap(JSONSerialization.jsonObject(with: result.auditJSON()) as? NSDictionary)
            let expected = try XCTUnwrap(row["audit"] as? NSDictionary)
            XCTAssertEqual(actual, expected)
            let second = try SlideDeidentification.deidentifyTIFF(result.data)
            XCTAssertEqual(second.outcome, "no_associated_images")
            XCTAssertEqual(second.data, result.data)
            XCTAssertEqual(second.directories, result.directories.filter { $0.role == "level" })
            for sentinel in ["SYNTHETIC_NAME_ALICE", "SYNTHETIC_ACCESSION_123", "SYNTHETIC_患者", "SYNTHETIC_OPERATOR_DATE_20000101", "SYNTHETIC_ORPHAN_IDENTIFIER"] {
                XCTAssertNil(result.data.range(of: Data(sentinel.utf8)))
                XCTAssertFalse(String(decoding: try result.auditJSON(), as: UTF8.self).contains(sentinel))
                XCTAssertFalse(result.description.contains(sentinel))
            }
        }
    }
    func testUnmarkedSecondaryGenericTiledImageIsRefused() throws {
        var source = decode(try XCTUnwrap(vectors().first?["source_hex"] as? String))
        // The first shared fixture is independently built little-endian classic
        // TIFF with two levels. Remove vendor detection and the secondary
        // reduced-resolution bit while retaining its otherwise valid geometry.
        let vendor = try XCTUnwrap(source.range(of: Data("Aperio".utf8)))
        source[vendor.lowerBound] = 71
        func little(_ at: Int, _ count: Int) -> Int {
            (0..<count).reduce(0) { $0 | (Int(source[at + $1]) << (8 * $1)) }
        }
        let first = little(4, 4)
        let second = little(first + 2 + little(first, 2) * 12, 4)
        XCTAssertEqual(little(second + 2, 2), 254)
        source.replaceSubrange(second + 10..<second + 14, with: Data([0, 0, 0, 0]))
        XCTAssertThrowsError(try SlideDeidentification.deidentifyTIFF(source)) {
            XCTAssertEqual(($0 as? SlideDeidError)?.category, "slide_layout_unknown")
        }
    }
    func testTruncationAndDeviceBudgetsFailClosed() throws {
        let source = decode(try XCTUnwrap(vectors().first?["source_hex"] as? String))
        for n in 0..<source.count - 32 {
            XCTAssertThrowsError(try SlideDeidentification.deidentifyTIFF(source.prefix(n)))
        }
        var limits = SlideDeidLimits()
        limits.maxDirectories = 1
        XCTAssertThrowsError(try SlideDeidentification.deidentifyTIFF(source, limits: limits))
        limits.maxDirectories = 0
        XCTAssertThrowsError(try SlideDeidentification.deidentifyTIFF(source, limits: limits))
    }
    func testDICOMLabelAndOverviewNeedBoundRedactionAndReviewer() throws {
        let sop = "1.2.840.10008.5.1.4.1.1.77.1.6"
        let pixels = Data([0, 0, 0, 0])
        let digest = SHA256.hash(data: pixels).map { String(format: "%02x", $0) }.joined()
        let evidence = SlidePixelRedactionEvidence(pixelSHA256: digest, framesProcessed: 1, residualVerified: true, residualEntityCount: 0)
        for role in ["LABEL", "OVERVIEW"] {
            let type = ["ORIGINAL", "PRIMARY", role, "NONE"]
            XCTAssertThrowsError(try SlideDeidentification.validateDICOMInstance(sopClassUID: sop, imageType: type))
            XCTAssertEqual(
                try SlideDeidentification.validateDICOMInstance(
                    sopClassUID: sop, imageType: type,
                    keepAssociatedImages: true, reviewerConfirmed: true, pixels: pixels, evidence: evidence), role.lowercased())
            XCTAssertThrowsError(
                try SlideDeidentification.validateDICOMInstance(
                    sopClassUID: sop, imageType: type,
                    keepAssociatedImages: true, reviewerConfirmed: false, pixels: pixels, evidence: evidence))
            XCTAssertThrowsError(
                try SlideDeidentification.validateDICOMInstance(
                    sopClassUID: sop, imageType: type,
                    keepAssociatedImages: true, reviewerConfirmed: true, pixels: Data([1]), evidence: evidence))
            let unverified = SlidePixelRedactionEvidence(pixelSHA256: digest, framesProcessed: 1, residualVerified: false, residualEntityCount: 0)
            XCTAssertThrowsError(
                try SlideDeidentification.validateDICOMInstance(
                    sopClassUID: sop, imageType: type,
                    keepAssociatedImages: true, reviewerConfirmed: true, pixels: pixels, evidence: unverified))
        }
        XCTAssertThrowsError(try SlideDeidentification.validateDICOMInstance(sopClassUID: sop, imageType: []))
        XCTAssertThrowsError(
            try SlideDeidentification.validateDICOMInstance(
                sopClassUID: sop,
                imageType: ["ORIGINAL", "PRIMARY", "VOLUME"], specimenLabelInImage: "YES"))
    }
}
