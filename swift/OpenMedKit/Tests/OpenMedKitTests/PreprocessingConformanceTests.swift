import Foundation
import XCTest

@testable import OpenMedKit

final class PreprocessingConformanceTests: XCTestCase {
    private let runtime = String(repeating: "a", count: 64)

    private struct ProfilePayload: Decodable {
        let sourceSize: [Int], crop: [Int], targetSize: [Int], orientation: Int
        let interpolation: String, colorOrder: String, background: [Int]
        let scale: Double, mean: [Double], std: [Double], layout: String, dtype: String, atol: Double
        func make() throws -> PreprocessingProfile {
            try PreprocessingProfile(
                sourceSize: sourceSize, crop: crop, targetSize: targetSize,
                orientation: orientation, interpolation: interpolation, colorOrder: colorOrder,
                background: background, scale: scale, mean: mean, std: std, layout: layout, dtype: dtype, atol: atol)
        }
    }
    private struct Vector: Decodable {
        let profile: ProfilePayload
        let pixels: [[Int]], mask: [Bool], tensor: [Double], geometry: [String], protectedMask: [Bool]
        let profileDigest: String, tensorDigest: String, coordinateDigest: String, qualificationDigest: String
    }
    private struct Fixture: Decodable { let vectors: [Vector] }

    func testSharedPythonVectors() throws {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        let data = try Data(contentsOf: root.appendingPathComponent("tests/fixtures/multimodal/preprocessing-v1.json"))
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let fixture = try decoder.decode(Fixture.self, from: data)
        XCTAssertEqual(fixture.vectors.count, 9)
        for v in fixture.vectors {
            let p = try v.profile.make()
            let out = try PreprocessingConformance.execute(p, pixels: v.pixels, protectedMask: v.mask)
            XCTAssertEqual(out.tensor, v.tensor)
            XCTAssertEqual(out.geometry, v.geometry)
            XCTAssertEqual(out.protectedMask, v.protectedMask)
            XCTAssertEqual(p.digest, v.profileDigest)
            XCTAssertEqual(out.tensorDigest, v.tensorDigest)
            XCTAssertEqual(out.coordinateDigest, v.coordinateDigest)
            let q = try PreprocessingConformance.qualify(p, runtimeDigest: runtime) { p, pixels, mask in
                try PreprocessingConformance.execute(p, pixels: pixels, protectedMask: mask)
            }
            XCTAssertEqual(q.digest, v.qualificationDigest)
            let e = try PreprocessingConformance.verify(
                p, pixels: v.pixels, protectedMask: v.mask,
                output: out, qualification: q, runtimeDigest: runtime)
            XCTAssertTrue(e.reviewerConfirmationRequired)
            XCTAssertEqual(e.notice, "synthetic_preprocessing_only_non_diagnostic")
            try e.requireCurrent(profile: p, output: out, qualification: q, runtimeDigest: runtime)
        }
    }

    func testGeometryExactAndTensorToleranceSeparate() throws {
        let p = try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], dtype: "float64")
        let pixels = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
        let mask = [true, false, false, false]
        let q = try PreprocessingConformance.qualify(p, runtimeDigest: runtime) { p, pixels, mask in
            try PreprocessingConformance.execute(p, pixels: pixels, protectedMask: mask)
        }
        let out = try PreprocessingConformance.execute(p, pixels: pixels, protectedMask: mask)
        var tensor = out.tensor
        tensor[0] += p.atol / 2
        let rounded = try PreprocessingOutput(
            profileDigest: p.digest, shape: out.shape, tensor: tensor,
            geometry: out.geometry, protectedMask: out.protectedMask, dtype: out.dtype)
        let e = try PreprocessingConformance.verify(
            p, pixels: pixels, protectedMask: mask, output: rounded,
            qualification: q, runtimeDigest: runtime)
        XCTAssertNotEqual(rounded.tensorDigest, out.tensorDigest)
        XCTAssertThrowsError(try e.requireCurrent(profile: p, output: out, qualification: q, runtimeDigest: runtime))
        var geometry = out.geometry
        geometry[0] = "1000001/1000000"
        let wrong = try PreprocessingOutput(
            profileDigest: p.digest, shape: out.shape, tensor: out.tensor,
            geometry: geometry, protectedMask: mask, dtype: out.dtype)
        XCTAssertThrowsError(
            try PreprocessingConformance.verify(
                p, pixels: pixels, protectedMask: mask,
                output: wrong, qualification: q, runtimeDigest: runtime)
        ) { error in
            XCTAssertEqual(error as? PreprocessingConformanceError, .coordinateMismatch)
        }
        for changed in [
            try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], interpolation: "bilinear"),
            try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], colorOrder: "BGR"),
            try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], mean: [0.01, 0, 0]),
        ] {
            XCTAssertThrowsError(try e.requireCurrent(profile: changed, output: out, qualification: q, runtimeDigest: runtime)) { error in
                XCTAssertEqual(error as? PreprocessingConformanceError, .qualificationStale)
            }
        }
    }

    func testUnsupportedAndAdapterFailure() throws {
        XCTAssertThrowsError(try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], interpolation: "bicubic"))
        XCTAssertThrowsError(try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2], std: [0, 1, 1]))
        let p = try PreprocessingProfile(sourceSize: [2, 2], crop: [0, 0, 2, 2], targetSize: [2, 2])
        XCTAssertThrowsError(
            try PreprocessingConformance.qualify(p, runtimeDigest: runtime) { _, _, _ in
                throw NSError(domain: "PRIVATE_PAYLOAD", code: 1)
            }
        ) { error in XCTAssertEqual(error as? PreprocessingConformanceError, .adapterFailed) }
        XCTAssertThrowsError(
            try PreprocessingConformance.qualify(p, runtimeDigest: runtime) { p, pixels, mask in
                let out = try PreprocessingConformance.execute(p, pixels: pixels, protectedMask: mask)
                var tensor = out.tensor
                tensor[0] += 1
                return try PreprocessingOutput(
                    profileDigest: p.digest, shape: out.shape, tensor: tensor,
                    geometry: out.geometry, protectedMask: out.protectedMask, dtype: out.dtype)
            }
        ) { error in XCTAssertEqual(error as? PreprocessingConformanceError, .tensorMismatch) }
    }
}
