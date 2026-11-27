import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

/// The bundled `task_packs.json` is the shared manifest, and decoding it fails
/// closed on anything the manifest schema does not declare.
final class OpenMedTaskPackManifestTests: XCTestCase {
    static let sharedManifestPath = "openmed/core/data/task_packs.json"

    static func repositoryRoot() -> URL {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { root.deleteLastPathComponent() }
        return root
    }

    static func sharedManifestData() throws -> Data {
        try Data(contentsOf: repositoryRoot().appending(path: sharedManifestPath))
    }

    /// Apple candidate repository ids of a pack, read straight from the shared
    /// file rather than through the decoder under test.
    static func sharedAppleRepoIDs(pack packID: String) throws -> [String] {
        let root = try XCTUnwrap(JSONSerialization.jsonObject(with: sharedManifestData()) as? [String: Any])
        let packs = try XCTUnwrap(root["packs"] as? [[String: Any]])
        let pack = try XCTUnwrap(packs.first { $0["pack_id"] as? String == packID })
        let runtimes = try XCTUnwrap(pack["runtimes"] as? [String: Any])
        let apple = try XCTUnwrap(runtimes["apple"] as? [String: Any])
        let candidates = try XCTUnwrap(apple["candidates"] as? [[String: Any]])
        return try candidates.sorted { ($0["rank"] as? Int ?? 0) < ($1["rank"] as? Int ?? 0) }
            .map { try XCTUnwrap($0["repo_id"] as? String) }
    }

    func testBundledManifestIsByteIdenticalToTheSharedManifest() throws {
        let bundled = try XCTUnwrap(OpenMedTaskPackManifest.bundledResourceData())
        let shared = try Self.sharedManifestData()
        XCTAssertEqual(
            Data(SHA256.hash(data: bundled)),
            Data(SHA256.hash(data: shared)),
            "Resources/task_packs/task_packs.json is stale; copy \(Self.sharedManifestPath)"
        )
    }

    func testBundledManifestDecodesEveryPackWithItsAppleCandidates() throws {
        let manifest = try OpenMedTaskPackManifest.bundled()
        XCTAssertEqual(manifest.schemaVersion.split(separator: ".").first, "1")
        XCTAssertEqual(Set(manifest.packs.keys), ["clinical_entity", "medication", "pii"])
        for (packID, pack) in manifest.packs {
            XCTAssertEqual(pack.packID, packID)
            XCTAssertEqual(pack.format, "mlx-fp")
            XCTAssertEqual(pack.candidates.map(\.repoID), try Self.sharedAppleRepoIDs(pack: packID))
            XCTAssertEqual(pack.candidates.map(\.rank), Array(1...pack.candidates.count))
            XCTAssertTrue(pack.candidates.allSatisfy { $0.format == "mlx-fp" })
            XCTAssertTrue(pack.notice.key.hasPrefix("openmed.notice.\(packID)."))
            XCTAssertFalse(pack.notice.text.isEmpty)
            XCTAssertEqual(pack.canonicalLabels, pack.canonicalLabels.sorted())
        }
    }

    func testUnknownPackIsATypedError() throws {
        let manifest = try OpenMedTaskPackManifest.bundled()
        XCTAssertThrowsError(try manifest.pack("radiology")) { error in
            XCTAssertEqual(error as? OpenMedTaskPackError, .unknownPack)
        }
    }

    func testDecodingFailsClosed() throws {
        let cases: [(String, OpenMedTaskPackError, (inout [String: Any]) throws -> Void)] = [
            ("another schema", .schemaUnsupported, { $0["schema"] = "openmed.other" }),
            ("another major version", .schemaUnsupported, { $0["schema_version"] = "2.0.0" }),
            ("malformed version", .manifestInvalid, { $0["schema_version"] = "1.0" }),
            ("undeclared top-level field", .manifestInvalid, { $0["extra"] = true }),
            ("undeclared pack field", .manifestInvalid, { try Self.editPack(&$0) { $0["intended_use"] = "x" } }),
            (
                "undeclared candidate field", .manifestInvalid,
                { try Self.editAppleCandidates(&$0) { $0[0]["sha"] = "x" } }
            ),
            ("rank gap", .manifestInvalid, { try Self.editAppleCandidates(&$0) { $0[0]["rank"] = 2 } }),
            ("copyleft license", .manifestInvalid, { try Self.editAppleCandidates(&$0) { $0[0]["license"] = "gpl-3.0" } }),
            (
                "size cap exceeded", .manifestInvalid,
                { try Self.editAppleCandidates(&$0) { $0[0]["param_count"] = 10_000_000_000 } }
            ),
            ("out of order", .manifestInvalid, { try Self.editAppleCandidates(&$0) { $0.swapAt(0, 1) } }),
            (
                "status without candidates", .manifestInvalid,
                { try Self.editAppleCandidates(&$0) { $0.removeAll() } }
            ),
            ("unknown notice key", .manifestInvalid, { try Self.editPack(&$0) { $0["notice_key"] = "openmed.notice.x.y" } }),
            ("lowercase label", .manifestInvalid, { try Self.editPack(&$0) { $0["canonical_labels"] = ["person"] } }),
            ("packs not an array", .manifestInvalid, { $0["packs"] = "x" }),
        ]
        for (name, expected, edit) in cases {
            var document = try XCTUnwrap(
                JSONSerialization.jsonObject(with: Self.sharedManifestData()) as? [String: Any]
            )
            try edit(&document)
            let data = try JSONSerialization.data(withJSONObject: document)
            XCTAssertThrowsError(try OpenMedTaskPackManifest.decode(data), name) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, expected, name)
            }
        }
        XCTAssertThrowsError(try OpenMedTaskPackManifest.decode(Data("[]".utf8))) { error in
            XCTAssertEqual(error as? OpenMedTaskPackError, .manifestInvalid)
        }
    }

    func testNoCandidateRuntimeDecodesAndResolvesToNoCandidate() throws {
        var document = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Self.sharedManifestData()) as? [String: Any]
        )
        try Self.editPack(&document) { pack in
            var runtimes = pack["runtimes"] as? [String: Any] ?? [:]
            runtimes["apple"] = ["candidates": [Any](), "format": "mlx-fp", "status": "no_candidate"]
            pack["runtimes"] = runtimes
        }
        let manifest = try OpenMedTaskPackManifest.decode(JSONSerialization.data(withJSONObject: document))
        let pack = try manifest.pack("pii")
        XCTAssertEqual(pack.candidates, [])
        let resolution = OpenMedTaskPackResolution.resolve(pack, mlxSupported: true, isSimulator: false)
        XCTAssertEqual(resolution.status, .noCandidate)
        XCTAssertEqual(resolution.refusal, .noCandidate)
    }

    func testResolutionSkipsMLXCandidatesWhereMLXCannotRun() throws {
        let pack = try OpenMedTaskPackManifest.bundled().pack("pii")
        let supported = OpenMedTaskPackResolution.resolve(pack, mlxSupported: true, isSimulator: false)
        XCTAssertEqual(supported.status, .resolved)
        XCTAssertEqual(supported.candidates, pack.candidates)
        XCTAssertEqual(supported.skipped, [])
        XCTAssertNil(supported.refusal)

        let simulator = OpenMedTaskPackResolution.resolve(pack, mlxSupported: false, isSimulator: true)
        XCTAssertEqual(simulator.status, .unsupportedOnSimulator)
        XCTAssertEqual(simulator.candidates, [])
        XCTAssertEqual(simulator.skipped.map(\.candidate), pack.candidates)
        XCTAssertTrue(simulator.skipped.allSatisfy { $0.reason == .unsupportedOnSimulator })
        XCTAssertEqual(simulator.refusal, .unsupportedOnSimulator)

        let intel = OpenMedTaskPackResolution.resolve(pack, mlxSupported: false, isSimulator: false)
        XCTAssertEqual(intel.status, .unsupportedPlatform)
        XCTAssertTrue(intel.skipped.allSatisfy { $0.reason == .unsupportedPlatform })
        XCTAssertEqual(intel.refusal, .unsupportedPlatform)
    }

    func testErrorsAreValueFree() throws {
        let pack = try OpenMedTaskPackManifest.bundled().pack("pii")
        for error in OpenMedTaskPackError.allCases {
            let renderings = [
                error.localizedDescription, String(describing: error), String(reflecting: error),
                (error as NSError).description,
            ]
            for rendering in renderings {
                XCTAssertFalse(rendering.contains("OpenMed/"), "\(error.reason)")
                XCTAssertFalse(rendering.contains(NSHomeDirectory()), "\(error.reason)")
                for candidate in pack.candidates {
                    XCTAssertFalse(rendering.contains(candidate.repoID), "\(error.reason)")
                }
            }
            XCTAssertEqual(error.reason, error.rawValue)
        }
    }

    // MARK: - Helpers

    private static func editPack(
        _ document: inout [String: Any],
        id: String = "pii",
        _ edit: (inout [String: Any]) throws -> Void
    ) throws {
        var packs = try XCTUnwrap(document["packs"] as? [[String: Any]])
        let index = try XCTUnwrap(packs.firstIndex { $0["pack_id"] as? String == id })
        try edit(&packs[index])
        document["packs"] = packs
    }

    private static func editAppleCandidates(
        _ document: inout [String: Any],
        _ edit: (inout [[String: Any]]) throws -> Void
    ) throws {
        try editPack(&document) { pack in
            var runtimes = try XCTUnwrap(pack["runtimes"] as? [String: Any])
            var apple = try XCTUnwrap(runtimes["apple"] as? [String: Any])
            var candidates = try XCTUnwrap(apple["candidates"] as? [[String: Any]])
            try edit(&candidates)
            apple["candidates"] = candidates
            runtimes["apple"] = apple
            pack["runtimes"] = runtimes
        }
    }
}
