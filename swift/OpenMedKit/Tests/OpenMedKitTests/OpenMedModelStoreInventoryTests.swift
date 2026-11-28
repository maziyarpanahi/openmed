import Foundation
import XCTest

@testable import OpenMedKit

final class OpenMedModelStoreInventoryTests: XCTestCase {
    private var scratch: URL!
    private var cacheRoot: URL!
    private var outside: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory.appending(
            path: "openmed-inventory-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        cacheRoot = scratch.appending(path: "cache", directoryHint: .isDirectory)
        outside = scratch.appending(path: "outside", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let scratch, let enumerator = FileManager.default.enumerator(atPath: scratch.path) {
            for case let path as String in enumerator {
                try? FileManager.default.setAttributes(
                    [.posixPermissions: 0o755], ofItemAtPath: scratch.appending(path: path).path)
            }
        }
        try? FileManager.default.removeItem(at: scratch)
    }

    // MARK: - Listing and sizes

    func testListingReportsReadyPartialAndStraySnapshots() throws {
        let ready = try writeSnapshot(repo: "OpenMed__Ready-mlx", revision: "main", complete: true)
        try writeSnapshot(repo: "OpenMed__Partial-mlx", revision: "main", complete: false)
        try makeDirectory("OpenMed__Stray-mlx/empty-revision")
        try makeDirectory("not-a-repository/main")
        try write("stray-file.bin", bytes: 7)
        try makeDirectory(".hidden__repo/main")
        try write("OpenMed__Ready-mlx/loose-file.txt", bytes: 3)

        let models = try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot)

        XCTAssertEqual(
            models.map { "\($0.repoID)@\($0.revision)=\($0.state.rawValue)" },
            [
                "OpenMed/Partial-mlx@main=partial",
                "OpenMed/Ready-mlx@main=ready",
                "OpenMed/Stray-mlx@empty-revision=missing",
            ]
        )
        let readyRow = try XCTUnwrap(models.first { $0.repoID == "OpenMed/Ready-mlx" })
        let marker =
            try FileManager.default.attributesOfItem(
                atPath: ready.appending(path: ".openmed-artifact-ready").path)[.size] as? Int64
        XCTAssertEqual(readyRow.sizeBytes, Self.completeSnapshotBytes + (marker ?? 0))
        XCTAssertNotNil(readyRow.modificationDate)
        XCTAssertEqual(models.first { $0.repoID == "OpenMed/Partial-mlx" }?.sizeBytes, 11)
        XCTAssertEqual(models.first { $0.repoID == "OpenMed/Stray-mlx" }?.sizeBytes, 0)
        for model in models {
            XCTAssertEqual(
                try OpenMedModelStore.mlxModelCacheState(
                    repoID: model.repoID, revision: model.revision, cacheDirectory: cacheRoot),
                model.state
            )
        }
    }

    func testRevisionNamesAreDecodedAndEmptyCachesListNothing() throws {
        XCTAssertEqual(try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot), [])
        XCTAssertEqual(
            try OpenMedModelStore.listCachedModels(cacheDirectory: scratch.appending(path: "absent")),
            []
        )
        try writeSnapshot(repo: "OpenMed__Pinned-mlx", revision: "refs__pr__1", complete: true)

        let models = try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot)

        XCTAssertEqual(models.map(\.revision), ["refs/pr/1"])
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: "OpenMed/Pinned-mlx", revision: "refs/pr/1", cacheDirectory: cacheRoot),
            .ready
        )
    }

    func testTotalSizeCountsEveryRegularFileButNoSymlinkTarget() throws {
        try writeSnapshot(repo: "OpenMed__Ready-mlx", revision: "main", complete: false)
        try write("stray-file.bin", bytes: 7)
        let bigOutside = outside.appending(path: "big.bin")
        try Data(count: 4_096).write(to: bigOutside)
        try FileManager.default.createSymbolicLink(
            at: cacheRoot.appending(path: "OpenMed__Ready-mlx/main/link.bin"), withDestinationURL: bigOutside)
        try FileManager.default.createSymbolicLink(
            at: cacheRoot.appending(path: "OpenMed__Linked-mlx"), withDestinationURL: outside)

        XCTAssertEqual(try OpenMedModelStore.totalCacheSizeBytes(cacheDirectory: cacheRoot), 11 + 7)
        XCTAssertEqual(try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot).map(\.sizeBytes), [11])
        XCTAssertEqual(try OpenMedModelStore.totalCacheSizeBytes(cacheDirectory: scratch.appending(path: "absent")), 0)
    }

    // MARK: - Deletion

    func testDeleteLeavesTheSnapshotMissingAndOtherModelsUntouched() throws {
        try writeSnapshot(repo: "OpenMed__Delete-mlx", revision: "main", complete: true)
        try writeSnapshot(repo: "OpenMed__Delete-mlx", revision: "v2", complete: true)
        try writeSnapshot(repo: "OpenMed__Keep-mlx", revision: "main", complete: true)
        let before = try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot)
            .filter { $0.repoID != "OpenMed/Delete-mlx" || $0.revision != "main" }

        try OpenMedModelStore.deleteModel(repoID: "OpenMed/Delete-mlx", revision: "main", cacheDirectory: cacheRoot)

        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: "OpenMed/Delete-mlx", revision: "main", cacheDirectory: cacheRoot),
            .missing
        )
        XCTAssertEqual(try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot), before)
        XCTAssertEqual(
            try FileManager.default.contentsOfDirectory(atPath: cacheRoot.appending(path: "OpenMed__Delete-mlx").path),
            ["v2"]
        )

        try OpenMedModelStore.deleteModel(repoID: "OpenMed/Delete-mlx", revision: "v2", cacheDirectory: cacheRoot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: cacheRoot.appending(path: "OpenMed__Delete-mlx").path))
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: "OpenMed/Keep-mlx", revision: "main", cacheDirectory: cacheRoot),
            .ready
        )
        // Deleting a snapshot that is not cached is a no-op.
        XCTAssertNoThrow(
            try OpenMedModelStore.deleteModel(repoID: "OpenMed/Absent-mlx", cacheDirectory: cacheRoot))
    }

    func testSymlinkedSnapshotOutsideTheRootIsRefusedAndItsTargetKept() throws {
        let target = outside.appending(path: "snapshot", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data(count: 5).write(to: target.appending(path: "weights.safetensors"))
        try makeDirectory("OpenMed__Evil-mlx")
        try FileManager.default.createSymbolicLink(
            at: cacheRoot.appending(path: "OpenMed__Evil-mlx/main"), withDestinationURL: target)
        try FileManager.default.createSymbolicLink(
            at: cacheRoot.appending(path: "OpenMed__EvilRepo-mlx"), withDestinationURL: outside)

        for repoID in ["OpenMed/Evil-mlx", "OpenMed/EvilRepo-mlx"] {
            XCTAssertThrowsError(
                try OpenMedModelStore.deleteModel(repoID: repoID, revision: "main", cacheDirectory: cacheRoot),
                repoID
            ) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .invalidManifestPath("unsafe_symlink"))
                XCTAssertFalse(error.localizedDescription.contains(self.scratch.path))
                XCTAssertFalse(error.localizedDescription.contains("Evil"))
            }
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: target.appending(path: "weights.safetensors").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: outside.appending(path: "snapshot").path))
        XCTAssertEqual(try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot), [])
    }

    func testDeleteRemovesInnerSymlinksWithoutTouchingTheirTargets() throws {
        let snapshot = try writeSnapshot(repo: "OpenMed__Links-mlx", revision: "main", complete: true)
        let outsideFile = outside.appending(path: "keep.txt")
        try Data("keep".utf8).write(to: outsideFile)
        try FileManager.default.createSymbolicLink(
            at: snapshot.appending(path: "escape.txt"), withDestinationURL: outsideFile)
        try FileManager.default.createSymbolicLink(
            at: snapshot.appending(path: "escape-dir"), withDestinationURL: outside)

        try OpenMedModelStore.deleteModel(repoID: "OpenMed/Links-mlx", cacheDirectory: cacheRoot)

        XCTAssertEqual(try Data(contentsOf: outsideFile), Data("keep".utf8))
        XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.path))
    }

    func testReadyMarkerIsRemovedBeforeAnyFileWhenDeletionFails() throws {
        let snapshot = try writeSnapshot(repo: "OpenMed__Locked-mlx", revision: "main", complete: true)
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(repoID: "OpenMed/Locked-mlx", cacheDirectory: cacheRoot), .ready)
        let marker = snapshot.appending(path: ".openmed-artifact-ready")
        XCTAssertTrue(FileManager.default.fileExists(atPath: marker.path))
        // A read-only repository directory refuses the rename that starts the delete.
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o555], ofItemAtPath: snapshot.deletingLastPathComponent().path)

        XCTAssertThrowsError(
            try OpenMedModelStore.deleteModel(repoID: "OpenMed/Locked-mlx", cacheDirectory: cacheRoot)
        ) { error in
            XCTAssertEqual(error as? OpenMedModelStoreError, .cacheDeletionFailed)
            XCTAssertEqual((error as? OpenMedModelStoreError)?.reason, "cache_deletion_failed")
            XCTAssertFalse(error.localizedDescription.contains(self.scratch.path))
        }

        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: snapshot.appending(path: "weights.safetensors").path))
    }

    func testInterruptedDeletionTombstonesAreHiddenAndSwept() throws {
        try writeSnapshot(repo: "OpenMed__Swept-mlx", revision: "main", complete: true)
        try write("OpenMed__Swept-mlx/.openmed-deleting-interrupted/weights.safetensors", bytes: 9)

        XCTAssertEqual(try OpenMedModelStore.listCachedModels(cacheDirectory: cacheRoot).map(\.revision), ["main"])

        try OpenMedModelStore.deleteModel(repoID: "OpenMed/Swept-mlx", cacheDirectory: cacheRoot)

        XCTAssertFalse(FileManager.default.fileExists(atPath: cacheRoot.appending(path: "OpenMed__Swept-mlx").path))
        XCTAssertEqual(try OpenMedModelStore.totalCacheSizeBytes(cacheDirectory: cacheRoot), 0)
    }

    func testRepositoryDirectoryNamesDecode() {
        XCTAssertEqual(OpenMedModelStore.decodedRepoID("OpenMed__Name-mlx"), "OpenMed/Name-mlx")
        XCTAssertEqual(OpenMedModelStore.decodedRepoID("OpenMed__Name__v2"), "OpenMed/Name__v2")
        XCTAssertNil(OpenMedModelStore.decodedRepoID("plain-directory"))
        XCTAssertNil(OpenMedModelStore.decodedRepoID("__leading"))
        XCTAssertNil(OpenMedModelStore.decodedRepoID("trailing__"))
    }

    // MARK: - Fixtures

    /// config.json (11) + weights.safetensors (13) + tokenizer.json (17).
    private static let completeSnapshotBytes: Int64 = 41

    @discardableResult
    private func writeSnapshot(repo: String, revision: String, complete: Bool) throws -> URL {
        let directory = cacheRoot.appending(path: "\(repo)/\(revision)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(count: 11).write(to: directory.appending(path: "config.json"))
        if complete {
            try Data(count: 13).write(to: directory.appending(path: "weights.safetensors"))
            try Data(count: 17).write(to: directory.appending(path: "tokenizer.json"))
        }
        return directory
    }

    private func makeDirectory(_ path: String) throws {
        try FileManager.default.createDirectory(
            at: cacheRoot.appending(path: path, directoryHint: .isDirectory), withIntermediateDirectories: true)
    }

    private func write(_ path: String, bytes: Int) throws {
        let url = cacheRoot.appending(path: path)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(count: bytes).write(to: url)
    }
}
