import Foundation
import XCTest

@testable import OpenMedKit

/// Serves a synthetic Hub: one model-info document and files keyed by name.
/// URLProtocol instances are created by the loading system, so state is
/// static and every access holds `lock`.
private final class PinnedHubURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var modelInfo: (status: Int, body: Data) = (404, Data())
    nonisolated(unsafe) private static var files: [String: Data] = [:]
    nonisolated(unsafe) private static var urls: [String] = []

    static var requestedURLs: [String] {
        lock.lock()
        defer { lock.unlock() }
        return urls
    }

    static func install(modelInfo: (status: Int, body: Data), files: [String: Data]) {
        lock.lock()
        defer { lock.unlock() }
        self.modelInfo = modelInfo
        self.files = files
        urls = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url else { return }
        Self.lock.lock()
        Self.urls.append(url.absoluteString)
        let reply: (status: Int, body: Data)
        if url.path.hasPrefix("/api/models/") {
            reply = Self.modelInfo
        } else if let data = Self.files[url.lastPathComponent] {
            reply = (200, data)
        } else {
            reply = (404, Data())
        }
        Self.lock.unlock()
        let response = HTTPURLResponse(url: url, statusCode: reply.status, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: reply.body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class OpenMedModelIntegrityTests: XCTestCase {
    private struct Vectors: Decodable {
        struct Case: Decodable {
            let id: String
            let repoId: String
            let sha: String?
            let lastModified: String?
            let createdAt: String?
            let siblings: [String]
            let released: String?
            let expectedHash: String
        }

        let schema: String
        let version: Int
        let cases: [Case]
    }

    private static let vectorsPath = "tests/fixtures/model_catalog/repro_hash_vectors.json"
    private static let repoID = "OpenMed/Synthetic-Pinned-mlx"
    private static let commit = "0123456789abcdef0123456789abcdef01234567"
    private static let lastModified = "2026-06-24T12:34:56.000Z"
    private static let siblings = ["config.json", "weights.safetensors", "tokenizer.json", "README.md"]

    private var cacheRoot: URL!

    override func setUpWithError() throws {
        cacheRoot = FileManager.default.temporaryDirectory.appending(
            path: "openmed-integrity-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        try FileManager.default.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: cacheRoot)
    }

    // MARK: - Shared vectors

    func testHashMatchesEverySharedVector() throws {
        let vectors = try Self.loadVectors()
        XCTAssertEqual(vectors.schema, "openmed.model_repro_hash_vectors")
        XCTAssertEqual(vectors.version, 1)
        XCTAssertGreaterThanOrEqual(vectors.cases.count, 8)

        for vector in vectors.cases {
            let direct = OpenMedModelIntegrity.reproducibilityHash(
                repoID: vector.repoId,
                revisionSHA: vector.sha,
                released: vector.lastModified ?? vector.createdAt,
                siblings: vector.siblings
            )
            XCTAssertEqual(direct, vector.expectedHash, vector.id)

            let info = try OpenMedHubModelInfo(jsonData: Self.modelInfoJSON(vector))
            XCTAssertEqual(info.released, vector.released, vector.id)
            XCTAssertEqual(
                OpenMedModelIntegrity.reproducibilityHash(
                    repoID: vector.repoId,
                    revisionSHA: info.revisionSHA,
                    released: info.released,
                    siblings: info.siblings
                ),
                vector.expectedHash,
                vector.id
            )
        }
    }

    func testChangedSiblingListChangesTheHashButOrderDoesNot() {
        let base = OpenMedModelIntegrity.reproducibilityHash(
            repoID: Self.repoID, revisionSHA: Self.commit, released: Self.lastModified, siblings: Self.siblings)

        XCTAssertEqual(
            OpenMedModelIntegrity.reproducibilityHash(
                repoID: Self.repoID, revisionSHA: Self.commit, released: "2026-06-24",
                siblings: Self.siblings.reversed()),
            base
        )
        XCTAssertNotEqual(
            OpenMedModelIntegrity.reproducibilityHash(
                repoID: Self.repoID, revisionSHA: Self.commit, released: Self.lastModified,
                siblings: Self.siblings + ["extra.bin"]),
            base
        )
        XCTAssertNotEqual(
            OpenMedModelIntegrity.reproducibilityHash(
                repoID: Self.repoID, revisionSHA: nil, released: Self.lastModified, siblings: Self.siblings),
            base
        )
    }

    func testModelInfoParserRejectsMalformedDocuments() {
        for body in [#"not json"#, #"[]"#, #"{"sha":"x","siblings":"config.json"}"#, #"{"siblings":["config.json"]}"#] {
            XCTAssertThrowsError(try OpenMedHubModelInfo(jsonData: Data(body.utf8)), body) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .invalidModelInfo)
            }
        }
        XCTAssertEqual(
            try OpenMedHubModelInfo(jsonData: Data(#"{"sha":null,"siblings":[{"size":1},{"rfilename":"a"}]}"#.utf8)),
            OpenMedHubModelInfo(revisionSHA: nil, released: nil, siblings: ["a"])
        )
    }

    func testModelInfoURLEncodesTheRevisionAsOneSegment() throws {
        XCTAssertEqual(
            try OpenMedModelStore.modelInfoURL(repoID: "OpenMed/Synthetic-mlx", revision: "refs/pr/1").absoluteString,
            "https://huggingface.co/api/models/OpenMed/Synthetic-mlx/revision/refs%2Fpr%2F1"
        )
    }

    // MARK: - Pinned downloads

    func testMismatchedSiblingListIsRefusedBeforeAnyFileIsWritten() async throws {
        let entry = Self.entry()
        PinnedHubURLProtocol.install(
            modelInfo: (200, Self.modelInfo(siblings: Self.siblings + ["extra.bin"])),
            files: Self.snapshotFiles()
        )

        do {
            _ = try await OpenMedModelStore.downloadMLXModel(
                entry: entry, cacheDirectory: cacheRoot, session: Self.session())
            XCTFail("a changed sibling list must be refused")
        } catch {
            XCTAssertEqual(error as? OpenMedModelStoreError, .reproducibilityHashMismatch)
            XCTAssertEqual((error as? OpenMedModelStoreError)?.reason, "reproducibility_hash_mismatch")
            XCTAssertFalse(error.localizedDescription.contains(Self.repoID))
        }

        XCTAssertEqual(
            PinnedHubURLProtocol.requestedURLs,
            ["https://huggingface.co/api/models/OpenMed/Synthetic-Pinned-mlx/revision/main"]
        )
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cacheRoot.path), [])
    }

    func testMatchingSnapshotDownloadsFromTheCommitAndRecordsIt() async throws {
        let entry = Self.entry()
        PinnedHubURLProtocol.install(modelInfo: (200, Self.modelInfo()), files: Self.snapshotFiles())

        let directory = try await OpenMedModelStore.downloadMLXModel(
            entry: entry, cacheDirectory: cacheRoot, session: Self.session())

        XCTAssertEqual(
            directory,
            try OpenMedModelStore.cachedMLXModelDirectory(
                repoID: Self.repoID, revision: Self.commit, cacheDirectory: cacheRoot)
        )
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: Self.repoID, revision: Self.commit, cacheDirectory: cacheRoot),
            .ready
        )
        let fileRequests = PinnedHubURLProtocol.requestedURLs.filter { !$0.contains("/api/models/") }
        XCTAssertFalse(fileRequests.isEmpty)
        XCTAssertTrue(fileRequests.allSatisfy { $0.contains("/resolve/\(Self.commit)/") }, "\(fileRequests)")

        let markerData = try Data(contentsOf: directory.appending(path: ".openmed-artifact-ready"))
        let marker = try XCTUnwrap(JSONSerialization.jsonObject(with: markerData) as? [String: String])
        XCTAssertEqual(marker["state"], "ready")
        XCTAssertEqual(marker["revision_sha"], Self.commit)
        XCTAssertEqual(marker["reproducibility_hash"], entry.reproducibilityHash)
        XCTAssertEqual(marker["requested_revision"], "main")
        XCTAssertEqual(
            try OpenMedModelStore.cachedMLXModelRevisionSHA(
                repoID: Self.repoID, revision: Self.commit, cacheDirectory: cacheRoot),
            Self.commit
        )
        // The moving branch name is never used as a cache key for pinned downloads.
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(repoID: Self.repoID, revision: "main", cacheDirectory: cacheRoot),
            .missing
        )
    }

    func testVerifiedSnapshotIsReusedWithoutNetwork() async throws {
        let entry = Self.entry()
        PinnedHubURLProtocol.install(modelInfo: (200, Self.modelInfo()), files: Self.snapshotFiles())
        let first = try await OpenMedModelStore.downloadMLXModel(
            entry: entry, cacheDirectory: cacheRoot, session: Self.session())

        PinnedHubURLProtocol.install(modelInfo: (500, Data()), files: [:])
        let second = try await OpenMedModelStore.downloadMLXModel(
            entry: entry, cacheDirectory: cacheRoot, session: Self.session())

        XCTAssertEqual(second, first)
        XCTAssertEqual(PinnedHubURLProtocol.requestedURLs, [])
        XCTAssertEqual(try OpenMedModelStore.cachedMLXModelDirectory(entry: entry, cacheDirectory: cacheRoot), first)
        let other = OpenMedModelCatalogEntry(
            repoID: Self.repoID, formats: ["mlx-fp"], reproducibilityHash: "sha256:" + String(repeating: "0", count: 64))
        XCTAssertNil(try OpenMedModelStore.cachedMLXModelDirectory(entry: other, cacheDirectory: cacheRoot))
    }

    func testUnpinnedDownloadRecordsNoCommit() async throws {
        PinnedHubURLProtocol.install(modelInfo: (404, Data()), files: Self.snapshotFiles())

        _ = try await OpenMedModelStore.downloadMLXModel(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .ready)
        XCTAssertNil(try OpenMedModelStore.cachedMLXModelRevisionSHA(repoID: Self.repoID, cacheDirectory: cacheRoot))
        XCTAssertNil(try OpenMedModelStore.cachedMLXModelDirectory(entry: Self.entry(), cacheDirectory: cacheRoot))
    }

    func testMatchingHashWithoutACommitShaIsRefused() async throws {
        let entry = OpenMedModelCatalogEntry(
            repoID: Self.repoID,
            formats: ["mlx-fp"],
            reproducibilityHash: OpenMedModelIntegrity.reproducibilityHash(
                repoID: Self.repoID, revisionSHA: nil, released: Self.lastModified, siblings: Self.siblings)
        )
        PinnedHubURLProtocol.install(modelInfo: (200, Self.modelInfo(sha: nil)), files: Self.snapshotFiles())

        do {
            _ = try await OpenMedModelStore.downloadMLXModel(
                entry: entry, cacheDirectory: cacheRoot, session: Self.session())
            XCTFail("a snapshot without a commit sha cannot be pinned")
        } catch {
            XCTAssertEqual(error as? OpenMedModelStoreError, .invalidModelInfo)
        }
        XCTAssertEqual(PinnedHubURLProtocol.requestedURLs.count, 1)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cacheRoot.path), [])
    }

    func testModelInfoHTTPFailureWritesNothing() async throws {
        PinnedHubURLProtocol.install(modelInfo: (404, Data()), files: Self.snapshotFiles())

        do {
            _ = try await OpenMedModelStore.downloadMLXModel(
                entry: Self.entry(), cacheDirectory: cacheRoot, session: Self.session())
            XCTFail("a missing model info document must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedModelStoreError, .httpError(statusCode: 404))
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: cacheRoot.path), [])
    }

    // MARK: - Helpers

    private static func entry() -> OpenMedModelCatalogEntry {
        OpenMedModelCatalogEntry(
            repoID: repoID,
            formats: ["mlx-fp"],
            reproducibilityHash: OpenMedModelIntegrity.reproducibilityHash(
                repoID: repoID, revisionSHA: commit, released: lastModified, siblings: siblings)
        )
    }

    private static func modelInfo(sha: String? = commit, siblings: [String] = siblings) -> Data {
        let object: [String: Any] = [
            "id": repoID,
            "sha": sha.map { $0 as Any } ?? NSNull(),
            "lastModified": lastModified,
            "siblings": siblings.map { ["rfilename": $0] },
        ]
        return try! JSONSerialization.data(withJSONObject: object)
    }

    private static func snapshotFiles() -> [String: Data] {
        [
            "config.json": Data(#"{"model_type":"bert"}"#.utf8),
            "weights.safetensors": Data([0x01, 0x02, 0x03]),
            "tokenizer.json": Data(#"{"version":"1.0"}"#.utf8),
        ]
    }

    private static func modelInfoJSON(_ vector: Vectors.Case) throws -> Data {
        var object: [String: Any] = [
            "id": vector.repoId,
            "siblings": vector.siblings.map { ["rfilename": $0] },
        ]
        object["sha"] = vector.sha.map { $0 as Any } ?? NSNull()
        object["lastModified"] = vector.lastModified.map { $0 as Any } ?? NSNull()
        object["createdAt"] = vector.createdAt.map { $0 as Any } ?? NSNull()
        return try JSONSerialization.data(withJSONObject: object)
    }

    private static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [PinnedHubURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private static func loadVectors() throws -> Vectors {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return try decoder.decode(Vectors.self, from: Data(contentsOf: url.appending(path: vectorsPath)))
    }
}
