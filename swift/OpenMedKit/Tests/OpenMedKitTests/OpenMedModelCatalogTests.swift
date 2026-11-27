import Foundation
import XCTest

@testable import OpenMedKit

/// Fails and counts every request, so tests can prove the catalog is offline.
private final class CatalogNoNetworkURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var count = 0

    static var requestCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return count
    }

    static func reset() {
        lock.lock()
        defer { lock.unlock() }
        count = 0
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        Self.count += 1
        Self.lock.unlock()
        client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
    }
    override func stopLoading() {}
}

final class OpenMedModelCatalogTests: XCTestCase {
    private struct Vectors: Decodable {
        struct Query: Decodable {
            let format: String?
            let tier: String?
            let maxParamCount: Int64?
            let language: String?
            let license: String?
        }

        struct Case: Decodable {
            let id: String
            let query: Query
            let expectedRepoIds: [String]
        }

        struct Lookup: Decodable {
            let repoId: String
            let found: Bool
        }

        let schema: String
        let version: Int
        let catalog: String
        let cases: [Case]
        let lookups: [Lookup]
    }

    static let vectorsPath = "tests/fixtures/model_catalog/catalog_filter_vectors.json"
    static let sharedFilterVectorsPath = "tests/fixtures/parity/catalog_filters_v1.json"
    static let resourcePath = "swift/OpenMedKit/Sources/OpenMedKit/Resources/openmed_model_catalog.jsonl"

    override func setUp() {
        super.setUp()
        CatalogNoNetworkURLProtocol.reset()
        URLProtocol.registerClass(CatalogNoNetworkURLProtocol.self)
    }

    override func tearDown() {
        URLProtocol.unregisterClass(CatalogNoNetworkURLProtocol.self)
        super.tearDown()
    }

    func testNoNetworkStubObservesSharedSessionRequests() async {
        // Negative control: the zero-request assertion below only means
        // something if the stub sees shared-session traffic.
        _ = try? await URLSession.shared.data(from: URL(string: "https://catalog-stub.invalid/probe")!)
        XCTAssertEqual(CatalogNoNetworkURLProtocol.requestCount, 1)
    }

    func testBundledCatalogLoadsWithoutNetwork() throws {
        let catalog = try OpenMedModelCatalog.bundled()

        XCTAssertFalse(catalog.entries.isEmpty)
        XCTAssertEqual(CatalogNoNetworkURLProtocol.requestCount, 0)
    }

    func testBundledCatalogIsTheCommittedResource() throws {
        let committed = try OpenMedModelCatalog(jsonLines: Self.repositoryBytes(Self.resourcePath))

        XCTAssertEqual(try OpenMedModelCatalog.bundled().entries, committed.entries)
    }

    func testEveryBundledEntryCarriesItsReproducibilityHash() throws {
        let hex = CharacterSet(charactersIn: "0123456789abcdef")
        for entry in try OpenMedModelCatalog.bundled().entries {
            XCTAssertTrue(entry.reproducibilityHash.hasPrefix("sha256:"), entry.repoID)
            let digest = entry.reproducibilityHash.dropFirst("sha256:".count)
            XCTAssertEqual(digest.count, 64, entry.repoID)
            XCTAssertTrue(digest.unicodeScalars.allSatisfy(hex.contains), entry.repoID)
        }
    }

    func testBundledEntriesListOnlyAppleRunnableFormats() throws {
        let entries = try OpenMedModelCatalog.bundled().entries
        XCTAssertFalse(entries.isEmpty)
        for entry in entries {
            XCTAssertFalse(entry.formats.isEmpty, entry.repoID)
            for format in entry.formats {
                let token = format.lowercased()
                XCTAssertTrue(token.hasPrefix("mlx") || token.hasPrefix("coreml"), entry.repoID)
            }
        }
        XCTAssertFalse(try OpenMedModelCatalog.bundled().filter(format: "mlx-fp").isEmpty)
        XCTAssertTrue(try OpenMedModelCatalog.bundled().filter(format: "onnx").isEmpty)
    }

    func testFilterVectorsMatchThePythonGenerator() throws {
        let vectors = try Self.loadVectors()
        XCTAssertEqual(vectors.schema, "openmed.model_catalog_filter_vectors")
        XCTAssertEqual(vectors.version, 1)
        let catalog = try OpenMedModelCatalog(jsonLines: Self.repositoryBytes(vectors.catalog))
        XCTAssertFalse(vectors.cases.isEmpty)

        for vector in vectors.cases {
            let query = vector.query
            let ids = catalog.filter(
                format: query.format,
                tier: query.tier,
                maxParamCount: query.maxParamCount,
                language: query.language,
                license: query.license
            ).map(\.repoID)
            XCTAssertEqual(ids, vector.expectedRepoIds, vector.id)
        }
        for lookup in vectors.lookups {
            XCTAssertEqual(catalog.entry(repoID: lookup.repoId) != nil, lookup.found, lookup.repoId)
        }
    }

    func testSharedCatalogFilterParityVectors() throws {
        // The cross-runtime filter contract that Android's CatalogFilterParityTest also runs.
        struct SharedVectors: Decodable {
            let schema: String
            let version: Int
            let syntheticOnly: Bool
            let cases: [Vectors.Case]
            let lookups: [Vectors.Lookup]
        }
        let data = try Self.repositoryBytes(Self.sharedFilterVectorsPath)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let vectors = try decoder.decode(SharedVectors.self, from: data)
        XCTAssertEqual(vectors.schema, "openmed.model_catalog_filters")
        XCTAssertEqual(vectors.version, 1)
        XCTAssertTrue(vectors.syntheticOnly)

        // Load the rows through the JSON Lines parser, as Android loads them with fromJsonLines.
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let rows = try XCTUnwrap(document["rows"] as? [Any])
        let lines = try rows.map { String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self) }
        let catalog = try OpenMedModelCatalog(jsonLines: Data(lines.joined(separator: "\n").utf8))
        XCTAssertEqual(catalog.entries.count, rows.count)
        XCTAssertFalse(vectors.cases.isEmpty)

        for vector in vectors.cases {
            let query = vector.query
            let ids = catalog.filter(
                format: query.format,
                tier: query.tier,
                maxParamCount: query.maxParamCount,
                language: query.language,
                license: query.license
            ).map(\.repoID)
            XCTAssertEqual(ids, vector.expectedRepoIds, vector.id)
        }
        XCTAssertFalse(vectors.lookups.isEmpty)
        for lookup in vectors.lookups {
            XCTAssertEqual(catalog.entry(repoID: lookup.repoId)?.repoID, lookup.found ? lookup.repoId : nil, lookup.repoId)
        }
    }

    func testEntryDecodesSnakeCaseFieldsAndDefaultsMissingLists() throws {
        let line = #"{"repo_id":"OpenMed/Synthetic-mlx","param_count":3000000000,"reproducibility_hash":"sha256:abc"}"#
        let catalog = try OpenMedModelCatalog(jsonLines: Data(line.utf8))

        XCTAssertEqual(
            catalog.entries,
            [
                OpenMedModelCatalogEntry(
                    repoID: "OpenMed/Synthetic-mlx",
                    formats: [],
                    paramCount: 3_000_000_000,
                    reproducibilityHash: "sha256:abc"
                )
            ]
        )
    }

    func testEntryEncodesTheCatalogFieldNames() throws {
        let entry = OpenMedModelCatalogEntry(
            repoID: "OpenMed/Synthetic-mlx",
            formats: ["mlx-fp"],
            tier: "Tiny",
            paramCount: 22_000_000,
            languages: ["en"],
            license: "apache-2.0",
            reproducibilityHash: "sha256:abc"
        )
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(entry)) as? [String: Any]
        )

        XCTAssertEqual(
            Set(object.keys),
            ["repo_id", "formats", "tier", "param_count", "languages", "license", "reproducibility_hash"]
        )
        XCTAssertEqual(try OpenMedModelCatalog(jsonLines: JSONEncoder().encode(entry)).entries, [entry])
    }

    func testMalformedLineReportsItsLineNumberOnly() {
        let sentinel = "OpenMed/Sentinel-Value-mlx"
        let text = "\n{\"repo_id\":\"\(sentinel)\",\"formats\":[\"mlx-fp\"]}\n"

        XCTAssertThrowsError(try OpenMedModelCatalog(jsonLines: Data(text.utf8))) { error in
            XCTAssertEqual(error as? OpenMedModelCatalogError, .invalidEntry(line: 2))
            XCTAssertEqual((error as? OpenMedModelCatalogError)?.reason, "catalog_invalid_entry")
            XCTAssertFalse(error.localizedDescription.contains(sentinel))
            XCTAssertFalse(String(describing: error).contains(sentinel))
        }
    }

    func testNonUTF8CatalogIsRejected() {
        XCTAssertThrowsError(try OpenMedModelCatalog(jsonLines: Data([0xFF, 0xFE, 0x00]))) { error in
            XCTAssertEqual(error as? OpenMedModelCatalogError, .invalidEncoding)
        }
    }

    // MARK: - Fixture loading

    private static func loadVectors() throws -> Vectors {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Vectors.self, from: repositoryBytes(vectorsPath))
    }

    /// Read a file relative to the repository root.
    private static func repositoryBytes(_ path: String) throws -> Data {
        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<5 { url.deleteLastPathComponent() }
        return try Data(contentsOf: url.appending(path: path))
    }
}
