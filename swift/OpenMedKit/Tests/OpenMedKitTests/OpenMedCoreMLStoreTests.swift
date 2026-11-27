import CoreML
import CryptoKit
import Foundation
import XCTest

@testable import OpenMedKit

/// Serves a synthetic Core ML snapshot keyed by repository-relative path.
/// URLProtocol instances are created by the loading system, so state is
/// static and every access holds `lock`.
private final class CoreMLSnapshotURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var files: [String: Data] = [:]
    nonisolated(unsafe) private static var paths: [String] = []

    static var requestedPaths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return paths
    }

    static func install(files: [String: Data]) {
        lock.lock()
        defer { lock.unlock() }
        self.files = files
        paths = []
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        guard let url = request.url else { return }
        let path = url.path.components(separatedBy: "/resolve/main/").last ?? ""
        Self.lock.lock()
        Self.paths.append(path)
        let data = Self.files[path]
        Self.lock.unlock()
        let response = HTTPURLResponse(url: url, statusCode: data == nil ? 404 : 200, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data ?? Data())
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class OpenMedCoreMLStoreTests: XCTestCase {
    private static let repoID = "OpenMed/Synthetic-Tiny-coreml"
    private var cacheRoot: URL!

    override func setUpWithError() throws {
        cacheRoot = FileManager.default.temporaryDirectory.appending(
            path: "openmed-coreml-store-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: cacheRoot)
    }

    // MARK: - Ready layouts

    #if os(macOS)
        func testPackageSnapshotCompilesIntoAReadyDirectoryOpenMedCanLoad() async throws {
            CoreMLSnapshotURLProtocol.install(files: try Self.packageSnapshot())

            let model = try await OpenMedModelStore.downloadCoreMLModel(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

            XCTAssertEqual(
                try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .ready)
            XCTAssertEqual(model.modelURL.pathExtension, "mlmodelc")
            XCTAssertEqual(model.modelURL.deletingLastPathComponent().lastPathComponent, ".openmed-compiled")
            let package = try XCTUnwrap(model.packageURL)
            XCTAssertTrue(FileManager.default.fileExists(atPath: package.appending(path: "Manifest.json").path))
            XCTAssertEqual(model.tokenizerFolderURL, model.directoryURL)
            XCTAssertEqual(model.id2labelURL.lastPathComponent, "id2label.json")
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: model.directoryURL.appending(path: ".openmed-artifact-ready").path))
            XCTAssertNoThrow(try MLModel(contentsOf: model.modelURL))
            #if canImport(MLX) && canImport(Tokenizers)
                XCTAssertNoThrow(try OpenMed(backend: model.backend))
            #endif

            XCTAssertEqual(
                try OpenMedModelStore.cachedCoreMLModel(repoID: Self.repoID, cacheDirectory: cacheRoot), model)
            CoreMLSnapshotURLProtocol.install(files: [:])
            let again = try await OpenMedModelStore.downloadCoreMLModel(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())
            XCTAssertEqual(again, model)
            XCTAssertEqual(CoreMLSnapshotURLProtocol.requestedPaths, [])
        }

        func testPackageIsCompiledOnlyAfterEveryFileDownloaded() async throws {
            var files = try Self.packageSnapshot()
            files["tokenizer_config.json"] = nil
            CoreMLSnapshotURLProtocol.install(files: files)

            await assertThrowsAsync(
                try await OpenMedModelStore.downloadCoreMLModel(
                    repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())
            ) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .httpError(statusCode: 404))
            }
            let directory = try OpenMedModelStore.cachedCoreMLModelDirectory(
                repoID: Self.repoID, cacheDirectory: cacheRoot)
            XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: ".openmed-compiled").path))
            XCTAssertEqual(
                try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .partial)
        }
    #endif

    #if os(watchOS)
        func testPackageSnapshotIsRefusedOnWatchOSBeforeItsFilesDownload() async throws {
            CoreMLSnapshotURLProtocol.install(files: try Self.packageSnapshot())

            await assertThrowsAsync(
                try await OpenMedModelStore.downloadCoreMLModel(
                    repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())
            ) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .coreMLCompilationFailed)
            }
            XCTAssertEqual(
                CoreMLSnapshotURLProtocol.requestedPaths, ["openmed-coreml.json", "Tiny.mlpackage/openmed-coreml.json"])
            XCTAssertNotEqual(
                try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .ready)
        }
    #endif

    func testCompiledVariantWithDeclaredFilesBecomesReadyWithoutCompiling() async throws {
        CoreMLSnapshotURLProtocol.install(files: Self.compiledSnapshot())

        let model = try await OpenMedModelStore.downloadCoreMLModel(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        XCTAssertEqual(
            try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .ready)
        XCTAssertNil(model.packageURL)
        XCTAssertEqual(model.modelURL.lastPathComponent, "Tiny.mlmodelc")
        XCTAssertEqual(model.id2labelURL.lastPathComponent, "Tiny_id2label.json")
        XCTAssertEqual(model.tokenizerFolderURL.lastPathComponent, "tokenizer")
        XCTAssertFalse(FileManager.default.fileExists(atPath: model.directoryURL.appending(path: ".openmed-compiled").path))
        // The declared tokenizer list replaces the optional defaults.
        XCTAssertFalse(CoreMLSnapshotURLProtocol.requestedPaths.contains("config.json"))
        // The Core ML cache is separate from the MLX cache.
        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot.appending(path: "mlx")),
            .missing
        )
    }

    // MARK: - Partial layouts

    func testMissingLabelMapOrTokenizerLeavesTheSnapshotPartial() async throws {
        for missing in ["Tiny_id2label.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            let root = cacheRoot.appending(path: UUID().uuidString)
            var files = Self.compiledSnapshot()
            files[missing] = nil
            CoreMLSnapshotURLProtocol.install(files: files)

            await assertThrowsAsync(
                try await OpenMedModelStore.downloadCoreMLModel(
                    repoID: Self.repoID, cacheDirectory: root, session: Self.session()),
                missing
            ) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .httpError(statusCode: 404), missing)
            }
            XCTAssertEqual(
                try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: root), .partial, missing)
            XCTAssertNil(try OpenMedModelStore.cachedCoreMLModel(repoID: Self.repoID, cacheDirectory: root), missing)
            let directory = try OpenMedModelStore.cachedCoreMLModelDirectory(repoID: Self.repoID, cacheDirectory: root)
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: directory.appending(path: ".openmed-artifact-ready").path), missing)
        }
    }

    func testRemovingARequiredFileDropsAReadySnapshotToPartial() async throws {
        CoreMLSnapshotURLProtocol.install(files: Self.compiledSnapshot())
        let model = try await OpenMedModelStore.downloadCoreMLModel(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        try FileManager.default.removeItem(at: model.tokenizerFolderURL.appending(path: "tokenizer.json"))

        XCTAssertEqual(
            try OpenMedModelStore.coreMLModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .partial)
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: model.directoryURL.appending(path: ".openmed-artifact-ready").path))
    }

    // MARK: - Confinement and shape

    func testManifestPathsThatEscapeTheSnapshotAreRefused() async throws {
        let compiled: [String: Any] = ["name": "coreml", "path": "Tiny.mlmodelc", "files": ["model.mil"]]
        let cases: [(String, [String: Any])] = [
            ("variant path", ["variants": [["name": "coreml", "path": "../Escape.mlpackage"]]]),
            ("absolute variant path", ["variants": [["name": "coreml", "path": "/tmp/Escape.mlmodelc", "files": ["a"]]]]),
            ("variant file", ["variants": [["name": "coreml", "path": "Tiny.mlmodelc", "files": ["../../escape.bin"]]]]),
            ("label map", ["variants": [compiled], "id2label": "../id2label.json"]),
            ("tokenizer directory", ["variants": [compiled], "tokenizer": ["path": "..", "files": ["tokenizer.json"]]]),
            ("tokenizer file", ["variants": [compiled], "tokenizer": ["path": ".", "files": ["../tokenizer.json"]]]),
            ("backslash", ["variants": [compiled], "id2label": "labels\\id2label.json"]),
        ]
        for (name, fields) in cases {
            let root = cacheRoot.appending(path: UUID().uuidString)
            var files = Self.compiledSnapshot()
            files["openmed-coreml.json"] = try Self.manifest(fields)
            CoreMLSnapshotURLProtocol.install(files: files)

            await assertThrowsAsync(
                try await OpenMedModelStore.downloadCoreMLModel(
                    repoID: Self.repoID, cacheDirectory: root, session: Self.session()),
                name
            ) { error in
                guard case .invalidManifestPath = error as? OpenMedModelStoreError else {
                    return XCTFail("\(name): expected a confinement refusal, got \(error)")
                }
                XCTAssertFalse(String(describing: error).contains(".."), name)
            }
            XCTAssertEqual(CoreMLSnapshotURLProtocol.requestedPaths, ["openmed-coreml.json"], name)
            let directory = try OpenMedModelStore.cachedCoreMLModelDirectory(repoID: Self.repoID, cacheDirectory: root)
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: directory.path), [], name)
        }
    }

    func testPackageInventoryPathsThatEscapeAreRefused() async throws {
        var files = try Self.packageSnapshot()
        files["Tiny.mlpackage/openmed-coreml.json"] = try Self.packageInventory(["../../escape.bin"])
        CoreMLSnapshotURLProtocol.install(files: files)

        await assertThrowsAsync(
            try await OpenMedModelStore.downloadCoreMLModel(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())
        ) { error in
            guard case .invalidManifestPath = error as? OpenMedModelStoreError else {
                return XCTFail("expected a confinement refusal, got \(error)")
            }
        }
        XCTAssertEqual(
            CoreMLSnapshotURLProtocol.requestedPaths, ["openmed-coreml.json", "Tiny.mlpackage/openmed-coreml.json"])
        let directory = try OpenMedModelStore.cachedCoreMLModelDirectory(repoID: Self.repoID, cacheDirectory: cacheRoot)
        let remaining = FileManager.default.enumerator(atPath: directory.path)?.allObjects as? [String] ?? []
        XCTAssertFalse(remaining.contains { $0.hasSuffix(".json") }, "\(remaining)")
    }

    func testUnsupportedManifestShapesAreRejected() async throws {
        let cases: [(String, [String: Any])] = [
            ("wrong format", ["format": "openmed-mlx", "variants": [["name": "a", "path": "Tiny.mlpackage"]]]),
            ("no variants", ["variants": [] as [Any]]),
            ("not a Core ML directory", ["variants": [["name": "a", "path": "model.onnx", "files": ["x"]]]]),
            ("compiled model without files", ["variants": [["name": "a", "path": "Tiny.mlmodelc"]]]),
        ]
        for (name, fields) in cases {
            let root = cacheRoot.appending(path: UUID().uuidString)
            CoreMLSnapshotURLProtocol.install(files: ["openmed-coreml.json": try Self.manifest(fields)])

            await assertThrowsAsync(
                try await OpenMedModelStore.downloadCoreMLModel(
                    repoID: Self.repoID, cacheDirectory: root, session: Self.session()),
                name
            ) { error in
                XCTAssertEqual(error as? OpenMedModelStoreError, .invalidCoreMLManifest, name)
                XCTAssertEqual((error as? OpenMedModelStoreError)?.reason, "invalid_coreml_manifest", name)
            }
            XCTAssertEqual(CoreMLSnapshotURLProtocol.requestedPaths, ["openmed-coreml.json"], name)
        }
    }

    // MARK: - Fixtures

    private static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [CoreMLSnapshotURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private static func manifest(_ fields: [String: Any]) throws -> Data {
        var object: [String: Any] = ["format": "openmed-coreml", "schema_version": 1]
        object.merge(fields) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// `.mlpackage` snapshot whose package manifest inventories its files.
    private static func packageSnapshot() throws -> [String: Data] {
        let model = tinyModelSpecification()
        let packageManifest = Data(
            #"{"fileFormatVersion":"1.0.0","itemInfoEntries":{"A":{"author":"com.apple.CoreML","description":"CoreML Model Specification","name":"model.mlmodel","path":"com.apple.CoreML/model.mlmodel"}},"rootModelIdentifier":"A"}"#
                .utf8)
        var files = try tokenizerFiles(prefix: "")
        files["openmed-coreml.json"] = try manifest(["variants": [["name": "coreml-fp16", "path": "Tiny.mlpackage"]]])
        files["Tiny.mlpackage/openmed-coreml.json"] = try packageInventory(
            ["Data/com.apple.CoreML/model.mlmodel", "Manifest.json"], contents: [model, packageManifest])
        files["Tiny.mlpackage/Manifest.json"] = packageManifest
        files["Tiny.mlpackage/Data/com.apple.CoreML/model.mlmodel"] = model
        files["id2label.json"] = Data(#"{"0":"O","1":"B-NAME","2":"I-NAME"}"#.utf8)
        return files
    }

    /// `.mlmodelc` snapshot that declares its files, label map and tokenizer.
    private static func compiledSnapshot() -> [String: Data] {
        var files = (try? tokenizerFiles(prefix: "tokenizer/")) ?? [:]
        files["openmed-coreml.json"] = try? manifest([
            "variants": [["name": "coreml-compiled", "path": "Tiny.mlmodelc", "files": ["model.mil", "coremldata.bin"]]],
            "id2label": "Tiny_id2label.json",
            "tokenizer": ["path": "tokenizer", "files": ["tokenizer.json", "tokenizer_config.json"]],
        ])
        files["Tiny.mlmodelc/model.mil"] = Data("program(1.0)".utf8)
        files["Tiny.mlmodelc/coremldata.bin"] = Data([0x00, 0x01])
        files["Tiny_id2label.json"] = Data(#"{"0":"O","1":"B-NAME"}"#.utf8)
        return files
    }

    private static func packageInventory(_ paths: [String], contents: [Data]? = nil) throws -> Data {
        let records = paths.enumerated().map { index, path -> [String: Any] in
            let data = contents?[index] ?? Data()
            return [
                "path": path, "size_bytes": data.count,
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
            ]
        }
        return try JSONSerialization.data(
            withJSONObject: [
                "format": "openmed-coreml", "schema_version": 2,
                "variants": [["name": "coreml-fp16", "path": "."]],
                "provenance": ["schema_version": 1, "files": records],
            ],
            options: [.sortedKeys])
    }

    /// A valid, minimal Core ML neural-network specification: one ReLU layer
    /// from `input_ids` to `logits`, hand-encoded as protobuf so the test
    /// needs no converter and no bundled model.
    private static func tinyModelSpecification() -> Data {
        func varint(_ value: UInt64) -> [UInt8] {
            var value = value
            var bytes: [UInt8] = []
            repeat {
                var byte = UInt8(value & 0x7F)
                value >>= 7
                if value != 0 { byte |= 0x80 }
                bytes.append(byte)
            } while value != 0
            return bytes
        }
        func field(_ number: UInt64, _ payload: [UInt8]) -> [UInt8] {
            varint(number << 3 | 2) + varint(UInt64(payload.count)) + payload
        }
        func number(_ field: UInt64, _ value: UInt64) -> [UInt8] { varint(field << 3) + varint(value) }
        func text(_ number: UInt64, _ value: String) -> [UInt8] { field(number, Array(value.utf8)) }
        func feature(_ name: String) -> [UInt8] {
            // FeatureType.multiArrayType { shape: [1], dataType: FLOAT32 }
            text(1, name) + field(3, field(5, number(1, 1) + number(2, 65_568)))
        }
        let description = field(1, feature("input_ids")) + field(10, feature("logits"))
        let layer = text(1, "relu") + text(2, "input_ids") + text(3, "logits") + field(130, field(10, []))
        return Data(number(1, 4) + field(2, description) + field(500, field(1, layer)))
    }

    private static func tokenizerFiles(prefix: String) throws -> [String: Data] {
        let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        var vocabulary: [String: Int] = ["patient": 5, "ada": 6]
        for (index, token) in specials.enumerated() { vocabulary[token] = index }
        let tokenizer: [String: Any] = [
            "version": "1.0",
            "added_tokens": specials.enumerated().map { index, token in
                [
                    "id": index, "content": token, "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true,
                ] as [String: Any]
            },
            "normalizer": ["type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "lowercase": true],
            "pre_tokenizer": ["type": "BertPreTokenizer"],
            "decoder": ["type": "WordPiece", "prefix": "##", "cleanup": true],
            "model": [
                "type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100, "vocab": vocabulary,
            ],
        ]
        let config: [String: Any] = [
            "tokenizer_class": "BertTokenizer", "do_lower_case": true, "unk_token": "[UNK]", "cls_token": "[CLS]",
            "sep_token": "[SEP]", "pad_token": "[PAD]", "mask_token": "[MASK]", "model_max_length": 32,
        ]
        return [
            prefix + "tokenizer.json": try JSONSerialization.data(withJSONObject: tokenizer, options: [.sortedKeys]),
            prefix + "tokenizer_config.json": try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]),
            prefix + "config.json": Data(#"{"model_type":"bert"}"#.utf8),
        ]
    }
}

/// `XCTAssertThrowsError` for async expressions.
private func assertThrowsAsync<T>(
    _ expression: @autoclosure () async throws -> T,
    _ message: String = "",
    file: StaticString = #filePath,
    line: UInt = #line,
    _ handler: (any Error) -> Void = { _ in }
) async {
    do {
        _ = try await expression()
        XCTFail("expected an error \(message)", file: file, line: line)
    } catch {
        handler(error)
    }
}
