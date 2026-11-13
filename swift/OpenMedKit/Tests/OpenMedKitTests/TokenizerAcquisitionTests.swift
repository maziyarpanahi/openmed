#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import Tokenizers
    import XCTest

    @testable import OpenMedKit

    /// Fails every request and counts it, so tests can prove zero traffic.
    private final class FailingTokenizerURLProtocol: URLProtocol {
        private static let lock = NSLock()
        private static var count = 0

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

    /// Serves synthetic tokenizer files keyed by the last path component.
    private final class PinnedTokenizerURLProtocol: URLProtocol {
        private static let lock = NSLock()
        private static var files: [String: Data] = [:]
        private static var urls: [URL] = []

        static var requestedURLs: [URL] {
            lock.lock()
            defer { lock.unlock() }
            return urls
        }

        static func install(files: [String: Data]) {
            lock.lock()
            defer { lock.unlock() }
            self.files = files
            urls = []
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            guard let url = request.url else { return }
            Self.lock.lock()
            Self.urls.append(url)
            let data = Self.files[url.lastPathComponent]
            Self.lock.unlock()
            let response = HTTPURLResponse(
                url: url,
                statusCode: data == nil ? 404 : 200,
                httpVersion: nil,
                headerFields: nil
            )!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data ?? Data())
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    final class TokenizerAcquisitionTests: XCTestCase {
        private static let repositoryID = "OpenMed/Synthetic-Tokenizer"
        private static let revision = "0123456789abcdef0123456789abcdef01234567"
        private static let text = "Patient Ada visited clinics"
        private static let expectedIDs = [2, 5, 6, 7, 8, 9, 3]

        private var scratch: URL!

        override func setUpWithError() throws {
            scratch = FileManager.default.temporaryDirectory.appending(
                path: "openmed-tokenizer-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
            try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
            FailingTokenizerURLProtocol.reset()
            URLProtocol.registerClass(FailingTokenizerURLProtocol.self)
        }

        override func tearDownWithError() throws {
            URLProtocol.unregisterClass(FailingTokenizerURLProtocol.self)
            try? FileManager.default.removeItem(at: scratch)
        }

        // MARK: - Zero-request defaults

        func testGlobalStubObservesSharedSessionRequests() async {
            // Negative control: the zero-request assertions below are only
            // meaningful if the stub sees shared-session traffic.
            let url = URL(string: "https://tokenizer-stub.invalid/probe")!
            _ = try? await URLSession.shared.data(from: url)
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 1)
        }

        func testLocalAssetsLoadWithZeroRequests() throws {
            let folder = try makeTokenizerFolder()

            let tokenizer = try OpenMed.loadTokenizer(
                tokenizerName: nil,
                tokenizerFolderURL: folder
            )

            XCTAssertEqual(tokenizer.encode(text: Self.text), Self.expectedIDs)
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testAsyncLoaderLoadsLocalAssetsWithZeroRequests() async throws {
            let folder = try makeTokenizerFolder()

            let tokenizer = try await OpenMed.loadTokenizerAsync(
                tokenizerName: "OpenMed/Any-Declared-Name",
                tokenizerFolderURL: folder
            )

            XCTAssertEqual(tokenizer.encode(text: Self.text), Self.expectedIDs)
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testOpenMedResolvesLocalTokenizerBeforeModelWithZeroRequests() throws {
            let folder = try makeTokenizerFolder()
            let backend = OpenMedBackend.coreML(
                modelURL: scratch.appending(path: "Missing.mlmodelc"),
                id2labelURL: scratch.appending(path: "id2label.json"),
                tokenizerFolderURL: folder
            )

            // The tokenizer stage succeeds offline; only the absent synthetic
            // model fails afterwards.
            XCTAssertThrowsError(try OpenMed(backend: backend)) { error in
                XCTAssertNil(error as? OpenMedTokenizerError, "Unexpected tokenizer error: \(error)")
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testOpenMedFailsClosedWithoutAssetsAndMakesNoRequests() async throws {
            let missingFolder = scratch.appending(path: "absent", directoryHint: .isDirectory)
            let backends: [OpenMedBackend] = [
                .coreML(
                    modelURL: scratch.appending(path: "Missing.mlmodelc"),
                    id2labelURL: scratch.appending(path: "id2label.json")
                ),
                .coreML(
                    modelURL: scratch.appending(path: "Missing.mlmodelc"),
                    id2labelURL: scratch.appending(path: "id2label.json"),
                    tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                    tokenizerFolderURL: missingFolder
                ),
            ]

            for backend in backends {
                XCTAssertThrowsError(try OpenMed(backend: backend)) { error in
                    XCTAssertEqual(error as? OpenMedTokenizerError, .missingLocalAssets)
                    self.assertValueFree(error)
                }
                do {
                    _ = try await OpenMed.load(backend: backend)
                    XCTFail("Expected missing tokenizer assets")
                } catch {
                    XCTAssertEqual(error as? OpenMedTokenizerError, .missingLocalAssets)
                }
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testCoreMLBackendHasNoImplicitTokenizerIdentity() {
            let backend = OpenMedBackend.coreML(
                modelURL: scratch.appending(path: "Model.mlmodelc"),
                id2labelURL: scratch.appending(path: "id2label.json")
            )

            guard case .coreML(_, _, let tokenizerName, let tokenizerFolderURL) = backend else {
                return XCTFail("Expected a Core ML backend")
            }
            XCTAssertNil(tokenizerName)
            XCTAssertNil(tokenizerFolderURL)
        }

        func testLegacyMLXArtifactNoLongerFallsBackToNetwork() throws {
            let directory = try makeLegacyMLXArtifactWithoutTokenizer()

            XCTAssertThrowsError(try OpenMed(backend: .mlx(modelDirectoryURL: directory))) { error in
                XCTAssertEqual(error as? OpenMedTokenizerError, .missingLocalAssets)
                self.assertValueFree(error, forbidden: [directory.path])
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testNetworkOptInWithoutPinFailsBeforeAnyRequest() async {
            do {
                _ = try await OpenMed.loadTokenizerAsync(
                    tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                    tokenizerFolderURL: nil,
                    allowNetworkAccess: true
                )
                XCTFail("Expected a pin requirement")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .pinnedRevisionRequired)
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        // MARK: - Pinned opt-in

        func testPinnedDownloadRequestsOnlyPinnedRevisionAndWritesUnderCacheRoot() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            let cacheRoot = scratch.appending(path: "Caches", directoryHint: .isDirectory)
            let store = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: pinnedSession(files))

            let tokenizer = try await OpenMed.loadTokenizerAsync(
                tokenizerName: Self.repositoryID,
                tokenizerFolderURL: nil,
                allowNetworkAccess: true,
                pin: pin,
                assetStore: store
            )

            XCTAssertEqual(tokenizer.encode(text: Self.text), Self.expectedIDs)
            let requested = PinnedTokenizerURLProtocol.requestedURLs.map(\.absoluteString)
            XCTAssertEqual(
                requested.sorted(),
                files.keys.sorted().map {
                    "https://huggingface.co/\(Self.repositoryID)/resolve/\(Self.revision)/\($0)"
                }
            )
            XCTAssertFalse(requested.contains { $0.contains("/main/") })

            let snapshot = store.directory(for: pin)
            XCTAssertEqual(try Self.relativeFiles(under: cacheRoot), Set(files.keys.map { "OpenMed__Synthetic-Tokenizer/\(Self.revision)/\($0)" }))
            XCTAssertTrue(snapshot.path.hasPrefix(cacheRoot.path))

            // A verified cached copy is reused offline with zero requests.
            let offlineStore = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: failingSession())
            let cached = try await OpenMed.loadTokenizerAsync(
                tokenizerName: nil,
                tokenizerFolderURL: nil,
                allowNetworkAccess: false,
                pin: pin,
                assetStore: offlineStore
            )
            XCTAssertEqual(cached.encode(text: Self.text), Self.expectedIDs)
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testDigestMismatchFailsClosedAndInstallsNothing() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            var tampered = files
            tampered["tokenizer.json"] = Self.flippingOneByte(files["tokenizer.json"]!)
            let cacheRoot = scratch.appending(path: "Caches", directoryHint: .isDirectory)
            let store = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: pinnedSession(tampered))

            do {
                _ = try await OpenMed.loadTokenizerAsync(
                    tokenizerName: nil,
                    tokenizerFolderURL: nil,
                    allowNetworkAccess: true,
                    pin: pin,
                    assetStore: store
                )
                XCTFail("Expected a digest mismatch")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .digestMismatch)
                assertValueFree(error, forbidden: [cacheRoot.path, Self.repositoryID, Self.revision])
            }

            XCTAssertFalse(FileManager.default.fileExists(atPath: store.directory(for: pin).path))
            XCTAssertEqual(try Self.relativeFiles(under: cacheRoot), [])
        }

        func testHTTPFailureFailsClosedAndInstallsNothing() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            var partial = files
            partial["tokenizer_config.json"] = nil
            let cacheRoot = scratch.appending(path: "Caches", directoryHint: .isDirectory)
            let store = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: pinnedSession(partial))

            do {
                _ = try await store.download(pin)
                XCTFail("Expected a download failure")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .downloadFailed(statusCode: 404))
                assertValueFree(error, forbidden: [Self.repositoryID, Self.revision])
            }
            XCTAssertEqual(try Self.relativeFiles(under: cacheRoot), [])
        }

        func testTamperedCacheFailsClosedBeforeUse() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            let cacheRoot = scratch.appending(path: "Caches", directoryHint: .isDirectory)
            let store = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: pinnedSession(files))
            let snapshot = try await store.download(pin)
            try Self.flippingOneByte(files["tokenizer.json"]!)
                .write(to: snapshot.appending(path: "tokenizer.json"))

            let offlineStore = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: failingSession())
            do {
                _ = try await OpenMed.loadTokenizerAsync(
                    tokenizerName: nil,
                    tokenizerFolderURL: nil,
                    pin: pin,
                    assetStore: offlineStore
                )
                XCTFail("Expected a digest mismatch")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .digestMismatch)
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.path))

            // The stale copy is gone; without network the load stays closed.
            do {
                _ = try await OpenMed.loadTokenizerAsync(
                    tokenizerName: nil,
                    tokenizerFolderURL: nil,
                    pin: pin,
                    assetStore: offlineStore
                )
                XCTFail("Expected missing assets")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .missingLocalAssets)
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testUnpinnedFileInCacheFailsClosed() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            let cacheRoot = scratch.appending(path: "Caches", directoryHint: .isDirectory)
            let store = OpenMedTokenizerAssetStore(cacheRoot: cacheRoot, session: pinnedSession(files))
            let snapshot = try await store.download(pin)
            try Data("{}".utf8).write(to: snapshot.appending(path: "special_tokens_map.json"))

            XCTAssertThrowsError(try store.verifiedCachedDirectory(for: pin)) { error in
                XCTAssertEqual(error as? OpenMedTokenizerError, .digestMismatch)
            }
            XCTAssertFalse(FileManager.default.fileExists(atPath: snapshot.path))
        }

        func testPinForAnotherRepositoryIsRejectedBeforeAnyRequest() async throws {
            let files = try Self.tokenizerFiles()
            let pin = try Self.pin(for: files)
            let store = OpenMedTokenizerAssetStore(
                cacheRoot: scratch.appending(path: "Caches", directoryHint: .isDirectory),
                session: failingSession()
            )

            do {
                _ = try await OpenMed.loadTokenizerAsync(
                    tokenizerName: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                    tokenizerFolderURL: nil,
                    allowNetworkAccess: true,
                    pin: pin,
                    assetStore: store
                )
                XCTFail("Expected an identity mismatch")
            } catch {
                XCTAssertEqual(error as? OpenMedTokenizerError, .identityMismatch)
            }
            XCTAssertEqual(FailingTokenizerURLProtocol.requestCount, 0)
        }

        func testPinValidationRejectsMovingRevisionsAndUnsafeFiles() throws {
            let digest = String(repeating: "a", count: 64)
            let required: [OpenMedTokenizerPin.File] = [
                .init(name: "tokenizer.json", sha256: digest),
                .init(name: "tokenizer_config.json", sha256: digest),
            ]
            let cases: [(String, String, [OpenMedTokenizerPin.File], OpenMedTokenizerPinIssue)] = [
                (Self.repositoryID, "main", required, .revision),
                (Self.repositoryID, "v1.0.0", required, .revision),
                (Self.repositoryID, String(Self.revision.dropLast()), required, .revision),
                ("OpenMed", Self.revision, required, .repositoryID),
                ("OpenMed/a/b", Self.revision, required, .repositoryID),
                ("../OpenMed", Self.revision, required, .repositoryID),
                ("https://huggingface.co/OpenMed", Self.revision, required, .repositoryID),
                (Self.repositoryID, Self.revision, required + [.init(name: "../tokenizer.json", sha256: digest)], .fileName),
                (Self.repositoryID, Self.revision, required + [.init(name: "model.safetensors", sha256: digest)], .fileName),
                (Self.repositoryID, Self.revision, required + [required[0]], .duplicateFile),
                (Self.repositoryID, Self.revision, [required[0]], .missingRequiredFile),
                (Self.repositoryID, Self.revision, [required[0], .init(name: "tokenizer_config.json", sha256: "abc")], .digest),
                (Self.repositoryID, Self.revision, [required[0], .init(name: "tokenizer_config.json", sha256: digest, byteCount: 0)], .byteCount),
            ]

            for (repositoryID, revision, files, issue) in cases {
                XCTAssertThrowsError(
                    try OpenMedTokenizerPin(repositoryID: repositoryID, revision: revision, files: files)
                ) { error in
                    XCTAssertEqual(error as? OpenMedTokenizerError, .invalidPin(issue), "\(issue)")
                    // Short probes such as "main" are not identifying values.
                    self.assertValueFree(
                        error,
                        forbidden: [repositoryID, revision].filter { $0.contains("/") || $0.count >= 32 }
                    )
                }
            }

            let pin = try OpenMedTokenizerPin(
                repositoryID: Self.repositoryID,
                revision: Self.revision.uppercased(),
                files: required
            )
            XCTAssertEqual(pin.revision, Self.revision)
        }

        func testVerifyRejectsWrongSize() throws {
            let data = Data("synthetic".utf8)
            let digest = OpenMedTokenizerAssetStore.sha256Hex(data)

            XCTAssertNoThrow(
                try OpenMedTokenizerAssetStore.verify(
                    data,
                    against: .init(name: "tokenizer.json", sha256: digest, byteCount: data.count)
                )
            )
            XCTAssertThrowsError(
                try OpenMedTokenizerAssetStore.verify(
                    data,
                    against: .init(name: "tokenizer.json", sha256: digest, byteCount: data.count + 1)
                )
            ) { error in
                XCTAssertEqual(error as? OpenMedTokenizerError, .sizeMismatch)
            }
        }

        func testDefaultCacheRootIsUnderCachesNotDocuments() throws {
            let root = try OpenMedTokenizerAssetStore.defaultCacheRoot().standardizedFileURL.path
            let caches = try XCTUnwrap(
                FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ).standardizedFileURL.path
            let documents = try XCTUnwrap(
                FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
            ).standardizedFileURL.path

            XCTAssertTrue(root.hasPrefix(caches), "Tokenizer cache must live under Caches")
            XCTAssertFalse(root.hasPrefix(documents), "Tokenizer cache must not live under Documents")
            XCTAssertTrue(root.hasSuffix("OpenMed/TokenizerAssets"))
        }

        func testEveryTokenizerErrorIsValueFree() {
            let errors: [OpenMedTokenizerError] = [
                .missingLocalAssets,
                .pinnedRevisionRequired,
                .identityMismatch,
                .downloadFailed(statusCode: nil),
                .downloadFailed(statusCode: 503),
                .sizeMismatch,
                .digestMismatch,
                .cacheUnavailable,
                .invalidAssets,
                .invalidPin(.revision),
            ]

            for error in errors {
                assertValueFree(error)
                XCTAssertTrue(error.reasonCode.hasPrefix("tokenizer_"))
                XCTAssertFalse(error.reasonCode.contains(" "))
            }
            XCTAssertEqual(Set(errors.map(\.reasonCode)).count, errors.count - 1)
        }

        // MARK: - Helpers

        private func assertValueFree(
            _ error: Error,
            forbidden: [String] = [],
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            let renderings = [error.localizedDescription, String(describing: error)]
            let sentinels =
                [
                    "https://", "huggingface.co", "/resolve/", "/private/", "/var/", "/Users/",
                    "OpenMed/", scratch.path,
                ] + forbidden
            for rendering in renderings {
                for sentinel in sentinels where !sentinel.isEmpty {
                    XCTAssertFalse(
                        rendering.contains(sentinel),
                        "Error rendering leaked a value",
                        file: file,
                        line: line
                    )
                }
            }
        }

        private func makeTokenizerFolder() throws -> URL {
            let folder = scratch.appending(path: "tokenizer", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (name, data) in try Self.tokenizerFiles() {
                try data.write(to: folder.appending(path: name))
            }
            return folder
        }

        private func makeLegacyMLXArtifactWithoutTokenizer() throws -> URL {
            let directory = scratch.appending(path: "legacy-mlx", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let config: [String: Any] = [
                "model_type": "bert",
                "_mlx_model_type": "bert",
                "_mlx_weights_format": "safetensors",
                "_name_or_path": "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
                "vocab_size": 32,
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "max_position_embeddings": 32,
                "type_vocab_size": 2,
                "num_labels": 3,
                "id2label": ["0": "O", "1": "B-NAME", "2": "I-NAME"],
            ]
            try JSONSerialization.data(withJSONObject: config)
                .write(to: directory.appending(path: "config.json"))
            try JSONSerialization.data(withJSONObject: ["0": "O", "1": "B-NAME", "2": "I-NAME"])
                .write(to: directory.appending(path: "id2label.json"))
            try Data().write(to: directory.appending(path: "weights.safetensors"))
            return directory
        }

        private func pinnedSession(_ files: [String: Data]) -> URLSession {
            PinnedTokenizerURLProtocol.install(files: files)
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [PinnedTokenizerURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        private func failingSession() -> URLSession {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [FailingTokenizerURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        private static func pin(for files: [String: Data]) throws -> OpenMedTokenizerPin {
            try OpenMedTokenizerPin(
                repositoryID: repositoryID,
                revision: revision,
                files: files.keys.sorted().map { name in
                    OpenMedTokenizerPin.File(
                        name: name,
                        sha256: OpenMedTokenizerAssetStore.sha256Hex(files[name]!),
                        byteCount: files[name]!.count
                    )
                }
            )
        }

        /// Same size, different digest: isolates the digest check from the size check.
        private static func flippingOneByte(_ data: Data) -> Data {
            var tampered = data
            let index = tampered.index(tampered.startIndex, offsetBy: tampered.count / 2)
            tampered[index] ^= 0x01
            return tampered
        }

        private static func relativeFiles(under root: URL) throws -> Set<String> {
            guard FileManager.default.fileExists(atPath: root.path) else {
                return []
            }
            let rootPath = root.resolvingSymlinksInPath().path + "/"
            var files = Set<String>()
            let enumerator = FileManager.default.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey]
            )
            while let url = enumerator?.nextObject() as? URL {
                guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                    continue
                }
                let path = url.resolvingSymlinksInPath().path
                XCTAssertTrue(path.hasPrefix(rootPath), "File escaped the cache root")
                files.insert(String(path.dropFirst(rootPath.count)))
            }
            return files
        }

        /// Synthetic WordPiece tokenizer: no real vocabulary or model assets.
        private static func tokenizerFiles() throws -> [String: Data] {
            [
                "tokenizer.json": try tokenizerJSON(),
                "tokenizer_config.json": try JSONSerialization.data(
                    withJSONObject: [
                        "tokenizer_class": "BertTokenizer",
                        "do_lower_case": true,
                        "unk_token": "[UNK]",
                        "cls_token": "[CLS]",
                        "sep_token": "[SEP]",
                        "pad_token": "[PAD]",
                        "mask_token": "[MASK]",
                        "model_max_length": 32,
                    ],
                    options: [.sortedKeys]
                ),
                "config.json": try JSONSerialization.data(
                    withJSONObject: ["model_type": "bert"],
                    options: [.sortedKeys]
                ),
            ]
        }

        private static func tokenizerJSON() throws -> Data {
            let specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            var vocabulary: [String: Int] = [
                "patient": 5, "ada": 6, "visited": 7, "clinic": 8, "##s": 9,
            ]
            for (index, token) in specials.enumerated() {
                vocabulary[token] = index
            }
            let object: [String: Any] = [
                "version": "1.0",
                "added_tokens": specials.enumerated().map { index, token in
                    [
                        "id": index, "content": token, "single_word": false, "lstrip": false,
                        "rstrip": false, "normalized": false, "special": true,
                    ] as [String: Any]
                },
                "normalizer": [
                    "type": "BertNormalizer", "clean_text": true,
                    "handle_chinese_chars": true, "lowercase": true,
                ],
                "pre_tokenizer": ["type": "BertPreTokenizer"],
                "post_processor": [
                    "type": "TemplateProcessing",
                    "single": [
                        ["SpecialToken": ["id": "[CLS]", "type_id": 0]],
                        ["Sequence": ["id": "A", "type_id": 0]],
                        ["SpecialToken": ["id": "[SEP]", "type_id": 0]],
                    ],
                    "pair": [
                        ["SpecialToken": ["id": "[CLS]", "type_id": 0]],
                        ["Sequence": ["id": "A", "type_id": 0]],
                        ["SpecialToken": ["id": "[SEP]", "type_id": 0]],
                        ["Sequence": ["id": "B", "type_id": 1]],
                        ["SpecialToken": ["id": "[SEP]", "type_id": 1]],
                    ],
                    "special_tokens": [
                        "[CLS]": ["id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]],
                        "[SEP]": ["id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]],
                    ],
                ],
                "decoder": ["type": "WordPiece", "prefix": "##", "cleanup": true],
                "model": [
                    "type": "WordPiece", "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                    "vocab": vocabulary,
                ],
            ]
            return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        }
    }
#endif
