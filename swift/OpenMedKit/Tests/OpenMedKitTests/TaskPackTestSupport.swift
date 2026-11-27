#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// Offline doubles shared by the task pack tests.
    enum TaskPackTestSupport {
        /// A temporary model cache root, removed by the caller.
        static func temporaryCache() throws -> URL {
            let root = FileManager.default.temporaryDirectory.appending(
                path: "openmed-task-pack-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            return root
        }

        /// Write a minimal legacy MLX snapshot so the store reports it ready.
        /// The bytes are placeholders: the injected loader never reads them.
        static func cacheSnapshot(of candidate: OpenMedTaskPackCandidate, in cache: URL) throws {
            let directory = try OpenMedModelStore.cachedMLXModelDirectory(
                repoID: candidate.repoID,
                cacheDirectory: cache
            )
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: directory.appending(path: "config.json"))
            try Data([0x00]).write(to: directory.appending(path: "weights.safetensors"))
            XCTAssertEqual(
                try OpenMedModelStore.mlxModelCacheState(repoID: candidate.repoID, cacheDirectory: cache),
                .ready
            )
        }

        /// A store whose session only reaches ``TaskPackHubStub``.
        static func stubbedStore(cache: URL) -> OpenMedTaskPackStore {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [TaskPackHubStub.self]
            return OpenMedTaskPackStore(
                cacheDirectory: cache,
                revision: "main",
                session: URLSession(configuration: configuration)
            )
        }

        /// Scalar offsets of the first occurrence of `needle` in `text`.
        static func scalarSpan(of needle: String, in text: String) -> (start: Int, end: Int)? {
            guard let range = text.range(of: needle) else { return nil }
            let start = text.unicodeScalars.distance(from: text.unicodeScalars.startIndex, to: range.lowerBound)
            return (start, start + needle.unicodeScalars.count)
        }
    }

    /// Deterministic recognizer that emits fixed labels for fixed substrings,
    /// standing in for a model so pack tests run without MLX.
    struct TaskPackFakeRecognizer: OpenMedSpanRecognizer {
        let detections: [(needle: String, label: String, confidence: Float)]

        func recognize(_ text: String) throws -> [EntityPrediction] {
            detections.compactMap { detection in
                guard let span = TaskPackTestSupport.scalarSpan(of: detection.needle, in: text) else {
                    return nil
                }
                return EntityPrediction(
                    label: detection.label,
                    text: detection.needle,
                    confidence: detection.confidence,
                    start: span.start,
                    end: span.end
                )
            }
        }
    }

    /// Records the directories a pack asks to load and returns a fake runtime.
    final class TaskPackLoaderProbe {
        private(set) var loadedDirectories: [URL] = []
        private let recognizer: TaskPackFakeRecognizer
        private let failingRepoIDs: Set<String>

        init(recognizer: TaskPackFakeRecognizer, failingRepoIDs: Set<String> = []) {
            self.recognizer = recognizer
            self.failingRepoIDs = failingRepoIDs
        }

        func load(_ directory: URL) async throws -> OpenMed {
            loadedDirectories.append(directory)
            if failingRepoIDs.contains(where: { directory.path.contains($0.replacingOccurrences(of: "/", with: "__")) }) {
                throw OpenMedMLXArtifactError.unreadableFile(.config)
            }
            return OpenMed(recognizer: recognizer)
        }
    }

    /// Hub double: counts every request and serves a minimal legacy snapshot,
    /// a fixed HTTP status, or a transport failure.
    final class TaskPackHubStub: URLProtocol {
        enum Mode {
            case refuse
            case serveLegacySnapshot
            case status(Int)
        }

        private static let lock = NSLock()
        nonisolated(unsafe) private static var paths: [String] = []
        nonisolated(unsafe) private static var mode: Mode = .refuse

        static var requestedPaths: [String] {
            lock.lock()
            defer { lock.unlock() }
            return paths
        }

        static func reset(_ newMode: Mode = .refuse) {
            lock.lock()
            defer { lock.unlock() }
            paths = []
            mode = newMode
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            Self.lock.lock()
            let path = request.url?.path ?? ""
            Self.paths.append(path)
            let mode = Self.mode
            Self.lock.unlock()

            let status: Int
            var body = Data()
            switch mode {
            case .refuse:
                client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
                return
            case .serveLegacySnapshot:
                if path.hasSuffix("/config.json") {
                    status = 200
                    body = Data("{}".utf8)
                } else if path.hasSuffix("/weights.safetensors") {
                    status = 200
                    body = Data([0x00])
                } else {
                    status = 404
                }
            case .status(let code):
                status = code
            }
            let response = HTTPURLResponse(
                url: request.url ?? URL(fileURLWithPath: "/"),
                statusCode: status,
                httpVersion: "HTTP/1.1",
                headerFields: nil
            )
            if let response {
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            }
            client?.urlProtocol(self, didLoad: body)
            client?.urlProtocolDidFinishLoading(self)
        }

        override func stopLoading() {}
    }
#endif
