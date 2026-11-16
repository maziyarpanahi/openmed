import Foundation
import XCTest

@testable import OpenMedKit

/// Serves a synthetic legacy-layout model in fixed-size chunks. A file named
/// in `stalled` sends its first chunk and then waits for cancellation.
/// URLProtocol instances are created by the loading system, so state is
/// static and every access holds `lock`.
private final class ChunkedModelURLProtocol: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var files: [String: Data] = [:]
    nonisolated(unsafe) private static var stalled: Set<String> = []
    nonisolated(unsafe) private static var failure: URLError?
    nonisolated(unsafe) private static var paths: [String] = []
    static let chunkSize = 4_096

    static var requestedPaths: [String] {
        lock.lock()
        defer { lock.unlock() }
        return paths
    }

    static func install(files: [String: Data], stalled: Set<String> = [], failure: URLError? = nil) {
        lock.lock()
        defer { lock.unlock() }
        self.files = files
        self.stalled = stalled
        self.failure = failure
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
        let stall = Self.stalled.contains(path)
        let failure = Self.failure
        Self.lock.unlock()
        if let failure {
            client?.urlProtocol(self, didFailWithError: failure)
            return
        }
        guard let data else {
            let response = HTTPURLResponse(url: url, statusCode: 404, httpVersion: nil, headerFields: nil)!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocolDidFinishLoading(self)
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Length": "\(data.count)"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        var offset = 0
        while offset < data.count {
            let end = min(offset + Self.chunkSize, data.count)
            client?.urlProtocol(self, didLoad: data.subdata(in: offset..<end))
            offset = end
            if stall { return }
        }
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class OpenMedModelDownloadTaskTests: XCTestCase {
    private static let repoID = "OpenMed/Synthetic-Progress-mlx"
    private static let files: [String: Data] = [
        "config.json": Data(#"{"model_type":"bert"}"#.utf8),
        "weights.safetensors": Data((0..<40_000).map { UInt8($0 % 251) }),
        "tokenizer.json": Data(repeating: 0x7B, count: 9_000),
    ]

    private var cacheRoot: URL!

    override func setUpWithError() throws {
        cacheRoot = FileManager.default.temporaryDirectory.appending(
            path: "openmed-progress-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: cacheRoot)
    }

    func testChunkedDownloadReportsMonotonicProgressEndingAtTheTotal() async throws {
        ChunkedModelURLProtocol.install(files: Self.files)
        let download = OpenMedModelStore.startMLXModelDownload(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        let events = await Self.collect(download.progress)
        let directory = try await download.value

        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot), .ready)
        XCTAssertEqual(directory, try OpenMedModelStore.cachedMLXModelDirectory(repoID: Self.repoID, cacheDirectory: cacheRoot))
        let totalBytes = Int64(Self.files.values.map(\.count).reduce(0, +))
        let last = try XCTUnwrap(events.last)
        XCTAssertEqual(last.totalBytesDone, totalBytes)
        XCTAssertEqual(last.filesDone, last.filesTotal)
        XCTAssertEqual(last.bytesTotal, last.bytesDone)

        for (previous, next) in zip(events, events.dropFirst()) {
            XCTAssertLessThanOrEqual(previous.totalBytesDone, next.totalBytesDone)
            XCTAssertLessThanOrEqual(previous.filesDone, next.filesDone)
            XCTAssertLessThanOrEqual(previous.filesTotal, next.filesTotal)
            if previous.fileName == next.fileName && previous.filesDone == next.filesDone {
                XCTAssertLessThanOrEqual(previous.bytesDone, next.bytesDone)
            }
        }
        // URLSession may coalesce chunks that arrive together, so only require
        // a byte-progress event with the declared length before completion.
        // The cancellation test pins a mid-file event deterministically.
        let weights = events.filter { $0.fileName == "weights.safetensors" }
        XCTAssertGreaterThanOrEqual(weights.count, 2)
        XCTAssertEqual(weights.first?.bytesTotal, 40_000)
        XCTAssertEqual(weights.first?.filesDone, weights.last.map { $0.filesDone - 1 })
        XCTAssertEqual(weights.last?.bytesDone, 40_000)
    }

    func testProgressCarriesNoURLTokenOrAbsolutePath() async throws {
        ChunkedModelURLProtocol.install(files: Self.files)
        let download = OpenMedModelStore.startMLXModelDownload(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        let events = await Self.collect(download.progress)
        _ = try await download.value

        XCTAssertFalse(events.isEmpty)
        let requested = Set(ChunkedModelURLProtocol.requestedPaths)
        for event in events {
            XCTAssertTrue(requested.contains(event.fileName), event.fileName)
            let rendered = String(describing: event)
            for forbidden in ["https://", "huggingface", "Synthetic-Progress", cacheRoot.path, "/private/", "token="] {
                XCTAssertFalse(rendered.contains(forbidden), forbidden)
            }
        }
    }

    func testCancellingMidDownloadRemovesItsFilesAndNeverMarksReady() async throws {
        ChunkedModelURLProtocol.install(files: Self.files, stalled: ["weights.safetensors"])
        let download = OpenMedModelStore.startMLXModelDownload(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())

        var sawPartialWeights = false
        for await event in download.progress where event.fileName == "weights.safetensors" && event.bytesDone > 0 {
            // The stub stalls after one chunk, so this event is mid-file.
            XCTAssertLessThanOrEqual(event.bytesDone, Int64(ChunkedModelURLProtocol.chunkSize))
            XCTAssertEqual(event.bytesTotal, 40_000)
            sawPartialWeights = true
            download.cancel()
        }

        XCTAssertTrue(sawPartialWeights)
        XCTAssertTrue(download.isCancelled)
        do {
            _ = try await download.value
            XCTFail("a cancelled download must not finish")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(repoID: Self.repoID, cacheDirectory: cacheRoot)
        XCTAssertTrue(
            [.missing, .partial].contains(
                try OpenMedModelStore.mlxModelCacheState(repoID: Self.repoID, cacheDirectory: cacheRoot)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: ".openmed-artifact-ready").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: "config.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: "weights.safetensors").path))
        XCTAssertFalse(ChunkedModelURLProtocol.requestedPaths.contains("tokenizer.json"))
    }

    func testCancellingTheCallingTaskStopsTheAsyncWrapper() async throws {
        ChunkedModelURLProtocol.install(files: Self.files, stalled: ["weights.safetensors"])
        let session = Self.session()
        let cacheRoot = self.cacheRoot!
        let caller = Task {
            try await OpenMedModelStore.downloadMLXModel(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: session)
        }
        while !ChunkedModelURLProtocol.requestedPaths.contains("weights.safetensors") {
            await Task.yield()
        }
        caller.cancel()

        do {
            _ = try await caller.value
            XCTFail("a cancelled download must not finish")
        } catch {
            XCTAssertTrue(error is CancellationError, "\(error)")
        }
        let directory = try OpenMedModelStore.cachedMLXModelDirectory(repoID: Self.repoID, cacheDirectory: cacheRoot)
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: ".openmed-artifact-ready").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: directory.appending(path: "config.json").path))
    }

    func testTransportAndHTTPErrorsAreValueFree() async throws {
        ChunkedModelURLProtocol.install(files: Self.files, failure: URLError(.notConnectedToInternet))
        do {
            _ = try await OpenMedModelStore.startMLXModelDownload(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session()
            ).value
            XCTFail("an unreachable host must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedModelStoreError, .transferFailed)
            XCTAssertEqual((error as? OpenMedModelStoreError)?.reason, "transfer_failed")
            Self.assertValueFree(error, cacheRoot: cacheRoot)
        }

        ChunkedModelURLProtocol.install(files: [:])
        do {
            _ = try await OpenMedModelStore.startMLXModelDownload(
                repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session()
            ).value
            XCTFail("a missing config must fail")
        } catch {
            XCTAssertEqual(error as? OpenMedModelStoreError, .httpError(statusCode: 404))
            Self.assertValueFree(error, cacheRoot: cacheRoot)
        }
    }

    func testCachedSnapshotFinishesWithoutRequestsOrEvents() async throws {
        ChunkedModelURLProtocol.install(files: Self.files)
        _ = try await OpenMedModelStore.startMLXModelDownload(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session()
        ).value

        ChunkedModelURLProtocol.install(files: [:])
        let again = OpenMedModelStore.startMLXModelDownload(
            repoID: Self.repoID, cacheDirectory: cacheRoot, session: Self.session())
        let events = await Self.collect(again.progress)

        _ = try await again.value
        XCTAssertEqual(events, [])
        XCTAssertEqual(ChunkedModelURLProtocol.requestedPaths, [])
    }

    // MARK: - Helpers

    private static func collect(_ stream: AsyncStream<OpenMedModelDownloadProgress>) async -> [OpenMedModelDownloadProgress] {
        var events: [OpenMedModelDownloadProgress] = []
        for await event in stream { events.append(event) }
        return events
    }

    private static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [ChunkedModelURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private static func assertValueFree(_ error: any Error, cacheRoot: URL, line: UInt = #line) {
        let nsError = error as NSError
        let renderings = [
            error.localizedDescription, String(describing: error), String(reflecting: error),
            "\(nsError.userInfo)",
        ]
        for rendering in renderings {
            for forbidden in ["https://", "huggingface", "Synthetic-Progress", cacheRoot.path, "/private/"] {
                XCTAssertFalse(rendering.contains(forbidden), "\(forbidden) leaked", line: line)
            }
        }
    }
}
