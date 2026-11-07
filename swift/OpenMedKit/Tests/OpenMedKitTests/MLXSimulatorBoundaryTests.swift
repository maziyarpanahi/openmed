#if canImport(MLX) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// Skips an MLX-dependent test wherever the production gate refuses MLX,
    /// so the iOS Simulator lane reports it as skipped instead of aborting in
    /// mlx-swift's Metal library lookup.
    func skipUnlessMLXCanRun() throws {
        try XCTSkipUnless(
            OpenMedMLXRuntimeSupport.isSupported,
            "MLX cannot run on the iOS Simulator or an Intel Mac; run this test on Apple silicon macOS or a physical device."
        )
    }

    /// Proves every MLX entry point refuses with the typed, value-free
    /// unsupported error before it reads model files, downloads assets or
    /// reaches MLX.
    ///
    /// Each check runs twice: once under the simulated-Simulator seam, which
    /// works on any host, and once without it. On a supported host the
    /// unflagged call is a negative control that must fail for a different
    /// reason; on the iOS Simulator lane it exercises the real guard.
    final class MLXSimulatorBoundaryTests: XCTestCase {
        private static let fixtureName = "simulator-boundary-fixture"

        func testSupportRequiresAppleSiliconOutsideAnySimulator() {
            XCTAssertTrue(OpenMedMLXRuntimeSupport.supportsMLX(isSimulator: false, isAppleSilicon: true))
            XCTAssertFalse(OpenMedMLXRuntimeSupport.supportsMLX(isSimulator: true, isAppleSilicon: true))
            XCTAssertFalse(OpenMedMLXRuntimeSupport.supportsMLX(isSimulator: false, isAppleSilicon: false))
            XCTAssertFalse(OpenMedMLXRuntimeSupport.supportsMLX(isSimulator: true, isAppleSilicon: false))
        }

        func testBuildEnvironmentMatchesCompiledTarget() {
            #if targetEnvironment(simulator)
                XCTAssertFalse(OpenMedMLXRuntimeSupport.isSupported)
            #elseif arch(arm64) || arch(arm64e)
                XCTAssertTrue(OpenMedMLXRuntimeSupport.isSupported)
            #else
                XCTAssertFalse(OpenMedMLXRuntimeSupport.isSupported)
            #endif
            XCTAssertEqual(
                MLXTokenClassificationPipeline.isRuntimeSupported,
                OpenMedMLXRuntimeSupport.isSupported
            )
        }

        func testSimulatedSimulatorOnlyWithdrawsSupportInsideItsScope() throws {
            let outside = OpenMedMLXRuntimeSupport.isSupported
            try OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                XCTAssertFalse(OpenMedMLXRuntimeSupport.isSupported)
                XCTAssertFalse(MLXTokenClassificationPipeline.isRuntimeSupported)
                XCTAssertThrowsError(try OpenMedMLXRuntimeSupport.requireSupported()) { error in
                    XCTAssertEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                }
            }
            OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(false) {
                XCTAssertEqual(OpenMedMLXRuntimeSupport.isSupported, outside)
            }
            XCTAssertEqual(OpenMedMLXRuntimeSupport.isSupported, outside)
        }

        func testUnsupportedErrorIsValueFree() {
            let error: Error = OpenMedMLXRuntimeError.unsupportedPlatform
            let directory = missingModelDirectory()
            let renderings = [
                error.localizedDescription,
                String(describing: error),
                String(reflecting: error),
                (error as NSError).description,
            ]
            for rendering in renderings {
                XCTAssertFalse(rendering.contains(directory.path))
                XCTAssertFalse(rendering.contains(Self.fixtureName))
                XCTAssertFalse(rendering.contains(NSHomeDirectory()))
            }
            XCTAssertEqual(
                error.localizedDescription,
                "Swift MLX inference requires Apple Silicon macOS or a real iPhone/iPad device."
            )
        }

        func testTokenClassificationPipelineRefusesBeforeReadingTheArtifact() throws {
            let directory = missingModelDirectory()
            assertRefused { _ = try MLXTokenClassificationPipeline(modelDirectoryURL: directory) }
            assertUnflagged {
                _ = try MLXTokenClassificationPipeline(modelDirectoryURL: directory)
            } otherwise: { error in
                XCTAssertNotEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
            }
        }

        func testRuntimeMemoryCacheClearIsANoOpWhereMLXCannotRun() throws {
            #if canImport(Tokenizers)
                // Under the seam this must return without reaching the MLX
                // allocator. On the iOS Simulator the unflagged call would
                // abort the test process if the guard were missing.
                OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                    OpenMed.clearRuntimeMemoryCache()
                }
                if !OpenMedMLXRuntimeSupport.isSupported {
                    OpenMed.clearRuntimeMemoryCache()
                }
            #else
                throw XCTSkip("OpenMed requires swift-transformers on this platform.")
            #endif
        }

        #if canImport(Tokenizers)
            func testOpenMedMLXBackendRefusesBeforeReadingTheArtifact() throws {
                let directory = missingModelDirectory()
                assertRefused { _ = try OpenMed(backend: .mlx(modelDirectoryURL: directory)) }
                assertUnflagged {
                    _ = try OpenMed(backend: .mlx(modelDirectoryURL: directory))
                } otherwise: { error in
                    XCTAssertNotEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                }
            }

            func testZeroShotRuntimesRefuseBeforeReadingTheArtifact() throws {
                let directory = missingModelDirectory()
                assertRefused { _ = try OpenMedZeroShotNER(modelDirectoryURL: directory) }
                assertRefused { _ = try OpenMedZeroShotClassifier(modelDirectoryURL: directory) }
                assertRefused { _ = try OpenMedRelationExtractor(modelDirectoryURL: directory) }
                assertUnflagged {
                    _ = try OpenMedZeroShotNER(modelDirectoryURL: directory)
                } otherwise: { error in
                    XCTAssertNotEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                }
            }
        #endif

        #if canImport(MLXLMCommon) && canImport(MLXLLM) && canImport(MLXNN) && canImport(Tokenizers)
            func testMapleRefusesBeforeReadingTheModelDirectory() async throws {
                let directory = missingModelDirectory()
                await assertRefused { _ = try await OpenMedMaple(modelDirectoryURL: directory) }
                await assertUnflagged {
                    _ = try await OpenMedMaple(modelDirectoryURL: directory)
                } otherwise: { error in
                    guard case .missingFiles? = error as? OpenMedMapleRuntimeError else {
                        return XCTFail("Expected the missing-file check, got \(type(of: error))")
                    }
                }
            }
        #endif

        #if canImport(MLXLMCommon) && canImport(Tokenizers)
            func testCompassLocalLoadRefusesBeforeReadingTheModelDirectory() async throws {
                let directory = missingModelDirectory()
                await assertRefused {
                    _ = try await OpenMedVisionLanguageModel.load(modelDirectory: directory)
                }
                await assertUnflagged {
                    _ = try await OpenMedVisionLanguageModel.load(modelDirectory: directory)
                } otherwise: { error in
                    XCTAssertNotEqual(error as? OpenMedMLXRuntimeError, .unsupportedPlatform)
                }
            }

            func testCompassHubLoadRefusesBeforeAnyDownload() async throws {
                RefusingURLProtocol.reset()
                XCTAssertTrue(URLProtocol.registerClass(RefusingURLProtocol.self))
                defer { URLProtocol.unregisterClass(RefusingURLProtocol.self) }
                let progress = ProgressCounter()
                let modelID = "OpenMed/\(Self.fixtureName)"

                await assertRefused {
                    _ = try await OpenMedVisionLanguageModel.load(
                        modelID: modelID,
                        progressHandler: { _ in progress.increment() }
                    )
                }
                if !OpenMedMLXRuntimeSupport.isSupported {
                    // The real Simulator guard, without the seam. A supported
                    // host would start a Hub download here, so it stops at the
                    // simulated check above.
                    await assertRefused(simulated: false) {
                        _ = try await OpenMedVisionLanguageModel.load(
                            modelID: modelID,
                            progressHandler: { _ in progress.increment() }
                        )
                    }
                }

                XCTAssertEqual(RefusingURLProtocol.startedRequests, 0)
                XCTAssertEqual(progress.value, 0)
            }
        #endif

        // MARK: - Helpers

        private func missingModelDirectory() -> URL {
            FileManager.default.temporaryDirectory.appending(
                path: "\(Self.fixtureName)-\(UUID().uuidString)",
                directoryHint: .isDirectory
            )
        }

        private func assertRefused(
            simulated: Bool = true,
            file: StaticString = #filePath,
            line: UInt = #line,
            _ body: () throws -> Void
        ) {
            OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(simulated) {
                do {
                    try body()
                    XCTFail("Expected the unsupported-platform refusal", file: file, line: line)
                } catch {
                    assertUnsupported(error, file: file, line: line)
                }
            }
        }

        private func assertRefused(
            simulated: Bool = true,
            file: StaticString = #filePath,
            line: UInt = #line,
            _ body: () async throws -> Void
        ) async {
            await OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(simulated) {
                do {
                    try await body()
                    XCTFail("Expected the unsupported-platform refusal", file: file, line: line)
                } catch {
                    assertUnsupported(error, file: file, line: line)
                }
            }
        }

        /// Runs `body` without the seam. Where MLX is supported it must fail
        /// for the reason `otherwise` checks; elsewhere the real guard must
        /// refuse it.
        private func assertUnflagged(
            file: StaticString = #filePath,
            line: UInt = #line,
            _ body: () throws -> Void,
            otherwise: (Error) -> Void
        ) {
            guard OpenMedMLXRuntimeSupport.isSupported else {
                return assertRefused(simulated: false, file: file, line: line, body)
            }
            do {
                try body()
                XCTFail("A missing model directory must not load", file: file, line: line)
            } catch {
                otherwise(error)
            }
        }

        private func assertUnflagged(
            file: StaticString = #filePath,
            line: UInt = #line,
            _ body: () async throws -> Void,
            otherwise: (Error) -> Void
        ) async {
            guard OpenMedMLXRuntimeSupport.isSupported else {
                return await assertRefused(simulated: false, file: file, line: line, body)
            }
            do {
                try await body()
                XCTFail("A missing model directory must not load", file: file, line: line)
            } catch {
                otherwise(error)
            }
        }

        private func assertUnsupported(
            _ error: Error,
            file: StaticString,
            line: UInt
        ) {
            XCTAssertEqual(
                error as? OpenMedMLXRuntimeError,
                .unsupportedPlatform,
                "Unexpected \(type(of: error))",
                file: file,
                line: line
            )
            XCTAssertFalse(
                error.localizedDescription.contains(Self.fixtureName),
                file: file,
                line: line
            )
        }
    }

    /// Fails and counts every request that reaches the shared URL session.
    private final class RefusingURLProtocol: URLProtocol {
        private static let lock = NSLock()
        nonisolated(unsafe) private static var started = 0

        static var startedRequests: Int {
            lock.lock()
            defer { lock.unlock() }
            return started
        }

        static func reset() {
            lock.lock()
            defer { lock.unlock() }
            started = 0
        }

        override class func canInit(with request: URLRequest) -> Bool { true }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

        override func startLoading() {
            Self.lock.lock()
            Self.started += 1
            Self.lock.unlock()
            client?.urlProtocol(self, didFailWithError: URLError(.notConnectedToInternet))
        }

        override func stopLoading() {}
    }

    private final class ProgressCounter: @unchecked Sendable {
        private let lock = NSLock()
        private var count = 0

        var value: Int {
            lock.lock()
            defer { lock.unlock() }
            return count
        }

        func increment() {
            lock.lock()
            count += 1
            lock.unlock()
        }
    }
#endif
