#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// `PIIPack` resolves the manifest's `pii` candidates, prepares one offline
    /// through an injected loader, and returns offset-only results.
    final class PIIPackTests: XCTestCase {
        /// Synthetic identifiers; none may appear in results or errors.
        private static let firstName = "Zorvexa"
        private static let email = "zq.sentinel@example.test"
        private static let note = "Café note: Patient \(firstName) was seen. Contact \(email) today."

        private var cache: URL!

        override func setUpWithError() throws {
            cache = try TaskPackTestSupport.temporaryCache()
            TaskPackHubStub.reset()
        }

        override func tearDownWithError() throws {
            if let cache {
                try? FileManager.default.removeItem(at: cache)
            }
            TaskPackHubStub.reset()
        }

        private static var recognizer: TaskPackFakeRecognizer {
            TaskPackFakeRecognizer(detections: [
                (firstName, "first_name", 0.97),
                (email, "email", 0.95),
            ])
        }

        private func makePack(probe: TaskPackLoaderProbe) throws -> PIIPack {
            try PIIPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
        }

        // MARK: - Resolve

        func testResolveReturnsTheManifestCandidatesInRankOrder() throws {
            try skipUnlessMLXCanRun()
            let pack = try PIIPack()
            let resolution = pack.resolve()
            XCTAssertEqual(resolution.status, .resolved)
            XCTAssertEqual(resolution.repoIDs, try OpenMedTaskPackManifestTests.sharedAppleRepoIDs(pack: "pii"))
            XCTAssertEqual(resolution.skipped, [])
            XCTAssertTrue(resolution.candidates.allSatisfy { $0.license == "apache-2.0" })
            XCTAssertEqual(pack.notice.key, "openmed.notice.pii.review_required")
            XCTAssertNil(pack.preparedCandidate)
        }

        func testSimulatorSkipsEveryMLXCandidateAndPrepareRefusesWithoutRequests() async throws {
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)
            for candidate in pack.definition.candidates {
                try TaskPackTestSupport.cacheSnapshot(of: candidate, in: cache)
            }
            let store = TaskPackTestSupport.stubbedStore(cache: cache)

            await OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                let resolution = pack.resolve()
                XCTAssertEqual(resolution.status, .unsupportedOnSimulator)
                XCTAssertEqual(resolution.candidates, [])
                XCTAssertEqual(resolution.skipped.map(\.candidate), pack.definition.candidates)
                XCTAssertTrue(resolution.skipped.allSatisfy { $0.reason == .unsupportedOnSimulator })
                do {
                    try await pack.prepare(store: store, allowDownload: true)
                    XCTFail("The Simulator must refuse MLX candidates")
                } catch {
                    XCTAssertEqual(error as? OpenMedTaskPackError, .unsupportedOnSimulator)
                }
            }
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])
            XCTAssertEqual(probe.loadedDirectories, [])
            XCTAssertNil(pack.preparedCandidate)
        }

        // MARK: - Prepare and run

        func testCachedCandidateExtractsSpansAndDeidentifiesWithAFakePipeline() async throws {
            try skipUnlessMLXCanRun()
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)
            let best = try XCTUnwrap(pack.definition.candidates.first)
            try TaskPackTestSupport.cacheSnapshot(of: best, in: cache)

            let prepared = try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            XCTAssertEqual(prepared, best)
            XCTAssertEqual(pack.preparedCandidate, best)
            XCTAssertEqual(
                probe.loadedDirectories.map(\.lastPathComponent),
                [try OpenMedModelStore.cachedMLXModelDirectory(repoID: best.repoID, cacheDirectory: cache).lastPathComponent]
            )
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])

            let extraction = try pack.extract(Self.note)
            XCTAssertEqual(extraction.sourceModelID, best.repoID)
            XCTAssertEqual(extraction.notice, pack.notice)
            let name = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: Self.firstName, in: Self.note))
            let mail = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: Self.email, in: Self.note))
            XCTAssertEqual(
                extraction.spans.map { [$0.canonicalLabel, "\($0.start)", "\($0.end)"] },
                [["FIRST_NAME", "\(name.start)", "\(name.end)"], ["EMAIL", "\(mail.start)", "\(mail.end)"]]
            )
            for span in extraction.spans {
                let range = try XCTUnwrap(span.range(in: Self.note))
                XCTAssertTrue([Self.firstName, Self.email].contains(String(Self.note[range])))
            }

            let deidentified = try pack.deidentify(Self.note)
            XCTAssertEqual(deidentified.result.policyName, Policy.defaultName)
            XCTAssertEqual(deidentified.sourceModelID, best.repoID)
            XCTAssertEqual(deidentified.notice, pack.notice)
            XCTAssertFalse(deidentified.redactedText.contains(Self.firstName))
            XCTAssertFalse(deidentified.redactedText.contains(Self.email))
            XCTAssertTrue(deidentified.redactedText.hasPrefix("Café note: Patient "))
            XCTAssertEqual(
                deidentified.result.actions.map { [$0.canonicalLabel, "\($0.start)", "\($0.end)"] },
                [["FIRST_NAME", "\(name.start)", "\(name.end)"], ["EMAIL", "\(mail.start)", "\(mail.end)"]]
            )

            let loaded = try Policy(named: "strict_no_leak")
            XCTAssertEqual(try pack.deidentify(Self.note, policy: loaded).result.policyName, "strict_no_leak")
        }

        func testResultsAndErrorsCarryNoInputTextBeyondTheRedactedOutput() async throws {
            try skipUnlessMLXCanRun()
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)
            XCTAssertThrowsError(try pack.extract(Self.note)) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .notPrepared)
                assertNoIdentifier(in: [error.localizedDescription, String(reflecting: error)])
            }
            XCTAssertThrowsError(try pack.deidentify(Self.note)) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .notPrepared)
            }

            try TaskPackTestSupport.cacheSnapshot(of: try XCTUnwrap(pack.definition.candidates.first), in: cache)
            try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))

            let extraction = try pack.extract(Self.note)
            XCTAssertEqual(extraction.spans.count, 2)
            var dumped = ""
            dump(extraction, to: &dumped)
            assertNoIdentifier(in: [String(describing: extraction), String(reflecting: extraction), dumped])
            XCTAssertFalse(String(reflecting: extraction).contains("Café"))

            let deidentified = try pack.deidentify(Self.note)
            var dumpedResult = ""
            dump(deidentified, to: &dumpedResult)
            assertNoIdentifier(in: [String(reflecting: deidentified), dumpedResult])
        }

        func testFallsBackToTheNextCachedCandidateInRankOrder() async throws {
            try skipUnlessMLXCanRun()
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)
            let candidates = pack.definition.candidates
            try XCTSkipIf(candidates.count < 2, "the manifest needs two Apple candidates")
            try TaskPackTestSupport.cacheSnapshot(of: candidates[1], in: cache)

            let prepared = try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            XCTAssertEqual(prepared, candidates[1])
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])
        }

        func testACachedCandidateThatFailsToLoadFallsBackAndThenFailsClosed() async throws {
            try skipUnlessMLXCanRun()
            let candidates = try OpenMedTaskPackManifest.bundled().pack("pii").candidates
            try XCTSkipIf(candidates.count < 2, "the manifest needs two Apple candidates")
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer, failingRepoIDs: [candidates[0].repoID])
            let pack = try makePack(probe: probe)
            try TaskPackTestSupport.cacheSnapshot(of: candidates[0], in: cache)
            try TaskPackTestSupport.cacheSnapshot(of: candidates[1], in: cache)
            let prepared = try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            XCTAssertEqual(prepared, candidates[1])
            XCTAssertEqual(probe.loadedDirectories.count, 2)

            let failing = TaskPackLoaderProbe(recognizer: Self.recognizer, failingRepoIDs: [candidates[0].repoID])
            let onlyBroken = try makePack(probe: failing)
            try FileManager.default.removeItem(
                at: try OpenMedModelStore.cachedMLXModelDirectory(repoID: candidates[1].repoID, cacheDirectory: cache)
            )
            do {
                try await onlyBroken.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
                XCTFail("A broken cached candidate must not prepare")
            } catch {
                XCTAssertEqual(error as? OpenMedTaskPackError, .loadFailed)
            }
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])
        }

        // MARK: - Download consent

        func testWithoutConsentAndWithoutCacheNoRequestIsMadeAndATypedErrorIsReturned() async throws {
            try skipUnlessMLXCanRun()
            XCTAssertTrue(URLProtocol.registerClass(TaskPackHubStub.self))
            defer { URLProtocol.unregisterClass(TaskPackHubStub.self) }
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)

            for store in [TaskPackTestSupport.stubbedStore(cache: cache), OpenMedTaskPackStore(cacheDirectory: cache)] {
                do {
                    try await pack.prepare(store: store, allowDownload: false)
                    XCTFail("An empty cache must not prepare without consent")
                } catch {
                    XCTAssertEqual(error as? OpenMedTaskPackError, .notCached)
                    assertNoIdentifier(in: [error.localizedDescription, String(reflecting: error)])
                    XCTAssertFalse(String(reflecting: error).contains(cache.path))
                }
            }
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])
            XCTAssertEqual(probe.loadedDirectories, [])
            XCTAssertNil(pack.preparedCandidate)
            XCTAssertThrowsError(try pack.extract(Self.note)) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .notPrepared)
            }
        }

        func testWithConsentTheBestCandidateIsDownloadedThroughTheStore() async throws {
            try skipUnlessMLXCanRun()
            TaskPackHubStub.reset(.serveLegacySnapshot)
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)
            let best = try XCTUnwrap(pack.definition.candidates.first)

            let prepared = try await pack.prepare(
                store: TaskPackTestSupport.stubbedStore(cache: cache),
                allowDownload: true
            )
            XCTAssertEqual(prepared, best)
            let paths = TaskPackHubStub.requestedPaths
            XCTAssertFalse(paths.isEmpty)
            XCTAssertTrue(paths.allSatisfy { $0.hasPrefix("/\(best.repoID)/resolve/main/") }, "\(paths)")
            XCTAssertTrue(try OpenMedModelStore.isMLXModelCached(repoID: best.repoID, cacheDirectory: cache))
            XCTAssertEqual(probe.loadedDirectories.count, 1)
            XCTAssertEqual(try pack.extract(Self.note).spans.count, 2)
        }

        func testDownloadFailuresFallBackThroughEveryCandidateThenFailValueFree() async throws {
            try skipUnlessMLXCanRun()
            TaskPackHubStub.reset(.status(503))
            let probe = TaskPackLoaderProbe(recognizer: Self.recognizer)
            let pack = try makePack(probe: probe)

            do {
                try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache), allowDownload: true)
                XCTFail("Unavailable downloads must not prepare")
            } catch {
                XCTAssertEqual(error as? OpenMedTaskPackError, .downloadFailed)
                let rendering = String(reflecting: error) + error.localizedDescription
                for candidate in pack.definition.candidates {
                    XCTAssertFalse(rendering.contains(candidate.repoID))
                }
            }
            let requestedRepos = Set(
                TaskPackHubStub.requestedPaths.compactMap { path in
                    pack.definition.candidates.first { path.hasPrefix("/\($0.repoID)/") }?.repoID
                }
            )
            XCTAssertEqual(requestedRepos, Set(pack.definition.candidates.map(\.repoID)))
            XCTAssertEqual(probe.loadedDirectories, [])
        }

        // MARK: - Helpers

        private func assertNoIdentifier(
            in renderings: [String],
            file: StaticString = #filePath,
            line: UInt = #line
        ) {
            for rendering in renderings {
                XCTAssertFalse(rendering.contains(Self.firstName), file: file, line: line)
                XCTAssertFalse(rendering.contains(Self.email), file: file, line: line)
                XCTAssertFalse(rendering.contains("sentinel"), file: file, line: line)
            }
        }
    }
#endif
