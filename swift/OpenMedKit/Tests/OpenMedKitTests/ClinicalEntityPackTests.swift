#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// `ClinicalEntityPack` maps model labels to the pack's canonical
    /// categories, drops the rest, and keeps Unicode scalar offsets.
    final class ClinicalEntityPackTests: XCTestCase {
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

        private func preparedPack(
            _ detections: [(needle: String, label: String, confidence: Float)]
        ) async throws -> (ClinicalEntityPack, OpenMedTaskPackCandidate) {
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: detections))
            let pack = try ClinicalEntityPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
            let best = try XCTUnwrap(pack.definition.candidates.first)
            try TaskPackTestSupport.cacheSnapshot(of: best, in: cache)
            let prepared = try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            XCTAssertEqual(prepared, best)
            return (pack, best)
        }

        // MARK: - Resolve

        func testResolveReturnsTheManifestCandidatesAndCategories() throws {
            try skipUnlessMLXCanRun()
            let pack = try ClinicalEntityPack()
            let resolution = pack.resolve()
            XCTAssertEqual(resolution.status, .resolved)
            XCTAssertEqual(
                resolution.repoIDs,
                try OpenMedTaskPackManifestTests.sharedAppleRepoIDs(pack: "clinical_entity")
            )
            XCTAssertTrue(resolution.candidates.allSatisfy { $0.license == "apache-2.0" })
            XCTAssertEqual(pack.definition.task, "clinical_entity_extraction")
            XCTAssertEqual(pack.notice.key, "openmed.notice.clinical_entity.extraction_only")
            XCTAssertFalse(pack.definition.canonicalLabels.isEmpty)
        }

        func testSimulatorSkipsEveryMLXCandidateWithATypedResult() async throws {
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: []))
            let pack = try ClinicalEntityPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
            for candidate in pack.definition.candidates {
                try TaskPackTestSupport.cacheSnapshot(of: candidate, in: cache)
            }
            let store = TaskPackTestSupport.stubbedStore(cache: cache)
            await OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                let resolution = pack.resolve()
                XCTAssertEqual(resolution.status, .unsupportedOnSimulator)
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
        }

        // MARK: - Label mapping

        func testOnlyMappedLabelsAreReturnedWithTheirCategories() async throws {
            try skipUnlessMLXCanRun()
            let note = "Asthma and diabetes noted; fibrosis, hypertension, metformin, left lung, biopsy, Zorvexa."
            let (pack, best) = try await preparedPack([
                ("Asthma", "DISEASE", 0.93),
                ("diabetes", "Disease", 0.91),
                ("fibrosis", "B-PATHOLOGY", 0.88),
                ("hypertension", "condition", 0.86),
                ("metformin", "CHEM", 0.95),
                ("left lung", "ANATOMY", 0.9),
                ("biopsy", "procedure", 0.9),
                ("Zorvexa", "first_name", 0.97),
                ("noted", "OTHER", 0.9),
                ("lung", "unmapped_label", 0.9),
            ])
            let categories = Set(pack.definition.canonicalLabels)
            let expected: [(String, String, String)] = [
                ("Asthma", "DISEASE", "DISEASE"),
                ("diabetes", "Disease", "DISEASE"),
                ("fibrosis", "B-PATHOLOGY", "PATHOLOGY"),
                ("hypertension", "condition", "CONDITION"),
            ].filter { categories.contains($0.2) }
            XCTAssertEqual(expected.count, 4, "the pack must declare DISEASE, PATHOLOGY and CONDITION")

            let extraction = try pack.extract(note)
            XCTAssertEqual(extraction.sourceModelID, best.repoID)
            XCTAssertEqual(extraction.notice, pack.notice)
            XCTAssertEqual(extraction.entities.map(\.label), expected.map(\.1))
            XCTAssertEqual(extraction.entities.map(\.category), expected.map(\.2))
            XCTAssertTrue(extraction.entities.allSatisfy { categories.contains($0.category) })
            XCTAssertTrue(extraction.entities.allSatisfy { $0.sourceModelID == best.repoID })
            for (entity, item) in zip(extraction.entities, expected) {
                let range = try XCTUnwrap(entity.range(in: note))
                XCTAssertEqual(String(note[range]), item.0)
            }

            XCTAssertEqual(pack.category(forModelLabel: "I-DISEASE"), "DISEASE")
            XCTAssertNil(pack.category(forModelLabel: "CHEMICAL"))
            XCTAssertNil(pack.category(forModelLabel: "ANATOMY"))
            XCTAssertNil(pack.category(forModelLabel: "something_else"))
        }

        func testConfidenceThresholdDropsLowScoringMentions() async throws {
            try skipUnlessMLXCanRun()
            let (pack, _) = try await preparedPack([
                ("asthma", "DISEASE", 0.4),
                ("diabetes", "DISEASE", 0.8),
            ])
            let note = "asthma and diabetes"
            XCTAssertEqual(try pack.extract(note).entities.map(\.start), [11])
            XCTAssertEqual(try pack.extract(note, confidenceThreshold: 0.3).entities.map(\.start), [0, 11])
        }

        func testOffsetsAreUnicodeScalarOffsetsForANonASCIINote() async throws {
            try skipUnlessMLXCanRun()
            // An emoji (two UTF-16 units), a decomposed accent (two scalars in
            // one grapheme) and CJK text precede the mentions.
            let note = "Pt 😷 note, Cafe\u{301} visit 診察: pneumonia 🫁 and sepsis."
            let (pack, _) = try await preparedPack([
                ("pneumonia", "DISEASE", 0.9),
                ("sepsis", "DISEASE", 0.9),
            ])
            let entities = try pack.extract(note).entities
            XCTAssertEqual(entities.count, 2)
            let scalars = Array(note.unicodeScalars)
            for (entity, needle) in zip(entities, ["pneumonia", "sepsis"]) {
                let expected = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: needle, in: note))
                XCTAssertEqual(entity.start, expected.start)
                XCTAssertEqual(entity.end, expected.end)
                XCTAssertEqual(
                    String(String.UnicodeScalarView(scalars[entity.start..<entity.end])),
                    needle
                )
                let range = try XCTUnwrap(note.range(of: needle))
                XCTAssertNotEqual(note.utf16.distance(from: note.startIndex, to: range.lowerBound), entity.start)
                XCTAssertNotEqual(note.distance(from: note.startIndex, to: range.lowerBound), entity.start)
            }
        }

        // MARK: - Consent and privacy

        func testNoNetworkRequestHappensWithoutDownloadConsent() async throws {
            try skipUnlessMLXCanRun()
            XCTAssertTrue(URLProtocol.registerClass(TaskPackHubStub.self))
            defer { URLProtocol.unregisterClass(TaskPackHubStub.self) }
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: []))
            let pack = try ClinicalEntityPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)

            for store in [TaskPackTestSupport.stubbedStore(cache: cache), OpenMedTaskPackStore(cacheDirectory: cache)] {
                do {
                    try await pack.prepare(store: store)
                    XCTFail("An empty cache must not prepare without consent")
                } catch {
                    XCTAssertEqual(error as? OpenMedTaskPackError, .notCached)
                }
            }
            XCTAssertEqual(TaskPackHubStub.requestedPaths, [])
            XCTAssertEqual(probe.loadedDirectories, [])
            XCTAssertThrowsError(try pack.extract("asthma")) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .notPrepared)
            }
        }

        func testResultsCarryOffsetsNotMentionText() async throws {
            try skipUnlessMLXCanRun()
            let mention = "zorvexitis"
            let (pack, _) = try await preparedPack([(mention, "DISEASE", 0.9)])
            let extraction = try pack.extract("History: \(mention).")
            XCTAssertEqual(extraction.entities.count, 1)
            var dumped = ""
            dump(extraction, to: &dumped)
            for rendering in [String(describing: extraction), String(reflecting: extraction), dumped] {
                XCTAssertFalse(rendering.contains(mention))
                XCTAssertFalse(rendering.contains("History"))
            }
        }
    }
#endif
