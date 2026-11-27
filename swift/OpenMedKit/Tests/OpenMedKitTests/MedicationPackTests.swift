#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// `MedicationPack` returns medication mentions as data with the
    /// extraction-only notice attached, and nothing that reads as advice.
    final class MedicationPackTests: XCTestCase {
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
        ) async throws -> (MedicationPack, OpenMedTaskPackCandidate) {
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: detections))
            let pack = try MedicationPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
            let best = try XCTUnwrap(pack.definition.candidates.first)
            try TaskPackTestSupport.cacheSnapshot(of: best, in: cache)
            let prepared = try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            XCTAssertEqual(prepared, best)
            return (pack, best)
        }

        // MARK: - Resolve and notice

        func testResolveReturnsTheManifestCandidatesAndTheExtractionOnlyNotice() throws {
            try skipUnlessMLXCanRun()
            let pack = try MedicationPack()
            let resolution = pack.resolve()
            XCTAssertEqual(resolution.status, .resolved)
            XCTAssertEqual(resolution.repoIDs, try OpenMedTaskPackManifestTests.sharedAppleRepoIDs(pack: "medication"))
            XCTAssertTrue(resolution.candidates.allSatisfy { $0.license == "apache-2.0" })
            XCTAssertEqual(pack.definition.task, "medication_mention_extraction")
            XCTAssertEqual(pack.notice.key, MedicationPack.noticeKey)
            XCTAssertTrue(pack.notice.text.localizedCaseInsensitiveContains("never gives dosing advice"))
            XCTAssertTrue(pack.notice.text.localizedCaseInsensitiveContains("never triggers an action"))
        }

        func testAManifestWithoutTheExtractionOnlyNoticeIsRejected() throws {
            var document = try XCTUnwrap(
                JSONSerialization.jsonObject(with: OpenMedTaskPackManifestTests.sharedManifestData()) as? [String: Any]
            )
            var packs = try XCTUnwrap(document["packs"] as? [[String: Any]])
            let index = try XCTUnwrap(packs.firstIndex { $0["pack_id"] as? String == "medication" })
            packs[index]["notice_key"] = "openmed.notice.pii.review_required"
            document["packs"] = packs
            let manifest = try OpenMedTaskPackManifest.decode(JSONSerialization.data(withJSONObject: document))
            XCTAssertThrowsError(try MedicationPack(manifest: manifest)) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .manifestInvalid)
            }
        }

        func testSimulatorSkipsEveryMLXCandidateWithATypedResult() async throws {
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: []))
            let pack = try MedicationPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
            for candidate in pack.definition.candidates {
                try TaskPackTestSupport.cacheSnapshot(of: candidate, in: cache)
            }
            let store = TaskPackTestSupport.stubbedStore(cache: cache)
            await OpenMedMLXRuntimeSupport.$simulatesSimulator.withValue(true) {
                XCTAssertEqual(pack.resolve().status, .unsupportedOnSimulator)
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

        // MARK: - Extraction

        func testMentionsComeBackWithOffsetsAndTheNoticeAttached() async throws {
            try skipUnlessMLXCanRun()
            let note = "Résumé 💊: continue metformin, start lisinopril; aspirin held; diabetes stable."
            let (pack, best) = try await preparedPack([
                ("metformin", "CHEM", 0.94),
                ("lisinopril", "Chemical", 0.9),
                ("aspirin", "DRUG", 0.9),
                ("diabetes", "DISEASE", 0.9),
                ("held", "OTHER", 0.9),
            ])
            let labels = Set(pack.definition.canonicalLabels)
            XCTAssertTrue(labels.contains("CHEMICAL"), "PharmaDetect models emit CHEM, which normalizes to CHEMICAL")

            let extraction = try pack.extract(note)
            XCTAssertEqual(extraction.notice, pack.notice)
            XCTAssertEqual(extraction.sourceModelID, best.repoID)
            XCTAssertEqual(extraction.mentions.map(\.label), ["CHEM", "Chemical"])
            XCTAssertEqual(extraction.mentions.map(\.normalizedLabel), ["CHEMICAL", "CHEMICAL"])
            for (mention, needle) in zip(extraction.mentions, ["metformin", "lisinopril"]) {
                let expected = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: needle, in: note))
                XCTAssertEqual([mention.start, mention.end], [expected.start, expected.end])
                XCTAssertEqual(String(note[try XCTUnwrap(mention.range(in: note))]), needle)
                XCTAssertEqual(mention.notice, pack.notice)
                XCTAssertEqual(mention.notice.key, MedicationPack.noticeKey)
                XCTAssertEqual(mention.sourceModelID, best.repoID)
            }

            XCTAssertEqual(pack.normalizedLabel(forModelLabel: "B-CHEM"), "CHEMICAL")
            XCTAssertNil(pack.normalizedLabel(forModelLabel: "DRUG"))
            XCTAssertNil(pack.normalizedLabel(forModelLabel: "MEDICATION"))
        }

        func testNoResultTypeHasADoseInteractionOrActionField() async throws {
            try skipUnlessMLXCanRun()
            let (pack, _) = try await preparedPack([("warfarin", "CHEM", 0.9)])
            let extraction = try pack.extract("warfarin 5 mg daily")
            let mention = try XCTUnwrap(extraction.mentions.first)
            let forbidden = ["dose", "dosing", "dosage", "interaction", "action", "recommend", "advice", "frequency"]
            for value in [extraction as Any, mention as Any, mention.notice as Any] {
                for child in Mirror(reflecting: value).children {
                    let name = (child.label ?? "").lowercased()
                    for word in forbidden {
                        XCTAssertFalse(name.contains(word), "\(type(of: value)).\(name)")
                    }
                }
            }
            XCTAssertEqual(
                Mirror(reflecting: mention).children.compactMap(\.label),
                ["normalizedLabel", "label", "start", "end", "confidence", "sourceModelID", "notice"]
            )
            XCTAssertEqual(
                Mirror(reflecting: extraction).children.compactMap(\.label),
                ["mentions", "sourceModelID", "notice"]
            )
            var dumped = ""
            dump(extraction, to: &dumped)
            XCTAssertFalse(dumped.contains("warfarin"))
            XCTAssertFalse(dumped.contains("5 mg"))
        }

        // MARK: - Consent

        func testNoNetworkRequestHappensWithoutConsent() async throws {
            try skipUnlessMLXCanRun()
            XCTAssertTrue(URLProtocol.registerClass(TaskPackHubStub.self))
            defer { URLProtocol.unregisterClass(TaskPackHubStub.self) }
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: []))
            let pack = try MedicationPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)

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
            XCTAssertThrowsError(try pack.extract("metformin")) { error in
                XCTAssertEqual(error as? OpenMedTaskPackError, .notPrepared)
            }
        }
    }
#endif
