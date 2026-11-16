#if canImport(MLX) && canImport(Tokenizers) && !os(watchOS) && !os(visionOS)
    import Foundation
    import XCTest

    @testable import OpenMedKit

    /// `ClinicalEntityPack` attaches assertion axes to mapped mentions and
    /// never drops a negated, uncertain, hypothetical or non-patient one.
    final class ClinicalAssertionPackTests: XCTestCase {
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
        ) async throws -> ClinicalEntityPack {
            let probe = TaskPackLoaderProbe(recognizer: TaskPackFakeRecognizer(detections: detections))
            let pack = try ClinicalEntityPack(manifest: OpenMedTaskPackManifest.bundled(), loader: probe.load)
            try TaskPackTestSupport.cacheSnapshot(of: try XCTUnwrap(pack.definition.candidates.first), in: cache)
            try await pack.prepare(store: TaskPackTestSupport.stubbedStore(cache: cache))
            return pack
        }

        func testDeniedChestPainIsNegatedAndMothersCancerIsFamily() async throws {
            try skipUnlessMLXCanRun()
            let pack = try await preparedPack([
                ("chest pain", "DISEASE", 0.9),
                ("breast cancer", "DISEASE", 0.9),
            ])

            let denied = try pack.extract("Patient denies chest pain.")
            XCTAssertNil(denied.assertionUnavailableReason)
            let pain = try XCTUnwrap(denied.entities.first)
            XCTAssertEqual([pain.start, pain.end], [15, 25])
            XCTAssertEqual(pain.assertion?.negation, .negated)
            XCTAssertEqual(pain.assertion?.experiencer, .patient)

            let family = try pack.extract("Mother had breast cancer.")
            let cancer = try XCTUnwrap(family.entities.first)
            XCTAssertEqual(cancer.assertion?.experiencer, .family)
            XCTAssertEqual(cancer.assertion?.negation, .affirmed)
        }

        func testFlaggedMentionsAreKeptWithTheirAxes() async throws {
            try skipUnlessMLXCanRun()
            let note = "No fever. Possible pneumonia. If asthma flares, return. Donor has hepatitis. History of gout."
            let pack = try await preparedPack([
                ("fever", "DISEASE", 0.9),
                ("pneumonia", "DISEASE", 0.9),
                ("asthma", "CONDITION", 0.9),
                ("hepatitis", "DISEASE", 0.9),
                ("gout", "PATHOLOGY", 0.9),
            ])
            let entities = try pack.extract(note).entities
            XCTAssertEqual(entities.count, 5)
            let assertions = try entities.map { try XCTUnwrap($0.assertion) }
            XCTAssertEqual(assertions.map(\.negation), [.negated, .affirmed, .affirmed, .affirmed, .affirmed])
            XCTAssertEqual(assertions.map(\.uncertainty), [.certain, .uncertain, .uncertain, .certain, .certain])
            XCTAssertEqual(
                assertions.map(\.temporality),
                [.recent, .recent, .hypothetical, .recent, .historical]
            )
            XCTAssertEqual(assertions.map(\.experiencer), [.patient, .patient, .patient, .other, .patient])
        }

        func testAssertionUsesScalarOffsetsAndCallerSentences() async throws {
            try skipUnlessMLXCanRun()
            let note = "Note 𝟙: patient denies dysuria. Cough 😷 is present."
            let pack = try await preparedPack([
                ("dysuria", "DISEASE", 0.9),
                ("Cough", "DISEASE", 0.9),
            ])
            let first = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: "Note", in: note))
            let period = try XCTUnwrap(TaskPackTestSupport.scalarSpan(of: "dysuria.", in: note))
            let sentences = [
                OpenMedClinicalContext.Span(start: first.start, end: period.end),
                OpenMedClinicalContext.Span(start: period.end + 1, end: note.unicodeScalars.count),
            ]
            for extraction in [try pack.extract(note), try pack.extract(note, sentences: sentences)] {
                XCTAssertEqual(extraction.entities.map { $0.assertion?.negation }, [.negated, .affirmed])
            }
            XCTAssertThrowsError(
                try pack.extract(note, sentences: [.init(start: 0, end: note.unicodeScalars.count + 1)])
            ) { error in
                XCTAssertEqual(error as? OpenMedClinicalContextError, .invalidSentence)
            }
        }

        func testASpanishDeclaredInputReturnsNoAssertionAndTheReason() async throws {
            try skipUnlessMLXCanRun()
            let pack = try await preparedPack([("dolor torácico", "DISEASE", 0.9)])
            let extraction = try pack.extract("Paciente niega dolor torácico.", language: "es")
            XCTAssertEqual(extraction.assertionUnavailableReason, .unsupportedLanguage)
            XCTAssertEqual(extraction.entities.count, 1, "mentions are still returned")
            XCTAssertNil(extraction.entities.first?.assertion)

            let english = try pack.extract("Paciente niega dolor torácico.", language: "en")
            XCTAssertNil(english.assertionUnavailableReason)
            XCTAssertNotNil(english.entities.first?.assertion)
        }
    }
#endif
