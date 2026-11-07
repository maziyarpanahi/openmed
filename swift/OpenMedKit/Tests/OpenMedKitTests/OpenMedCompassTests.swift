#if canImport(MLXLMCommon) && !os(watchOS) && !os(visionOS)
    import CryptoKit
    import Foundation
    import XCTest

    @testable import OpenMedKit

    final class OpenMedCompassTests: XCTestCase {
        private let privacyPrompt =
            "In one concise sentence, explain how running a vision-language model "
            + "entirely on-device can improve privacy for clinical documents."

        private let factPrompt =
            "A synthetic note states: \"The follow-up appointment is scheduled for "
            + "Tuesday at 10:30 AM.\" What day is the follow-up? Answer with only the day."

        private let documentPrompt =
            "This is synthetic test data. In one concise sentence, report the exact "
            + "patient name, record ID, medication with dose and frequency, and allergy "
            + "shown in the image."

        private let chartPrompt =
            "Which category has the tallest bar, and what exact value is printed above "
            + "it? Answer concisely."

        func testSmartResizeMatchesCompassReferenceDimensions() throws {
            let document = try OpenMedCompassProcessor.targetSize(
                height: 900,
                width: 1_280,
                factor: 32
            )
            let chart = try OpenMedCompassProcessor.targetSize(
                height: 850,
                width: 1_200,
                factor: 32
            )

            XCTAssertEqual(document.0, 896)
            XCTAssertEqual(document.1, 1_280)
            XCTAssertEqual(chart.0, 864)
            XCTAssertEqual(chart.1, 1_216)
        }

        func testProcessorConfigurationUsesCompassPixelDefaults() throws {
            let data = Data(
                #"{"processor_class":"CohereCompassProcessor","image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"merge_size":2,"patch_size":16,"temporal_patch_size":2}"#.utf8
            )
            let configuration = try JSONDecoder().decode(
                OpenMedCompassProcessorConfiguration.self,
                from: data
            )

            XCTAssertEqual(configuration.minPixels, 16_384)
            XCTAssertEqual(configuration.maxPixels, 3_868_706)
            XCTAssertEqual(configuration.patchSize * configuration.mergeSize, 32)
        }

        func testLocalArtifactGeneratesCoherentTextAndImageAnswers() async throws {
            let environment = ProcessInfo.processInfo.environment
            guard let artifactPath = environment["OPENMED_COMPASS_MLX_ARTIFACT"] else {
                throw XCTSkip(
                    "Set OPENMED_COMPASS_MLX_ARTIFACT to run the local Compass parity test."
                )
            }
            guard let fixturePath = environment["OPENMED_COMPASS_FIXTURE_DIRECTORY"] else {
                throw XCTSkip(
                    "Set OPENMED_COMPASS_FIXTURE_DIRECTORY to run the image parity test."
                )
            }

            let model = try await OpenMedVisionLanguageModel.load(
                modelDirectory: URL(fileURLWithPath: artifactPath, isDirectory: true)
            )
            let expected = try referenceResponses(environment: environment)
            let documentURL = URL(fileURLWithPath: fixturePath)
                .appending(path: "synthetic_clinical_document.png")
            // Known synthetic identifiers; this detector is not qualification.
            let releaseGuard = GeneratedTextGuard(detector: { _ in [] }, detectorVersion: "synthetic-parity-v1")
            let documentDigest = digest(try Data(contentsOf: documentURL))
            let identifiers = [
                GeneratedIdentifierDigest(
                    assetDigest: documentDigest, label: .name,
                    digest: GeneratedTextGuard.identifierDigest("Alex Rivera"), normalizedLength: 10),
                GeneratedIdentifierDigest(
                    assetDigest: documentDigest, label: .mrn,
                    digest: GeneratedTextGuard.identifierDigest("SYN-2048"), normalizedLength: 8),
            ]
            let privacy = try await model.generate(
                privacyPrompt, maxTokens: 96, releaseGuard: releaseGuard,
                assetDigest: digest(Data(privacyPrompt.utf8)))
            XCTAssertEqual(privacy.promptTokenCount, 30)
            XCTAssertTrue(
                privacy.text.localizedCaseInsensitiveContains("private")
                    || privacy.text.localizedCaseInsensitiveContains("privacy"),
                "Missing privacy concept"
            )
            for term in ["clinical", "device", "sensitive"] {
                XCTAssertTrue(
                    privacy.text.localizedCaseInsensitiveContains(term),
                    "Missing expected concept"
                )
            }
            XCTAssertTrue(
                ["cloud", "network", "transmit", "local"].contains { term in
                    privacy.text.localizedCaseInsensitiveContains(term)
                },
                "Missing local-data rationale"
            )

            let fact = try await model.generate(
                factPrompt, maxTokens: 32, releaseGuard: releaseGuard,
                assetDigest: digest(Data(factPrompt.utf8)))
            XCTAssertEqual(fact.text, "Tuesday")
            XCTAssertEqual(fact.promptTokenCount, 44)
            if let response = expected["text_fact_extraction"] {
                XCTAssertEqual(fact.text, response)
            }

            let document = try await model.generate(
                documentPrompt,
                imageURL: documentURL,
                maxTokens: 96,
                releaseGuard: releaseGuard,
                assetDigest: documentDigest,
                identifiers: identifiers
            )
            XCTAssertEqual(document.promptTokenCount, 1_161)
            for expected in [
                "Metformin", "500", "twice", "Penicillin",
            ] {
                XCTAssertTrue(
                    document.text.localizedCaseInsensitiveContains(expected),
                    "Missing expected clinical fixture concept"
                )
            }
            XCTAssertFalse(document.text.localizedCaseInsensitiveContains("Alex Rivera"))
            XCTAssertFalse(document.text.localizedCaseInsensitiveContains("SYN-2048"))
            XCTAssertGreaterThanOrEqual(document.text.components(separatedBy: "[REDACTED]").count - 1, 2)
            XCTAssertEqual(document.releaseEnvelope?.guardReport.decision, "redact")
            XCTAssertTrue(document.tokenIDs.isEmpty)

            let chartURL = URL(fileURLWithPath: fixturePath)
                .appending(path: "synthetic_clinic_chart.png")
            let chart = try await model.generate(
                chartPrompt,
                imageURL: chartURL,
                maxTokens: 32,
                releaseGuard: releaseGuard,
                assetDigest: digest(try Data(contentsOf: chartURL))
            )
            XCTAssertEqual(chart.text, "Screening, 42")
            XCTAssertEqual(chart.promptTokenCount, 1_053)
            if let response = expected["image_chart"] {
                XCTAssertEqual(chart.text, response)
            }

            XCTAssertTrue(privacy.tokenIDs.isEmpty)
            XCTAssertTrue(fact.tokenIDs.isEmpty)
            XCTAssertTrue(chart.tokenIDs.isEmpty)
        }

        private func digest(_ data: Data) -> String {
            SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        }

        private func referenceResponses(
            environment: [String: String]
        ) throws -> [String: String] {
            guard let path = environment["OPENMED_COMPASS_REFERENCE_REPORT"] else {
                return [:]
            }
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let object = try JSONSerialization.jsonObject(with: data)
            guard
                let root = object as? [String: Any],
                let cases = root["cases"] as? [[String: Any]]
            else {
                XCTFail("Invalid Compass reference report: \(path)")
                return [:]
            }
            return Dictionary(
                uniqueKeysWithValues: cases.compactMap { item in
                    guard
                        let identifier = item["id"] as? String,
                        let response = item["response"] as? String
                    else { return nil }
                    return (identifier, response)
                }
            )
        }
    }
#endif
