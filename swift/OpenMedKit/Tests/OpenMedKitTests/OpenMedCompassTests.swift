#if canImport(MLXVLM) && canImport(MLXLMCommon) && !os(watchOS) && !os(visionOS)
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
            let privacy = try await model.generate(privacyPrompt, maxTokens: 96)
            XCTAssertEqual(privacy.promptTokenCount, 30)
            XCTAssertTrue(
                privacy.text.localizedCaseInsensitiveContains("private")
                    || privacy.text.localizedCaseInsensitiveContains("privacy"),
                "Missing privacy concept in: \(privacy.text)"
            )
            for term in ["clinical", "device", "sensitive"] {
                XCTAssertTrue(
                    privacy.text.localizedCaseInsensitiveContains(term),
                    "Missing \(term) in: \(privacy.text)"
                )
            }
            XCTAssertTrue(
                ["cloud", "network", "transmit", "local"].contains { term in
                    privacy.text.localizedCaseInsensitiveContains(term)
                },
                "Missing local-data rationale in: \(privacy.text)"
            )

            let fact = try await model.generate(factPrompt, maxTokens: 32)
            XCTAssertEqual(fact.text, "Tuesday")
            XCTAssertEqual(fact.promptTokenCount, 44)
            if let response = expected["text_fact_extraction"] {
                XCTAssertEqual(fact.text, response)
            }

            let document = try await model.generate(
                documentPrompt,
                imageURL: documentURL,
                maxTokens: 96
            )
            XCTAssertEqual(document.promptTokenCount, 1_161)
            for expected in [
                "Alex Rivera", "SYN-2048", "Metformin", "500", "twice", "Penicillin",
            ] {
                XCTAssertTrue(
                    document.text.localizedCaseInsensitiveContains(expected),
                    "Missing \(expected) in: \(document.text)"
                )
            }

            let chart = try await model.generate(
                chartPrompt,
                imageURL: URL(fileURLWithPath: fixturePath)
                    .appending(path: "synthetic_clinic_chart.png"),
                maxTokens: 32
            )
            XCTAssertEqual(chart.text, "Screening, 42")
            XCTAssertEqual(chart.promptTokenCount, 1_053)
            if let response = expected["image_chart"] {
                XCTAssertEqual(chart.text, response)
            }

            print(
                "OPENMED_COMPASS_PARITY "
                    + "privacy=\(privacy.text.debugDescription)/\(privacy.tokenIDs) "
                    + "text=\(fact.text.debugDescription)/\(fact.tokenIDs) "
                    + "document=\(document.text.debugDescription)/\(document.tokenIDs) "
                    + "chart=\(chart.text.debugDescription)/\(chart.tokenIDs)"
            )
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
