import Foundation
import MLX
import Tokenizers
import XCTest
import ZIPFoundation
#if canImport(AppKit) && canImport(Vision)
import AppKit
import Vision
#endif
@testable import OpenMedKit

final class OpenMedMLXTests: XCTestCase {

    func testArtifactLoadsManifestAndTokenizerAssets() throws {
        let directory = try makeManifestOnlyArtifact()
        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)

        XCTAssertEqual(artifact.manifest.format, "openmed-mlx")
        XCTAssertEqual(artifact.configuration.modelType, "bert")
        XCTAssertNotNil(artifact.tokenizerDirectoryURL)
        XCTAssertEqual(
            artifact.tokenizerDirectoryURL?.standardizedFileURL,
            directory.standardizedFileURL
        )
        XCTAssertEqual(artifact.id2label[1], "B-NAME")
    }

    func testIsMLXModelCachedRecognizesCompleteCachedArtifact() throws {
        let cacheRoot = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )
        let repoID = "OpenMed/Test-Artifact"
        let revision = "demo"
        let cacheDirectory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheRoot
        )

        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        _ = try makeManifestOnlyArtifact(in: cacheDirectory)

        XCTAssertTrue(
            try OpenMedModelStore.isMLXModelCached(
                repoID: repoID,
                revision: revision,
                cacheDirectory: cacheRoot
            )
        )
    }

    func testMLXModelCacheStateRecognizesCompleteCachedArtifact() throws {
        let cacheRoot = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )
        let repoID = "OpenMed/Test-Ready-Artifact"
        let revision = "demo"
        let cacheDirectory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheRoot
        )

        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        _ = try makeManifestOnlyArtifact(in: cacheDirectory)

        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: repoID,
                revision: revision,
                cacheDirectory: cacheRoot
            ),
            .ready
        )
    }

    func testMLXModelCacheStateRecognizesPartialArtifact() throws {
        let cacheRoot = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )
        let repoID = "OpenMed/Test-Partial-Artifact"
        let revision = "demo"
        let cacheDirectory = try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: repoID,
            revision: revision,
            cacheDirectory: cacheRoot
        )

        try FileManager.default.createDirectory(
            at: cacheDirectory,
            withIntermediateDirectories: true
        )
        _ = try makeManifestOnlyArtifact(in: cacheDirectory)
        try FileManager.default.removeItem(at: cacheDirectory.appending(path: "weights.safetensors"))

        XCTAssertEqual(
            try OpenMedModelStore.mlxModelCacheState(
                repoID: repoID,
                revision: revision,
                cacheDirectory: cacheRoot
            ),
            .partial
        )
        XCTAssertFalse(
            try OpenMedModelStore.isMLXModelCached(
                repoID: repoID,
                revision: revision,
                cacheDirectory: cacheRoot
            )
        )
    }

    func testArtifactRejectsUnsupportedArchitecture() throws {
        let directory = try makeManifestOnlyArtifact(family: "deberta-v2")

        XCTAssertThrowsError(try OpenMedMLXArtifact(modelDirectoryURL: directory)) { error in
            guard case OpenMedMLXArtifactError.unsupportedArchitecture(let family) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(family, "deberta-v2")
        }
    }

    func testArtifactAcceptsNativeGLiNERFamilies() throws {
        let cases = [
            (
                task: "zero-shot-ner",
                family: "gliner-uni-encoder-span",
                promptSpec: [
                    "kind": "gliner-words",
                    "entity_token": "<<ENT>>",
                    "separator_token": "<<SEP>>",
                    "class_token_index": 128001,
                ] as [String: Any]
            ),
            (
                task: "zero-shot-sequence-classification",
                family: "gliclass-uni-encoder",
                promptSpec: [
                    "kind": "gliclass-uni-encoder",
                    "label_token": "<<LABEL>>",
                    "separator_token": "<<SEP>>",
                    "class_token_index": 128001,
                    "text_token_index": 128002,
                ] as [String: Any]
            ),
            (
                task: "zero-shot-relation-extraction",
                family: "gliner-uni-encoder-token-relex",
                promptSpec: [
                    "kind": "gliner-relex",
                    "entity_token": "<<ENT>>",
                    "relation_token": "<<REL>>",
                    "separator_token": "<<SEP>>",
                    "class_token_index": 128001,
                    "rel_token_index": 128003,
                ] as [String: Any]
            ),
        ]

        for testCase in cases {
            let directory = try makeManifestOnlyArtifact(
                task: testCase.task,
                family: testCase.family,
                configModelType: "deberta-v2",
                promptSpec: testCase.promptSpec
            )
            let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)

            XCTAssertEqual(artifact.manifest.task, testCase.task)
            XCTAssertEqual(artifact.manifest.family, testCase.family)
            XCTAssertNotNil(artifact.manifest.promptSpec)
        }
    }

    func testArtifactAcceptsPrivacyFilterFamily() throws {
        let directory = try makePrivacyFilterManifestOnlyArtifact()
        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)

        XCTAssertEqual(artifact.task, .tokenClassification)
        XCTAssertEqual(artifact.family, .openaiPrivacyFilter)
        XCTAssertEqual(artifact.manifest.family, "openai-privacy-filter")
    }

    func testPrivacyFilterConfigDecodesArchitectureFields() throws {
        let directory = try makePrivacyFilterManifestOnlyArtifact()
        let config = try OpenMedMLXArtifact(modelDirectoryURL: directory).configuration

        XCTAssertEqual(config.modelType, "openai-privacy-filter")
        XCTAssertEqual(config.encoding, "o200k_base")
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.numKeyValueHeads, 1)
        XCTAssertEqual(config.numExperts, 4)
        XCTAssertEqual(config.expertsPerToken, 2)
        XCTAssertEqual(config.bidirectionalLeftContext, 1)
        XCTAssertEqual(config.bidirectionalRightContext, 1)
        XCTAssertEqual(config.initialContextLength, 64)
        XCTAssertEqual(config.ropeTheta, 150_000)
        XCTAssertEqual(config.ropeScalingFactor, 1.0)
        XCTAssertEqual(config.parameterDType, "float32")
        XCTAssertEqual(config.viterbiBiases["transition_bias_background_stay"], 0.0)
    }

    func testPrivacyFilterTokenizerMatchesTiktokenGoldens() throws {
        let directory = try localArtifactURL(from: "OPENMED_PRIVACY_FILTER_MLX_ARTIFACT")
        let tokenizer = try OpenMedPrivacyFilterTokenizer(directoryURL: directory)

        let encoded = try tokenizer.encode(
            "My name is Alice Smith and my email is alice.smith@example.com.",
            maxTokens: 128
        )

        XCTAssertEqual(encoded.tokenIDs, [
            5444, 1308, 382, 44045, 16627, 326, 922, 3719, 382, 134271, 640, 68671,
            81309, 1136, 13,
        ])
        let tokenPieces = encoded.charStarts.indices.map { index in
            let lower = encoded.decodedText.index(
                encoded.decodedText.startIndex,
                offsetBy: encoded.charStarts[index]
            )
            let upper = encoded.decodedText.index(
                encoded.decodedText.startIndex,
                offsetBy: encoded.charEnds[index]
            )
            return String(encoded.decodedText[lower..<upper])
        }
        XCTAssertEqual(tokenPieces, [
            "My", " name", " is", " Alice", " Smith", " and", " my", " email",
            " is", " alice", ".s", "mith", "@example", ".com", ".",
        ])
    }

    func testPrivacyFilterViterbiRejectsInvalidInsideStart() {
        let labelInfo = OpenMedPrivacyFilterLabelInfo(id2label: [
            0: "O",
            1: "B-private_person",
            2: "I-private_person",
            3: "E-private_person",
            4: "S-private_email",
        ])

        let path = OpenMedPrivacyFilterViterbi.decode(
            tokenLogProbabilities: [
                [-10.0, -9.0, 0.0, -9.0, -8.0],
                [-10.0, -9.0, -8.0, 0.0, -8.0],
            ],
            labelInfo: labelInfo,
            biases: [:]
        )

        XCTAssertEqual(path, [1, 3])
    }

    func testTinyPrivacyFilterModelForwardShape() throws {
        try requireUsableMLXRuntime()
        let directory = try makePrivacyFilterManifestOnlyArtifact()
        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)
        let model = OpenMedPrivacyFilterForTokenClassification(artifact.configuration)

        let logits = model(
            MLXArray([Int32(1), Int32(2), Int32(3), Int32(4)], [1, 4]),
            attentionMask: MLXArray.ones([1, 4], type: Bool.self)
        )
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 4, artifact.configuration.numLabels])
    }

    func testArtifactRejectsUnknownGLiNERVariant() throws {
        let directory = try makeManifestOnlyArtifact(
            task: "zero-shot-ner",
            family: "gliner-bi-encoder-span",
            configModelType: "deberta-v2"
        )

        XCTAssertThrowsError(try OpenMedMLXArtifact(modelDirectoryURL: directory)) { error in
            guard case OpenMedMLXArtifactError.unsupportedArchitecture(let family) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(family, "gliner-bi-encoder-span")
        }
    }

    func testArtifactRequiresTokenizerAssets() throws {
        let directory = try makeManifestOnlyArtifact()
        try FileManager.default.removeItem(at: directory.appending(path: "tokenizer.json"))

        XCTAssertThrowsError(try OpenMedMLXArtifact(modelDirectoryURL: directory)) { error in
            guard case OpenMedMLXArtifactError.missingTokenizerAssets = error else {
                return XCTFail("Unexpected error: \(error)")
            }
        }
    }

    func testLegacyArtifactFallsBackToSourceTokenizer() throws {
        let directory = try makeLegacyArtifactWithoutTokenizerAssets()

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)

        XCTAssertNil(artifact.tokenizerDirectoryURL)
        XCTAssertEqual(
            artifact.tokenizerName,
            "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
        )
        XCTAssertEqual(artifact.manifest.preferredWeights, "weights.safetensors")
        XCTAssertTrue(artifact.weightCandidateURLs.contains(directory.appending(path: "weights.safetensors")))
    }

    func testLegacyArtifactIgnoresIncompleteLocalBPETokenizerAssets() throws {
        let directory = try makeLegacyArtifactWithoutTokenizerAssets()
        try "{}".write(
            to: directory.appending(path: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        try "{}".write(
            to: directory.appending(path: "vocab.json"),
            atomically: true,
            encoding: .utf8
        )

        let artifact = try OpenMedMLXArtifact(modelDirectoryURL: directory)

        XCTAssertNil(artifact.tokenizerDirectoryURL)
        XCTAssertEqual(
            artifact.tokenizerName,
            "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1"
        )
    }

    func testPrepareTokenizerDirectoryAddsStubConfigForTokenizerOnlyFolder() throws {
        let directory = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )

        try makeTokenizerData(modelType: "BPE").write(
            to: directory.appending(path: "tokenizer.json")
        )
        try makeTokenizerConfig(tokenizerClass: "RobertaTokenizer").write(
            to: directory.appending(path: "tokenizer_config.json")
        )

        let preparedDirectory = try OpenMed.prepareTokenizerDirectory(directory)

        XCTAssertNotEqual(preparedDirectory.standardizedFileURL, directory.standardizedFileURL)
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: preparedDirectory.appending(path: "config.json").path
            )
        )
    }

    func testPrepareTokenizerDirectoryPatchesBigMedStyleUnigramTokenizer() throws {
        let directory = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )

        try makeTokenizerData(modelType: "Unigram").write(
            to: directory.appending(path: "tokenizer.json")
        )
        try makeTokenizerConfig(tokenizerClass: "RobertaTokenizer").write(
            to: directory.appending(path: "tokenizer_config.json")
        )

        let preparedDirectory = try OpenMed.prepareTokenizerDirectory(directory)
        let preparedConfigData = try Data(
            contentsOf: preparedDirectory.appending(path: "tokenizer_config.json")
        )
        let preparedConfig = try JSONSerialization.jsonObject(with: preparedConfigData)
            as? [String: Any]

        XCTAssertEqual(preparedConfig?["tokenizer_class"] as? String, "T5Tokenizer")
    }

    func testPatchTokenizerConfigDataLeavesBPEAlone() throws {
        let patched = try OpenMed.patchTokenizerConfigDataIfNeeded(
            tokenizerConfigData: makeTokenizerConfig(tokenizerClass: "RobertaTokenizer"),
            tokenizerData: makeTokenizerData(modelType: "BPE")
        )

        XCTAssertNil(patched)
    }

    func testPatchTokenizerConfigDataNormalizesListShapedExtraSpecialTokens() throws {
        let tokenizerConfig = try JSONSerialization.data(
            withJSONObject: [
                "tokenizer_class": "DebertaV2Tokenizer",
                "extra_special_tokens": ["<<ENT>>", "<<SEP>>"],
            ],
            options: [.prettyPrinted]
        )

        let patched = try OpenMed.patchTokenizerConfigDataIfNeeded(
            tokenizerConfigData: tokenizerConfig,
            tokenizerData: makeTokenizerData(modelType: "Unigram")
        )

        let patchedData = try XCTUnwrap(patched)
        let object = try JSONSerialization.jsonObject(with: patchedData) as? [String: Any]
        XCTAssertNil(object?["extra_special_tokens"])
        XCTAssertEqual(object?["additional_special_tokens"] as? [String], ["<<ENT>>", "<<SEP>>"])
        XCTAssertEqual(object?["tokenizer_class"] as? String, "T5Tokenizer")
    }

    func testGLiNERPromptEncoderBuildsWordMaskForFirstSubtokensOnly() {
        let encoder = OpenMedGLiNERPromptEncoder(tokenizer: FakeTokenizer())
        let encoded = encoder.encodeWords(
            ["<<ENT>>", "symptom", "<<SEP>>", "headache", "follow-up", "."],
            skipFirstWords: 3,
            maxSeqLength: 16
        )

        XCTAssertEqual(encoded.inputIDs, [101, 9001, 10, 9002, 20, 21, 30, 31, 32, 33, 102])
        XCTAssertEqual(encoded.attentionMask, Array(repeating: 1, count: encoded.inputIDs.count))
        XCTAssertEqual(encoded.wordsMask, [0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 0])
    }

    func testGLiNERSpanCandidateGenerationMatchesMaxWidthLayout() throws {
        try requireUsableMLXRuntime()
        let spans = OpenMedGLiNERPromptEncoder.buildCandidateSpanBatch(
            wordCount: 3,
            maxWidth: 2
        )

        XCTAssertEqual(spans.index.asArray(Int32.self).map(Int.init), [
            0, 0,
            0, 1,
            1, 1,
            1, 2,
            2, 2,
            2, 3,
        ])
        XCTAssertEqual(spans.mask.asArray(Bool.self), [true, true, true, true, true, false])
    }

    func testBuildOffsetsPrefersEarliestCaseInsensitiveMatch() {
        let text = "my name is John, I am 89 years old and I live at 22 blbd asdad, 75015, paris."
        let tokens = [
            "[CLS]", "my", "name", "is", "john", ",", "i", "am", "89", "years", "old",
            "and", "i", "live", "at", "22", "bl", "##bd", "asd", "##ad", ",", "750",
            "##15", ",", "paris", ".", "[SEP]",
        ]

        let offsets = OpenMed.buildOffsets(tokens: tokens, in: text)

        XCTAssertEqual(offsets[4].0, 11, "john should align to John")
        XCTAssertEqual(offsets[4].1, 15, "john should align to John")
        XCTAssertEqual(offsets[6].0, 17, "i should align to the uppercase I after John")
        XCTAssertEqual(offsets[6].1, 18, "i should align to the uppercase I after John")
        XCTAssertEqual(offsets[7].0, 19, "am should follow the same clause")
        XCTAssertEqual(offsets[7].1, 21, "am should follow the same clause")
        XCTAssertEqual(offsets[8].0, 22, "89 should keep its true numeric span")
        XCTAssertEqual(offsets[8].1, 24, "89 should keep its true numeric span")
        XCTAssertEqual(offsets[15].0, 49, "22 should still align after the earlier uppercase tokens")
        XCTAssertEqual(offsets[15].1, 51, "22 should still align after the earlier uppercase tokens")
        XCTAssertEqual(offsets[21].0, 64, "postcode should remain aligned")
        XCTAssertEqual(offsets[21].1, 67, "postcode should remain aligned")
        XCTAssertEqual(offsets[24].0, 71, "city should remain aligned")
        XCTAssertEqual(offsets[24].1, 76, "city should remain aligned")
    }

    func testPipelinePredictsEntitiesFromLocalArtifact() throws {
        try requireUsableMLXRuntime()
        let directory = try makeTinyArtifact()
        let pipeline = try MLXTokenClassificationPipeline(modelDirectoryURL: directory)

        let entities = try pipeline.predict(
            inputIDs: [101, 2001, 2002, 102],
            attentionMask: [1, 1, 1, 1],
            tokenTypeIDs: [0, 0, 0, 0],
            offsets: [(0, 0), (0, 4), (5, 8), (0, 0)],
            text: "John Doe"
        )

        XCTAssertEqual(entities.count, 2)
        XCTAssertEqual(entities[0].label, "NAME")
        XCTAssertEqual(entities[0].text, "John")
        XCTAssertEqual(entities[1].text, "Doe")
    }

    func testNPZFallbackLoadsWeights() throws {
        try requireUsableMLXRuntime()
        let directory = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )
        let archiveURL = directory.appending(path: "weights.npz")
        let npyData = makeNPY(
            descriptor: "<f4",
            shape: [2, 2],
            body: Data([0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64])
        )

        let npyFile = directory.appending(path: "classifier.bias.npy")
        try npyData.write(to: npyFile)

        let archive = try Archive(url: archiveURL, accessMode: .create)
        try archive.addEntry(with: "classifier.bias.npy", relativeTo: directory)

        let arrays = try OpenMedMLXWeightArchive.loadWeights(from: [archiveURL])
        let values = arrays["classifier.bias"]?.asArray(Float.self)

        XCTAssertEqual(values ?? [], [0, 1, 2, 3])
    }

    func testLocalSmokeArtifactPredictsExpectedBiomedElectraEntities() throws {
        try requireUsableMLXRuntime()

        let environment = ProcessInfo.processInfo.environment
        guard let artifactPath = environment["OPENMED_LOCAL_MLX_ARTIFACT"], !artifactPath.isEmpty else {
            throw XCTSkip("Set OPENMED_LOCAL_MLX_ARTIFACT to run local Swift MLX smoke tests.")
        }

        let text = "my name is John, I am 89 years old and I live at 22 blbd asdad, 75015, paris."
        let openmed = try OpenMed(
            backend: .mlx(modelDirectoryURL: URL(fileURLWithPath: artifactPath))
        )

        let entities = try openmed.analyzeText(text, confidenceThreshold: 0.0)

        XCTAssertTrue(
            entities.contains { $0.label == "first_name" && $0.text == "John" },
            "Expected first_name=John in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "age" && $0.text == "89" },
            "Expected age=89 in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "street_address" && $0.text.contains("22 blbd asdad") },
            "Expected street_address in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "postcode" && $0.text == "750" },
            "Expected postcode=750 in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "city" && $0.text.lowercased() == "paris" },
            "Expected city=paris in \(entities)"
        )
    }

    func testLocalPrivacyFilterArtifactRunsNativePII() throws {
        try requireUsableMLXRuntime()
        let artifactURL = try localArtifactURL(from: "OPENMED_PRIVACY_FILTER_MLX_ARTIFACT")
        let openmed = try OpenMed(
            backend: .mlx(modelDirectoryURL: artifactURL),
            maxSeqLength: 256
        )

        let entities = try openmed.extractPII(
            "My name is Alice Smith, call 415-555-0101 or email alice.smith@example.com.",
            confidenceThreshold: 0.0
        )

        XCTAssertTrue(
            entities.contains { $0.label == "private_person" && $0.text.contains("Alice") },
            "Expected private_person in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "private_email" && $0.text.contains("@example.com") },
            "Expected private_email in \(entities)"
        )
        XCTAssertTrue(
            entities.contains { $0.label == "private_phone" && $0.text.contains("415") },
            "Expected private_phone in \(entities)"
        )
    }

    #if canImport(AppKit) && canImport(Vision)
    func testSampleClinicalDocumentPIIEndToEnd() async throws {
        try requireUsableMLXRuntime()

        let repoID = "OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx"
        let cacheState = try OpenMedModelStore.mlxModelCacheState(repoID: repoID)
        guard cacheState == .ready else {
            throw XCTSkip(
                "Cache \(cacheState.rawValue) for \(repoID). Download the artifact into the OpenMed MLX cache to run this smoke test."
            )
        }

        let ocrText = try Self.extractSampleClinicalDocumentOCRText()
        let artifactURL = try OpenMedModelStore.cachedMLXModelDirectory(repoID: repoID)
        let openmed = try OpenMed(backend: .mlx(modelDirectoryURL: artifactURL))

        let rawEntities = try openmed.analyzeText(ocrText, confidenceThreshold: 0.0)
        let piiEntities = try openmed.extractPII(
            ocrText,
            confidenceThreshold: 0.0,
            useSmartMerging: true
        )

        print("=== SAMPLE OCR TEXT ===")
        print(ocrText)
        print("=== SAMPLE RAW ENTITIES ===")
        print(rawEntities.map(\.description).joined(separator: "\n"))
        print("=== SAMPLE FINAL PII ENTITIES ===")
        print(piiEntities.map(\.description).joined(separator: "\n"))

        XCTAssertTrue(
            piiEntities.contains { $0.label == "full_name" && $0.text == "Eleanor Ruiz" },
            "Expected patient full name in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "phone_number" && $0.text == "(415) 555-0142" },
            "Expected patient phone number in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "phone_number" && $0.text == "(415) 555-0199" },
            "Expected emergency contact phone number in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "email" && $0.text == "eleanor.ruiz@sampleclinic.test" },
            "Expected email in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "date_of_birth" && $0.text == "03/14/1981" },
            "Expected DOB in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "medical_record_number" && $0.text == "MRN-448271" },
            "Expected MRN in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "insurance_id" && $0.text == "HMO-99318442" },
            "Expected insurance ID in \(piiEntities)"
        )
        XCTAssertTrue(
            piiEntities.contains { $0.label == "street_address" && $0.text == "1942 Harbor View Drive, Marseille, CA 92111" },
            "Expected street address in \(piiEntities)"
        )
    }

    func testSampleClinicalDocumentGLiNERMedicationCoverageAtDemoThreshold() async throws {
        try requireUsableMLXRuntime()

        let repoID = "OpenMed/gliner-multi-pii-v1-mlx"
        let artifactURL: URL
        if let artifactPath = ProcessInfo.processInfo.environment["OPENMED_GLINER_SPAN_ARTIFACT"],
           !artifactPath.isEmpty {
            artifactURL = URL(fileURLWithPath: (artifactPath as NSString).expandingTildeInPath)
        } else if let homeDirectory = FileManager.default.homeDirectoryForCurrentUser as URL? {
            let localCacheArtifactURL = homeDirectory
                .appending(path: ".cache")
                .appending(path: "openmed-mlx")
                .appending(path: "OpenMed")
                .appending(path: "gliner-multi-pii-v1-mlx")
                .appending(path: "main")
            if FileManager.default.fileExists(
                atPath: localCacheArtifactURL.appending(path: "openmed-mlx.json").path
            ) {
                artifactURL = localCacheArtifactURL
            } else {
                let cacheState = try OpenMedModelStore.mlxModelCacheState(repoID: repoID)
                guard cacheState == .ready else {
                    throw XCTSkip(
                        "Cache \(cacheState.rawValue) for \(repoID). Download the artifact into the OpenMed MLX cache or set OPENMED_GLINER_SPAN_ARTIFACT to run this smoke test."
                    )
                }
                artifactURL = try OpenMedModelStore.cachedMLXModelDirectory(repoID: repoID)
            }
        } else {
            let cacheState = try OpenMedModelStore.mlxModelCacheState(repoID: repoID)
            guard cacheState == .ready else {
                throw XCTSkip(
                    "Cache \(cacheState.rawValue) for \(repoID). Download the artifact into the OpenMed MLX cache or set OPENMED_GLINER_SPAN_ARTIFACT to run this smoke test."
                )
            }
            artifactURL = try OpenMedModelStore.cachedMLXModelDirectory(repoID: repoID)
        }

        let labels = [
            "symptom",
            "condition",
            "medical history",
            "medication",
            "dosage",
            "allergy",
            "treatment",
            "procedure",
            "follow-up plan",
            "care plan",
            "care setting",
            "work status",
        ]
        let ocrText = try Self.extractSampleClinicalDocumentOCRText()
        let pipeline = try OpenMedZeroShotNER(modelDirectoryURL: artifactURL)

        // Characterize the current demo note so we can distinguish model windowing
        // behavior from Swift runtime regressions.
        let fullNoteEntities = try pipeline.extract(
            ocrText,
            labels: labels,
            threshold: 0.6,
            flatNER: true
        )
        let fullNoteMedications = fullNoteEntities
            .filter { $0.label == "medication" }
            .map { $0.text.lowercased() }

        XCTAssertTrue(
            fullNoteMedications.contains { $0.contains("metformin") },
            "Expected metformin in \(fullNoteMedications)"
        )
        XCTAssertTrue(
            fullNoteMedications.contains { $0.contains("lisinopril") },
            "Expected lisinopril in \(fullNoteMedications)"
        )
        XCTAssertTrue(
            fullNoteMedications.contains { $0.contains("sumatriptan") },
            "Expected sumatriptan in \(fullNoteMedications)"
        )
        XCTAssertFalse(
            fullNoteMedications.contains { $0.contains("atorvastatin") },
            "The long OCR note currently drops atorvastatin at the demo threshold, which helps distinguish model-confidence issues from runtime regressions."
        )
        XCTAssertFalse(
            fullNoteMedications.contains { $0.contains("ibuprofen") },
            "The current long OCR note should drop later-plan medication mentions when they fall outside the GLiNER window."
        )

        let homeMedicationSnippet = """
        HOME MEDICATIONS
        Metformin 500 mg twice daily, lisinopril 10 mg daily, atorvastatin 20 mg nightly, and sumatriptan 50 mg as needed for migraine.
        """
        let homeMedicationEntities = try pipeline.extract(
            homeMedicationSnippet,
            labels: labels,
            threshold: 0.6,
            flatNER: true
        )
        let homeMedicationMatches = homeMedicationEntities
            .filter { $0.label == "medication" }
            .map { $0.text.lowercased() }

        XCTAssertTrue(
            homeMedicationMatches.contains { $0.contains("metformin") },
            "Expected metformin in the focused medication snippet: \(homeMedicationMatches)"
        )
        XCTAssertTrue(
            homeMedicationMatches.contains { $0.contains("lisinopril") },
            "Expected lisinopril in the focused medication snippet: \(homeMedicationMatches)"
        )
        XCTAssertTrue(
            homeMedicationMatches.contains { $0.contains("atorvastatin") },
            "Expected atorvastatin in the focused medication snippet: \(homeMedicationMatches)"
        )
        XCTAssertTrue(
            homeMedicationMatches.contains { $0.contains("sumatriptan") },
            "Expected sumatriptan in the focused medication snippet: \(homeMedicationMatches)"
        )

        let planSnippet = """
        Continue oral hydration, ibuprofen 400 mg every 6 hours as needed, ondansetron 4 mg every 8 hours as needed for nausea.
        """
        let planEntities = try pipeline.extract(
            planSnippet,
            labels: labels,
            threshold: 0.6,
            flatNER: true
        )
        let planMedications = planEntities
            .filter { $0.label == "medication" }
            .map { $0.text.lowercased() }

        XCTAssertTrue(
            planMedications.contains { $0.contains("ibuprofen") },
            "Expected ibuprofen once the plan snippet fits in the model window: \(planMedications)"
        )
        XCTAssertTrue(
            planMedications.contains { $0.contains("ondansetron") },
            "Expected ondansetron once the plan snippet fits in the model window: \(planMedications)"
        )
    }
    #endif

    func testLocalGLiNERSpanArtifactRunsNativeNER() throws {
        try requireUsableMLXRuntime()
        let artifactURL = try localArtifactURL(from: "OPENMED_GLINER_SPAN_ARTIFACT")
        let pipeline = try OpenMedZeroShotNER(modelDirectoryURL: artifactURL, maxSeqLength: 128)

        let text = "John Doe was treated with metformin for diabetes."
        let entities = try pipeline.extract(
            text,
            labels: ["person", "medication", "condition"],
            threshold: 0.95
        )

        XCTAssertTrue(entities.allSatisfy { $0.start >= 0 && $0.end <= text.count && $0.start < $0.end })
    }

    func testLocalGLiClassArtifactRunsNativeClassification() throws {
        try requireUsableMLXRuntime()
        let artifactURL = try localArtifactURL(from: "OPENMED_GLICLASS_ARTIFACT")
        let classifier = try OpenMedZeroShotClassifier(modelDirectoryURL: artifactURL, maxSeqLength: 128)

        let classifications = try classifier.classify(
            "The patient needs follow-up for diabetes and hypertension.",
            labels: ["medical condition", "medication", "appointment"],
            threshold: 0.0,
            prompt: "Classify this clinical sentence."
        )

        XCTAssertEqual(Set(classifications.map(\.label)), ["medical condition", "medication", "appointment"])
    }

    func testLocalGLiNERRelexArtifactRunsNativeExtraction() throws {
        try requireUsableMLXRuntime()
        let artifactURL = try localArtifactURL(from: "OPENMED_GLINER_RELEX_ARTIFACT")
        let extractor = try OpenMedRelationExtractor(modelDirectoryURL: artifactURL, maxSeqLength: 128)

        let text = "John Doe takes metformin."
        let result = try extractor.extract(
            text,
            entityLabels: ["person", "medication"],
            relationLabels: ["takes"],
            threshold: 0.0,
            relationThreshold: 1.0
        )

        XCTAssertTrue(result.entities.allSatisfy { $0.start >= 0 && $0.end <= text.count && $0.start < $0.end })
    }

    private func requireUsableMLXRuntime() throws {
        guard !Self.isSwiftPMCLI else {
            throw XCTSkip("MLX runtime resources are not loadable from the SwiftPM CLI test bundle.")
        }
        guard Self.hasPackagedMetalLibrary else {
            throw XCTSkip("MLX runtime resources are not bundled in this swift test environment.")
        }
    }

    private func localArtifactURL(from environmentKey: String) throws -> URL {
        let environment = ProcessInfo.processInfo.environment
        guard let artifactPath = environment[environmentKey], !artifactPath.isEmpty else {
            throw XCTSkip("Set \(environmentKey) to run local private artifact smoke tests.")
        }
        return URL(fileURLWithPath: (artifactPath as NSString).expandingTildeInPath)
    }

    #if canImport(AppKit) && canImport(Vision)
    private static func extractSampleClinicalDocumentOCRText() throws -> String {
        let imageURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(path: "OpenMedScanDemo/OpenMedScanDemo/Assets.xcassets/SampleClinicalDocument.imageset/sample-clinical-document.png")

        guard let image = NSImage(contentsOf: imageURL) else {
            XCTFail("Missing sample clinical document image at \(imageURL.path)")
            return ""
        }

        var rect = NSRect(origin: .zero, size: image.size)
        guard let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            XCTFail("Unable to create CGImage for sample clinical document")
            return ""
        }

        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        request.recognitionLanguages = ["en-US"]

        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try handler.perform([request])

        let observations = (request.results ?? []).sorted(by: recognitionSort)
        let lines = observations.compactMap { observation in
            observation.topCandidates(1).first?.string.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return lines
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func recognitionSort(
        lhs: VNRecognizedTextObservation,
        rhs: VNRecognizedTextObservation
    ) -> Bool {
        let verticalDelta = lhs.boundingBox.maxY - rhs.boundingBox.maxY
        if abs(verticalDelta) > 0.02 {
            return verticalDelta > 0
        }
        return lhs.boundingBox.minX < rhs.boundingBox.minX
    }
    #endif

    private static var isSwiftPMCLI: Bool {
        Bundle(for: OpenMedMLXTests.self).bundlePath.contains("/.build/")
    }

    private static var hasPackagedMetalLibrary: Bool {
        for bundle in Bundle.allBundles + Bundle.allFrameworks {
            guard let resourceURL = bundle.resourceURL else {
                continue
            }
            if FileManager.default.fileExists(
                atPath: resourceURL.appending(path: "default.metallib").path
            ) {
                return true
            }
            guard let enumerator = FileManager.default.enumerator(
                at: resourceURL,
                includingPropertiesForKeys: nil
            ) else {
                continue
            }
            for case let fileURL as URL in enumerator where fileURL.lastPathComponent == "default.metallib" {
                return true
            }
        }

        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let buildDirectory = packageRoot.appending(path: ".build", directoryHint: .isDirectory)
        guard let enumerator = FileManager.default.enumerator(
            at: buildDirectory,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }
        for case let fileURL as URL in enumerator where fileURL.lastPathComponent == "default.metallib" {
            return true
        }
        return false
    }

    private func makeManifestOnlyArtifact(
        task: String = "token-classification",
        family: String = "bert",
        configModelType: String? = nil,
        promptSpec: [String: Any]? = nil,
        in directory: URL? = nil
    ) throws -> URL {
        let directory = try directory ?? FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )

        try "{}".write(to: directory.appending(path: "tokenizer.json"), atomically: true, encoding: .utf8)
        try "{}".write(
            to: directory.appending(path: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        try "{}".write(
            to: directory.appending(path: "special_tokens_map.json"),
            atomically: true,
            encoding: .utf8
        )

        let modelType = configModelType ?? family
        let configObject: [String: Any] = [
            "model_type": modelType,
            "_mlx_model_type": modelType,
            "_mlx_weights_format": "safetensors",
            "vocab_size": 32,
            "hidden_size": 8,
            "encoder_hidden_size": 8,
            "embedding_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "type_vocab_size": 2,
            "num_labels": 3,
            "id2label": [
                "0": "O",
                "1": "B-NAME",
                "2": "I-NAME",
            ],
        ]
        try JSONSerialization.data(withJSONObject: configObject, options: [.prettyPrinted])
            .write(to: directory.appending(path: "config.json"))
        try JSONSerialization.data(
            withJSONObject: ["0": "O", "1": "B-NAME", "2": "I-NAME"],
            options: [.prettyPrinted]
        ).write(to: directory.appending(path: "id2label.json"))

        var manifestObject: [String: Any] = [
            "format": "openmed-mlx",
            "format_version": 1,
            "task": task,
            "family": family,
            "source_model_id": "OpenMed/Test-Model",
            "config_path": "config.json",
            "label_map_path": "id2label.json",
            "preferred_weights": "weights.safetensors",
            "fallback_weights": ["weights.npz"],
            "available_weights": ["weights.safetensors"],
            "weights_format": "safetensors",
            "quantization": NSNull(),
            "max_sequence_length": 32,
            "tokenizer": [
                "path": ".",
                "files": ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
            ],
        ]
        if let promptSpec {
            manifestObject["prompt_spec"] = promptSpec
        }
        try JSONSerialization.data(withJSONObject: manifestObject, options: [.prettyPrinted])
            .write(to: directory.appending(path: "openmed-mlx.json"))
        try Data().write(to: directory.appending(path: "weights.safetensors"))

        return directory
    }

    private func makePrivacyFilterManifestOnlyArtifact() throws -> URL {
        let directory = try makeManifestOnlyArtifact(
            task: "token-classification",
            family: "openai-privacy-filter",
            configModelType: "openai_privacy_filter"
        )

        let configObject: [String: Any] = [
            "model_type": "openai_privacy_filter",
            "_mlx_model_type": "openai-privacy-filter",
            "_mlx_weights_format": "safetensors",
            "_name_or_path": "openai/privacy-filter",
            "encoding": "o200k_base",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 8,
            "head_dim": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "num_experts": 4,
            "experts_per_token": 2,
            "bidirectional_left_context": 1,
            "bidirectional_right_context": 1,
            "initial_context_length": 64,
            "max_position_embeddings": 128,
            "rope_theta": 150_000,
            "rope_scaling_factor": 1.0,
            "rope_ntk_alpha": 1.0,
            "rope_ntk_beta": 32.0,
            "param_dtype": "float32",
            "rms_norm_eps": 1e-5,
            "swiglu_limit": 7.0,
            "num_labels": 5,
            "id2label": [
                "0": "O",
                "1": "B-private_person",
                "2": "I-private_person",
                "3": "E-private_person",
                "4": "S-private_email",
            ],
            "_mlx_viterbi_biases": [
                "transition_bias_background_stay": 0.0,
                "transition_bias_background_to_start": 0.0,
                "transition_bias_inside_to_continue": 0.0,
                "transition_bias_inside_to_end": 0.0,
                "transition_bias_end_to_background": 0.0,
                "transition_bias_end_to_start": 0.0,
            ],
        ]
        try JSONSerialization.data(withJSONObject: configObject, options: [.prettyPrinted])
            .write(to: directory.appending(path: "config.json"))
        try JSONSerialization.data(
            withJSONObject: [
                "0": "O",
                "1": "B-private_person",
                "2": "I-private_person",
                "3": "E-private_person",
                "4": "S-private_email",
            ],
            options: [.prettyPrinted]
        ).write(to: directory.appending(path: "id2label.json"))
        return directory
    }

    private func makeTinyArtifact() throws -> URL {
        let directory = try makeManifestOnlyArtifact()
        let configData = try Data(contentsOf: directory.appending(path: "config.json"))

        let config = try JSONDecoder().decode(OpenMedMLXBertConfiguration.self, from: configData)
        let model = OpenMedBertForTokenClassification(config)
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        weights["classifier.weight"] = MLXArray.zeros([config.numLabels, config.hiddenSize], type: Float.self)
        weights["classifier.bias"] = MLXArray([Float(0.0), Float(6.0), Float(-6.0)], [config.numLabels])
        try MLX.save(arrays: weights, url: directory.appending(path: "weights.safetensors"))

        return directory
    }

    private func makeLegacyArtifactWithoutTokenizerAssets() throws -> URL {
        let directory = try FileManager.default.url(
            for: .itemReplacementDirectory,
            in: .userDomainMask,
            appropriateFor: FileManager.default.temporaryDirectory,
            create: true
        )

        let configObject: [String: Any] = [
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
            "id2label": [
                "0": "O",
                "1": "B-NAME",
                "2": "I-NAME",
            ],
        ]
        try JSONSerialization.data(withJSONObject: configObject, options: [.prettyPrinted])
            .write(to: directory.appending(path: "config.json"))
        try JSONSerialization.data(
            withJSONObject: ["0": "O", "1": "B-NAME", "2": "I-NAME"],
            options: [.prettyPrinted]
        ).write(to: directory.appending(path: "id2label.json"))
        try Data().write(to: directory.appending(path: "weights.safetensors"))

        return directory
    }

    private func makeNPY(descriptor: String, shape: [Int], body: Data) -> Data {
        let shapeString: String
        if shape.count == 1 {
            shapeString = "\(shape[0]),"
        } else {
            shapeString = shape.map(String.init).joined(separator: ", ")
        }

        var header = "{'descr': '\(descriptor)', 'fortran_order': False, 'shape': (\(shapeString)), }"
        let magicLength = 10
        let newlineLength = 1
        let paddedLength = ((magicLength + header.utf8.count + newlineLength + 15) / 16) * 16
        let paddingCount = max(0, paddedLength - magicLength - header.utf8.count - newlineLength)
        header += String(repeating: " ", count: paddingCount)
        header += "\n"

        let headerData = Data(header.utf8)
        var data = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00])
        let headerLength = UInt16(headerData.count)
        data.append(UInt8(headerLength & 0xff))
        data.append(UInt8((headerLength >> 8) & 0xff))
        data.append(headerData)
        data.append(body)
        return data
    }

    private func makeTokenizerData(modelType: String) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: [
                "model": [
                    "type": modelType,
                ],
            ],
            options: [.prettyPrinted]
        )
    }

    private func makeTokenizerConfig(tokenizerClass: String) throws -> Data {
        try JSONSerialization.data(
            withJSONObject: [
                "tokenizer_class": tokenizerClass,
            ],
            options: [.prettyPrinted]
        )
    }

    private struct FakeTokenizer: Tokenizer {
        let tokenMap: [String: [Int]] = [
            "": [101, 102],
            "<<ENT>>": [9001],
            "<<SEP>>": [9002],
            "symptom": [10],
            "headache": [20, 21],
            "follow-up": [30, 31, 32],
            ".": [33],
        ]

        func tokenize(text: String) -> [String] {
            [text]
        }

        func encode(text: String) -> [Int] {
            encode(text: text, addSpecialTokens: true)
        }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            if addSpecialTokens, text.isEmpty {
                return [101, 102]
            }
            return tokenMap[text] ?? [999]
        }

        func decode(tokens: [Int]) -> String {
            tokens.map(String.init).joined(separator: " ")
        }

        func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
            decode(tokens: tokens)
        }

        func convertTokenToId(_ token: String) -> Int? {
            tokenMap[token]?.first
        }

        func convertTokensToIds(_ tokens: [String]) -> [Int?] {
            tokens.map(convertTokenToId)
        }

        func convertIdToToken(_ id: Int) -> String? {
            tokenMap.first { $0.value.first == id }?.key
        }

        func convertIdsToTokens(_ ids: [Int]) -> [String?] {
            ids.map(convertIdToToken)
        }

        var bosToken: String? { "[CLS]" }
        var bosTokenId: Int? { 101 }
        var eosToken: String? { "[SEP]" }
        var eosTokenId: Int? { 102 }
        var unknownToken: String? { "[UNK]" }
        var unknownTokenId: Int? { 999 }

        func applyChatTemplate(messages: [Message]) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(messages: [Message], tools: [ToolSpec]?) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(
            messages: [Message],
            tools: [ToolSpec]?,
            additionalContext: [String: Any]?
        ) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(messages: [Message], chatTemplate: ChatTemplateArgument) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(messages: [Message], chatTemplate: String) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?
        ) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }

        func applyChatTemplate(
            messages: [Message],
            chatTemplate: ChatTemplateArgument?,
            addGenerationPrompt: Bool,
            truncation: Bool,
            maxLength: Int?,
            tools: [ToolSpec]?,
            additionalContext: [String: Any]?
        ) throws -> [Int] {
            throw TokenizerError.chatTemplate("FakeTokenizer does not support chat templates.")
        }
    }
}
