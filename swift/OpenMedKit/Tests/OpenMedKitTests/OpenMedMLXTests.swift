import Foundation
import MLX
import XCTest
import ZIPFoundation
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

    func testArtifactRejectsUnsupportedArchitecture() throws {
        let directory = try makeManifestOnlyArtifact(family: "deberta-v2")

        XCTAssertThrowsError(try OpenMedMLXArtifact(modelDirectoryURL: directory)) { error in
            guard case OpenMedMLXArtifactError.unsupportedArchitecture(let family) = error else {
                return XCTFail("Unexpected error: \(error)")
            }
            XCTAssertEqual(family, "deberta-v2")
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

    private func requireUsableMLXRuntime() throws {
        guard Self.hasPackagedMetalLibrary else {
            throw XCTSkip("MLX runtime resources are not bundled in this swift test environment.")
        }
    }

    private static var hasPackagedMetalLibrary: Bool {
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

    private func makeManifestOnlyArtifact(family: String = "bert") throws -> URL {
        let directory = try FileManager.default.url(
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

        let configObject: [String: Any] = [
            "model_type": family,
            "_mlx_model_type": family,
            "_mlx_weights_format": "safetensors",
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

        let manifestObject: [String: Any] = [
            "format": "openmed-mlx",
            "format_version": 1,
            "task": "token-classification",
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
        try JSONSerialization.data(withJSONObject: manifestObject, options: [.prettyPrinted])
            .write(to: directory.appending(path: "openmed-mlx.json"))

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
}
