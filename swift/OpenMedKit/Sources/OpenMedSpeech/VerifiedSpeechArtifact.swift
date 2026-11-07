import CryptoKit
import Foundation
import OpenMedKit

/// Public reproducibility metadata, containing no model bytes or local paths.
public struct SpeechArtifactFile: Sendable {
    public let path: String
    public let sha256: String
    public let byteCount: Int
}

/// A verified local tiny.en bundle. Verification is against built-in immutable pins,
/// not an untrusted manifest next to the weights. Keep the directory immutable while
/// the engine is alive; verification cannot prevent a privileged concurrent writer.
public struct VerifiedSpeechArtifact: Sendable, CustomStringConvertible, CustomDebugStringConvertible {
    public static let modelRepository = "argmaxinc/whisperkit-coreml"
    public static let modelRevision = "0f63a7800b00dd0226abd051b906c246e1907482"
    public static let modelSubdirectory = "openai_whisper-tiny.en"
    public static let tokenizerRepository = "openai/whisper-tiny.en"
    public static let tokenizerRevision = "87c7102498dcde7456f24cfd30239ca606ed9063"
    public static let runtimeRevision = "1bbc57cb7fe6410248a685b1c5a3c6d56adc2ed4"
    /// Revision-bound model-card snapshot; provenance, not permission or legal advice.
    public static let modelCardSHA256 = "13627ce3d7dce389223eab7b6ed06738f0f1c786e5245e650b029e4e314c8ff1"
    public static let files: [SpeechArtifactFile] = [
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/analytics/coremldata.bin", sha256: "eaaaa6671a96a359a0bbd5e97885246dcc17f7435b6ffad8d871bb940964500b", byteCount: 243),
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/coremldata.bin", sha256: "325b182d0a4266730a81795ae6b7a787b5111dd091500fc0c04dedf610015d46", byteCount: 347),
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/metadata.json", sha256: "c0468bfcd4c940e479812dc35db6c9bb380c1a9da1786471c0b916e1ee784ff0", byteCount: 1807),
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/model.mil", sha256: "6423991df85b6811ef08a935c86ab4f0cc0a7a072fe24913f3bcfab2c5bebd84", byteCount: 308780),
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/model.mlmodel", sha256: "030d64a3ddd296d6f709691a66a870aab7ee9f19e5fe07e8086245fb85302802", byteCount: 54965),
        SpeechArtifactFile(path: "AudioEncoder.mlmodelc/weights/weight.bin", sha256: "f3706dac8d9d4bec269d3cee10fa4eda39b4240a46091c8323c1731a8c6d59c2", byteCount: 16_422_784),
        SpeechArtifactFile(path: "MelSpectrogram.mlmodelc/analytics/coremldata.bin", sha256: "160d9737169d22dc01a899e1c6a0a9c44d0637d41f0dedb2a0b7c1422c4035d2", byteCount: 243),
        SpeechArtifactFile(path: "MelSpectrogram.mlmodelc/coremldata.bin", sha256: "cb3b3f51b080f58b12a6888a5e8ad57419be9e4c6843b96a7577f171b300e660", byteCount: 328),
        SpeechArtifactFile(path: "MelSpectrogram.mlmodelc/metadata.json", sha256: "d92e196c14b4d03167a8ded97d9f925617ff302942588aa460257eb0c58de35b", byteCount: 1848),
        SpeechArtifactFile(path: "MelSpectrogram.mlmodelc/model.mil", sha256: "5fe83a96c41965219d39734aa3248bd4956d2427705539fba2d406aa4eeb4862", byteCount: 10132),
        SpeechArtifactFile(path: "MelSpectrogram.mlmodelc/weights/weight.bin", sha256: "801024dbc7a89c677be1f8b285de3409e35f7d1786c9c8d9d0d6842ac57a1c83", byteCount: 354080),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/analytics/coremldata.bin", sha256: "edb99a30ccee8e157fbec80dc3dce49349ba0982391b327d753e10ccab0a01c3", byteCount: 243),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/coremldata.bin", sha256: "65c043a081845d190918b4c7d244f94a55df1a15fae796abedc1f414995542c6", byteCount: 633),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/metadata.json", sha256: "41dccfd0fe75a8f333abf74a1b0ab5ca007a589e5cb50fa34ec6033936763b58", byteCount: 4754),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/model.mil", sha256: "0faaeee40f86f6e4bc69b81c7142bc9ca8af7859a597ee0a22eb3df8ad30ab65", byteCount: 135272),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/model.mlmodel", sha256: "5c3e91bc036014426708e2ceb0e35cb1bbbf34e8121d2070d2b174a7957581d0", byteCount: 108558),
        SpeechArtifactFile(path: "TextDecoder.mlmodelc/weights/weight.bin", sha256: "763f915f0126093fc2c506572b3ab0fad134c04cfc2221333ccc7d73552c9252", byteCount: 59_215_664),
        SpeechArtifactFile(path: "tokenizer/config.json", sha256: "b5366317df280f79e6cc366f259f379674f81a8721d458acf2e00680ccb10b1d", byteCount: 1937),
        SpeechArtifactFile(path: "tokenizer/tokenizer.json", sha256: "5eb60cec1e77aeeb6869a2bb5a8e01a84c3fe5d072d75369343021fe6f5310d0", byteCount: 2_405_679),
        SpeechArtifactFile(path: "tokenizer/tokenizer_config.json", sha256: "14f84bdf4b9ecdbd4738ddc81c17a1baedfc02bb93c6e049c951e15a1b40b70d", byteCount: 805),
    ]
    public static var fingerprint: String {
        let inventory =
            [modelRepository, modelRevision, tokenizerRepository, tokenizerRevision]
            + files.map { "\($0.path):\($0.byteCount):\($0.sha256)" }
        return "sha256:" + SHA256.hash(data: Data(inventory.joined(separator: "\n").utf8)).map { String(format: "%02x", $0) }.joined()
    }
    let directory: URL
    private init(directory: URL) { self.directory = directory }
    public var description: String { "VerifiedSpeechArtifact(\(Self.fingerprint))" }
    public var debugDescription: String { description }

    /// Verify all files consumed by Core ML and the tokenizer. No download, file copy,
    /// license acceptance or custom-model substitution occurs here.
    public static func verifyTinyEnglish(at directory: URL) throws -> Self {
        guard directory.isFileURL else { throw StreamingSpeechError.unverifiedArtifact }
        do {
            let values = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else { throw StreamingSpeechError.unverifiedArtifact }
            let root = directory.resolvingSymlinksInPath().standardizedFileURL
            let expected = Set(files.map(\.path))
            for folder in ["AudioEncoder.mlmodelc", "MelSpectrogram.mlmodelc", "TextDecoder.mlmodelc", "tokenizer"] {
                try inspect(root: root, relative: folder, expected: expected)
            }
            for file in files { try verifyFile(root.appending(path: file.path), expected: file) }
            return Self(directory: root)
        } catch let error as StreamingSpeechError { throw error } catch { throw StreamingSpeechError.artifactMismatch }
    }

    func revalidate() throws { _ = try Self.verifyTinyEnglish(at: directory) }

    private static func inspect(root: URL, relative: String, expected: Set<String>) throws {
        let url = root.appending(path: relative)
        let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isSymbolicLink != true else { throw StreamingSpeechError.artifactMismatch }
        if values.isDirectory == true {
            guard expected.contains(where: { $0.hasPrefix(relative + "/") }) else { throw StreamingSpeechError.artifactMismatch }
            let children = try FileManager.default.contentsOfDirectory(at: url, includingPropertiesForKeys: nil)
            guard children.count <= files.count else { throw StreamingSpeechError.artifactMismatch }
            for child in children { try inspect(root: root, relative: relative + "/" + child.lastPathComponent, expected: expected) }
        } else {
            // A local model card is optional and is never opened by the runtime.
            guard values.isRegularFile == true, expected.contains(relative) || relative == "tokenizer/README.md" else { throw StreamingSpeechError.artifactMismatch }
        }
    }

    static func verifyFile(_ url: URL, expected: SpeechArtifactFile) throws {
        do {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true, values.fileSize == expected.byteCount else { throw StreamingSpeechError.artifactMismatch }
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            var hash = SHA256()
            var count = 0
            while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
                count += chunk.count
                guard count <= expected.byteCount else { throw StreamingSpeechError.artifactMismatch }
                hash.update(data: chunk)
            }
            guard count == expected.byteCount,
                hash.finalize().map({ String(format: "%02x", $0) }).joined() == expected.sha256
            else { throw StreamingSpeechError.artifactMismatch }
        } catch { throw StreamingSpeechError.artifactMismatch }
    }
}
