import Foundation
import OpenMedKit

/// Compile-only references to the MLX-gated OpenMedKit APIs that a consumer of the
/// repository root package must see on macOS and iOS.
///
/// `scripts/check_swift_root_consumer.sh` builds this file in a throwaway package that
/// requires the root package by version, as documented installs do. Every symbol below
/// compiles only under `canImport(MLXLMCommon)`, so the build fails if the root
/// manifest stops exposing the `mlx-swift-lm` products. Nothing here loads a model.
public enum RootPackageConsumerSymbols {
    public static let mapleRepositoryID: String = OpenMedMaple.repositoryID
    public static let maplePinnedRevision: String = OpenMedMaple.pinnedRevision
    public static let mapleRequiredFiles: [String] = OpenMedMaple.requiredModelFiles
    public static let compassDefaultModelID: String = OpenMedVisionLanguageModel.defaultModelID

    public static func mapleIsReady(_ directory: URL) -> Bool {
        OpenMedMaple.isModelDirectoryReady(directory)
    }

    public static func loadMaple(_ directory: URL) async throws -> OpenMedMaple {
        try await OpenMedMaple(modelDirectoryURL: directory)
    }

    public static func brief(
        with maple: OpenMedMaple,
        source: String,
        evaluate: @escaping @Sendable (String, String) async throws -> Data,
        privacyCheck: @escaping @Sendable (String) throws -> Bool
    ) async throws -> ClinicalBrief {
        try await maple.brief(
            source: source,
            originalIdentifiers: [],
            evaluate: evaluate,
            privacyCheck: privacyCheck
        )
    }

    public static func describe(_ error: OpenMedMapleRuntimeError) -> String? {
        error.errorDescription
    }

    public static func decodeMapleConfiguration(_ data: Data) throws -> OpenMedMapleConfiguration {
        try JSONDecoder().decode(OpenMedMapleConfiguration.self, from: data)
    }

    public static func loadCompass(_ directory: URL) async throws -> OpenMedVisionLanguageModel {
        try await OpenMedVisionLanguageModel.load(modelDirectory: directory)
    }

    public static func generatedText(_ generation: OpenMedVisionLanguageGeneration) -> String {
        generation.text
    }
}
