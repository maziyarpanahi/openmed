import Foundation
import OpenMedKit

/// Compile-only migration targets from docs/migration/openmedkit-installed-packages.md.
///
/// Building this consumer proves that the documented replacements compile
/// against the installed package. The facade helpers need real model assets,
/// so the verifier compiles them without executing them.
enum MigrationExamples {
    #if os(iOS) || os(macOS)
        /// Load a pre-staged MLX snapshot without letting the tokenizer fetch assets.
        static func offlineMLXFacade(modelDirectory: URL) throws -> OpenMed {
            try OpenMed(
                backend: .mlx(modelDirectoryURL: modelDirectory),
                allowNetworkAccess: false
            )
        }

        /// Load a CoreML model with a local tokenizer folder and no network access.
        static func offlineCoreMLFacade(
            modelURL: URL,
            id2labelURL: URL,
            tokenizerFolderURL: URL
        ) throws -> OpenMed {
            try OpenMed(
                backend: .coreML(
                    modelURL: modelURL,
                    id2labelURL: id2labelURL,
                    tokenizerFolderURL: tokenizerFolderURL
                ),
                allowNetworkAccess: false
            )
        }

        /// Policy-driven redaction with the documented default profile.
        static func policyRedaction(
            _ openmed: OpenMed,
            text: String
        ) throws -> PolicyDeidentificationResult {
            try openmed.deidentify(text, policy: Policy.defaultName)
        }

        /// Method-based redaction keeps the Python-compatible result schema.
        static func methodRedaction(
            _ openmed: OpenMed,
            text: String
        ) throws -> DeidentificationResult {
            try openmed.deidentify(text, method: .mask)
        }
    #endif

    /// Resolve a snapshot that was staged before going offline.
    static func stagedSnapshot(repoID: String, cacheDirectory: URL) throws -> URL? {
        guard
            try OpenMedModelStore.mlxModelCacheState(
                repoID: repoID,
                cacheDirectory: cacheDirectory
            ) == .ready
        else {
            return nil
        }
        return try OpenMedModelStore.cachedMLXModelDirectory(
            repoID: repoID,
            cacheDirectory: cacheDirectory
        )
    }

    /// Map the typed store and policy errors a consumer must handle.
    static func incompatibilityCode(for error: Error) -> String? {
        switch error {
        case OpenMedModelStoreError.invalidManifestPath:
            return "unsafe_manifest_path"
        case OpenMedModelStoreError.missingWeights, OpenMedModelStoreError.missingManifest:
            return "incomplete_snapshot"
        case PolicyError.unknownProfile:
            return "unknown_policy_profile"
        case PolicyError.missingResource:
            return "policy_resource_missing"
        default:
            return nil
        }
    }
}
