import Foundation

/// Supported runtime backends for OpenMedKit.
public enum OpenMedBackend: Sendable {
    /// A compiled Core ML token-classification model.
    ///
    /// There is no default tokenizer identity: ship tokenizer assets in
    /// `tokenizerFolderURL`, or pass an `OpenMedTokenizerPin` when explicitly
    /// opting into network acquisition.
    ///
    /// - Parameters:
    ///   - modelURL: Compiled model (`.mlmodelc`) or package (`.mlpackage`).
    ///   - id2labelURL: Label map file.
    ///   - tokenizerName: Optional Hugging Face repository ID the tokenizer
    ///     must belong to. It never triggers a download; a tokenizer pin for a
    ///     different repository fails with `.identityMismatch`.
    ///   - tokenizerFolderURL: Local directory holding `tokenizer.json` and
    ///     `tokenizer_config.json`.
    case coreML(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String? = nil,
        tokenizerFolderURL: URL? = nil
    )
    /// A self-contained OpenMed MLX artifact directory.
    case mlx(modelDirectoryURL: URL)
}
