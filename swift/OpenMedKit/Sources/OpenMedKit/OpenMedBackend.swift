import Foundation

/// Supported runtime backends for OpenMedKit.
public enum OpenMedBackend: Sendable {
    case coreML(
        modelURL: URL,
        id2labelURL: URL,
        tokenizerName: String = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
        tokenizerFolderURL: URL? = nil
    )
    case mlx(modelDirectoryURL: URL)
}
