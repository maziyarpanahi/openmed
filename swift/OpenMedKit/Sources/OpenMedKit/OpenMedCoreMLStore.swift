import CoreML
import Foundation

/// A ready Core ML token-classification snapshot from ``OpenMedModelStore``.
public struct OpenMedCoreMLModel: Equatable, Sendable {
    /// The snapshot directory, `<root>/<org>__<name>/<revision>`.
    public let directoryURL: URL
    /// The compiled model (`.mlmodelc`) to load.
    public let modelURL: URL
    /// The downloaded source `.mlpackage`, kept next to its compiled model,
    /// or `nil` when the snapshot ships a compiled `.mlmodelc`.
    public let packageURL: URL?
    /// The label map.
    public let id2labelURL: URL
    /// The directory holding the tokenizer files.
    public let tokenizerFolderURL: URL

    /// Create a model description.
    public init(directoryURL: URL, modelURL: URL, packageURL: URL?, id2labelURL: URL, tokenizerFolderURL: URL) {
        self.directoryURL = directoryURL
        self.modelURL = modelURL
        self.packageURL = packageURL
        self.id2labelURL = id2labelURL
        self.tokenizerFolderURL = tokenizerFolderURL
    }

    /// The backend description for `OpenMed(backend:)`.
    public var backend: OpenMedBackend {
        .coreML(modelURL: modelURL, id2labelURL: id2labelURL, tokenizerFolderURL: tokenizerFolderURL)
    }
}

/// The snapshot-root `openmed-coreml.json` fields the store reads.
///
/// Every other field (latency, residency, provenance) is ignored. Every path
/// is relative to the snapshot and confined to it.
struct OpenMedCoreMLStoreManifest: Decodable, Sendable {
    struct Variant: Decodable, Sendable {
        let name: String
        /// `<Name>.mlpackage` or `<Name>.mlmodelc`.
        let path: String
        /// Files inside `path`, relative to it. Optional for `.mlpackage`
        /// variants, whose own `openmed-coreml.json` inventories them.
        let files: [String]?
    }

    struct Tokenizer: Decodable, Sendable {
        let path: String
        let files: [String]
    }

    let format: String
    let variants: [Variant]
    let id2label: String?
    let tokenizer: Tokenizer?

    static let defaultLabelMap = "id2label.json"
    static let defaultTokenizer = Tokenizer(path: ".", files: ["tokenizer.json", "tokenizer_config.json"])
    static let optionalTokenizerFiles = ["config.json", "special_tokens_map.json", "vocab.txt", "added_tokens.json"]

    var variant: Variant { variants[0] }
    var labelMapPath: String { id2label ?? Self.defaultLabelMap }
    var tokenizerSpec: Tokenizer { tokenizer ?? Self.defaultTokenizer }
    var isPackage: Bool { variant.path.lowercased().hasSuffix(".mlpackage") }

    /// The compiled model of a `.mlpackage` variant, inside the hidden
    /// `.openmed-compiled` directory so it never collides with snapshot files.
    var compiledModelPath: String {
        let name = String(variant.path.split(separator: "/").last ?? "")
        return ".openmed-compiled/" + (name as NSString).deletingPathExtension + ".mlmodelc"
    }

    func tokenizerFile(_ file: String) -> String {
        tokenizerSpec.path == "." ? file : "\(tokenizerSpec.path)/\(file)"
    }

    /// Reject unsupported shapes and every path that is not confined to `boundary`.
    func validate(_ boundary: OpenMedMLXPathBoundary) throws {
        guard format == "openmed-coreml", !variants.isEmpty else {
            throw OpenMedModelStoreError.invalidCoreMLManifest
        }
        let lowered = variant.path.lowercased()
        guard lowered.hasSuffix(".mlpackage") || lowered.hasSuffix(".mlmodelc") else {
            throw OpenMedModelStoreError.invalidCoreMLManifest
        }
        guard isPackage || variant.files?.isEmpty == false else {
            throw OpenMedModelStoreError.invalidCoreMLManifest
        }
        _ = try boundary.fileURL(variant.path)
        for file in variant.files ?? [] { _ = try boundary.fileURL("\(variant.path)/\(file)") }
        _ = try boundary.fileURL(labelMapPath)
        if tokenizerSpec.path != "." { _ = try boundary.fileURL(tokenizerSpec.path) }
        for file in tokenizerSpec.files + Self.optionalTokenizerFiles {
            try OpenMedMLXPathBoundary.validateRelativePath(file)
            _ = try boundary.fileURL(tokenizerFile(file))
        }
    }
}

/// `provenance.files` of a `.mlpackage/openmed-coreml.json` package manifest.
private struct OpenMedCoreMLPackageInventory: Decodable {
    struct Provenance: Decodable {
        struct File: Decodable { let path: String }
        let files: [File]
    }

    let provenance: Provenance
}

extension OpenMedModelStore {
    static let coreMLManifestFileName = "openmed-coreml.json"

    /// Download a Core ML token-classification snapshot from the Hugging Face
    /// Hub and compile it into the cache.
    ///
    /// The snapshot root holds `openmed-coreml.json` with
    /// `"format": "openmed-coreml"` and a `variants` list. The store uses the
    /// first variant. Its `path` names a `.mlpackage` or `.mlmodelc`
    /// directory. Its files come from the variant's `files` list or, for a
    /// package, from the package's own `openmed-coreml.json` provenance
    /// inventory. The label map is `id2label` (default `id2label.json`). The
    /// tokenizer is `tokenizer.path` and `tokenizer.files` (default: the
    /// snapshot root with `tokenizer.json` and `tokenizer_config.json`, plus
    /// optional standard files). Every declared path is checked against the
    /// snapshot boundary before any artifact request.
    ///
    /// A `.mlpackage` is compiled with `MLModel.compileModel(at:)` only after
    /// every file has downloaded. The compiled model goes to
    /// `.openmed-compiled/<Name>.mlmodelc` and the source package is kept.
    /// The snapshot becomes `ready` only when the model, label map, tokenizer
    /// files and compiled model are all present.
    ///
    /// - Parameters:
    ///   - repoID: Hugging Face repository id.
    ///   - revision: Branch, tag or commit (default `main`).
    ///   - cacheDirectory: Cache root; defaults to `Caches/OpenMed/CoreMLModels`.
    /// - Returns: The ready model; pass ``OpenMedCoreMLModel/backend`` to `OpenMed`.
    /// - Throws: ``OpenMedModelStoreError``.
    public static func downloadCoreMLModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) async throws -> OpenMedCoreMLModel {
        try await downloadCoreMLModel(
            repoID: repoID, revision: revision, cacheDirectory: cacheDirectory, session: .shared)
    }

    static func downloadCoreMLModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil,
        session: URLSession
    ) async throws -> OpenMedCoreMLModel {
        let directory = try cachedCoreMLModelDirectory(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
        let boundary = try OpenMedMLXPathBoundary(directoryURL: directory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        if try coreMLCacheState(at: directory) == .ready {
            return try readyCoreMLModel(boundary: boundary, directory: directory)
        }

        var writtenFiles: [String] = []
        func fetch(_ path: String, optional: Bool = false) async throws -> Bool {
            let exists = try boundary.containsFile(path)
            let present = try await fetchCoreMLFile(
                repoID: repoID, revision: revision, relativePath: path,
                boundary: boundary, session: session, optional: optional)
            if present && !exists { writtenFiles.append(path) }
            return present
        }

        do {
            _ = try await fetch(coreMLManifestFileName)
            let manifest = try decodeCoreMLManifest(boundary: boundary)
            // Preflight every declaration before requesting model files.
            try manifest.validate(boundary)
            if manifest.variant.files == nil {
                _ = try await fetch("\(manifest.variant.path)/\(coreMLManifestFileName)")
            }
            for path in try variantFiles(manifest, boundary: boundary) { _ = try await fetch(path) }
            _ = try await fetch(manifest.labelMapPath)
            for file in manifest.tokenizerSpec.files { _ = try await fetch(manifest.tokenizerFile(file)) }
            if manifest.tokenizer == nil {
                for file in OpenMedCoreMLStoreManifest.optionalTokenizerFiles {
                    _ = try await fetch(manifest.tokenizerFile(file), optional: true)
                }
            }

            try manifest.validate(boundary)
            if manifest.isPackage {
                try await compileCoreMLPackage(manifest, boundary: boundary)
            }
            guard try hasCompleteCoreMLArtifact(boundary: boundary) else {
                throw OpenMedModelStoreError.invalidCoreMLManifest
            }
            try writeReadyMarker(to: boundary.fileURL(readyMarkerFileName))
            return try readyCoreMLModel(boundary: boundary, directory: directory)
        } catch OpenMedModelStoreError.invalidManifestPath(let reason) {
            boundary.removeFileIfConfined(readyMarkerFileName)
            for path in writtenFiles.reversed() { boundary.removeFileIfConfined(path) }
            throw OpenMedModelStoreError.invalidManifestPath(reason)
        }
    }

    /// Cache directory for a Core ML snapshot, `<root>/<org>__<name>/<revision>`.
    public static func cachedCoreMLModelDirectory(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> URL {
        let root = try cacheDirectory ?? defaultCoreMLCacheDirectory()
        return try OpenMedMLXPathBoundary(directoryURL: root).fileURL(
            "\(sanitizedPathComponent(repoID))/\(sanitizedPathComponent(revision))", rejectSymlinks: true)
    }

    /// `missing`, `partial` or `ready` for a Core ML snapshot, recomputed from
    /// the files on disk with the same confinement rules as the download.
    public static func coreMLModelCacheState(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> OpenMedMLXModelCacheState {
        try coreMLCacheState(
            at: cachedCoreMLModelDirectory(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory))
    }

    /// The ready Core ML snapshot, or `nil` when it is missing or partial.
    /// Reads only the local cache.
    public static func cachedCoreMLModel(
        repoID: String,
        revision: String = "main",
        cacheDirectory: URL? = nil
    ) throws -> OpenMedCoreMLModel? {
        let directory = try cachedCoreMLModelDirectory(repoID: repoID, revision: revision, cacheDirectory: cacheDirectory)
        guard try coreMLCacheState(at: directory) == .ready else { return nil }
        return try readyCoreMLModel(boundary: OpenMedMLXPathBoundary(directoryURL: directory), directory: directory)
    }

    // MARK: - Internals

    static func defaultCoreMLCacheDirectory() throws -> URL {
        try defaultCacheDirectory().deletingLastPathComponent()
            .appending(path: "CoreMLModels", directoryHint: .isDirectory)
    }

    static func coreMLCacheState(at directory: URL) throws -> OpenMedMLXModelCacheState {
        guard FileManager.default.fileExists(atPath: directory.path) else { return .missing }
        let boundary = try OpenMedMLXPathBoundary(directoryURL: directory)
        let markerURL = try boundary.fileURL(readyMarkerFileName)
        let isComplete: Bool
        do {
            isComplete = try hasCompleteCoreMLArtifact(boundary: boundary)
        } catch OpenMedModelStoreError.invalidManifestPath(let reason) {
            boundary.removeFileIfConfined(readyMarkerFileName)
            throw OpenMedModelStoreError.invalidManifestPath(reason)
        }
        if isComplete {
            if !FileManager.default.fileExists(atPath: markerURL.path) { try writeReadyMarker(to: markerURL) }
            return .ready
        }
        boundary.removeFileIfConfined(readyMarkerFileName)
        let contents = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        return contents.isEmpty ? .missing : .partial
    }

    static func hasCompleteCoreMLArtifact(boundary: OpenMedMLXPathBoundary) throws -> Bool {
        guard try boundary.containsFile(coreMLManifestFileName),
            let manifest = try? decodeCoreMLManifest(boundary: boundary)
        else {
            return false
        }
        do {
            try manifest.validate(boundary)
        } catch OpenMedModelStoreError.invalidCoreMLManifest {
            return false
        }
        guard let files = try? variantFiles(manifest, boundary: boundary) else { return false }
        let required =
            files + [manifest.labelMapPath] + manifest.tokenizerSpec.files.map(manifest.tokenizerFile)
        guard try required.allSatisfy({ try boundary.containsFile($0) }) else { return false }
        guard manifest.isPackage else { return true }
        let compiled = try boundary.fileURL(manifest.compiledModelPath, rejectSymlinks: true)
        return (try? FileManager.default.attributesOfItem(atPath: compiled.path)[.type]) as? FileAttributeType
            == .typeDirectory
    }

    private static func decodeCoreMLManifest(boundary: OpenMedMLXPathBoundary) throws -> OpenMedCoreMLStoreManifest {
        guard let data = try? Data(contentsOf: boundary.fileURL(coreMLManifestFileName)),
            let manifest = try? JSONDecoder().decode(OpenMedCoreMLStoreManifest.self, from: data)
        else {
            throw OpenMedModelStoreError.invalidCoreMLManifest
        }
        return manifest
    }

    /// Snapshot-relative paths of every file in the selected variant.
    private static func variantFiles(
        _ manifest: OpenMedCoreMLStoreManifest,
        boundary: OpenMedMLXPathBoundary
    ) throws -> [String] {
        let variant = manifest.variant
        if let files = variant.files {
            return files.map { "\(variant.path)/\($0)" }
        }
        let inventoryPath = "\(variant.path)/\(coreMLManifestFileName)"
        guard let data = try? Data(contentsOf: boundary.fileURL(inventoryPath)),
            let inventory = try? JSONDecoder().decode(OpenMedCoreMLPackageInventory.self, from: data),
            !inventory.provenance.files.isEmpty
        else {
            throw OpenMedModelStoreError.invalidCoreMLManifest
        }
        let paths = inventory.provenance.files.map { "\(variant.path)/\($0.path)" }
        for path in paths { _ = try boundary.fileURL(path) }
        return [inventoryPath] + paths
    }

    private static func compileCoreMLPackage(
        _ manifest: OpenMedCoreMLStoreManifest,
        boundary: OpenMedMLXPathBoundary
    ) async throws {
        let package = try boundary.fileURL(manifest.variant.path)
        let compiled: URL
        do {
            compiled = try await MLModel.compileModel(at: package)
        } catch {
            throw OpenMedModelStoreError.coreMLCompilationFailed
        }
        defer { try? FileManager.default.removeItem(at: compiled) }
        let destination = try boundary.fileURL(manifest.compiledModelPath, rejectSymlinks: true)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = try boundary.fileURL(manifest.compiledModelPath, rejectSymlinks: true)
        if (try? FileManager.default.attributesOfItem(atPath: destination.path)) != nil {
            try? FileManager.default.removeItem(at: destination)
        }
        do {
            try FileManager.default.moveItem(at: compiled, to: destination)
        } catch {
            throw OpenMedModelStoreError.coreMLCompilationFailed
        }
    }

    private static func readyCoreMLModel(
        boundary: OpenMedMLXPathBoundary,
        directory: URL
    ) throws -> OpenMedCoreMLModel {
        let manifest = try decodeCoreMLManifest(boundary: boundary)
        let variantURL = try boundary.fileURL(manifest.variant.path)
        let tokenizerPath = manifest.tokenizerSpec.path
        return OpenMedCoreMLModel(
            directoryURL: directory,
            modelURL: manifest.isPackage ? try boundary.fileURL(manifest.compiledModelPath) : variantURL,
            packageURL: manifest.isPackage ? variantURL : nil,
            id2labelURL: try boundary.fileURL(manifest.labelMapPath),
            tokenizerFolderURL: tokenizerPath == "." ? directory : try boundary.fileURL(tokenizerPath)
        )
    }

    /// Fetch one snapshot file unless it is already present. A 404 on an
    /// optional file returns `false`.
    private static func fetchCoreMLFile(
        repoID: String,
        revision: String,
        relativePath: String,
        boundary: OpenMedMLXPathBoundary,
        session: URLSession,
        optional: Bool
    ) async throws -> Bool {
        let destination = try boundary.fileURL(relativePath)
        if try boundary.containsFile(relativePath) { return true }
        var request = URLRequest(url: try resolveHubURL(repoID: repoID, revision: revision, relativePath: relativePath))
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        let (data, response) = try await session.data(for: request)
        guard let httpResponse = response as? HTTPURLResponse else {
            throw OpenMedModelStoreError.invalidResponse
        }
        if optional && httpResponse.statusCode == 404 { return false }
        guard (200..<300).contains(httpResponse.statusCode) else {
            throw OpenMedModelStoreError.httpError(statusCode: httpResponse.statusCode)
        }
        // URLSession suspends: check again before creating directories or writing.
        _ = try boundary.fileURL(relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = try boundary.fileURL(relativePath)
        try data.write(to: destination, options: .atomic)
        return true
    }
}
