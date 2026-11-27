import CryptoKit
import Foundation

/// Why an ``OpenMedTokenizerPin`` was rejected.
///
/// Raw values are stable, value-free reason codes.
public enum OpenMedTokenizerPinIssue: String, Sendable, Equatable {
    /// The repository ID is not a plain `owner/name` Hugging Face identifier.
    case repositoryID = "invalid_repository_id"
    /// The revision is not a full 40-character commit SHA.
    case revision = "invalid_revision"
    /// A digest is not a 64-character hexadecimal SHA-256 value.
    case digest = "invalid_digest"
    /// A file name is not one of the supported tokenizer asset names.
    case fileName = "unsupported_file_name"
    /// The same file is pinned more than once.
    case duplicateFile = "duplicate_file"
    /// `tokenizer.json` or `tokenizer_config.json` is not pinned.
    case missingRequiredFile = "missing_required_file"
    /// An expected byte count is not positive or exceeds the download bound.
    case byteCount = "invalid_byte_count"
}

/// Typed, value-free failures raised while resolving tokenizer assets.
///
/// No case carries a model path, URL, repository ID or tokenizer content, so
/// these errors are safe to log or show to users. Use ``reasonCode`` for
/// telemetry-free diagnostics.
public enum OpenMedTokenizerError: Error, Equatable, Sendable {
    /// Local tokenizer assets are absent and no verified pinned copy is cached.
    case missingLocalAssets
    /// Network acquisition was allowed without an ``OpenMedTokenizerPin``.
    case pinnedRevisionRequired
    /// The tokenizer pin is malformed.
    case invalidPin(OpenMedTokenizerPinIssue)
    /// The pin names a different repository than the model declares.
    case identityMismatch
    /// A pinned download failed. `statusCode` is the HTTP status, if any.
    case downloadFailed(statusCode: Int?)
    /// A pinned file's size does not match the pin or exceeds the bound.
    case sizeMismatch
    /// A pinned file's SHA-256 digest does not match the pin.
    case digestMismatch
    /// The tokenizer cache under the Caches directory is unavailable.
    case cacheUnavailable
    /// Local tokenizer assets exist but could not be read or parsed.
    case invalidAssets

    /// Stable, value-free reason code for this failure.
    public var reasonCode: String {
        switch self {
        case .missingLocalAssets:
            return "tokenizer_assets_missing"
        case .pinnedRevisionRequired:
            return "tokenizer_pin_required"
        case .invalidPin(let issue):
            return "tokenizer_pin_\(issue.rawValue)"
        case .identityMismatch:
            return "tokenizer_identity_mismatch"
        case .downloadFailed:
            return "tokenizer_download_failed"
        case .sizeMismatch:
            return "tokenizer_size_mismatch"
        case .digestMismatch:
            return "tokenizer_digest_mismatch"
        case .cacheUnavailable:
            return "tokenizer_cache_unavailable"
        case .invalidAssets:
            return "tokenizer_assets_invalid"
        }
    }
}

extension OpenMedTokenizerError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .missingLocalAssets:
            return
                "Tokenizer assets are not available locally. Ship tokenizer.json and tokenizer_config.json with the model, or opt into a pinned download."
        case .pinnedRevisionRequired:
            return "Network tokenizer acquisition requires an OpenMedTokenizerPin with a commit revision and file digests."
        case .invalidPin(let issue):
            return "The tokenizer pin was rejected (\(issue.rawValue))."
        case .identityMismatch:
            return "The tokenizer pin does not match the tokenizer identity declared by the model."
        case .downloadFailed(let statusCode):
            if let statusCode {
                return "The pinned tokenizer download failed with HTTP status \(statusCode)."
            }
            return "The pinned tokenizer download failed."
        case .sizeMismatch:
            return "A tokenizer file does not match its pinned size."
        case .digestMismatch:
            return "A tokenizer file does not match its pinned SHA-256 digest."
        case .cacheUnavailable:
            return "The tokenizer cache directory is unavailable."
        case .invalidAssets:
            return "The local tokenizer assets could not be read."
        }
    }
}

/// An exact tokenizer snapshot that OpenMedKit may download when the caller
/// explicitly opts into network access.
///
/// A pin names a Hugging Face repository, a full commit SHA (never a branch
/// such as `main`) and the SHA-256 digest of every file to fetch. Downloads go
/// only to the app's Caches directory and every file is verified against its
/// digest before it is written and again before each use.
///
/// ```swift
/// let pin = try OpenMedTokenizerPin(
///     repositoryID: "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1",
///     revision: "<40-character commit SHA>",
///     files: [
///         .init(name: "tokenizer.json", sha256: "<sha256>"),
///         .init(name: "tokenizer_config.json", sha256: "<sha256>"),
///     ]
/// )
/// ```
public struct OpenMedTokenizerPin: Sendable, Hashable {
    /// One pinned tokenizer file.
    public struct File: Sendable, Hashable {
        /// Tokenizer asset file name, for example `tokenizer.json`.
        public let name: String
        /// Lowercase hexadecimal SHA-256 digest of the file.
        public let sha256: String
        /// Optional exact size in bytes, checked before the digest.
        public let byteCount: Int?

        /// Creates a pinned file entry.
        ///
        /// - Parameters:
        ///   - name: Tokenizer asset file name, for example `tokenizer.json`.
        ///   - sha256: Hexadecimal SHA-256 digest of the file contents.
        ///   - byteCount: Optional exact size in bytes.
        public init(name: String, sha256: String, byteCount: Int? = nil) {
            self.name = name
            self.sha256 = sha256.lowercased()
            self.byteCount = byteCount
        }
    }

    /// Hugging Face repository ID in `owner/name` form.
    public let repositoryID: String
    /// Full 40-character lowercase commit SHA.
    public let revision: String
    /// Every file to download, each with its expected digest.
    public let files: [File]

    /// Creates a validated tokenizer pin.
    ///
    /// - Parameters:
    ///   - repositoryID: Hugging Face repository ID in `owner/name` form.
    ///   - revision: Full 40-character commit SHA. Branch and tag names are
    ///     rejected because they can move.
    ///   - files: Files to download. Must include `tokenizer.json` and
    ///     `tokenizer_config.json`.
    /// - Throws: ``OpenMedTokenizerError/invalidPin(_:)`` when any field is
    ///   malformed.
    public init(repositoryID: String, revision: String, files: [File]) throws {
        guard OpenMedTokenizerAssetStore.isRepositoryID(repositoryID) else {
            throw OpenMedTokenizerError.invalidPin(.repositoryID)
        }
        let normalizedRevision = revision.lowercased()
        guard OpenMedTokenizerAssetStore.isHex(normalizedRevision, length: 40) else {
            throw OpenMedTokenizerError.invalidPin(.revision)
        }
        var seen = Set<String>()
        for file in files {
            guard OpenMedTokenizerAssetStore.pinnableFileNames.contains(file.name) else {
                throw OpenMedTokenizerError.invalidPin(.fileName)
            }
            guard seen.insert(file.name).inserted else {
                throw OpenMedTokenizerError.invalidPin(.duplicateFile)
            }
            guard OpenMedTokenizerAssetStore.isHex(file.sha256, length: 64) else {
                throw OpenMedTokenizerError.invalidPin(.digest)
            }
            if let byteCount = file.byteCount {
                guard byteCount > 0, byteCount <= OpenMedTokenizerAssetStore.maximumFileBytes else {
                    throw OpenMedTokenizerError.invalidPin(.byteCount)
                }
            }
        }
        guard OpenMedTokenizerAssetStore.requiredFileNames.isSubset(of: seen) else {
            throw OpenMedTokenizerError.invalidPin(.missingRequiredFile)
        }

        self.repositoryID = repositoryID
        self.revision = normalizedRevision
        self.files = files
    }
}

/// Downloads, verifies and caches pinned tokenizer snapshots.
///
/// Every write lands under ``cacheRoot``, which defaults to
/// `Caches/OpenMed/TokenizerAssets`. Nothing is written under Documents.
struct OpenMedTokenizerAssetStore: Sendable {
    /// Tokenizer files the swift-transformers loader and prepared-directory
    /// copy understand.
    static let tokenizerAssetFileNames = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
        "spm.model",
        "sentencepiece.bpe.model",
        "added_tokens.json",
    ]

    /// Files a pin may name: tokenizer assets plus the model `config.json`,
    /// whose `model_type` informs tokenizer class inference.
    static let pinnableFileNames = Set(tokenizerAssetFileNames + ["config.json"])

    static let requiredFileNames: Set<String> = ["tokenizer.json", "tokenizer_config.json"]

    /// Upper bound for any single pinned file.
    static let maximumFileBytes = 256 * 1024 * 1024

    let cacheRoot: URL
    let session: URLSession

    /// Store rooted at `Caches/OpenMed/TokenizerAssets` using the shared session.
    static func makeDefault() throws -> Self {
        try Self(cacheRoot: defaultCacheRoot(), session: .shared)
    }

    static func defaultCacheRoot() throws -> URL {
        let base: URL
        do {
            base = try FileManager.default.url(
                for: .cachesDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: true
            )
        } catch {
            throw OpenMedTokenizerError.cacheUnavailable
        }
        return
            base
            .appending(path: "OpenMed", directoryHint: .isDirectory)
            .appending(path: "TokenizerAssets", directoryHint: .isDirectory)
    }

    /// Directory holding the verified snapshot for `pin`.
    func directory(for pin: OpenMedTokenizerPin) -> URL {
        cacheRoot
            .appending(
                path: pin.repositoryID.replacingOccurrences(of: "/", with: "__"),
                directoryHint: .isDirectory
            )
            .appending(path: pin.revision, directoryHint: .isDirectory)
    }

    /// Hub URL for one pinned file. Always addresses the pinned commit.
    static func remoteURL(for pin: OpenMedTokenizerPin, file: OpenMedTokenizerPin.File) -> URL? {
        URL(string: "https://huggingface.co/\(pin.repositoryID)/resolve/\(pin.revision)/\(file.name)")
    }

    /// Returns the cached snapshot after re-verifying every pinned file.
    ///
    /// - Returns: The snapshot directory, or `nil` when nothing is cached.
    /// - Throws: ``OpenMedTokenizerError/digestMismatch`` or
    ///   ``OpenMedTokenizerError/sizeMismatch`` when the cached copy differs
    ///   from the pin. The stale copy is removed so a later opted-in load can
    ///   download it again.
    func verifiedCachedDirectory(for pin: OpenMedTokenizerPin) throws -> URL? {
        let directory = directory(for: pin)
        let fileManager = FileManager.default
        guard let attributes = try? fileManager.attributesOfItem(atPath: directory.path) else {
            return nil
        }
        do {
            guard attributes[.type] as? FileAttributeType == .typeDirectory else {
                throw OpenMedTokenizerError.digestMismatch
            }
            let entries = Set(try fileManager.contentsOfDirectory(atPath: directory.path))
            guard entries == Set(pin.files.map(\.name)) else {
                throw OpenMedTokenizerError.digestMismatch
            }
            for file in pin.files {
                let fileURL = directory.appending(path: file.name)
                let fileAttributes = try fileManager.attributesOfItem(atPath: fileURL.path)
                guard fileAttributes[.type] as? FileAttributeType == .typeRegular else {
                    throw OpenMedTokenizerError.digestMismatch
                }
                try Self.verify(Data(contentsOf: fileURL), against: file)
            }
            return directory
        } catch {
            try? fileManager.removeItem(at: directory)
            if let tokenizerError = error as? OpenMedTokenizerError {
                throw tokenizerError
            }
            throw OpenMedTokenizerError.digestMismatch
        }
    }

    /// Downloads every pinned file, verifies it, and installs the snapshot.
    ///
    /// Files are staged in a sibling directory and moved into place only after
    /// all of them verify, so a failed or tampered download leaves nothing
    /// usable behind.
    func download(_ pin: OpenMedTokenizerPin) async throws -> URL {
        let fileManager = FileManager.default
        let destination = directory(for: pin)
        let parent = destination.deletingLastPathComponent()
        let staging = parent.appending(
            path: ".staging-\(UUID().uuidString)",
            directoryHint: .isDirectory
        )
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        } catch {
            throw OpenMedTokenizerError.cacheUnavailable
        }
        defer { try? fileManager.removeItem(at: staging) }

        for file in pin.files {
            let data = try await fetch(file, of: pin)
            try Self.verify(data, against: file)
            do {
                try data.write(to: staging.appending(path: file.name), options: .atomic)
            } catch {
                throw OpenMedTokenizerError.cacheUnavailable
            }
        }

        do {
            if fileManager.fileExists(atPath: destination.path) {
                try fileManager.removeItem(at: destination)
            }
            try fileManager.moveItem(at: staging, to: destination)
        } catch {
            throw OpenMedTokenizerError.cacheUnavailable
        }
        return destination
    }

    private func fetch(
        _ file: OpenMedTokenizerPin.File,
        of pin: OpenMedTokenizerPin
    ) async throws -> Data {
        guard let url = Self.remoteURL(for: pin, file: file) else {
            throw OpenMedTokenizerError.invalidPin(.repositoryID)
        }
        var request = URLRequest(url: url)
        request.cachePolicy = .reloadIgnoringLocalCacheData
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            // URLError carries the request URL; surface only a value-free code.
            throw OpenMedTokenizerError.downloadFailed(statusCode: nil)
        }
        guard let http = response as? HTTPURLResponse else {
            throw OpenMedTokenizerError.downloadFailed(statusCode: nil)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw OpenMedTokenizerError.downloadFailed(statusCode: http.statusCode)
        }
        return data
    }

    static func verify(_ data: Data, against file: OpenMedTokenizerPin.File) throws {
        guard data.count <= maximumFileBytes else {
            throw OpenMedTokenizerError.sizeMismatch
        }
        if let byteCount = file.byteCount, data.count != byteCount {
            throw OpenMedTokenizerError.sizeMismatch
        }
        guard sha256Hex(data) == file.sha256 else {
            throw OpenMedTokenizerError.digestMismatch
        }
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Whether `value` is a plain `owner/name` Hugging Face repository ID.
    static func isRepositoryID(_ value: String) -> Bool {
        let components = value.split(separator: "/", omittingEmptySubsequences: false)
        guard components.count == 2, !value.contains("..") else {
            return false
        }
        return components.allSatisfy { component in
            guard (1...96).contains(component.count),
                let first = component.first, let last = component.last,
                first.isASCII, first.isLetter || first.isNumber,
                last.isASCII, last.isLetter || last.isNumber
            else {
                return false
            }
            return component.allSatisfy { character in
                character.isASCII
                    && (character.isLetter || character.isNumber || "._-".contains(character))
            }
        }
    }

    static func isHex(_ value: String, length: Int) -> Bool {
        value.count == length && value.allSatisfy { "0123456789abcdef".contains($0) }
    }
}
