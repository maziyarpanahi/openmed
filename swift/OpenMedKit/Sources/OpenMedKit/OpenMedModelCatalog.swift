import Foundation

/// One Apple-runnable model in the bundled OpenMed model catalog.
///
/// Rows are generated from the canonical `models.jsonl` manifest by
/// `scripts/swift/build_swift_catalog.py` and carry the same fields as the
/// Android `ModelCatalogEntry`. `formats` lists only the MLX and Core ML
/// formats Apple platforms run.
public struct OpenMedModelCatalogEntry: Codable, Sendable, Hashable {
    /// Hugging Face repository id, for example `OpenMed/<name>-mlx`.
    public let repoID: String
    /// Apple-runnable formats, for example `mlx-fp` or `mlx-8bit`.
    public let formats: [String]
    /// Size tier from the manifest (`Tiny`, `Small`, `Base`, ...), if any.
    public let tier: String?
    /// Parameter count, if the manifest records one. `Int64` so the value
    /// stays exact on 32-bit watchOS devices.
    public let paramCount: Int64?
    /// Language codes the model covers.
    public let languages: [String]
    /// SPDX-style license id from the manifest's permissive allowlist.
    public let license: String?
    /// `sha256:` reproducibility hash recorded for the published snapshot.
    public let reproducibilityHash: String

    /// Create a catalog entry.
    public init(
        repoID: String,
        formats: [String],
        tier: String? = nil,
        paramCount: Int64? = nil,
        languages: [String] = [],
        license: String? = nil,
        reproducibilityHash: String
    ) {
        self.repoID = repoID
        self.formats = formats
        self.tier = tier
        self.paramCount = paramCount
        self.languages = languages
        self.license = license
        self.reproducibilityHash = reproducibilityHash
    }

    enum CodingKeys: String, CodingKey {
        case repoID = "repo_id"
        case formats
        case tier
        case paramCount = "param_count"
        case languages
        case license
        case reproducibilityHash = "reproducibility_hash"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        repoID = try container.decode(String.self, forKey: .repoID)
        formats = try container.decodeIfPresent([String].self, forKey: .formats) ?? []
        tier = try container.decodeIfPresent(String.self, forKey: .tier)
        paramCount = try container.decodeIfPresent(Int64.self, forKey: .paramCount)
        languages = try container.decodeIfPresent([String].self, forKey: .languages) ?? []
        license = try container.decodeIfPresent(String.self, forKey: .license)
        reproducibilityHash = try container.decode(String.self, forKey: .reproducibilityHash)
    }
}

/// Value-free catalog failures: no case carries a repository id, path or
/// line content.
public enum OpenMedModelCatalogError: LocalizedError, Equatable, Sendable {
    /// The bundled `openmed_model_catalog.jsonl` resource is absent.
    case resourceMissing
    /// The catalog text is not UTF-8.
    case invalidEncoding
    /// A line is not a catalog row; carries the 1-based line number only.
    case invalidEntry(line: Int)

    /// Stable reason code for logs and metrics.
    public var reason: String {
        switch self {
        case .resourceMissing: return "catalog_resource_missing"
        case .invalidEncoding: return "catalog_invalid_encoding"
        case .invalidEntry: return "catalog_invalid_entry"
        }
    }

    public var errorDescription: String? {
        switch self {
        case .resourceMissing:
            return "The bundled OpenMed model catalog is missing."
        case .invalidEncoding:
            return "The OpenMed model catalog is not UTF-8 text."
        case .invalidEntry(let line):
            return "The OpenMed model catalog has an invalid entry on line \(line)."
        }
    }
}

/// Read-only catalog of Apple-runnable OpenMed models, bundled with
/// OpenMedKit so apps can discover models by format, size, language and
/// license without hard-coding repository names.
///
/// ```swift
/// let catalog = try OpenMedModelCatalog.bundled()
/// let small = catalog.filter(format: "mlx-fp", maxParamCount: 50_000_000, language: "de")
/// ```
///
/// Loading never touches the network. The filter semantics match the Android
/// `ModelCatalog.filter`: every argument left `nil` matches all rows, string
/// arguments compare case-insensitively, `format` and `language` match when
/// any listed value equals the argument, and `maxParamCount` is inclusive and
/// excludes rows without a parameter count. Results keep catalog order.
public struct OpenMedModelCatalog: Sendable {
    /// Bundled resource name (without extension).
    public static let resourceName = "openmed_model_catalog"
    /// Bundled resource extension.
    public static let resourceExtension = "jsonl"

    /// Catalog rows in file order (sorted case-insensitively by repository id).
    public let entries: [OpenMedModelCatalogEntry]

    /// Create a catalog from explicit entries, for tests or custom catalogs.
    public init(entries: [OpenMedModelCatalogEntry]) {
        self.entries = entries
    }

    /// Parse catalog JSON Lines: one entry per line, blank lines ignored.
    ///
    /// - Throws: ``OpenMedModelCatalogError`` naming the failing line number.
    public init(jsonLines data: Data) throws {
        guard let text = String(data: data, encoding: .utf8) else {
            throw OpenMedModelCatalogError.invalidEncoding
        }
        let decoder = JSONDecoder()
        var entries: [OpenMedModelCatalogEntry] = []
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            do {
                entries.append(try decoder.decode(OpenMedModelCatalogEntry.self, from: Data(trimmed.utf8)))
            } catch {
                throw OpenMedModelCatalogError.invalidEntry(line: index + 1)
            }
        }
        self.entries = entries
    }

    private static let bundledCatalog: Result<OpenMedModelCatalog, OpenMedModelCatalogError> = {
        guard
            let url = Bundle.module.url(forResource: resourceName, withExtension: resourceExtension),
            let data = try? Data(contentsOf: url)
        else {
            return .failure(.resourceMissing)
        }
        do {
            return .success(try OpenMedModelCatalog(jsonLines: data))
        } catch let error as OpenMedModelCatalogError {
            return .failure(error)
        } catch {
            return .failure(.invalidEncoding)
        }
    }()

    /// The catalog bundled in OpenMedKit's resources, parsed once per process.
    ///
    /// - Throws: ``OpenMedModelCatalogError`` if the resource is missing or
    ///   malformed.
    public static func bundled() throws -> OpenMedModelCatalog {
        try bundledCatalog.get()
    }

    /// Return entries matching every non-`nil` argument, in catalog order.
    ///
    /// - Parameters:
    ///   - format: A format any of the entry's formats must equal, ignoring case.
    ///   - tier: The entry's tier, ignoring case.
    ///   - maxParamCount: Inclusive upper bound; entries without a count fail it.
    ///   - language: A language any of the entry's languages must equal, ignoring case.
    ///   - license: The entry's license, ignoring case.
    public func filter(
        format: String? = nil,
        tier: String? = nil,
        maxParamCount: Int64? = nil,
        language: String? = nil,
        license: String? = nil
    ) -> [OpenMedModelCatalogEntry] {
        entries.filter { entry in
            Self.matchesAny(entry.formats, format)
                && Self.matches(entry.tier, tier)
                && Self.matchesMaxParamCount(entry.paramCount, maxParamCount)
                && Self.matchesAny(entry.languages, language)
                && Self.matches(entry.license, license)
        }
    }

    /// The entry with exactly this repository id, or `nil`.
    public func entry(repoID: String) -> OpenMedModelCatalogEntry? {
        entries.first { $0.repoID == repoID }
    }

    private static func matches(_ actual: String?, _ expected: String?) -> Bool {
        guard let expected else { return true }
        guard let actual else { return false }
        return actual.lowercased() == expected.lowercased()
    }

    private static func matchesAny(_ values: [String], _ expected: String?) -> Bool {
        guard let expected else { return true }
        let needle = expected.lowercased()
        return values.contains { $0.lowercased() == needle }
    }

    private static func matchesMaxParamCount(_ actual: Int64?, _ maxParamCount: Int64?) -> Bool {
        guard let maxParamCount else { return true }
        guard let actual else { return false }
        return actual <= maxParamCount
    }
}
