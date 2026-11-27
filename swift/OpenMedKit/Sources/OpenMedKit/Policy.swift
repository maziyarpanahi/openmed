import CryptoKit
import Foundation

/// A de-identification action from a bundled OpenMed policy profile.
public enum PolicyAction: String, Codable, Equatable, Sendable {
    case keep
    case redact
    case replace
    case mask
    case remove
    case hash

    var redactionEquivalent: PolicyAction {
        switch self {
        case .redact:
            return .mask
        case .keep, .replace, .mask, .remove, .hash:
            return self
        }
    }

    /// Redaction strength used by `high_recall_union` arbitration.
    var strength: Int {
        switch self {
        case .keep: return 0
        case .replace: return 1
        case .hash: return 2
        case .mask: return 3
        case .redact: return 4
        case .remove: return 5
        }
    }
}

/// A value-free policy failure: a stable reason code and a JSON field path.
///
/// Loading, verification, and decision errors never echo profile content,
/// labels, source text, or surrogate keys.
public struct PolicyError: Error, Equatable, Sendable, CustomStringConvertible,
    LocalizedError
{
    public let code: String
    public let fieldPath: String

    public init(_ code: String, _ fieldPath: String = "$") {
        self.code = code
        self.fieldPath = fieldPath
    }

    public var description: String { "\(code) at \(fieldPath)" }

    public var errorDescription: String? { description }
}

/// Threshold entry of a portable policy profile, in parts per million.
public struct PolicyThreshold: Equatable, Sendable {
    /// Minimum confidence for a candidate span to be retained.
    public let keepFloorPPM: Int
    /// Retained spans below this are flagged for review (metadata only).
    public let escalateBelowPPM: Int

    public init(keepFloorPPM: Int, escalateBelowPPM: Int) {
        self.keepFloorPPM = keepFloorPPM
        self.escalateBelowPPM = escalateBelowPPM
    }
}

/// The portable label taxonomy shared by every bundled profile.
///
/// Generated from `openmed/core/labels.py`; normalization equals the Python
/// `normalize_label` for ASCII detector labels.
public struct PolicyTaxonomy: Equatable, Sendable {
    public static let schema = "openmed.portable_taxonomy"
    public static let version = 1

    public let labels: [String]
    public let policyLabels: [String: String]
    public let specificity: [String: Int]
    public let aliases: [String: String]
    public let unknownLabel: String
    public let digest: String

    /// Normalize a detector label; unknown labels map to `unknownLabel`.
    public func normalize(_ label: String) -> (canonicalLabel: String, recognized: Bool) {
        if let canonical = aliases[Self.labelKey(label)] {
            return (canonical, true)
        }
        return (unknownLabel, false)
    }

    /// Strip whitespace and one BIOES prefix, lowercase, keep `a-z0-9`.
    public static func labelKey(_ label: String) -> String {
        let trimmed = label.trimmingCharacters(in: .whitespacesAndNewlines)
        var scalars = Array(trimmed.unicodeScalars)
        if scalars.count >= 2, "BIES".unicodeScalars.contains(scalars[0]), scalars[1] == "-" {
            scalars.removeFirst(2)
        }
        let lowered = String(String.UnicodeScalarView(scalars)).lowercased()
        return String(
            String.UnicodeScalarView(
                lowered.unicodeScalars.filter {
                    ("a"..."z").contains($0) || ("0"..."9").contains($0)
                }
            )
        )
    }
}

/// A bundled, versioned, digest-verified OpenMed policy profile.
///
/// Profiles are generated from the Python package's `openmed/core/policies`
/// definitions by `scripts/parity/generate_portable_policies.py` and shipped
/// as byte-identical SwiftPM resources. Loading verifies the manifest, the
/// profile and taxonomy digests, the IR version, and the required
/// capabilities, and fails closed with a value-free `PolicyError`.
public struct Policy: Equatable, Sendable {
    public static let defaultName = "hipaa_safe_harbor"
    public static let manifestFile = "portable_policy_manifest.json"
    public static let taxonomyFile = "portable_policy_taxonomy.json"
    public static let manifestSchema = "openmed.portable_policy_manifest"
    public static let manifestVersion = 1
    public static let irSchema = "openmed.portable_policy"
    public static let irVersion = 1
    public static let mergeAlgorithm = "openmed.portable_merge"
    public static let mergeVersion = 1
    public static let surrogateAlgorithm = "openmed.portable_surrogate.hmac_sha256"
    public static let surrogateVersion = 1

    /// Capabilities this runtime implements.
    public static let supportedCapabilities = [
        "actions.v1",
        "labels.portable_taxonomy.v1",
        "merge.cluster_union.v1",
        "merge.semantic_units.v1",
        "strict_no_leak.v1",
        "surrogate.hmac_sha256.v1",
        "thresholds.keep_floor.v1",
    ]

    /// Portable profile names bundled with OpenMedKit.
    public static let bundledProfileNames: [String] =
        (try? bundledManifest())?.profiles.keys.sorted() ?? []

    /// Profile aliases that resolve to a bundled profile.
    public static let aliases: [String: String] = {
        guard let manifest = try? bundledManifest() else {
            return [:]
        }
        return manifest.profileAliases.filter { manifest.profiles[$0.value] != nil }
    }()

    /// Bundled taxonomy, or `nil` when the resource is missing or stale.
    public static let bundledTaxonomy: PolicyTaxonomy? = try? loadBundledTaxonomy()

    public let schemaVersion: Int
    public let name: String
    public let digest: String
    public let posture: String
    public let thresholdProfile: String
    public let defaultAction: PolicyAction
    public let defaultActionBias: String
    public let arbitrationMode: String
    public let strictNoLeak: Bool
    public let safetySweepMandatory: Bool
    public let keepMapping: Bool
    public let reversibleID: Bool
    public let forcedCascadeTiers: [String]
    public let policyLabelActions: [String: PolicyAction]
    public let actions: [String: PolicyAction]
    public let taxonomy: PolicyTaxonomy
    public let defaultThreshold: PolicyThreshold
    public let labelThresholds: [String: PolicyThreshold]
    public let requiredCapabilities: [String]
    /// Python-only features of the source profile the portable subset omits.
    public let excludedFeatures: [String]

    /// Load a bundled policy profile by canonical name or alias.
    ///
    /// - Parameters:
    ///   - name: Profile name or alias.
    ///   - expectedDigest: Optional `sha256:` digest to pin; a different
    ///     bundled profile fails with `digest_mismatch`.
    public init(named name: String, expectedDigest: String? = nil) throws {
        self = try Self.load(named: name, expectedDigest: expectedDigest)
    }

    init(
        name: String,
        schemaVersion: Int = 1,
        digest: String = "",
        posture: String = "test",
        thresholdProfile: String = "balanced",
        defaultAction: PolicyAction,
        defaultActionBias: String? = nil,
        arbitrationMode: String = "balanced",
        strictNoLeak: Bool = false,
        safetySweepMandatory: Bool = false,
        keepMapping: Bool = false,
        reversibleID: Bool = false,
        forcedCascadeTiers: [String] = [],
        policyLabelActions: [String: PolicyAction] = [:],
        actions: [String: PolicyAction] = [:],
        taxonomy: PolicyTaxonomy? = nil,
        defaultThreshold: PolicyThreshold = PolicyThreshold(keepFloorPPM: 0, escalateBelowPPM: 0),
        labelThresholds: [String: PolicyThreshold] = [:],
        requiredCapabilities: [String] = [],
        excludedFeatures: [String] = []
    ) {
        self.schemaVersion = schemaVersion
        self.name = name
        self.digest = digest
        self.posture = posture
        self.thresholdProfile = thresholdProfile
        self.defaultAction = defaultAction
        self.defaultActionBias = defaultActionBias ?? defaultAction.rawValue
        self.arbitrationMode = arbitrationMode
        self.strictNoLeak = strictNoLeak
        self.safetySweepMandatory = safetySweepMandatory
        self.keepMapping = keepMapping
        self.reversibleID = reversibleID
        self.forcedCascadeTiers = forcedCascadeTiers
        self.policyLabelActions = policyLabelActions
        self.actions = actions
        self.taxonomy =
            taxonomy ?? Self.bundledTaxonomy
            ?? PolicyTaxonomy(
                labels: ["OTHER"],
                policyLabels: ["OTHER": "CLINICAL_CONCEPT"],
                specificity: ["OTHER": 0],
                aliases: ["other": "OTHER"],
                unknownLabel: "OTHER",
                digest: ""
            )
        self.defaultThreshold = defaultThreshold
        self.labelThresholds = labelThresholds
        self.requiredCapabilities = requiredCapabilities
        self.excludedFeatures = excludedFeatures
    }

    /// Resolve the action configured for a source or canonical entity label.
    public func action(for label: String) -> PolicyAction {
        let action = configuredAction(forCanonical: taxonomy.normalize(label).canonicalLabel)
        if strictNoLeak && action == .keep {
            return .mask
        }
        return action
    }

    /// Return the threshold entry for a canonical label.
    public func threshold(for canonicalLabel: String) -> PolicyThreshold {
        labelThresholds[canonicalLabel] ?? defaultThreshold
    }

    func configuredAction(forCanonical canonicalLabel: String) -> PolicyAction {
        actions[canonicalLabel]
            ?? taxonomy.policyLabels[canonicalLabel].flatMap { policyLabelActions[$0] }
            ?? defaultAction
    }

    /// Resolve an entity label to the portable OpenMed label taxonomy.
    public static func canonicalLabel(for label: String) -> String {
        bundledTaxonomy?.normalize(label).canonicalLabel ?? "OTHER"
    }

    /// Resolve a bundled profile name or alias.
    public static func canonicalProfileName(_ name: String) throws -> String {
        try resolveProfileName(name, manifest: bundledManifest())
    }

    // MARK: - Loading

    /// Resource bytes by file name.
    public typealias ResourceProvider = (String) -> Data?

    /// Load and verify a profile.
    ///
    /// Verification order: manifest, profile name, caller-pinned digest,
    /// profile resource digest, taxonomy resource digest, profile document.
    public static func load(
        named name: String,
        expectedDigest: String? = nil,
        resources: ResourceProvider = Policy.bundledResource,
        supportedCapabilities: [String] = Policy.supportedCapabilities
    ) throws -> Policy {
        guard let manifestData = resources(manifestFile) else {
            throw PolicyError("resource_missing", "$.manifest")
        }
        let manifest = try decodeManifest(manifestData)
        let canonical = try resolveProfileName(name, manifest: manifest)
        guard let digest = manifest.profiles[canonical] else {
            throw PolicyError("profile_unknown", "$.name")
        }
        if let expectedDigest, expectedDigest != digest {
            throw PolicyError("digest_mismatch", "$.digest")
        }
        guard let data = resources("\(canonical).json") else {
            throw PolicyError("resource_missing", "$.profiles")
        }
        guard sha256Digest(data) == digest else {
            throw PolicyError("resource_digest_mismatch", "$.profiles")
        }
        guard let taxonomyData = resources(taxonomyFile) else {
            throw PolicyError("resource_missing", "$.taxonomy_digest")
        }
        let taxonomyDigest = sha256Digest(taxonomyData)
        guard taxonomyDigest == manifest.taxonomyDigest else {
            throw PolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        let taxonomy = try decodeTaxonomy(taxonomyData, digest: taxonomyDigest)
        return try decodeProfile(
            data,
            name: canonical,
            digest: digest,
            taxonomy: taxonomy,
            supportedCapabilities: supportedCapabilities
        )
    }

    /// Return a bundled resource by file name.
    public static func bundledResource(_ file: String) -> Data? {
        let stem = (file as NSString).deletingPathExtension
        let url =
            Bundle.module.url(forResource: stem, withExtension: "json", subdirectory: "policies")
            ?? Bundle.module.url(
                forResource: stem,
                withExtension: "json",
                subdirectory: "Resources/policies"
            )
            ?? Bundle.module.url(forResource: stem, withExtension: "json")
        return url.flatMap { try? Data(contentsOf: $0) }
    }

    /// `sha256:` digest of resource bytes.
    public static func sha256Digest(_ data: Data) -> String {
        "sha256:" + SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    struct Manifest: Equatable, Sendable {
        let defaultProfile: String
        let profiles: [String: String]
        let excludedProfiles: [String: String]
        let profileAliases: [String: String]
        let taxonomyDigest: String
    }

    static func bundledManifest() throws -> Manifest {
        guard let data = bundledResource(manifestFile) else {
            throw PolicyError("resource_missing", "$.manifest")
        }
        return try decodeManifest(data)
    }

    static func decodeManifest(_ data: Data) throws -> Manifest {
        let document: ManifestDocument
        do {
            document = try JSONDecoder().decode(ManifestDocument.self, from: data)
        } catch {
            throw PolicyError("manifest_invalid")
        }
        guard document.schema == manifestSchema else {
            throw PolicyError("manifest_invalid", "$.schema")
        }
        guard document.schemaVersion == manifestVersion else {
            throw PolicyError("manifest_version_unsupported", "$.schema_version")
        }
        guard document.irSchema == irSchema else {
            throw PolicyError("manifest_invalid", "$.ir_schema")
        }
        guard document.irVersion == irVersion else {
            throw PolicyError("ir_version_unsupported", "$.ir_version")
        }
        guard let profiles = document.profiles else {
            throw PolicyError("manifest_invalid", "$.profiles")
        }
        guard let excluded = document.excludedProfiles else {
            throw PolicyError("manifest_invalid", "$.excluded_profiles")
        }
        guard let aliases = document.profileAliases else {
            throw PolicyError("manifest_invalid", "$.profile_aliases")
        }
        guard profiles.values.allSatisfy(isDigest) else {
            throw PolicyError("manifest_invalid", "$.profiles")
        }
        guard let taxonomyDigest = document.taxonomyDigest, isDigest(taxonomyDigest) else {
            throw PolicyError("manifest_invalid", "$.taxonomy_digest")
        }
        guard let defaultProfile = document.defaultProfile, profiles[defaultProfile] != nil
        else {
            throw PolicyError("manifest_invalid", "$.default_profile")
        }
        return Manifest(
            defaultProfile: defaultProfile,
            profiles: profiles,
            excludedProfiles: excluded,
            profileAliases: aliases,
            taxonomyDigest: taxonomyDigest
        )
    }

    static func resolveProfileName(_ name: String, manifest: Manifest) throws -> String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw PolicyError("profile_name_invalid", "$.name")
        }
        let normalized = trimmed.lowercased().replacingOccurrences(of: "-", with: "_")
        let canonical = manifest.profileAliases[normalized] ?? normalized
        if manifest.profiles[canonical] != nil {
            return canonical
        }
        if manifest.excludedProfiles[canonical] != nil {
            throw PolicyError("profile_not_portable", "$.name")
        }
        throw PolicyError("profile_unknown", "$.name")
    }

    static func decodeTaxonomy(_ data: Data, digest: String) throws -> PolicyTaxonomy {
        let document: TaxonomyDocument
        do {
            document = try JSONDecoder().decode(TaxonomyDocument.self, from: data)
        } catch {
            throw PolicyError("taxonomy_invalid")
        }
        guard document.name == PolicyTaxonomy.schema else {
            throw PolicyError("taxonomy_invalid", "$.name")
        }
        guard document.version == PolicyTaxonomy.version else {
            throw PolicyError("taxonomy_invalid", "$.version")
        }
        guard let labels = document.labels, !labels.isEmpty else {
            throw PolicyError("taxonomy_invalid", "$.labels")
        }
        guard let aliases = document.aliases, aliases.values.allSatisfy({ labels[$0] != nil })
        else {
            throw PolicyError("taxonomy_invalid", "$.aliases")
        }
        guard let unknown = document.unknownLabel, labels[unknown] != nil else {
            throw PolicyError("taxonomy_invalid", "$.unknown_label")
        }
        return PolicyTaxonomy(
            labels: labels.keys.sorted(),
            policyLabels: labels.mapValues(\.policyLabel),
            specificity: labels.mapValues(\.specificity),
            aliases: aliases,
            unknownLabel: unknown,
            digest: digest
        )
    }

    /// Validate profile document bytes in the order every kit shares.
    static func decodeProfile(
        _ data: Data,
        name: String,
        digest: String,
        taxonomy: PolicyTaxonomy,
        supportedCapabilities: [String] = Policy.supportedCapabilities
    ) throws -> Policy {
        let document: ProfileDocument
        do {
            document = try JSONDecoder().decode(ProfileDocument.self, from: data)
        } catch {
            throw PolicyError("profile_invalid")
        }
        guard document.irSchema == irSchema else {
            throw PolicyError("profile_invalid", "$.ir_schema")
        }
        guard document.irVersion == irVersion else {
            throw PolicyError("ir_version_unsupported", "$.ir_version")
        }
        guard document.name == name else {
            throw PolicyError("profile_invalid", "$.name")
        }
        guard document.taxonomyDigest == taxonomy.digest else {
            throw PolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        guard let required = document.requiredCapabilities else {
            throw PolicyError("profile_invalid", "$.required_capabilities")
        }
        let supported = Set(supportedCapabilities)
        guard required.allSatisfy(supported.contains) else {
            throw PolicyError("capability_missing", "$.required_capabilities")
        }
        guard document.merge?.algorithm == mergeAlgorithm, document.merge?.version == mergeVersion
        else {
            throw PolicyError("merge_algorithm_unsupported", "$.merge")
        }
        guard document.surrogate?.algorithm == surrogateAlgorithm,
            document.surrogate?.version == surrogateVersion
        else {
            throw PolicyError("surrogate_algorithm_unsupported", "$.surrogate")
        }
        guard let mode = document.arbitrationMode, ["balanced", "high_recall_union"].contains(mode)
        else {
            throw PolicyError("profile_invalid", "$.arbitration_mode")
        }
        guard let strict = document.strictNoLeak else {
            throw PolicyError("profile_invalid", "$.strict_no_leak")
        }
        guard let rawActions = document.actions, Set(rawActions.keys) == Set(taxonomy.labels) else {
            throw PolicyError("profile_invalid", "$.actions")
        }
        var actions: [String: PolicyAction] = [:]
        for (label, raw) in rawActions {
            guard let action = PolicyAction(rawValue: raw) else {
                throw PolicyError("profile_invalid", "$.actions")
            }
            actions[label] = action
        }
        guard let rawDefault = document.defaultAction, let defaultAction = PolicyAction(rawValue: rawDefault)
        else {
            throw PolicyError("profile_invalid", "$.default_action")
        }
        guard let thresholds = document.thresholds, let rawDefaultThreshold = thresholds.default,
            let rawLabelThresholds = thresholds.labels,
            rawLabelThresholds.keys.allSatisfy({ taxonomy.policyLabels[$0] != nil })
        else {
            throw PolicyError("profile_invalid", "$.thresholds")
        }
        let defaultThreshold = try threshold(rawDefaultThreshold)
        var labelThresholds: [String: PolicyThreshold] = [:]
        for (label, entry) in rawLabelThresholds {
            labelThresholds[label] = try threshold(entry)
        }
        var policyLabelActions: [String: PolicyAction] = [:]
        for (label, raw) in document.policyLabelActions ?? [:] {
            if let action = PolicyAction(rawValue: raw) {
                policyLabelActions[label] = action
            }
        }
        return Policy(
            name: name,
            schemaVersion: document.schemaVersion ?? 1,
            digest: digest,
            posture: document.posture ?? "",
            thresholdProfile: document.thresholdProfile ?? "",
            defaultAction: defaultAction,
            defaultActionBias: document.defaultActionBias,
            arbitrationMode: mode,
            strictNoLeak: strict,
            safetySweepMandatory: document.safetySweepMandatory ?? false,
            keepMapping: document.keepMapping ?? false,
            reversibleID: document.reversibleID ?? false,
            forcedCascadeTiers: document.forcedCascadeTiers ?? [],
            policyLabelActions: policyLabelActions,
            actions: actions,
            taxonomy: taxonomy,
            defaultThreshold: defaultThreshold,
            labelThresholds: labelThresholds,
            requiredCapabilities: required,
            excludedFeatures: document.excludedFeatures ?? []
        )
    }

    private static func threshold(_ entry: ThresholdDocument) throws -> PolicyThreshold {
        guard let keep = entry.keepFloorPPM, let escalate = entry.escalateBelowPPM,
            (0...KitParity.confidenceScale).contains(keep),
            (0...KitParity.confidenceScale).contains(escalate)
        else {
            throw PolicyError("profile_invalid", "$.thresholds")
        }
        return PolicyThreshold(keepFloorPPM: keep, escalateBelowPPM: escalate)
    }

    private static func isDigest(_ value: String) -> Bool {
        guard value.hasPrefix("sha256:") else {
            return false
        }
        let hex = value.dropFirst("sha256:".count)
        return hex.count == 64
            && hex.unicodeScalars.allSatisfy {
                ("0"..."9").contains($0) || ("a"..."f").contains($0)
            }
    }

    private static func loadBundledTaxonomy() throws -> PolicyTaxonomy {
        let manifest = try bundledManifest()
        guard let data = bundledResource(taxonomyFile) else {
            throw PolicyError("resource_missing", "$.taxonomy_digest")
        }
        let digest = sha256Digest(data)
        guard digest == manifest.taxonomyDigest else {
            throw PolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest")
        }
        return try decodeTaxonomy(data, digest: digest)
    }
}

/// A single policy action applied to a detected span.
public struct DeidentifiedSpanAction: Equatable, Sendable {
    public let label: String
    public let canonicalLabel: String
    public let action: PolicyAction
    public let start: Int
    public let end: Int
    public let confidence: Float
    public let replacement: String?
}

/// Result of an on-device policy-driven de-identification run.
public struct PolicyDeidentificationResult: Equatable, Sendable {
    public let redactedText: String
    public let policyName: String
    public let actions: [DeidentifiedSpanAction]
}

private struct ManifestDocument: Decodable {
    let schema: String?
    let schemaVersion: Int?
    let irSchema: String?
    let irVersion: Int?
    let defaultProfile: String?
    let profiles: [String: String]?
    let excludedProfiles: [String: String]?
    let profileAliases: [String: String]?
    let taxonomyDigest: String?

    enum CodingKeys: String, CodingKey {
        case schema
        case schemaVersion = "schema_version"
        case irSchema = "ir_schema"
        case irVersion = "ir_version"
        case defaultProfile = "default_profile"
        case profiles
        case excludedProfiles = "excluded_profiles"
        case profileAliases = "profile_aliases"
        case taxonomyDigest = "taxonomy_digest"
    }
}

private struct TaxonomyDocument: Decodable {
    struct Label: Decodable {
        let policyLabel: String
        let specificity: Int

        enum CodingKeys: String, CodingKey {
            case policyLabel = "policy_label"
            case specificity
        }
    }

    let name: String?
    let version: Int?
    let labels: [String: Label]?
    let aliases: [String: String]?
    let unknownLabel: String?

    enum CodingKeys: String, CodingKey {
        case name
        case version
        case labels
        case aliases
        case unknownLabel = "unknown_label"
    }
}

private struct AlgorithmDocument: Decodable {
    let algorithm: String?
    let version: Int?
}

private struct ThresholdDocument: Decodable {
    let keepFloorPPM: Int?
    let escalateBelowPPM: Int?

    enum CodingKeys: String, CodingKey {
        case keepFloorPPM = "keep_floor_ppm"
        case escalateBelowPPM = "escalate_below_ppm"
    }
}

private struct ThresholdsDocument: Decodable {
    let `default`: ThresholdDocument?
    let labels: [String: ThresholdDocument]?
}

private struct ProfileDocument: Decodable {
    let irSchema: String?
    let irVersion: Int?
    let schemaVersion: Int?
    let name: String?
    let posture: String?
    let thresholdProfile: String?
    let defaultAction: String?
    let defaultActionBias: String?
    let arbitrationMode: String?
    let strictNoLeak: Bool?
    let safetySweepMandatory: Bool?
    let keepMapping: Bool?
    let reversibleID: Bool?
    let forcedCascadeTiers: [String]?
    let policyLabelActions: [String: String]?
    let actions: [String: String]?
    let requiredCapabilities: [String]?
    let excludedFeatures: [String]?
    let merge: AlgorithmDocument?
    let surrogate: AlgorithmDocument?
    let taxonomyDigest: String?
    let thresholds: ThresholdsDocument?

    enum CodingKeys: String, CodingKey {
        case irSchema = "ir_schema"
        case irVersion = "ir_version"
        case schemaVersion = "schema_version"
        case name
        case posture
        case thresholdProfile = "threshold_profile"
        case defaultAction = "default_action"
        case defaultActionBias = "default_action_bias"
        case arbitrationMode = "arbitration_mode"
        case strictNoLeak = "strict_no_leak"
        case safetySweepMandatory = "safety_sweep_mandatory"
        case keepMapping = "keep_mapping"
        case reversibleID = "reversible_id"
        case forcedCascadeTiers = "forced_cascade_tiers"
        case policyLabelActions = "policy_label_actions"
        case actions
        case requiredCapabilities = "required_capabilities"
        case excludedFeatures = "excluded_features"
        case merge
        case surrogate
        case taxonomyDigest = "taxonomy_digest"
        case thresholds
    }
}
