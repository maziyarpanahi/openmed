import Foundation

/// Value-free reason codes for decision calibration profiles in signed bundles.
///
/// Mirrors `DecisionCalibrationReason` in
/// `openmed/core/portable_bundle_calibration.py`.
public enum DecisionCalibrationReason: String, CaseIterable, Sendable {
    case bundleUnverified = "calibration_bundle_unverified"
    case extensionMissing = "calibration_extension_missing"
    case malformed = "calibration_malformed"
    case versionUnsupported = "calibration_version_unsupported"
    case digestMismatch = "calibration_digest_mismatch"
    case modelMismatch = "calibration_model_mismatch"
    case runtimeMismatch = "calibration_runtime_mismatch"
    case scopeMismatch = "calibration_scope_mismatch"
    case evidenceMismatch = "calibration_evidence_mismatch"
    case profileMissing = "calibration_profile_missing"
    case runtimeUnbound = "calibration_runtime_unbound"
    case schemaIncompatible = "calibration_schema_incompatible"
    case pinMismatch = "calibration_pin_mismatch"
    case requestInvalid = "calibration_request_invalid"
}

/// Error that carries only a controlled calibration reason code.
public struct DecisionCalibrationError: Error, Equatable, Sendable, CustomStringConvertible {
    public let reason: DecisionCalibrationReason

    public init(_ reason: DecisionCalibrationReason) {
        self.reason = reason
    }

    public var description: String { reason.rawValue }
}

/// One isotonic-regression breakpoint, in parts per million.
public struct DecisionCalibrationPoint: Equatable, Sendable {
    public let inputPPM: Int64
    public let outputPPM: Int64
}

/// Calibration method with its integer parameters.
public enum DecisionCalibrationMethod: Equatable, Sendable {
    case deterministicMargin
    case temperatureScaling(temperaturePPM: Int64)
    case plattScaling(slopePPM: Int64, interceptPPM: Int64)
    case isotonicRegression(points: [DecisionCalibrationPoint])

    /// The method name the extension stores.
    public var name: String {
        switch self {
        case .deterministicMargin: return "deterministic_margin"
        case .temperatureScaling: return "temperature_scaling"
        case .plattScaling: return "platt_scaling"
        case .isotonicRegression: return "isotonic_regression"
        }
    }
}

/// One runtime variant a profile was calibrated on.
public struct DecisionCalibrationRuntimeBinding: Equatable, Sendable {
    public let runtimeID: String
    public let digest: String
}

/// A validated calibration profile read from a signed bundle.
///
/// Probabilities are integers in parts per million; the `Double` properties
/// divide by one million. `canonicalJSON()` rebuilds the exact extension
/// profile object from these typed fields.
public struct BundleCalibrationProfile: Equatable, Sendable {
    public let calibrationID: String
    public let version: String
    public let method: DecisionCalibrationMethod
    public let minimumConfidencePPM: Int64
    public let minimumMarginPPM: Int64
    public let multiLabelThresholdPPM: Int64
    public let scopeKind: String
    public let scopeDigest: String
    public let modelID: String
    public let modelRevision: String?
    public let runtimes: [DecisionCalibrationRuntimeBinding]
    public let decisionSchemaMin: String
    public let decisionSchemaMaxExclusive: String
    public let datasetDigest: String
    public let inDomainDigest: String
    public let outOfDomainDigest: String
    public let evaluationPath: String?
    public let evaluationSHA256: String?
    public let profileDigest: String

    public var minimumConfidence: Double { Double(minimumConfidencePPM) / 1_000_000 }
    public var minimumMargin: Double { Double(minimumMarginPPM) / 1_000_000 }
    public var multiLabelThreshold: Double { Double(multiLabelThresholdPPM) / 1_000_000 }
    public var runtimeIDs: [String] { runtimes.map(\.runtimeID) }

    /// Returns the bound digest for `runtimeID`, or `nil`.
    public func runtimeDigest(for runtimeID: String) -> String? {
        runtimes.first(where: { PortableBundle.same($0.runtimeID, runtimeID) })?.digest
    }

    /// Returns the canonical JSON of the extension profile object.
    public func canonicalJSON() -> Data {
        Data((try? PortableBundleJSON.encode(json)) ?? [])
    }

    var json: PortableBundleJSON {
        let parameters: [(String, PortableBundleJSON)]
        switch method {
        case .deterministicMargin:
            parameters = []
        case .temperatureScaling(let temperature):
            parameters = [("temperature_ppm", .int(temperature))]
        case .plattScaling(let slope, let intercept):
            parameters = [("intercept_ppm", .int(intercept)), ("slope_ppm", .int(slope))]
        case .isotonicRegression(let points):
            parameters = [
                ("points", .array(points.map { .object([("input_ppm", .int($0.inputPPM)), ("output_ppm", .int($0.outputPPM))]) }))
            ]
        }
        let evaluation: PortableBundleJSON =
            evaluationPath.map { .object([("path", .string($0)), ("sha256", .string(evaluationSHA256 ?? ""))]) } ?? .null
        return .object([
            ("calibration_id", .string(calibrationID)),
            ("decision_schema", .object([("max_exclusive", .string(decisionSchemaMaxExclusive)), ("min", .string(decisionSchemaMin))])),
            ("method", .string(method.name)),
            (
                "model",
                .object([
                    ("id", .string(modelID)),
                    ("revision", modelRevision.map { .string($0) } ?? .null),
                    ("runtimes", .array(runtimes.map { .object([("digest", .string($0.digest)), ("id", .string($0.runtimeID))]) })),
                ])
            ),
            ("parameters", .object(parameters)),
            ("profile_digest", .string(profileDigest)),
            (
                "provenance",
                .object([
                    ("dataset_digest", .string(datasetDigest)),
                    ("evaluation", evaluation),
                    ("in_domain_digest", .string(inDomainDigest)),
                    ("out_of_domain_digest", .string(outOfDomainDigest)),
                ])
            ),
            ("scope", .object([("digest", .string(scopeDigest)), ("kind", .string(scopeKind))])),
            (
                "thresholds",
                .object([
                    ("minimum_confidence_ppm", .int(minimumConfidencePPM)),
                    ("minimum_margin_ppm", .int(minimumMarginPPM)),
                    ("multi_label_threshold_ppm", .int(multiLabelThresholdPPM)),
                ])
            ),
            ("version", .string(version)),
        ])
    }
}

/// Value-free result of reading calibration profiles from a bundle.
public struct DecisionCalibrationRead: Equatable, Sendable {
    public let ok: Bool
    public let reason: DecisionCalibrationReason?
    public let contentID: String
    public let profiles: [BundleCalibrationProfile]
}

/// What the caller loaded and needs a calibration profile for.
public struct DecisionCalibrationRequest: Equatable, Sendable {
    public let calibrationID: String
    public let runtimeID: String
    public let decisionSchemaVersion: String
    public let modelID: String?
    public let modelRevision: String?
    public let runtimeDigest: String?
    public let scopeKind: String?
    public let scopeDigest: String?
    public let profileDigest: String?

    /// Validates every field; throws `calibration_request_invalid` otherwise.
    public init(
        calibrationID: String, runtimeID: String, decisionSchemaVersion: String, modelID: String? = nil,
        modelRevision: String? = nil, runtimeDigest: String? = nil, scopeKind: String? = nil, scopeDigest: String? = nil,
        profileDigest: String? = nil
    ) throws {
        let optional = { (value: String?, pattern: PortableBundlePattern) in value.map { pattern.matches($0) } ?? true }
        guard PortableBundlePattern.controlled.matches(calibrationID), PortableBundlePattern.token.matches(runtimeID),
            PortableBundlePattern.semver.matches(decisionSchemaVersion), optional(modelID, .modelID),
            optional(modelRevision, .revision), optional(runtimeDigest, .sha256), optional(profileDigest, .sha256),
            (scopeKind == nil) == (scopeDigest == nil),
            scopeKind.map({ PortableBundleCalibration.scopeKinds.contains($0) }) ?? true,
            optional(scopeDigest, .sha256)
        else { throw DecisionCalibrationError(.requestInvalid) }
        self.calibrationID = calibrationID
        self.runtimeID = runtimeID
        self.decisionSchemaVersion = decisionSchemaVersion
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.runtimeDigest = runtimeDigest
        self.scopeKind = scopeKind
        self.scopeDigest = scopeDigest
        self.profileDigest = profileDigest
    }
}

/// Value-free result of resolving the profile for a loaded runtime.
public struct DecisionCalibrationSelection: Equatable, Sendable {
    public let ok: Bool
    public let reason: DecisionCalibrationReason?
    public let contentID: String
    public let profile: BundleCalibrationProfile?
}

/// Reader for the `org.openmed.decision_calibration` portable-bundle extension.
///
/// Mirrors `openmed/core/portable_bundle_calibration.py`: the same checks run in
/// the same order and produce the same result for
/// `tests/fixtures/portable_bundle_calibration`. It never touches the network,
/// never runs a model, and reports only reason codes, identifiers, and digests.
public enum PortableBundleCalibration {
    public static let extensionName = "org.openmed.decision_calibration"
    public static let extensionVersion: Int64 = 1
    public static let ppmScale: Int64 = 1_000_000
    static let maxProfiles = 16
    static let maxRuntimes = 32
    static let maxPoints = 64
    static let maxTemperature: Int64 = 100_000_000
    static let maxPlatt: Int64 = 100_000_000
    static let methods: Set<String> = ["deterministic_margin", "isotonic_regression", "platt_scaling", "temperature_scaling"]
    static let scopeKinds: Set<String> = ["labels", "options", "questions"]
    static let profileKeys = [
        "calibration_id", "decision_schema", "method", "model", "parameters", "profile_digest", "provenance", "scope",
        "thresholds", "version",
    ]

    /// Reads and validates every calibration profile of a verified bundle.
    public static func read(manifest: Data, verification: PortableBundleVerification) -> DecisionCalibrationRead {
        read(manifest: manifest, verification: verification, requireFiles: false)
    }

    /// Selects the profile for the runtime a caller loaded, failing closed.
    ///
    /// The bundle must have verified with its file tree. The profile with the
    /// request's calibration id that binds its runtime is selected, then any
    /// loaded model identity, decision schema version, scope, and pinned
    /// profile digest the caller supplies must match it.
    public static func resolve(
        manifest: Data, verification: PortableBundleVerification, request: DecisionCalibrationRequest
    ) -> DecisionCalibrationSelection {
        let read = read(manifest: manifest, verification: verification, requireFiles: true)
        let refused = { (reason: DecisionCalibrationReason?) in
            DecisionCalibrationSelection(ok: false, reason: reason, contentID: read.contentID, profile: nil)
        }
        guard read.ok else { return refused(read.reason) }
        let candidates = read.profiles.filter { PortableBundle.same($0.calibrationID, request.calibrationID) }
        guard !candidates.isEmpty else { return refused(.profileMissing) }
        guard let profile = candidates.first(where: { $0.runtimeDigest(for: request.runtimeID) != nil }) else {
            return refused(.runtimeUnbound)
        }
        if let modelID = request.modelID, !PortableBundle.same(modelID, profile.modelID) { return refused(.modelMismatch) }
        if let revision = request.modelRevision, revision != profile.modelRevision { return refused(.modelMismatch) }
        if let digest = request.runtimeDigest, digest != profile.runtimeDigest(for: request.runtimeID) {
            return refused(.modelMismatch)
        }
        let version = semverKey(request.decisionSchemaVersion)
        guard !version.lexicographicallyPrecedes(semverKey(profile.decisionSchemaMin)),
            version.lexicographicallyPrecedes(semverKey(profile.decisionSchemaMaxExclusive))
        else { return refused(.schemaIncompatible) }
        if let kind = request.scopeKind, kind != profile.scopeKind || request.scopeDigest != profile.scopeDigest {
            return refused(.scopeMismatch)
        }
        if let pin = request.profileDigest, pin != profile.profileDigest { return refused(.pinMismatch) }
        return DecisionCalibrationSelection(ok: true, reason: nil, contentID: read.contentID, profile: profile)
    }

    /// Returns the digest that binds a profile to one runtime of canonical
    /// manifest bytes, or `nil` when the manifest or runtime is invalid.
    public static func runtimeDigest(manifest: Data, runtimeID: String) -> String? {
        guard let value = try? PortableBundle.parseCanonical(manifest), case .object(let root) = value else {
            return nil
        }
        return runtimeDigest(root, runtimeID)
    }

    // MARK: - Internals

    typealias Members = [(String, PortableBundleJSON)]
    struct Malformed: Error {}

    static func read(manifest: Data, verification: PortableBundleVerification, requireFiles: Bool) -> DecisionCalibrationRead {
        let identity = PortableBundle.contentID(of: manifest)
        let refused = { (reason: DecisionCalibrationReason) in
            DecisionCalibrationRead(ok: false, reason: reason, contentID: identity, profiles: [])
        }
        guard verification.ok, verification.contentID.map({ PortableBundle.same($0, identity) }) ?? false,
            !requireFiles || verification.filesVerified
        else { return refused(.bundleUnverified) }
        guard let root = structural(manifest) else { return refused(.bundleUnverified) }
        guard case .object(let extensions)? = PortableBundleJSON.member(root, "extensions"),
            PortableBundleJSON.member(extensions, extensionName) != nil
        else { return refused(.extensionMissing) }
        do {
            let profiles = try validate(root)
            return DecisionCalibrationRead(ok: true, reason: nil, contentID: identity, profiles: profiles)
        } catch let error as DecisionCalibrationError {
            return refused(error.reason)
        } catch {
            return refused(.malformed)
        }
    }

    static func structural(_ manifest: Data) -> Members? {
        guard let value = try? PortableBundle.parseCanonical(manifest), case .object(let root) = value else {
            return nil
        }
        var understood: Set<String> = []
        if case .array(let items)? = PortableBundleJSON.member(root, "critical_extensions") {
            for case .string(let name) in items { understood.insert(name) }
        }
        guard (try? PortableBundle.validate(value, understood: understood)) != nil else { return nil }
        return root
    }

    /// Validates the extension of a structurally valid manifest, in the
    /// normative order of the Python reference.
    static func validate(_ root: Members) throws -> [BundleCalibrationProfile] {
        guard case .object(let extensions)? = PortableBundleJSON.member(root, "extensions") else {
            throw DecisionCalibrationError(.malformed)
        }
        let raw = PortableBundleJSON.member(extensions, extensionName)
        let profiles: [BundleCalibrationProfile]
        var items: [PortableBundleJSON] = []
        do {
            guard case .object(let members)? = raw, case .int(let version)? = PortableBundleJSON.member(members, "schema_version")
            else { throw Malformed() }
            guard version == extensionVersion else { throw DecisionCalibrationError(.versionUnsupported) }
            guard PortableBundleJSON.hasExactKeys(members, ["profiles", "schema_version"]),
                case .array(let list)? = PortableBundleJSON.member(members, "profiles"), (1...maxProfiles).contains(list.count)
            else { throw Malformed() }
            items = list
            profiles = try list.map(parseProfile)
            var bindings: Set<String> = []
            for profile in profiles {
                for runtimeID in profile.runtimeIDs {
                    guard bindings.insert(profile.calibrationID + "\n" + runtimeID).inserted else { throw Malformed() }
                }
            }
        } catch is Malformed {
            throw DecisionCalibrationError(.malformed)
        }

        guard case .object(let model)? = PortableBundleJSON.member(root, "model") else {
            throw DecisionCalibrationError(.malformed)
        }
        let manifestModelID = stringValue(PortableBundleJSON.member(model, "id"))
        let manifestRevision = stringValue(PortableBundleJSON.member(model, "revision"))
        let digests = fileDigests(root)
        var runtimeIDs: Set<String> = []
        if case .array(let runtimes)? = PortableBundleJSON.member(root, "runtimes") {
            for case .object(let runtime) in runtimes {
                if let id = stringValue(PortableBundleJSON.member(runtime, "id")) { runtimeIDs.insert(id) }
            }
        }
        var labelsDigest: String?
        if case .object(let labels)? = PortableBundleJSON.member(root, "labels"),
            let path = stringValue(PortableBundleJSON.member(labels, "path"))
        {
            labelsDigest = digests[path]
        }
        for (profile, item) in zip(profiles, items) {
            guard case .object(let members) = item, profileDigest(members) == profile.profileDigest else {
                throw DecisionCalibrationError(.digestMismatch)
            }
            guard profile.modelID == manifestModelID, profile.modelRevision == manifestRevision else {
                throw DecisionCalibrationError(.modelMismatch)
            }
            for binding in profile.runtimes {
                guard runtimeIDs.contains(binding.runtimeID), runtimeDigest(root, binding.runtimeID) == binding.digest else {
                    throw DecisionCalibrationError(.runtimeMismatch)
                }
            }
            if profile.scopeKind == "labels", labelsDigest == nil || labelsDigest != profile.scopeDigest {
                throw DecisionCalibrationError(.scopeMismatch)
            }
            if let path = profile.evaluationPath, digests[path] != profile.evaluationSHA256 {
                throw DecisionCalibrationError(.evidenceMismatch)
            }
        }
        return profiles
    }

    static func profileDigest(_ members: Members) -> String? {
        let body = members.filter { !PortableBundle.same($0.0, "profile_digest") }
        guard let bytes = try? PortableBundleJSON.encode(.object(body)) else { return nil }
        return "sha256:" + PortableBundle.sha256Hex(Data(bytes))
    }

    static func runtimeDigest(_ root: Members, _ runtimeID: String) -> String? {
        guard case .array(let runtimes)? = PortableBundleJSON.member(root, "runtimes") else { return nil }
        let digests = fileDigests(root)
        for case .object(let runtime) in runtimes where stringValue(PortableBundleJSON.member(runtime, "id")) == runtimeID {
            guard case .array(let files)? = PortableBundleJSON.member(runtime, "files") else { return nil }
            var entries: [PortableBundleJSON] = []
            for case .string(let path) in files {
                guard let digest = digests[path] else { return nil }
                entries.append(.object([("path", .string(path)), ("sha256", .string(digest))]))
            }
            let body: PortableBundleJSON = .object([
                ("entrypoint", PortableBundleJSON.member(runtime, "entrypoint") ?? .null),
                ("files", .array(entries)),
                ("format", PortableBundleJSON.member(runtime, "format") ?? .null),
                ("id", .string(runtimeID)),
                ("precision", PortableBundleJSON.member(runtime, "precision") ?? .null),
            ])
            guard let bytes = try? PortableBundleJSON.encode(body) else { return nil }
            return "sha256:" + PortableBundle.sha256Hex(Data(bytes))
        }
        return nil
    }

    private static func fileDigests(_ root: Members) -> [String: String] {
        var digests: [String: String] = [:]
        if case .array(let files)? = PortableBundleJSON.member(root, "files") {
            for case .object(let file) in files {
                if let path = stringValue(PortableBundleJSON.member(file, "path")),
                    let digest = stringValue(PortableBundleJSON.member(file, "sha256"))
                {
                    digests[path] = digest
                }
            }
        }
        return digests
    }

    private static func stringValue(_ value: PortableBundleJSON?) -> String? {
        if case .string(let text)? = value { return text }
        return nil
    }

    static func semverKey(_ value: String) -> [Int64] {
        value.split(separator: ".").map { Int64($0) ?? 0 }
    }

    private static func object(_ value: PortableBundleJSON?, _ keys: [String]) throws -> Members {
        guard case .object(let members)? = value, PortableBundleJSON.hasExactKeys(members, keys) else { throw Malformed() }
        return members
    }

    private static func text(_ value: PortableBundleJSON?, _ pattern: PortableBundlePattern) throws -> String {
        guard case .string(let text)? = value, pattern.matches(text) else { throw Malformed() }
        return text
    }

    private static func integer(_ value: PortableBundleJSON?, _ minimum: Int64, _ maximum: Int64) throws -> Int64 {
        guard case .int(let number)? = value, number >= minimum, number <= maximum else { throw Malformed() }
        return number
    }

    private static func parseMethod(_ name: PortableBundleJSON?, _ value: PortableBundleJSON?) throws -> DecisionCalibrationMethod {
        guard case .string(let method)? = name, methods.contains(method) else { throw Malformed() }
        switch method {
        case "temperature_scaling":
            let parameters = try object(value, ["temperature_ppm"])
            return .temperatureScaling(temperaturePPM: try integer(PortableBundleJSON.member(parameters, "temperature_ppm"), 1, maxTemperature))
        case "platt_scaling":
            let parameters = try object(value, ["intercept_ppm", "slope_ppm"])
            let slope = try integer(PortableBundleJSON.member(parameters, "slope_ppm"), -maxPlatt, maxPlatt)
            guard slope != 0 else { throw Malformed() }
            let intercept = try integer(PortableBundleJSON.member(parameters, "intercept_ppm"), -maxPlatt, maxPlatt)
            return .plattScaling(slopePPM: slope, interceptPPM: intercept)
        case "isotonic_regression":
            let parameters = try object(value, ["points"])
            guard case .array(let raw)? = PortableBundleJSON.member(parameters, "points"), (2...maxPoints).contains(raw.count)
            else { throw Malformed() }
            var points: [DecisionCalibrationPoint] = []
            for item in raw {
                let point = try object(item, ["input_ppm", "output_ppm"])
                let next = DecisionCalibrationPoint(
                    inputPPM: try integer(PortableBundleJSON.member(point, "input_ppm"), 0, ppmScale),
                    outputPPM: try integer(PortableBundleJSON.member(point, "output_ppm"), 0, ppmScale))
                if let last = points.last, next.inputPPM <= last.inputPPM || next.outputPPM < last.outputPPM {
                    throw Malformed()
                }
                points.append(next)
            }
            return .isotonicRegression(points: points)
        default:
            _ = try object(value, [])
            return .deterministicMargin
        }
    }

    private static func parseProfile(_ value: PortableBundleJSON) throws -> BundleCalibrationProfile {
        let profile = try object(value, profileKeys)
        let member = { (members: Members, key: String) in PortableBundleJSON.member(members, key) }
        let calibrationID = try text(member(profile, "calibration_id"), .controlled)
        let version = try text(member(profile, "version"), .semver)
        let method = try parseMethod(member(profile, "method"), member(profile, "parameters"))

        let thresholds = try object(member(profile, "thresholds"), ["minimum_confidence_ppm", "minimum_margin_ppm", "multi_label_threshold_ppm"])
        let confidence = try integer(member(thresholds, "minimum_confidence_ppm"), 0, ppmScale)
        let margin = try integer(member(thresholds, "minimum_margin_ppm"), 0, ppmScale)
        let multiLabel = try integer(member(thresholds, "multi_label_threshold_ppm"), 0, ppmScale)
        guard margin <= confidence else { throw Malformed() }

        let scope = try object(member(profile, "scope"), ["digest", "kind"])
        guard case .string(let kind)? = member(scope, "kind"), scopeKinds.contains(kind) else { throw Malformed() }
        let scopeDigest = try text(member(scope, "digest"), .sha256)

        let model = try object(member(profile, "model"), ["id", "revision", "runtimes"])
        let modelID = try text(member(model, "id"), .modelID)
        var revision: String?
        if case .null? = member(model, "revision") {
            revision = nil
        } else {
            revision = try text(member(model, "revision"), .revision)
        }
        guard case .array(let rawRuntimes)? = member(model, "runtimes"), (1...maxRuntimes).contains(rawRuntimes.count) else {
            throw Malformed()
        }
        var bindings: [DecisionCalibrationRuntimeBinding] = []
        for item in rawRuntimes {
            let binding = try object(item, ["digest", "id"])
            bindings.append(
                DecisionCalibrationRuntimeBinding(
                    runtimeID: try text(member(binding, "id"), .token), digest: try text(member(binding, "digest"), .sha256)))
        }
        guard Set(bindings.map(\.runtimeID)).count == bindings.count else { throw Malformed() }

        let range = try object(member(profile, "decision_schema"), ["max_exclusive", "min"])
        let schemaMin = try text(member(range, "min"), .semver)
        let schemaMax = try text(member(range, "max_exclusive"), .semver)
        guard semverKey(schemaMin).lexicographicallyPrecedes(semverKey(schemaMax)) else { throw Malformed() }

        let provenance = try object(member(profile, "provenance"), ["dataset_digest", "evaluation", "in_domain_digest", "out_of_domain_digest"])
        let datasetDigest = try text(member(provenance, "dataset_digest"), .sha256)
        let inDomain = try text(member(provenance, "in_domain_digest"), .sha256)
        let outOfDomain = try text(member(provenance, "out_of_domain_digest"), .sha256)
        var evaluationPath: String?
        var evaluationSHA256: String?
        if case .null? = member(provenance, "evaluation") {
            evaluationPath = nil
        } else {
            let evaluation = try object(member(provenance, "evaluation"), ["path", "sha256"])
            evaluationPath = try text(member(evaluation, "path"), .evaluationPath)
            evaluationSHA256 = try text(member(evaluation, "sha256"), .sha256)
        }

        return BundleCalibrationProfile(
            calibrationID: calibrationID, version: version, method: method, minimumConfidencePPM: confidence,
            minimumMarginPPM: margin, multiLabelThresholdPPM: multiLabel, scopeKind: kind, scopeDigest: scopeDigest,
            modelID: modelID, modelRevision: revision, runtimes: bindings, decisionSchemaMin: schemaMin,
            decisionSchemaMaxExclusive: schemaMax, datasetDigest: datasetDigest, inDomainDigest: inDomain,
            outOfDomainDigest: outOfDomain, evaluationPath: evaluationPath, evaluationSHA256: evaluationSHA256,
            profileDigest: try text(member(profile, "profile_digest"), .sha256))
    }
}

extension PortableBundle {
    /// Runs the content validator of every built-in extension the reader
    /// understands and the manifest carries; failures are `extension_invalid`.
    static func validateBuiltinExtensions(_ value: PortableBundleJSON, understood: Set<String>) throws {
        let name = PortableBundleCalibration.extensionName
        guard understood.contains(name), case .object(let root) = value,
            case .object(let extensions)? = PortableBundleJSON.member(root, "extensions"),
            PortableBundleJSON.member(extensions, name) != nil
        else { return }
        do {
            _ = try PortableBundleCalibration.validate(root)
        } catch {
            throw Refusal(.extensionInvalid)
        }
    }
}

extension PortableBundlePattern {
    static let controlled = Self("[a-z][a-z0-9_.:/-]{0,127}")
    static let evaluationPath = Self("[A-Za-z0-9_][A-Za-z0-9._+/-]{0,511}")
}
