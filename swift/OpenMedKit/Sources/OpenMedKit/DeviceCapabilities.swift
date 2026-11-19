import Foundation

/// Value-free errors raised for malformed profiles, probes, or constraints.
public enum DeviceCapabilityError: String, Error, Equatable, Sendable, CustomStringConvertible {
    case profileInvalid = "profile_invalid"
    case probeInvalid = "probe_invalid"
    case constraintsInvalid = "constraints_invalid"

    public var description: String { rawValue }
}

/// Shared vocabulary for normalized device-capability profiles.
///
/// Mirrors `openmed/core/device_capabilities.py`. The golden vectors in
/// `tests/fixtures/device_capabilities/` pin every table here.
public enum DeviceCapabilityVocabulary {
    public static let schema = "openmed.device_capabilities"
    public static let schemaVersion: Int64 = 1

    /// Accelerator kind behind each normalized execution provider.
    public static let providerAccelerators: [String: String] = [
        "coreml": "npu", "cpu": "cpu", "metal": "gpu", "nnapi": "npu", "qnn": "npu", "wasm": "cpu",
        "webgpu": "gpu",
    ]
    /// Execution providers a profile may report on each platform.
    public static let platformProviders: [String: Set<String>] = [
        "android": ["cpu", "nnapi", "qnn"],
        "ios": ["coreml", "cpu", "metal"],
        "linux": ["cpu"],
        "macos": ["coreml", "cpu", "metal"],
        "visionos": ["coreml", "cpu"],
        "watchos": ["coreml", "cpu"],
        "web": ["wasm", "webgpu"],
        "windows": ["cpu"],
    ]
    public static let cpuArchitectures: Set<String> = ["arm64", "armv7", "unknown", "wasm32", "x86_64"]
    public static let features: Set<String> = [
        "cpu_avx2", "cpu_fp16", "cpu_neon", "wasm_simd", "wasm_threads", "webgpu_shader_f16",
    ]
    /// Model tiers in increasing capacity order.
    public static let modelTiers = ["nano", "tiny", "base"]
    public static let networkStates: Set<String> = ["offline", "online", "unknown"]
    public static let minBudgetBytes: Int64 = 16 << 20
    public static let maxBudgetBytes: Int64 = 1 << 40
    public static let maxSequenceLength: Int64 = 1 << 20
    public static let maxOperatorSet: Int64 = 1000
    /// SDK name the Apple adapter reports; matches bundle `compatibility`.
    public static let appleSDKName = "openmedkit-swift"

    static let accelerators: Set<String> = ["cpu", "gpu", "npu"]

    /// Rounds a measured byte budget down to the ladder: `0`, then every power
    /// of two from 16 MiB to 1 TiB.
    public static func budgetBucket(_ value: Int64?) throws -> Int64? {
        guard let value else { return nil }
        guard value >= 0, value <= PortableBundle.maxSafeInteger else { throw DeviceCapabilityError.probeInvalid }
        guard value >= minBudgetBytes else { return 0 }
        var bucket = minBudgetBytes
        while bucket < maxBudgetBytes, bucket * 2 <= value { bucket *= 2 }
        return bucket
    }

    /// Keeps only the major and minor components of a dotted OS version.
    public static func coarseOSVersion(_ value: String?) throws -> String? {
        guard let value else { return nil }
        guard PortableBundlePattern.version.matches(value) else { throw DeviceCapabilityError.probeInvalid }
        return value.split(separator: ".").prefix(2).joined(separator: ".")
    }

    /// Returns the capacity rank of a model tier (`nano` is 0), or `nil`.
    public static func tierRank(_ tier: String) -> Int? {
        modelTiers.firstIndex(of: tier)
    }

    static func isBudget(_ value: Int64?) -> Bool {
        guard let value else { return true }
        return (try? budgetBucket(value)) == value
    }
}

/// Coarse, closed description of what a device can execute.
///
/// Collections are sets: they are sorted and de-duplicated on construction so
/// equivalent profiles encode to identical canonical bytes. A profile holds
/// only controlled vocabulary, dotted versions, and ladder budgets, so it can
/// never carry a device name, vendor identifier, or exact memory size.
public struct DeviceCapabilityProfile: Equatable, Sendable {
    public let platform: String
    public let sdkName: String
    public let sdkVersion: String
    /// `MAJOR` or `MAJOR.MINOR`, or `nil` when unknown or not applicable.
    public let osVersion: String?
    public let cpuArchitecture: String
    public let providers: [String]
    public let runtimes: [String]
    public let precisions: [String]
    public let features: [String]
    /// Highest supported operator-set version per runtime format.
    public let operatorSets: [String: Int64]
    public let memoryBudgetBytes: Int64?
    public let storageBudgetBytes: Int64?
    public let maxSequenceLength: Int64?
    public let maxModelTier: String?
    /// Informational only; never changes a plan and is excluded from `digest`.
    public let network: String

    private static let documentKeys = [
        "accelerators", "cpu_architecture", "features", "max_model_tier", "max_sequence_length",
        "memory_budget_bytes", "network", "operator_sets", "os_version", "platform", "precisions", "providers",
        "runtimes", "schema", "schema_version", "sdk_name", "sdk_version", "storage_budget_bytes",
    ]

    public init(
        platform: String, sdkName: String, sdkVersion: String, osVersion: String? = nil,
        cpuArchitecture: String = "unknown", providers: [String] = [], runtimes: [String] = [],
        precisions: [String] = [], features: [String] = [], operatorSets: [String: Int64] = [:],
        memoryBudgetBytes: Int64? = nil, storageBudgetBytes: Int64? = nil, maxSequenceLength: Int64? = nil,
        maxModelTier: String? = nil, network: String = "unknown"
    ) throws {
        typealias V = DeviceCapabilityVocabulary
        guard PortableBundle.platforms.contains(platform),
            PortableBundlePattern.token.matches(sdkName),
            PortableBundlePattern.version.matches(sdkVersion),
            osVersion.map({ Self.isCoarseOS($0) }) ?? true,
            V.cpuArchitectures.contains(cpuArchitecture),
            V.isBudget(memoryBudgetBytes), V.isBudget(storageBudgetBytes),
            maxSequenceLength.map({ $0 > 0 && $0 <= V.maxSequenceLength }) ?? true,
            maxModelTier.map({ V.tierRank($0) != nil }) ?? true,
            V.networkStates.contains(network),
            operatorSets.allSatisfy({
                PortableBundle.formatPlatforms[$0.key] != nil && $0.value > 0 && $0.value <= V.maxOperatorSet
            })
        else { throw DeviceCapabilityError.profileInvalid }
        let loadable = Set(PortableBundle.formatPlatforms.filter { $0.value.contains(platform) }.keys)
        self.platform = platform
        self.sdkName = sdkName
        self.sdkVersion = sdkVersion
        self.osVersion = osVersion
        self.cpuArchitecture = cpuArchitecture
        self.providers = try Self.set(providers, allowed: V.platformProviders[platform] ?? [])
        self.runtimes = try Self.set(runtimes, allowed: loadable)
        self.precisions = try Self.set(precisions, allowed: PortableBundle.precisions)
        self.features = try Self.set(features, allowed: V.features)
        self.operatorSets = operatorSets
        self.memoryBudgetBytes = memoryBudgetBytes
        self.storageBudgetBytes = storageBudgetBytes
        self.maxSequenceLength = maxSequenceLength
        self.maxModelTier = maxModelTier
        self.network = network
    }

    /// Parses a profile document strictly; unknown or missing members fail.
    public init(json: Data) throws {
        guard let value = try? PortableBundleJSON.parse(json) else { throw DeviceCapabilityError.profileInvalid }
        try self.init(document: value)
    }

    init(document value: PortableBundleJSON) throws {
        let fail = DeviceCapabilityError.profileInvalid
        guard case .object(let members) = value, PortableBundleJSON.hasExactKeys(members, Self.documentKeys),
            value.plan("schema")?.planString == DeviceCapabilityVocabulary.schema,
            value.plan("schema_version")?.planInt == DeviceCapabilityVocabulary.schemaVersion,
            let platform = value.plan("platform")?.planString,
            let sdkName = value.plan("sdk_name")?.planString,
            let sdkVersion = value.plan("sdk_version")?.planString,
            let architecture = value.plan("cpu_architecture")?.planString,
            let network = value.plan("network")?.planString,
            let providers = value.plan("providers")?.planStrings,
            let runtimes = value.plan("runtimes")?.planStrings,
            let precisions = value.plan("precisions")?.planStrings,
            let features = value.plan("features")?.planStrings,
            let accelerators = value.plan("accelerators")?.planStrings,
            case .object(let operatorMembers)? = value.plan("operator_sets")
        else { throw fail }
        var operatorSets: [String: Int64] = [:]
        for (key, item) in operatorMembers {
            guard let number = item.planInt, operatorSets[key] == nil else { throw fail }
            operatorSets[key] = number
        }
        let optionalString = { (key: String) throws -> String? in
            guard let item = value.plan(key) else { throw fail }
            if item.planIsNull { return nil }
            guard let text = item.planString else { throw fail }
            return text
        }
        let optionalInt = { (key: String) throws -> Int64? in
            guard let item = value.plan(key) else { throw fail }
            if item.planIsNull { return nil }
            guard let number = item.planInt else { throw fail }
            return number
        }
        try self.init(
            platform: platform, sdkName: sdkName, sdkVersion: sdkVersion, osVersion: try optionalString("os_version"),
            cpuArchitecture: architecture, providers: providers, runtimes: runtimes, precisions: precisions,
            features: features, operatorSets: operatorSets,
            memoryBudgetBytes: try optionalInt("memory_budget_bytes"),
            storageBudgetBytes: try optionalInt("storage_budget_bytes"),
            maxSequenceLength: try optionalInt("max_sequence_length"),
            maxModelTier: try optionalString("max_model_tier"), network: network)
        guard Set(accelerators) == Set(self.accelerators), accelerators.count == Set(accelerators).count else {
            throw fail
        }
    }

    /// Sorted accelerator kinds behind `providers`.
    public var accelerators: [String] {
        Set(providers.compactMap { DeviceCapabilityVocabulary.providerAccelerators[$0] }).sorted(by: PortableBundle.utf8Less)
    }

    /// The canonical profile document bytes (`device_capability_profile`).
    public func canonicalJSON() -> Data {
        Data((try? PortableBundleJSON.encode(document(includeNetwork: true))) ?? [])
    }

    /// `sha256:` digest of every resolution-relevant member (not `network`).
    public var digest: String {
        "sha256:" + PortableBundle.sha256Hex(Data((try? PortableBundleJSON.encode(document(includeNetwork: false))) ?? []))
    }

    func document(includeNetwork: Bool) -> PortableBundleJSON {
        let strings = { (items: [String]) in PortableBundleJSON.array(items.map { .string($0) }) }
        let optionalString = { (item: String?) in item.map { PortableBundleJSON.string($0) } ?? .null }
        let optionalInt = { (item: Int64?) in item.map { PortableBundleJSON.int($0) } ?? .null }
        var members: [(String, PortableBundleJSON)] = [
            ("accelerators", strings(accelerators)),
            ("cpu_architecture", .string(cpuArchitecture)),
            ("features", strings(features)),
            ("max_model_tier", optionalString(maxModelTier)),
            ("max_sequence_length", optionalInt(maxSequenceLength)),
            ("memory_budget_bytes", optionalInt(memoryBudgetBytes)),
            ("operator_sets", .object(operatorSets.map { ($0.key, .int($0.value)) })),
            ("os_version", optionalString(osVersion)),
            ("platform", .string(platform)),
            ("precisions", strings(precisions)),
            ("providers", strings(providers)),
            ("runtimes", strings(runtimes)),
            ("schema", .string(DeviceCapabilityVocabulary.schema)),
            ("schema_version", .int(DeviceCapabilityVocabulary.schemaVersion)),
            ("sdk_name", .string(sdkName)),
            ("sdk_version", .string(sdkVersion)),
            ("storage_budget_bytes", optionalInt(storageBudgetBytes)),
        ]
        if includeNetwork { members.append(("network", .string(network))) }
        return .object(members)
    }

    private static func isCoarseOS(_ value: String) -> Bool {
        PortableBundlePattern.version.matches(value) && value.split(separator: ".").count <= 2
    }

    private static func set(_ items: [String], allowed: Set<String>) throws -> [String] {
        guard items.allSatisfy({ allowed.contains($0) }) else { throw DeviceCapabilityError.profileInvalid }
        return Set(items).sorted(by: PortableBundle.utf8Less)
    }
}

// MARK: - Apple adapter

extension DeviceCapabilityProfile {
    private static let mlxPlatforms: Set<OpenMedApplePlatform> = [.iOS, .macOS]

    /// Maps OpenMedKit's Apple platform limits into the normalized profile.
    ///
    /// Mirrors `profile_from_apple_probe` in the Python reference. The memory
    /// budget (rounded down to the ladder), sequence limit, tier ceiling, and
    /// INT8 requirement come from `PlatformModelConfiguration`. MLX counts only
    /// on iOS and macOS, where the package links it. Device names, vendor
    /// identifiers, and hardware model strings are never read.
    public static func apple(
        configuration: PlatformModelConfiguration, osVersion: String?, mlxAvailable: Bool, architecture: String,
        sdkVersion: String, storageFreeBytes: Int64? = nil, operatorSets: [String: Int64] = [:],
        network: String = "unknown"
    ) throws -> DeviceCapabilityProfile {
        let platform = configuration.platform
        let mlx = mlxAvailable && mlxPlatforms.contains(platform)
        let normalizedArchitecture = ["arm64", "x86_64"].contains(architecture) ? architecture : "unknown"
        var precisions = ["fp16", "fp32", "int4", "int8"]
        if mlx { precisions += ["bf16", "int2"] }
        if configuration.requiresINT8 { precisions = ["int8"] }
        do {
            return try DeviceCapabilityProfile(
                platform: platform.rawValue.lowercased(), sdkName: DeviceCapabilityVocabulary.appleSDKName,
                sdkVersion: sdkVersion, osVersion: try DeviceCapabilityVocabulary.coarseOSVersion(osVersion),
                cpuArchitecture: normalizedArchitecture, providers: ["coreml", "cpu"] + (mlx ? ["metal"] : []),
                runtimes: ["coreml"] + (mlx ? ["mlx"] : []), precisions: precisions,
                features: normalizedArchitecture == "arm64" ? ["cpu_neon"] : [], operatorSets: operatorSets,
                memoryBudgetBytes: try DeviceCapabilityVocabulary.budgetBucket(
                    Int64(configuration.maximumResidentMemoryMB) << 20),
                storageBudgetBytes: try DeviceCapabilityVocabulary.budgetBucket(storageFreeBytes),
                maxSequenceLength: Int64(configuration.maximumSequenceLength),
                maxModelTier: configuration.maximumTier.rawValue, network: network)
        } catch {
            throw DeviceCapabilityError.probeInvalid
        }
    }

    /// Probes the running Apple device with compile-time and `ProcessInfo`
    /// facts only: platform, OS major and minor version, CPU architecture, and
    /// whether MLX is linked and usable (never on the Simulator).
    public static func currentApple(
        sdkVersion: String, configuration: PlatformModelConfiguration = .current, storageFreeBytes: Int64? = nil,
        operatorSets: [String: Int64] = [:], network: String = "unknown"
    ) throws -> DeviceCapabilityProfile {
        let version = ProcessInfo.processInfo.operatingSystemVersion
        #if arch(arm64) || arch(arm64_32)
            let architecture = "arm64"
        #elseif arch(x86_64)
            let architecture = "x86_64"
        #else
            let architecture = "unknown"
        #endif
        #if canImport(MLX) && !targetEnvironment(simulator)
            let mlx = true
        #else
            let mlx = false
        #endif
        return try apple(
            configuration: configuration, osVersion: "\(version.majorVersion).\(version.minorVersion)",
            mlxAvailable: mlx, architecture: architecture, sdkVersion: sdkVersion,
            storageFreeBytes: storageFreeBytes, operatorSets: operatorSets, network: network)
    }
}

// MARK: - JSON helpers

extension PortableBundleJSON {
    func plan(_ key: String) -> PortableBundleJSON? {
        guard case .object(let members) = self else { return nil }
        return PortableBundleJSON.member(members, key)
    }

    var planString: String? {
        if case .string(let text) = self { return text }
        return nil
    }

    var planInt: Int64? {
        if case .int(let number) = self { return number }
        return nil
    }

    var planBool: Bool? {
        if case .bool(let flag) = self { return flag }
        return nil
    }

    var planArray: [PortableBundleJSON]? {
        if case .array(let items) = self { return items }
        return nil
    }

    var planIsNull: Bool {
        if case .null = self { return true }
        return false
    }

    var planStrings: [String]? {
        guard let items = planArray else { return nil }
        let strings = items.compactMap(\.planString)
        return strings.count == items.count ? strings : nil
    }
}
