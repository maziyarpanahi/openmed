import Foundation

/// Apple platforms supported by the OpenMedKit package.
public enum OpenMedApplePlatform: String, CaseIterable, Codable, Sendable {
    case iOS
    case macOS
    case watchOS
    case visionOS

    /// The platform for the current compilation target.
    public static var current: OpenMedApplePlatform {
        #if os(watchOS)
            .watchOS
        #elseif os(visionOS)
            .visionOS
        #elseif os(iOS)
            .iOS
        #else
            .macOS
        #endif
    }
}

/// OpenMed model tiers available to the Apple runtime.
public enum OpenMedAppleModelTier: String, CaseIterable, Codable, Sendable {
    case nano
    case tiny
    case base

    fileprivate var rank: Int {
        switch self {
        case .nano:
            0
        case .tiny:
            1
        case .base:
            2
        }
    }
}

/// Metadata required before a CoreML model may be loaded on an Apple platform.
public struct PlatformModelDescriptor: Equatable, Sendable {
    public let identifier: String
    public let modelURL: URL
    public let id2labelURL: URL
    public let tier: OpenMedAppleModelTier
    public let parameterCount: Int
    public let estimatedResidentMemoryMB: Int
    public let isINT8: Bool

    public init(
        identifier: String,
        modelURL: URL,
        id2labelURL: URL,
        tier: OpenMedAppleModelTier,
        parameterCount: Int,
        estimatedResidentMemoryMB: Int,
        isINT8: Bool
    ) {
        self.identifier = identifier
        self.modelURL = modelURL
        self.id2labelURL = id2labelURL
        self.tier = tier
        self.parameterCount = parameterCount
        self.estimatedResidentMemoryMB = estimatedResidentMemoryMB
        self.isINT8 = isINT8
    }
}

/// Fail-closed CoreML selection limits for one Apple platform.
public struct PlatformModelConfiguration: Equatable, Sendable {
    public let platform: OpenMedApplePlatform
    public let maximumTier: OpenMedAppleModelTier
    public let maximumParameterCount: Int
    public let maximumResidentMemoryMB: Int
    public let maximumSequenceLength: Int
    public let requiresINT8: Bool

    /// Selection limits for the current compilation target.
    public static var current: PlatformModelConfiguration {
        configuration(for: .current)
    }

    /// Return the model limits for a supported Apple platform.
    public static func configuration(
        for platform: OpenMedApplePlatform
    ) -> PlatformModelConfiguration {
        switch platform {
        case .watchOS, .visionOS:
            return PlatformModelConfiguration(
                platform: platform,
                maximumTier: .nano,
                maximumParameterCount: 30_000_000,
                maximumResidentMemoryMB: 150,
                maximumSequenceLength: 256,
                requiresINT8: true
            )
        case .iOS:
            return PlatformModelConfiguration(
                platform: platform,
                maximumTier: .tiny,
                maximumParameterCount: 135_000_000,
                maximumResidentMemoryMB: 350,
                maximumSequenceLength: 512,
                requiresINT8: false
            )
        case .macOS:
            return PlatformModelConfiguration(
                platform: platform,
                maximumTier: .base,
                maximumParameterCount: 280_000_000,
                maximumResidentMemoryMB: 900,
                maximumSequenceLength: 512,
                requiresINT8: false
            )
        }
    }

    /// Return whether a model descriptor fits every platform limit.
    public func allows(_ descriptor: PlatformModelDescriptor) -> Bool {
        descriptor.tier.rank <= maximumTier.rank
            && descriptor.parameterCount <= maximumParameterCount
            && descriptor.estimatedResidentMemoryMB <= maximumResidentMemoryMB
            && (!requiresINT8 || descriptor.isINT8)
    }
}

/// Errors raised before or while using a constrained Apple CoreML model.
public enum PlatformModelError: Error, Equatable, LocalizedError, Sendable {
    case noCompatibleModel(OpenMedApplePlatform)
    case invalidInputShape
    case sequenceTooLong(actual: Int, maximum: Int)

    public var errorDescription: String? {
        switch self {
        case .noCompatibleModel(let platform):
            return "No CoreML model satisfies the \(platform.rawValue) memory and tier limits."
        case .invalidInputShape:
            return "CoreML token IDs, attention mask, and offsets must have matching lengths."
        case .sequenceTooLong(let actual, let maximum):
            return "CoreML input has \(actual) tokens; the platform maximum is \(maximum)."
        }
    }
}

/// CoreML-only OpenMed runtime for constrained Apple platforms.
///
/// The loader selects a compatible descriptor before opening CoreML. watchOS
/// and visionOS therefore fail closed unless the model is an INT8 Nano artifact
/// within the canonical 30M-parameter and 150 MB resident-memory ceilings.
public final class PlatformModel {
    public let descriptor: PlatformModelDescriptor
    public let configuration: PlatformModelConfiguration

    private let pipeline: NERPipeline

    public init(
        candidates: [PlatformModelDescriptor],
        configuration: PlatformModelConfiguration = .current
    ) throws {
        let descriptor = try Self.selectModel(
            from: candidates,
            configuration: configuration
        )
        self.descriptor = descriptor
        self.configuration = configuration
        self.pipeline = try NERPipeline(
            validatedDescriptor: descriptor,
            configuration: configuration
        )
    }

    /// Select the highest-capacity compatible CoreML model without loading it.
    public static func selectModel(
        from candidates: [PlatformModelDescriptor],
        configuration: PlatformModelConfiguration = .current
    ) throws -> PlatformModelDescriptor {
        guard
            let selected =
                candidates
                .filter(configuration.allows)
                .sorted(by: preferredModel)
                .first
        else {
            throw PlatformModelError.noCompatibleModel(configuration.platform)
        }
        return selected
    }

    /// Run CoreML token classification with caller-provided tokenization.
    ///
    /// watchOS and visionOS deliberately do not link the full tokenizer/MLX
    /// dependency graph. Callers provide bounded token IDs and character offsets
    /// from their app's bundled Nano model tokenizer.
    public func predict(
        inputIDs: [Int],
        attentionMask: [Int],
        offsets: [(Int, Int)],
        text: String,
        strategy: PostProcessing.AggregationStrategy = .average
    ) throws -> [EntityPrediction] {
        guard inputIDs.count == attentionMask.count, inputIDs.count == offsets.count else {
            throw PlatformModelError.invalidInputShape
        }
        guard inputIDs.count <= configuration.maximumSequenceLength else {
            throw PlatformModelError.sequenceTooLong(
                actual: inputIDs.count,
                maximum: configuration.maximumSequenceLength
            )
        }
        return try pipeline.predict(
            inputIds: inputIDs,
            attentionMask: attentionMask,
            offsets: offsets,
            text: text,
            strategy: strategy
        )
    }

    /// Redact already detected spans without loading another runtime dependency.
    public static func redact(
        _ text: String,
        entities: [EntityPrediction],
        method: DeidentificationMethod = .mask
    ) -> DeidentificationResult {
        let selected = nonOverlappingEntities(entities, in: text)
        var redacted = text

        for entity in selected.reversed() {
            let lower = redacted.index(redacted.startIndex, offsetBy: entity.start)
            let upper = redacted.index(lower, offsetBy: entity.end - entity.start)
            let replacement: String
            switch method {
            case .mask:
                replacement = "[\(entity.entityType.uppercased())]"
            case .remove:
                replacement = ""
            }
            redacted.replaceSubrange(lower..<upper, with: replacement)
        }

        return DeidentificationResult(
            originalText: text,
            deidentifiedText: redacted,
            entities: selected,
            method: method.rawValue
        )
    }

    private static func preferredModel(
        _ lhs: PlatformModelDescriptor,
        _ rhs: PlatformModelDescriptor
    ) -> Bool {
        if lhs.tier.rank != rhs.tier.rank {
            return lhs.tier.rank > rhs.tier.rank
        }
        if lhs.parameterCount != rhs.parameterCount {
            return lhs.parameterCount > rhs.parameterCount
        }
        if lhs.estimatedResidentMemoryMB != rhs.estimatedResidentMemoryMB {
            return lhs.estimatedResidentMemoryMB < rhs.estimatedResidentMemoryMB
        }
        return lhs.identifier < rhs.identifier
    }

    private static func nonOverlappingEntities(
        _ entities: [EntityPrediction],
        in text: String
    ) -> [EntityPrediction] {
        let valid =
            entities
            .filter { $0.start >= 0 && $0.end > $0.start && $0.end <= text.count }
            .sorted {
                if $0.start == $1.start {
                    if $0.end == $1.end {
                        return $0.confidence > $1.confidence
                    }
                    return $0.end > $1.end
                }
                return $0.start < $1.start
            }

        var selected: [EntityPrediction] = []
        for entity in valid where selected.last.map({ $0.end <= entity.start }) ?? true {
            selected.append(entity)
        }
        return selected
    }
}
