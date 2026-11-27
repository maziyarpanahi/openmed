import Foundation

/// An OpenMedKit error that also carries a portable registry code.
///
/// The code is the one Python, the REST service and the other kits use for the
/// same kind of failure (`openmed/core/error_codes.json`). Each error keeps its
/// own, more specific reason code and description; ``openMedErrorCode`` only
/// adds the portable family, so an app can branch the same way on every
/// platform:
///
/// ```swift
/// do {
///     _ = try Policy(named: name)
/// } catch let error as any OpenMedCodedError where error.openMedErrorCode == .policyError {
///     // Ask for another profile.
/// }
/// ```
public protocol OpenMedCodedError: Error {
    /// The portable registry code for this failure.
    var openMedErrorCode: OpenMedErrorCode { get }
}

extension PolicyError: OpenMedCodedError {
    /// The portable code for this policy failure.
    ///
    /// An empty or out-of-range surrogate key is a `configuration_error`;
    /// caller detections with invalid labels, offsets, confidences or sources
    /// are an `input_error`; a profile that needs a capability this runtime
    /// lacks is a `capability_error`. Every other code, such as
    /// `profile_unknown` or a manifest, taxonomy or digest failure, is a
    /// `policy_error`.
    public var openMedErrorCode: OpenMedErrorCode {
        switch code {
        case "surrogate_key_invalid":
            return .configurationError
        case "detection_label_invalid", "detection_offset_invalid", "detection_offset_out_of_bounds",
            "detection_confidence_invalid", "detection_source_invalid":
            return .inputError
        case "capability_missing":
            return .capabilityError
        default:
            return .policyError
        }
    }
}

extension OpenMedModelStoreError: OpenMedCodedError {
    /// Every model-store failure is a `model_load_error`: the model could not
    /// be fetched, confined or completed.
    public var openMedErrorCode: OpenMedErrorCode { .modelLoadError }
}

extension OpenMedTokenizerError: OpenMedCodedError {
    /// A tokenizer that is not available locally, fails verification or cannot
    /// be read is a `model_load_error`. A network opt-in without a pin, a
    /// malformed pin, or a pin for another repository is a
    /// `configuration_error`.
    public var openMedErrorCode: OpenMedErrorCode {
        switch self {
        case .pinnedRevisionRequired, .invalidPin, .identityMismatch:
            return .configurationError
        case .missingLocalAssets, .downloadFailed, .sizeMismatch, .digestMismatch, .cacheUnavailable, .invalidAssets:
            return .modelLoadError
        }
    }
}

extension OpenMedMLXArtifactError: OpenMedCodedError {
    /// An artifact for a task or family this runtime cannot run is a
    /// `capability_error`; an absent, unreadable or malformed artifact is a
    /// `model_load_error`.
    var openMedErrorCode: OpenMedErrorCode {
        switch self {
        case .unsupportedArchitecture:
            return .capabilityError
        case .invalidManifestFormat, .missingTokenizerAssets, .missingTokenizerReference, .unreadableFile,
            .malformedFile:
            return .modelLoadError
        }
    }
}

#if canImport(MLX) && !os(watchOS) && !os(visionOS)
    extension OpenMedMLXRuntimeError: OpenMedCodedError {
        /// MLX cannot run in this process, such as on the iOS Simulator: a
        /// `capability_error`.
        public var openMedErrorCode: OpenMedErrorCode { .capabilityError }
    }
#endif
