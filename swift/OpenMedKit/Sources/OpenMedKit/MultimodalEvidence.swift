import CoreFoundation
import Foundation

/// Controlled failures shared with Python; rejected source values are never echoed.
public enum MultimodalContractRefusal: String, Error, LocalizedError, Sendable {
    case invalid = "invalid_multimodal_contract"
    case schema = "unsupported_multimodal_schema"
    case capability = "unsupported_multimodal_capability"
    case transform = "invalid_multimodal_transform"
    case provenance = "incomplete_multimodal_provenance"
    public var errorDescription: String? { rawValue }
}

public enum MultimodalProjectionKind: String, Sendable, CaseIterable {
    case asset, visual, transcript, provider, clock
}

public struct MultimodalAsset: Sendable, Equatable {
    public let manifestDigest: String
    public let assetRef: String
    public let mediaType: String
    public let sourceDigest: String
    public let byteSize: Int64
    public let pages: Int64?
    public let width: Int64?
    public let height: Int64?
    public let frames: Int64?
    public let durationSeconds: Double?
}

/// Top-left source-space, half-open rectangle, in pixels or PDF points.
public struct MultimodalRectangle: Sendable, Equatable {
    public let left: Double
    public let top: Double
    public let right: Double
    public let bottom: Double
}

/// Half-open scalar, sample or relative-time interval; its parent defines units.
public struct MultimodalInterval: Sendable, Equatable {
    public let start: Int64
    public let end: Int64
}

public enum MultimodalVisualKind: String, Sendable {
    case page = "page_geometry"
    case pixels
    case ocr = "ocr_span"
}

public enum MultimodalSpatialUnit: String, Sendable { case pixel, point }

/// Original coordinates remain separate from their checked display rotation.
public struct MultimodalVisual: Sendable, Equatable {
    public let nodeRef: String
    public let sourceDigest: String
    public let kind: MultimodalVisualKind
    public let page: Int64
    public let width: Double
    public let height: Double
    public let unit: MultimodalSpatialUnit
    public let sourceBox: MultimodalRectangle?
    public let scalarSpan: MultimodalInterval?
    public let rotation: Int64
    public let displayWidth: Double
    public let displayHeight: Double
    public let displayBox: MultimodalRectangle?
}

public enum MultimodalSpanInvalidation: String, Sendable {
    case textChanged = "cited_text_changed"
    case tokenChanged = "cited_token_changed"
}

/// Invalidated citations have no current coordinates or sample bounds.
public struct MultimodalTranscript: Sendable, Equatable {
    public let snapshotDigest: String
    public let sessionRef: String
    public let segmentRef: String
    public let spanRef: String
    public let originalRevision: Int64
    public let revision: Int64
    public let originalScalarSpan: MultimodalInterval
    public let tokenRefs: [String]
    public let scalarLength: Int64
    public let finalized: Bool
    public let invalidation: MultimodalSpanInvalidation?
    public let currentScalarSpan: MultimodalInterval?
    public let relativeMilliseconds: MultimodalInterval?
    public let sampleRateHz: Int64
    /// Outward conversion to a declared grid, not acoustic alignment evidence.
    public let sampleBounds: MultimodalInterval?
}

public enum MultimodalProviderOutcome: String, Sendable, CaseIterable {
    case success, abstention
    case unavailable = "provider_unavailable"
    case validationFailure = "validation_failure"
}

public enum MultimodalAbstention: String, Sendable, CaseIterable {
    case unsupportedMedia = "unsupported_media"
    case malformedMedia = "malformed_media"
    case resourceLimit = "resource_limit"
    case lowQuality = "low_quality"
    case phiUncertainty = "phi_uncertainty"
    case speakerUncertainty = "speaker_uncertainty"
    case temporalInstability = "temporal_instability"
}

public struct MultimodalProviderResult: Sendable, Equatable {
    public let providerID: String
    public let modelID: String
    public let inputDigest: String
    public let outputDigest: String?
    public let outcome: MultimodalProviderOutcome
    public let abstention: MultimodalAbstention?
    public let durationMilliseconds: Double
    public let counts: [String: Int64]
}

public enum MultimodalClockState: String, Sendable {
    case aligned = "aligned-for-review"
    case review = "review-required"
}

public enum MultimodalClockReason: String, Sendable, CaseIterable {
    case anchors = "insufficient-anchors"
    case inconsistent = "inconsistent-clock-anchors"
    case coverage = "clock-coverage-missing"
    case discontinuity = "clock-discontinuity"
    case sparse = "anchor-gap-exceeds-policy"
    case epoch = "clock-epoch-mismatch"
    case declaration = "clock-declaration-mismatch"
    case wallDomain = "wall-reference-mismatch"
    case timing = "source-timing-missing"
    case binding = "source-binding-mismatch"
    case reference = "invalid-source-reference"
    case unfinalized = "transcript-not-finalized"
    case evidence = "source-requires-review"
    case uncertainty = "alignment-uncertainty-exceeds-policy"
    case noOverlap = "evidence-does-not-overlap"
    case possibleOverlap = "overlap-is-uncertain"
    case changed = "source-changed-during-alignment"
}

/// Inclusive uncertainty envelope in a monotonic clock, never wall time.
public struct MultimodalNanosecondBound: Sendable, Equatable {
    public let lower: Int64
    public let upper: Int64
}

public struct MultimodalClockAlignment: Sendable, Equatable {
    public let sourceResultDigest: String
    public let state: MultimodalClockState
    public let reasons: [MultimodalClockReason]
    public let sourceReferenceDigest: String
    public let targetReferenceDigest: String
    public let sourceClockDigest: String
    public let targetClockDigest: String
    public let policyDigest: String
    public let targetStart: MultimodalNanosecondBound?
    public let targetEnd: MultimodalNanosecondBound?
    public let guaranteedOverlap: MultimodalInterval?
    public let uncertaintyNanoseconds: Int64?
}

public enum MultimodalProjection: Sendable, Equatable {
    case asset(MultimodalAsset)
    case visual(MultimodalVisual)
    case transcript(MultimodalTranscript)
    case provider(MultimodalProviderResult)
    case clock(MultimodalClockAlignment)

    public var kind: MultimodalProjectionKind {
        switch self {
        case .asset: return .asset
        case .visual: return .visual
        case .transcript: return .transcript
        case .provider: return .provider
        case .clock: return .clock
        }
    }
}

/// Validated shared wire metadata. Parsing grants no clinical or custody approval.
/// Create through `fromJSON`; no unchecked payload initializer is exposed.
public struct MultimodalPacket: Sendable, Equatable {
    public static let schema = "openmed.multimodal.apple.v1"
    public let projection: MultimodalProjection
    public let evidenceDigest: String
    private let wire: Data

    /// Exact canonical ASCII JSON containing protected metadata, never media bytes.
    public func toJSON() -> Data { wire }

    /// Canonical comparison rejects duplicate keys and numeric coercion.
    public static func fromJSON(_ data: Data) throws -> MultimodalPacket {
        do {
            try Wire.check(!data.isEmpty && data.count <= 65536)
            let root = try Wire.object(JSONSerialization.jsonObject(with: data), "schema_version kind payload evidence_digest")
            try Wire.check(root["schema_version"] as? String == schema, .schema)
            guard let rawKind = root["kind"] as? String, let kind = MultimodalProjectionKind(rawValue: rawKind) else {
                throw MultimodalContractRefusal.capability
            }
            let digest = try Wire.hex(root["evidence_digest"])
            guard let body = root["payload"] as? [String: Any] else { throw MultimodalContractRefusal.invalid }
            let projection: MultimodalProjection
            switch kind {
            case .asset: projection = .asset(try Wire.asset(body))
            case .visual: projection = .visual(try Wire.visual(body))
            case .transcript: projection = .transcript(try Wire.transcript(body))
            case .provider: projection = .provider(try Wire.provider(body))
            case .clock: projection = .clock(try Wire.clock(body))
            }
            var identity = root
            identity.removeValue(forKey: "evidence_digest")
            try Wire.check(try Wire.hash(identity) == digest, .provenance)
            try Wire.check(try ClinicalBrief.canonical(root) == data)
            return MultimodalPacket(projection: projection, evidenceDigest: digest, wire: data)
        } catch let refusal as MultimodalContractRefusal { throw refusal } catch { throw MultimodalContractRefusal.invalid }
    }
}

/// Injected local execution. Declarations are not a network sandbox.
public protocol LocalMultimodalAdapter: Sendable {
    var localOnly: Bool { get }
    var capabilities: Set<MultimodalProjectionKind> { get }
    func run(_ request: MultimodalPacket) async throws -> MultimodalPacket
}

public enum LocalMultimodalExecution {
    /// Refuse unsupported capabilities before invocation and require result binding.
    public static func run(_ adapter: any LocalMultimodalAdapter, request: MultimodalPacket) async throws -> MultimodalPacket {
        guard adapter.localOnly, adapter.capabilities.contains(request.projection.kind) else {
            throw MultimodalContractRefusal.capability
        }
        let response: MultimodalPacket
        do { response = try await adapter.run(request) } catch { throw MultimodalContractRefusal.invalid }
        guard case .provider(let result) = response.projection else { throw MultimodalContractRefusal.invalid }
        guard result.inputDigest == request.evidenceDigest else { throw MultimodalContractRefusal.provenance }
        return response
    }
}

private enum Wire {
    static let maxCoordinate: Int64 = 2_147_483_647
    static let clockLimit: Int64 = 9_000_000_000_000_000_000
    static let clockSchema = "openmed.multimodal.clock-alignment.v1"

    static func check(_ condition: Bool, _ refusal: MultimodalContractRefusal = .invalid) throws {
        if !condition { throw refusal }
    }

    static func object(_ value: Any?, _ fields: String) throws -> [String: Any] {
        guard let object = value as? [String: Any], Set(object.keys) == Set(fields.split(separator: " ").map(String.init)) else {
            throw MultimodalContractRefusal.invalid
        }
        return object
    }

    static func integer(_ value: Any?, _ minimum: Int64 = 0, _ maximum: Int64 = Int64.max) throws -> Int64 {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
            String(cString: number.objCType) != "d", String(cString: number.objCType) != "f",
            let result = Int64(number.stringValue), minimum <= result, result <= maximum
        else { throw MultimodalContractRefusal.invalid }
        return result
    }

    static func boolean(_ value: Any?) throws -> Bool {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else {
            throw MultimodalContractRefusal.invalid
        }
        return number.boolValue
    }

    static func matches(_ string: String, _ pattern: String) -> Bool {
        guard let range = string.range(of: pattern, options: .regularExpression) else { return false }
        return range.lowerBound == string.startIndex && range.upperBound == string.endIndex
    }

    static func hex(_ value: Any?, _ width: Int = 64) throws -> String {
        guard let string = value as? String, string.count == width, matches(string, "^[0-9a-f]+$") else {
            throw MultimodalContractRefusal.provenance
        }
        return string
    }

    static func bits(_ value: Double) -> String {
        String(format: "%016llx", value == 0 ? Double(0).bitPattern : value.bitPattern)
    }

    static func number(_ value: Any?, _ maximum: Double = Double(maxCoordinate)) throws -> Double {
        guard let string = value as? String, string.count == 16, matches(string, "^[0-9a-f]+$"),
            let bitPattern = UInt64(string, radix: 16)
        else { throw MultimodalContractRefusal.invalid }
        let result = Double(bitPattern: bitPattern)
        try check(result.isFinite && result >= 0 && result <= maximum && bits(result) == string)
        return result
    }

    static func hash(_ value: Any) throws -> String {
        String(ClinicalBrief.hash(try ClinicalBrief.canonical(value)).dropFirst(7))
    }

    static func interval(_ value: Any?, maximum: Int64, allowEmpty: Bool = false) throws -> MultimodalInterval {
        guard let values = value as? [Any], values.count == 2 else { throw MultimodalContractRefusal.invalid }
        let start = try integer(values[0], 0, maximum)
        let end = try integer(values[1], 0, maximum)
        try check(allowEmpty ? start <= end : start < end)
        return MultimodalInterval(start: start, end: end)
    }

    static func asset(_ p: [String: Any]) throws -> MultimodalAsset {
        _ = try object(p, "source_schema manifest_digest asset_ref media_type sha256 byte_size pages width height frames duration_f64")
        try check((p["source_schema"] as? NSNumber)?.doubleValue == 1, .schema)
        _ = try integer(p["source_schema"], 1, 1)
        let manifest = try hex(p["manifest_digest"])
        let ref = try hex(p["asset_ref"], 32)
        let source = try hex(p["sha256"])
        guard let media = p["media_type"] as? String,
            matches(media, "^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$"),
            media.hasPrefix("image/") || media.hasPrefix("audio/") || ["application/pdf", "application/dicom", "application/dicom+json"].contains(media)
        else { throw MultimodalContractRefusal.invalid }
        func count(_ key: String) throws -> Int64? {
            if p[key] is NSNull { return nil }
            return try integer(p[key], 1, maxCoordinate)
        }
        let duration = p["duration_f64"] is NSNull ? nil : try number(p["duration_f64"])
        if let duration { try check(duration > 0) }
        return MultimodalAsset(
            manifestDigest: manifest, assetRef: ref, mediaType: media, sourceDigest: source,
            byteSize: try integer(p["byte_size"], 1), pages: try count("pages"), width: try count("width"),
            height: try count("height"), frames: try count("frames"), durationSeconds: duration)
    }

    static func rectangle(_ value: Any?, width: Double, height: Double) throws -> MultimodalRectangle {
        guard let values = value as? [Any], values.count == 4 else { throw MultimodalContractRefusal.invalid }
        let v = try values.map { try number($0) }
        try check(v[0] < v[2] && v[1] < v[3] && v[2] <= width && v[3] <= height)
        return MultimodalRectangle(left: v[0], top: v[1], right: v[2], bottom: v[3])
    }

    static func visual(_ p: [String: Any]) throws -> MultimodalVisual {
        _ = try object(p, "source_schema node_ref source_digest kind page width_f64 height_f64 unit box_f64 span rotation display_width_f64 display_height_f64 display_box_f64")
        try check(p["source_schema"] as? String == "openmed.multimodal.fused-evidence.v1", .schema)
        let node = try hex(p["node_ref"], 32)
        let source = try hex(p["source_digest"])
        guard let rawKind = p["kind"] as? String, let kind = MultimodalVisualKind(rawValue: rawKind) else {
            throw MultimodalContractRefusal.capability
        }
        guard let rawUnit = p["unit"] as? String, let unit = MultimodalSpatialUnit(rawValue: rawUnit) else {
            throw MultimodalContractRefusal.invalid
        }
        let width = try number(p["width_f64"])
        let height = try number(p["height_f64"])
        try check(width > 0 && height > 0)
        let box: MultimodalRectangle?
        if kind == .page {
            try check(p["box_f64"] is NSNull)
            box = nil
        } else {
            box = try rectangle(p["box_f64"], width: width, height: height)
        }
        let span: MultimodalInterval?
        if kind == .ocr {
            span = try interval(p["span"], maximum: maxCoordinate)
        } else {
            try check(p["span"] is NSNull)
            span = nil
        }
        guard let rotation = try? integer(p["rotation"]), [0, 90, 180, 270].contains(rotation) else {
            throw MultimodalContractRefusal.transform
        }
        let displayWidth = rotation == 90 || rotation == 270 ? height : width
        let displayHeight = rotation == 90 || rotation == 270 ? width : height
        try check(p["display_width_f64"] as? String == bits(displayWidth) && p["display_height_f64"] as? String == bits(displayHeight), .transform)
        var displayBox: MultimodalRectangle?
        if let box {
            let coordinates: [Double]
            switch rotation {
            case 0: coordinates = [box.left, box.top, box.right, box.bottom]
            case 90: coordinates = [height - box.bottom, box.left, height - box.top, box.right]
            case 180: coordinates = [width - box.right, height - box.bottom, width - box.left, height - box.top]
            default: coordinates = [box.top, width - box.right, box.bottom, width - box.left]
            }
            try check(coordinates[0] < coordinates[2] && coordinates[1] < coordinates[3], .transform)
            try check(p["display_box_f64"] as? [String] == coordinates.map(bits), .transform)
            displayBox = MultimodalRectangle(left: coordinates[0], top: coordinates[1], right: coordinates[2], bottom: coordinates[3])
        } else {
            try check(p["display_box_f64"] is NSNull, .transform)
        }
        return MultimodalVisual(
            nodeRef: node, sourceDigest: source, kind: kind, page: try integer(p["page"], 0, maxCoordinate),
            width: width, height: height, unit: unit, sourceBox: box, scalarSpan: span, rotation: rotation,
            displayWidth: displayWidth, displayHeight: displayHeight, displayBox: displayBox)
    }

    static func transcript(_ p: [String: Any]) throws -> MultimodalTranscript {
        _ = try object(p, "source_schema snapshot_digest session_ref segment_ref span_ref original_revision revision original_span token_refs length finalized invalidation current_span time_ms sample_rate_hz sample_bounds")
        try check(p["source_schema"] as? String == "openmed.transcript.revisions.v1", .schema)
        let snapshot = try hex(p["snapshot_digest"])
        let session = try hex(p["session_ref"], 32)
        let segment = try hex(p["segment_ref"], 32)
        let spanRef = try hex(p["span_ref"], 32)
        let revision = try integer(p["revision"], 0, 4096)
        let originalRevision = try integer(p["original_revision"], 0, revision)
        let originalSpan = try interval(p["original_span"], maximum: 65536)
        guard let rawRefs = p["token_refs"] as? [Any], !rawRefs.isEmpty, rawRefs.count <= 4096 else {
            throw MultimodalContractRefusal.invalid
        }
        let refs = try rawRefs.map { try hex($0, 32) }
        try check(Set(refs).count == refs.count)
        let length = try integer(p["length"], 0, 65536)
        let finalized = try boolean(p["finalized"])
        let rate = try integer(p["sample_rate_hz"], 1, 384000)
        let invalidation: MultimodalSpanInvalidation?
        let current: MultimodalInterval?
        let times: MultimodalInterval?
        let samples: MultimodalInterval?
        if p["invalidation"] is NSNull {
            invalidation = nil
            current = try interval(p["current_span"], maximum: length)
            let timing = try interval(p["time_ms"], maximum: 86_400_000, allowEmpty: true)
            let sampled = try interval(p["sample_bounds"], maximum: 86_400_000 * 384000 / 1000, allowEmpty: true)
            try check(sampled.start == timing.start * rate / 1000 && sampled.end == (timing.end * rate + 999) / 1000)
            times = timing
            samples = sampled
        } else {
            guard let reason = p["invalidation"] as? String, let parsed = MultimodalSpanInvalidation(rawValue: reason) else {
                throw MultimodalContractRefusal.invalid
            }
            try check(["current_span", "time_ms", "sample_bounds"].allSatisfy { p[$0] is NSNull })
            invalidation = parsed
            current = nil
            times = nil
            samples = nil
        }
        return MultimodalTranscript(
            snapshotDigest: snapshot, sessionRef: session, segmentRef: segment, spanRef: spanRef,
            originalRevision: originalRevision, revision: revision, originalScalarSpan: originalSpan, tokenRefs: refs,
            scalarLength: length, finalized: finalized, invalidation: invalidation, currentScalarSpan: current,
            relativeMilliseconds: times, sampleRateHz: rate, sampleBounds: samples)
    }

    static func identifier(_ value: Any?) throws -> String {
        guard let string = value as? String, string.utf8.count <= 128,
            matches(string, "^[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?$"),
            !string.split(whereSeparator: { ".-_".contains($0) }).contains(where: {
                ["bearer", "credential", "mrn", "password", "patient", "prompt", "secret", "token"].contains(String($0))
            })
        else { throw MultimodalContractRefusal.invalid }
        return string
    }

    static func provider(_ p: [String: Any]) throws -> MultimodalProviderResult {
        _ = try object(p, "schema_version provider_id model_id input_digest output_digest outcome abstention_code duration_f64 count_metadata")
        try check(p["schema_version"] as? String == "openmed.multimodal.provider_result.v1", .schema)
        let providerID = try identifier(p["provider_id"])
        let modelID = try identifier(p["model_id"])
        let input = try hex(p["input_digest"])
        guard let rawOutcome = p["outcome"] as? String, let outcome = MultimodalProviderOutcome(rawValue: rawOutcome) else {
            throw MultimodalContractRefusal.invalid
        }
        let output: String?
        if outcome == .success {
            output = try hex(p["output_digest"])
        } else {
            try check(p["output_digest"] is NSNull)
            output = nil
        }
        let abstention: MultimodalAbstention?
        if outcome == .abstention {
            guard let raw = p["abstention_code"] as? String, let parsed = MultimodalAbstention(rawValue: raw) else {
                throw MultimodalContractRefusal.invalid
            }
            abstention = parsed
        } else {
            try check(p["abstention_code"] is NSNull)
            abstention = nil
        }
        guard let rawCounts = p["count_metadata"] as? [String: Any] else { throw MultimodalContractRefusal.invalid }
        let allowed = Set(["detection_count", "frame_count", "input_bytes", "input_items", "output_items", "page_count", "sample_count", "segment_count", "token_count"])
        try check(Set(rawCounts.keys).isSubset(of: allowed))
        return MultimodalProviderResult(
            providerID: providerID, modelID: modelID, inputDigest: input, outputDigest: output,
            outcome: outcome, abstention: abstention, durationMilliseconds: try number(p["duration_f64"], 86_400_000),
            counts: try rawCounts.mapValues { try integer($0) })
    }

    static func clock(_ p: [String: Any]) throws -> MultimodalClockAlignment {
        _ = try object(p, "source_schema source_result_digest state reasons source_reference_digest target_reference_digest source_clock_digest target_clock_digest policy_digest target_start_ns target_end_ns guaranteed_overlap_ns uncertainty_ns")
        try check(p["source_schema"] as? String == clockSchema, .schema)
        let sourceResult = try hex(p["source_result_digest"])
        let sourceRef = try hex(p["source_reference_digest"])
        let targetRef = try hex(p["target_reference_digest"])
        let sourceClock = try hex(p["source_clock_digest"])
        let targetClock = try hex(p["target_clock_digest"])
        let policy = try hex(p["policy_digest"])
        guard let rawReasons = p["reasons"] as? [String], rawReasons.count <= MultimodalClockReason.allCases.count,
            Set(rawReasons).count == rawReasons.count, let rawState = p["state"] as? String,
            let state = MultimodalClockState(rawValue: rawState)
        else { throw MultimodalContractRefusal.invalid }
        let reasons = try rawReasons.map { raw in
            guard let reason = MultimodalClockReason(rawValue: raw) else { throw MultimodalContractRefusal.invalid }
            return reason
        }
        let start: MultimodalNanosecondBound?
        let end: MultimodalNanosecondBound?
        let overlap: MultimodalInterval?
        let uncertainty: Int64?
        if state == .review {
            try check(!reasons.isEmpty && ["target_start_ns", "target_end_ns", "guaranteed_overlap_ns", "uncertainty_ns"].allSatisfy { p[$0] is NSNull })
            start = nil
            end = nil
            overlap = nil
            uncertainty = nil
        } else {
            try check(reasons.isEmpty)
            func bound(_ key: String) throws -> MultimodalNanosecondBound {
                let v = try object(p[key], "lower upper")
                let lower = try integer(v["lower"], -clockLimit, clockLimit)
                return MultimodalNanosecondBound(lower: lower, upper: try integer(v["upper"], lower, clockLimit))
            }
            let first = try bound("target_start_ns")
            let last = try bound("target_end_ns")
            guard let rawOverlap = p["guaranteed_overlap_ns"] as? [Any], rawOverlap.count == 2 else {
                throw MultimodalContractRefusal.invalid
            }
            let a = try integer(rawOverlap[0], first.upper, last.lower)
            try check(a < Int64.max)
            let b = try integer(rawOverlap[1], a + 1, last.lower)
            let fw = first.upper.subtractingReportingOverflow(first.lower)
            let lw = last.upper.subtractingReportingOverflow(last.lower)
            try check(!fw.overflow && !lw.overflow)
            uncertainty = try integer(p["uncertainty_ns"], max(fw.partialValue, lw.partialValue), clockLimit)
            start = first
            end = last
            overlap = MultimodalInterval(start: a, end: b)
        }
        var original = p
        original.removeValue(forKey: "source_schema")
        original.removeValue(forKey: "source_result_digest")
        try check(try hash(["algorithm": clockSchema, "result": original]) == sourceResult, .provenance)
        return MultimodalClockAlignment(
            sourceResultDigest: sourceResult, state: state, reasons: reasons,
            sourceReferenceDigest: sourceRef, targetReferenceDigest: targetRef, sourceClockDigest: sourceClock,
            targetClockDigest: targetClock, policyDigest: policy, targetStart: start, targetEnd: end,
            guaranteedOverlap: overlap, uncertaintyNanoseconds: uncertainty)
    }
}
