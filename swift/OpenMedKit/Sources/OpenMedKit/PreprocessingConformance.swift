import CryptoKit
import Foundation

/// Controlled errors never include source pixels, paths or runtime error text.
public enum PreprocessingConformanceError: String, Error {
    case versionUnsupported = "version_unsupported"
    case dimensionsInvalid = "dimensions_invalid"
    case cropInvalid = "crop_invalid"
    case numericUnsupported = "numeric_behavior_unsupported"
    case operatorUnsupported = "operator_unsupported"
    case sourceInvalid = "source_invalid"
    case profileMismatch = "profile_mismatch"
    case coordinateMismatch = "coordinate_mismatch"
    case tensorMismatch = "tensor_mismatch"
    case runtimeDigestInvalid = "runtime_digest_invalid"
    case adapterFailed = "adapter_failed"
    case qualificationStale = "qualification_stale"
    case evidenceStale = "evidence_stale"
}

private func preprocessingHash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func preprocessingSHA(_ value: String) -> Bool {
    value.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

private func digestBytes(_ value: String) -> Data {
    var result = Data()
    let bytes = Array(value.utf8)
    for index in stride(from: 0, to: bytes.count, by: 2) {
        result.append(UInt8(String(bytes: bytes[index...index + 1], encoding: .utf8)!, radix: 16)!)
    }
    return result
}

extension Data {
    fileprivate mutating func appendInteger<T: FixedWidthInteger>(_ value: T) {
        var little = value.littleEndian
        Swift.withUnsafeBytes(of: &little) { append(contentsOf: $0) }
    }
}

/// Bounded executable v1 profile shared with Python; no model or decoder is included.
/// Execution order is alpha composition, EXIF orientation, crop, half-pixel resize,
/// channel order, scale/mean/std, tensor layout, then final float rounding.
public struct PreprocessingProfile: Sendable {
    public let version: Int
    public let sourceSize: [Int]
    public let crop: [Int]
    public let targetSize: [Int]
    public let orientation: Int
    public let interpolation: String
    public let colorOrder: String
    public let background: [Int]
    public let scale: Double
    public let mean: [Double]
    public let std: [Double]
    public let layout: String
    public let dtype: String
    public let atol: Double

    /// Create an explicit profile. Only RGB/RGBA sRGB bytes, nearest/bilinear,
    /// HWC/CHW and float32/float64 are supported. No gamma or antialiasing step.
    public init(
        sourceSize: [Int], crop: [Int], targetSize: [Int], orientation: Int = 1,
        interpolation: String = "nearest", colorOrder: String = "RGB",
        background: [Int] = [0, 0, 0], scale: Double = 1.0 / 255,
        mean: [Double] = [0, 0, 0], std: [Double] = [1, 1, 1],
        layout: String = "HWC", dtype: String = "float32", atol: Double = 1e-6, version: Int = 1
    ) throws {
        guard version == 1 else { throw PreprocessingConformanceError.versionUnsupported }
        for size in [sourceSize, targetSize] {
            guard size.count == 2, size.allSatisfy({ (1...64).contains($0) }) else {
                throw PreprocessingConformanceError.dimensionsInvalid
            }
        }
        guard (1...8).contains(orientation), ["nearest", "bilinear"].contains(interpolation),
            ["RGB", "BGR"].contains(colorOrder), ["HWC", "CHW"].contains(layout),
            ["float32", "float64"].contains(dtype)
        else { throw PreprocessingConformanceError.operatorUnsupported }
        let size = (5...8).contains(orientation) ? sourceSize.reversed().map { $0 } : sourceSize
        guard crop.count == 4, crop.allSatisfy({ (0...64).contains($0) }),
            crop[0] < crop[2], crop[1] < crop[3], crop[2] <= size[0], crop[3] <= size[1]
        else { throw PreprocessingConformanceError.cropInvalid }
        guard background.count == 3, background.allSatisfy({ (0...255).contains($0) }),
            mean.count == 3, std.count == 3,
            mean.allSatisfy({ $0.isFinite && abs($0) <= 1e6 }),
            std.allSatisfy({ $0.isFinite && $0 >= 1e-6 && $0 <= 1e6 }),
            scale.isFinite, scale > 0, scale <= 1e6, atol.isFinite, atol >= 0, atol <= 1e-5
        else { throw PreprocessingConformanceError.numericUnsupported }
        self.version = version
        self.sourceSize = sourceSize
        self.crop = crop
        self.targetSize = targetSize
        self.orientation = orientation
        self.interpolation = interpolation
        self.colorOrder = colorOrder
        self.background = background
        self.scale = scale
        self.mean = mean
        self.std = std
        self.layout = layout
        self.dtype = dtype
        self.atol = atol
    }

    /// Exact oriented width and height.
    public var orientedSize: [Int] {
        (5...8).contains(orientation) ? sourceSize.reversed().map { $0 } : sourceSize
    }

    /// SHA-256 of the v1 binary operator identity, identical to Python.
    public var digest: String {
        var data = Data("openmed.preprocessing.v1\0".utf8)
        for value in [version] + sourceSize + crop + targetSize + [orientation] + background {
            data.appendInteger(UInt32(value))
        }
        data.append(Data(([interpolation, colorOrder, layout, dtype].joined(separator: "\0") + "\0").utf8))
        for value in [scale] + mean + std + [atol] { data.appendInteger(value.bitPattern) }
        return preprocessingHash(data)
    }
}

/// Transient runtime tensor and exact rational geometry; never serialize pixels to audit logs.
public struct PreprocessingOutput: Sendable, CustomStringConvertible {
    public let profileDigest: String
    public let shape: [Int]
    public let tensor: [Double]
    public let geometry: [String]
    public let protectedMask: [Bool]
    public let dtype: String

    /// Construct an adapter result from actual runtime values, not declared metadata alone.
    public init(profileDigest: String, shape: [Int], tensor: [Double], geometry: [String], protectedMask: [Bool], dtype: String) throws {
        guard preprocessingSHA(profileDigest), shape.count == 3,
            shape.allSatisfy({ (1...64).contains($0) }), shape.reduce(1, *) == tensor.count,
            tensor.count <= 4096 * 3, tensor.allSatisfy({ $0.isFinite }),
            geometry.count == 6,
            geometry.allSatisfy({ value in
                !value.isEmpty && value.count <= 32 && value.utf8.allSatisfy { (48...57).contains($0) || $0 == 45 || $0 == 47 }
            }), ["float32", "float64"].contains(dtype), protectedMask.count * 3 == tensor.count,
            dtype != "float32" || tensor.allSatisfy({ Float($0).isFinite && Double(Float($0)) == $0 })
        else { throw PreprocessingConformanceError.sourceInvalid }
        self.profileDigest = profileDigest
        self.shape = shape
        self.tensor = tensor
        self.geometry = geometry
        self.protectedMask = protectedMask
        self.dtype = dtype
    }

    /// Content-free printable description.
    public var description: String { "PreprocessingOutput(profileDigest: \(profileDigest))" }

    /// Hash actual tensor bytes and profile; tolerated rounding still has a distinct identity.
    public var tensorDigest: String {
        var data = digestBytes(profileDigest)
        for value in shape { data.appendInteger(UInt32(value)) }
        data.append(Data((dtype + "\0").utf8))
        for value in tensor {
            if dtype == "float32" { data.appendInteger(Float(value).bitPattern) } else { data.appendInteger(value.bitPattern) }
        }
        return preprocessingHash(data)
    }

    /// Hash profile-bound exact geometry and conservative protected coverage.
    public var coordinateDigest: String {
        var data = Data((profileDigest + "|" + geometry.joined(separator: "|") + "\0").utf8)
        data.append(contentsOf: protectedMask.map { $0 ? UInt8(1) : UInt8(0) })
        return preprocessingHash(data)
    }
}

/// Synthetic-only conformance binding; caller supplies a digest of the runtime build/configuration.
public struct PreprocessingQualification: Sendable {
    public let profileDigest: String
    public let runtimeDigest: String
    public let vectorsDigest: String
    public let notice = "synthetic_preprocessing_only_non_diagnostic"
    public let reviewerConfirmationRequired = true
    public var digest: String {
        preprocessingHash(Data((profileDigest + runtimeDigest + vectorsDigest).utf8))
    }
}

/// Content-free downstream evidence requires separate explicit reviewer confirmation.
public struct PreprocessingEvidence: Sendable {
    public let qualificationDigest: String
    public let tensorDigest: String
    public let coordinateDigest: String
    public let notice = "synthetic_preprocessing_only_non_diagnostic"
    public let reviewerConfirmationRequired = true

    /// Reject evidence after any profile, runtime, tensor or coordinate change.
    public func requireCurrent(profile: PreprocessingProfile, output: PreprocessingOutput, qualification: PreprocessingQualification, runtimeDigest: String) throws {
        try PreprocessingConformance.binding(profile, qualification, runtimeDigest)
        guard qualificationDigest == qualification.digest, tensorDigest == output.tensorDigest,
            coordinateDigest == output.coordinateDigest, output.profileDigest == profile.digest
        else { throw PreprocessingConformanceError.evidenceStale }
    }
}

/// Deterministic local operator execution and optional runtime conformance seam.
/// No diagnostic interpretation, model qualification, persistence or network transport.
public enum PreprocessingConformance {
    public typealias Adapter = (PreprocessingProfile, [[Int]], [Bool]) throws -> PreprocessingOutput

    private static func rational(_ n: Int, _ d: Int) -> String {
        var a = abs(n)
        var b = d
        while b != 0 { (a, b) = (b, a % b) }
        let numerator = n / a
        let denominator = d / a
        return denominator == 1 ? String(numerator) : "\(numerator)/\(denominator)"
    }

    private static func geometry(_ p: PreprocessingProfile) -> [String] {
        let w = p.sourceSize[0]
        let h = p.sourceSize[1]
        let matrices = [
            [1, 0, 0, 0, 1, 0], [-1, 0, w, 0, 1, 0], [-1, 0, w, 0, -1, h],
            [1, 0, 0, 0, -1, h], [0, 1, 0, 1, 0, 0], [0, -1, h, 1, 0, 0],
            [0, -1, h, -1, 0, w], [0, 1, 0, -1, 0, w],
        ]
        let m = matrices[p.orientation - 1]
        let cw = p.crop[2] - p.crop[0]
        let ch = p.crop[3] - p.crop[1]
        return [
            rational(m[0] * p.targetSize[0], cw), rational(m[1] * p.targetSize[0], cw),
            rational((m[2] - p.crop[0]) * p.targetSize[0], cw), rational(m[3] * p.targetSize[1], ch),
            rational(m[4] * p.targetSize[1], ch), rational((m[5] - p.crop[1]) * p.targetSize[1], ch),
        ]
    }

    private static func samples(_ index: Int, _ source: Int, _ target: Int, _ mode: String) -> [(Int, Double)] {
        let denominator = 2 * target
        let numerator = max(0, min((source - 1) * denominator, (2 * index + 1) * source - target))
        if mode == "nearest" { return [((numerator + target) / denominator, 1)] }
        let low = numerator / denominator
        let remainder = numerator % denominator
        var result = [(low, Double(denominator - remainder) / Double(denominator))]
        if remainder > 0 { result.append((min(low + 1, source - 1), Double(remainder) / Double(denominator))) }
        return result
    }

    /// Execute on row-major decoded RGB/RGBA sRGB byte arrays and a source mask.
    public static func execute(_ p: PreprocessingProfile, pixels: [[Int]], protectedMask: [Bool]) throws -> PreprocessingOutput {
        let w = p.sourceSize[0]
        let h = p.sourceSize[1]
        guard pixels.count == w * h, protectedMask.count == pixels.count,
            let channels = pixels.first?.count, [3, 4].contains(channels),
            pixels.allSatisfy({ $0.count == channels && $0.allSatisfy { (0...255).contains($0) } })
        else { throw PreprocessingConformanceError.sourceInvalid }
        let ow = p.orientedSize[0]
        var oriented = Array(repeating: [Double](repeating: 0, count: 3), count: pixels.count)
        var masks = Array(repeating: false, count: pixels.count)
        for (i, pixel) in pixels.enumerated() {
            let x = i % w
            let y = i / w
            let positions = [
                (x, y), (w - 1 - x, y), (w - 1 - x, h - 1 - y), (x, h - 1 - y),
                (y, x), (h - 1 - y, x), (h - 1 - y, w - 1 - x), (y, w - 1 - x),
            ]
            let (ox, oy) = positions[p.orientation - 1]
            let alpha = channels == 4 ? Double(pixel[3]) / 255 : 1
            oriented[oy * ow + ox] = (0..<3).map { Double(pixel[$0]) * alpha + Double(p.background[$0]) * (1 - alpha) }
            masks[oy * ow + ox] = protectedMask[i]
        }
        let tw = p.targetSize[0]
        let th = p.targetSize[1]
        var tensor = [Double]()
        var coverage = [Bool]()
        let order = p.colorOrder == "RGB" ? [0, 1, 2] : [2, 1, 0]
        for y in 0..<th {
            for x in 0..<tw {
                var values = [Double](repeating: 0, count: 3)
                var protected = false
                for (iy, wy) in samples(y, p.crop[3] - p.crop[1], th, p.interpolation) {
                    for (ix, wx) in samples(x, p.crop[2] - p.crop[0], tw, p.interpolation) {
                        let index = (iy + p.crop[1]) * ow + ix + p.crop[0]
                        for c in 0..<3 { values[c] += oriented[index][c] * wx * wy }
                        protected = protected || masks[index]
                    }
                }
                tensor.append(contentsOf: order.enumerated().map { (k, c) in (values[c] * p.scale - p.mean[k]) / p.std[k] })
                coverage.append(protected)
            }
        }
        if p.layout == "CHW" { tensor = (0..<3).flatMap { c in (0..<tw * th).map { tensor[$0 * 3 + c] } } }
        if p.dtype == "float32" { tensor = tensor.map { Double(Float($0)) } }
        return try PreprocessingOutput(
            profileDigest: p.digest,
            shape: p.layout == "HWC" ? [th, tw, 3] : [3, th, tw], tensor: tensor,
            geometry: geometry(p), protectedMask: coverage, dtype: p.dtype)
    }

    private static func compare(_ p: PreprocessingProfile, _ actual: PreprocessingOutput, _ expected: PreprocessingOutput) throws {
        guard actual.profileDigest == p.digest, actual.dtype == p.dtype else { throw PreprocessingConformanceError.profileMismatch }
        guard actual.shape == expected.shape, actual.geometry == expected.geometry,
            actual.protectedMask == expected.protectedMask
        else { throw PreprocessingConformanceError.coordinateMismatch }
        guard actual.tensor.count == expected.tensor.count,
            zip(actual.tensor, expected.tensor).allSatisfy({ abs($0 - $1) <= p.atol })
        else { throw PreprocessingConformanceError.tensorMismatch }
    }

    fileprivate static func binding(_ p: PreprocessingProfile, _ q: PreprocessingQualification, _ runtime: String) throws {
        guard preprocessingSHA(runtime), q.profileDigest == p.digest, q.runtimeDigest == runtime else { throw PreprocessingConformanceError.qualificationStale }
    }

    /// Qualify an optional local adapter using deterministic RGB and RGBA grids.
    /// Re-run after a runtime build change; this is no clinical/provider attestation.
    public static func qualify(_ p: PreprocessingProfile, runtimeDigest: String, adapter: Adapter) throws -> PreprocessingQualification {
        guard preprocessingSHA(runtimeDigest) else { throw PreprocessingConformanceError.runtimeDigestInvalid }
        let count = p.sourceSize[0] * p.sourceSize[1]
        var identity = Data("openmed.preprocessing.vectors.v1\0".utf8)
        for channels in [3, 4] {
            let pixels = (0..<count).map { i in (0..<3).map { (i * 37 + $0 * 71) % 256 } + (channels == 4 ? [[0, 128, 255][i % 3]] : []) }
            let mask = (0..<count).map { $0 % 3 == 0 }
            let expected = try execute(p, pixels: pixels, protectedMask: mask)
            let actual: PreprocessingOutput
            do { actual = try adapter(p, pixels, mask) } catch { throw PreprocessingConformanceError.adapterFailed }
            try compare(p, actual, expected)
            identity.append(contentsOf: pixels.flatMap { $0.map { UInt8($0) } })
            identity.append(contentsOf: mask.map { $0 ? UInt8(1) : UInt8(0) })
            identity.append(digestBytes(actual.tensorDigest))
            identity.append(digestBytes(actual.coordinateDigest))
        }
        return PreprocessingQualification(profileDigest: p.digest, runtimeDigest: runtimeDigest, vectorsDigest: preprocessingHash(identity))
    }

    /// Compare an actual output with the executable reference before binding evidence.
    /// A consequential consumer must separately request explicit reviewer confirmation.
    public static func verify(_ p: PreprocessingProfile, pixels: [[Int]], protectedMask: [Bool], output: PreprocessingOutput, qualification: PreprocessingQualification, runtimeDigest: String) throws -> PreprocessingEvidence {
        try binding(p, qualification, runtimeDigest)
        try compare(p, output, execute(p, pixels: pixels, protectedMask: protectedMask))
        return PreprocessingEvidence(qualificationDigest: qualification.digest, tensorDigest: output.tensorDigest, coordinateDigest: output.coordinateDigest)
    }
}
