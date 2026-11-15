import Foundation

/// Explicit metadata layouts, without a vendor compatibility or decoding claim.
public enum BigTiffProfile: String, Codable, Sendable {
    case ifdChain = "bigtiff-ifd-chain-v1"
    case subIFDs = "bigtiff-subifds-v1"
}

/// Resource ceilings for metadata-only traversal. No allocation depends on pixels.
public struct BigTiffLimits: Sendable {
    public var maxHeaderBytes: Int = 65_536
    public var maxPixels: UInt64 = 100_000_000
    public var maxIFDEntries: Int = 512
    public var maxDirectories: Int = 32
    public var maxLevels: Int = 32
    public var maxTiles: Int = 1_000_000
    public var maxOffset: UInt64 = UInt64(Int64.max)

    public init() {}
}

/// Value-free failure; identifying metadata and paths are never retained.
public struct BigTiffPreflightError: Error, LocalizedError, CustomStringConvertible {
    public let category: String
    public var description: String { category }
    public var errorDescription: String? { category }
    fileprivate init(_ code: String) { category = "bigtiff_" + code }
}

/// Declared tile geometry and compression category, without image payloads.
public struct BigTiffLevel: Codable, Equatable, Sendable {
    public let index: Int
    public let width: UInt64
    public let height: UInt64
    public let tileWidth: UInt64
    public let tileHeight: UInt64
    public let columns: UInt64
    public let rows: UInt64
    public let tileCount: Int
    public let compression: String

    enum CodingKeys: String, CodingKey {
        case index, width, height, columns, rows, compression
        case tileWidth = "tile_width"
        case tileHeight = "tile_height"
        case tileCount = "tile_count"
    }
}

/// A deterministic geometry manifest. Acceptance is not clinical or decoder approval.
public struct BigTiffPyramid: Codable, Equatable, Sendable {
    public let profile: BigTiffProfile
    public let byteOrder: String
    public let levels: [BigTiffLevel]
    public let totalTiles: Int

    enum CodingKeys: String, CodingKey {
        case profile, levels
        case byteOrder = "byte_order"
        case totalTiles = "total_tiles"
    }
}

/// Opt-in local BigTIFF metadata preflight, without pixel decoding or clinical action.
public enum BigTiffPreflight {
    /// Read caller-owned bytes, copying at most the configured metadata prefix.
    /// - Parameters:
    ///   - source: Complete caller-owned file bytes. No file is opened by this API.
    ///   - profile: A declared supported layout, not automatically detected.
    ///   - limits: Positive ceilings checked before count-based allocation.
    public static func read(_ source: Data, profile: BigTiffProfile, limits: BigTiffLimits = BigTiffLimits()) throws -> BigTiffPyramid {
        try read(prefix: source, sourceSize: UInt64(source.count), profile: profile, limits: limits)
    }

    /// Read an already bounded prefix with a caller-verified complete file size.
    /// The caller may load only metadata from a local slide; tile payloads are never read.
    /// - Parameters:
    ///   - prefix: File prefix beginning at byte zero; identifiers are ignored.
    ///   - sourceSize: Verified full byte length, bounding all tile ranges.
    ///   - profile: Explicit supported layout.
    ///   - limits: Resource ceilings. Metadata outside the prefix fails closed.
    public static func read(prefix: Data, sourceSize: UInt64, profile: BigTiffProfile, limits: BigTiffLimits = BigTiffLimits()) throws -> BigTiffPyramid {
        guard limits.maxHeaderBytes > 0, limits.maxPixels > 0, limits.maxIFDEntries > 0,
            limits.maxDirectories > 0, limits.maxLevels > 0, limits.maxTiles > 0, limits.maxOffset > 0
        else { throw BigTiffPreflightError("limit_invalid") }
        guard sourceSize >= 16, UInt64(prefix.count) <= sourceSize else { throw BigTiffPreflightError("source_size_invalid") }
        var reader = BigTiffReader(bytes: Array(prefix.prefix(limits.maxHeaderBytes)), sourceSize: sourceSize, profile: profile, limits: limits)
        return try reader.parse()
    }
}

private struct BigTiffReader {
    let bytes: [UInt8]
    let sourceSize: UInt64
    let profile: BigTiffProfile
    let limits: BigTiffLimits
    var little = true
    var metadataRanges: [(UInt64, UInt64)] = [(0, 16)]
    var tileRanges: [(UInt64, UInt64)] = []
    var visited: Set<UInt64> = []
    var levels: [BigTiffLevel] = []
    var totalTiles = 0

    func checkedRange(_ offset: UInt64, _ size: UInt64, metadata: Bool = false) throws -> (UInt64, UInt64) {
        guard size <= UInt64.max - offset else { throw BigTiffPreflightError("offset_overflow") }
        let end = offset + size
        guard end <= limits.maxOffset else { throw BigTiffPreflightError("offset_limit_exceeded") }
        guard end <= sourceSize else { throw BigTiffPreflightError("range_out_of_source") }
        guard !metadata || end <= UInt64(bytes.count) else { throw BigTiffPreflightError("header_limit_or_truncated") }
        return (offset, end)
    }

    func uint(_ offset: UInt64, _ size: Int) throws -> UInt64 {
        _ = try checkedRange(offset, UInt64(size), metadata: true)
        var value: UInt64 = 0
        for i in 0..<size {
            let shift = little ? i * 8 : (size - 1 - i) * 8
            value |= UInt64(bytes[Int(offset) + i]) << shift
        }
        return value
    }

    mutating func parse() throws -> BigTiffPyramid {
        guard bytes.count >= 16 else { throw BigTiffPreflightError("header_limit_or_truncated") }
        if bytes[0] == 73 && bytes[1] == 73 { little = true } else if bytes[0] == 77 && bytes[1] == 77 { little = false } else { throw BigTiffPreflightError("byte_order_invalid") }
        guard try uint(2, 2) == 43, try uint(4, 2) == 8, try uint(6, 2) == 0 else { throw BigTiffPreflightError("header_unsupported") }
        var pending = [try uint(8, 8)]
        while !pending.isEmpty {
            let offset = pending.removeFirst()
            guard !visited.contains(offset) else { throw BigTiffPreflightError("directory_cycle") }
            guard visited.count < limits.maxDirectories else { throw BigTiffPreflightError("directory_limit_exceeded") }
            guard levels.count < limits.maxLevels else { throw BigTiffPreflightError("level_limit_exceeded") }
            visited.insert(offset)
            let (values, next) = try directory(offset)
            try addLevel(values)
            let children = values[330] ?? []
            if profile == .ifdChain {
                guard children.isEmpty else { throw BigTiffPreflightError("layout_unsupported") }
                if next != 0 { pending.append(next) }
            } else {
                guard next == 0, levels.count == 1 || children.isEmpty else { throw BigTiffPreflightError("layout_unsupported") }
                pending.append(contentsOf: children)
            }
            guard pending.count <= limits.maxLevels - levels.count else { throw BigTiffPreflightError("level_limit_exceeded") }
            guard pending.count <= limits.maxDirectories - visited.count else { throw BigTiffPreflightError("directory_limit_exceeded") }
        }
        let ranges = (metadataRanges + tileRanges).sorted { a, b in a.0 == b.0 ? a.1 < b.1 : a.0 < b.0 }
        for i in 1..<ranges.count {
            guard ranges[i - 1].1 <= ranges[i].0 else { throw BigTiffPreflightError("range_overlap") }
        }
        return BigTiffPyramid(profile: profile, byteOrder: little ? "little" : "big", levels: levels, totalTiles: totalTiles)
    }

    mutating func directory(_ offset: UInt64) throws -> ([UInt64: [UInt64]], UInt64) {
        guard offset >= 16, offset % 8 == 0 else { throw BigTiffPreflightError("directory_offset_invalid") }
        let count = try uint(offset, 8)
        guard count > 0, count <= UInt64(limits.maxIFDEntries) else { throw BigTiffPreflightError("entry_limit_exceeded") }
        guard count <= (UInt64.max - 16) / 20 else { throw BigTiffPreflightError("offset_overflow") }
        metadataRanges.append(try checkedRange(offset, 16 + count * 20, metadata: true))
        var values: [UInt64: [UInt64]] = [:]
        var seen: Set<UInt64> = []
        let scalars: Set<UInt64> = [254, 256, 257, 259, 277, 284, 322, 323]
        for i in 0..<count {
            let start = offset + 8 + i * 20
            let tag = try uint(start, 2)
            let kind = try uint(start + 2, 2)
            let length = try uint(start + 4, 8)
            guard seen.insert(tag).inserted else { throw BigTiffPreflightError("tag_duplicate") }
            guard ![UInt64(273), 278, 279, 32997, 32998].contains(tag), tag < 32768 else { throw BigTiffPreflightError("layout_unsupported") }
            guard scalars.contains(tag) || [UInt64(324), 325, 330].contains(tag) else { continue }
            guard [UInt64(3), 4, 16, 18].contains(kind), kind != 18 || tag == 330 else { throw BigTiffPreflightError("value_type_unsupported") }
            let size = kind == 3 ? 2 : kind == 4 ? 4 : 8
            guard length > 0 else { throw BigTiffPreflightError("value_count_invalid") }
            guard length <= UInt64.max / UInt64(size) else { throw BigTiffPreflightError("offset_overflow") }
            let ceiling = scalars.contains(tag) ? 1 : tag == 330 ? limits.maxLevels : limits.maxTiles
            guard length <= UInt64(ceiling) else { throw BigTiffPreflightError("value_count_limit_exceeded") }
            let byteCount = length * UInt64(size)
            var location = start + 12
            if byteCount > 8 {
                location = try uint(location, 8)
                guard location >= 16, location % 8 == 0 else { throw BigTiffPreflightError("value_offset_invalid") }
                metadataRanges.append(try checkedRange(location, byteCount, metadata: true))
            }
            var entry: [UInt64] = []
            for j in 0..<length { entry.append(try uint(location + j * UInt64(size), size)) }
            values[tag] = entry
        }
        return (values, try uint(offset + 8 + count * 20, 8))
    }

    mutating func addLevel(_ values: [UInt64: [UInt64]]) throws {
        guard let width = values[256]?.first, let height = values[257]?.first,
            let tileWidth = values[322]?.first, let tileHeight = values[323]?.first,
            width > 0, height > 0, tileWidth > 0, tileHeight > 0
        else { throw BigTiffPreflightError("geometry_invalid") }
        guard width <= limits.maxPixels / height else { throw BigTiffPreflightError("pixel_limit_exceeded") }
        let index = levels.count
        guard (values[254] ?? [0]) == [index == 0 ? 0 : 1], (values[284] ?? [1]) == [1],
            let samples = (values[277] ?? [1]).first, samples >= 1, samples <= 64
        else { throw BigTiffPreflightError("layout_unsupported") }
        if let previous = levels.last {
            guard width <= previous.width, height <= previous.height, width < previous.width || height < previous.height
            else { throw BigTiffPreflightError("pyramid_geometry_inconsistent") }
        }
        let categories: [UInt64: String] = [1: "uncompressed", 5: "lzw", 7: "jpeg", 8: "deflate", 32946: "deflate"]
        guard let compression = categories[(values[259] ?? [1])[0]] else { throw BigTiffPreflightError("compression_unsupported") }
        let columns = (width - 1) / tileWidth + 1
        let rows = (height - 1) / tileHeight + 1
        guard columns <= UInt64(limits.maxTiles - totalTiles) / rows else { throw BigTiffPreflightError("tile_limit_exceeded") }
        let count = Int(columns * rows)
        let offsets = values[324] ?? []
        let sizes = values[325] ?? []
        guard offsets.count == count, sizes.count == count else { throw BigTiffPreflightError("tile_table_inconsistent") }
        for (offset, size) in zip(offsets, sizes) {
            guard offset >= 16, size > 0 else { throw BigTiffPreflightError("tile_range_invalid") }
            tileRanges.append(try checkedRange(offset, size))
        }
        totalTiles += count
        levels.append(BigTiffLevel(index: index, width: width, height: height, tileWidth: tileWidth, tileHeight: tileHeight, columns: columns, rows: rows, tileCount: count, compression: compression))
    }
}
