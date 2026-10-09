import CryptoKit
import Foundation

/// Controlled, value-free slide refusal. No paths, descriptions or pixel values.
public struct SlideDeidError: Error, LocalizedError, Sendable {
    public let category: String
    public var errorDescription: String? { category }
}

/// Device budgets shared with the Python TIFF-family implementation.
public struct SlideDeidLimits: Sendable {
    public var maxFileBytes = 256 * 1024 * 1024
    public var maxDirectories = 64
    public var maxIFDEntries = 128
    public var maxValueBytes = 1024 * 1024
    public var maxBlocks = 1_000_000
    public init() {}
}

/// Roles, counts and digests only; no original TIFF descriptions or identifiers.
public struct SlideDirectoryReport: Codable, Equatable, Sendable {
    public let role: String
    public let blockCount: Int
    public let blockSHA256: [String]
    enum CodingKeys: String, CodingKey {
        case role
        case blockCount = "block_count"
        case blockSHA256 = "block_sha256"
    }
}

/// Protected output. Structural de-identification does not certify pixel privacy.
public struct SlideDeidResult: Sendable, CustomStringConvertible {
    public let data: Data
    public let outcome: String
    public let directories: [SlideDirectoryReport]
    public let outputSHA256: String
    public var description: String { "SlideDeidResult(\(outcome), directories: \(directories.count))" }
    public func auditJSON() throws -> Data {
        let rows: [[String: Any]] = directories.map {
            ["role": $0.role, "block_count": $0.blockCount, "block_sha256": $0.blockSHA256]
        }
        return try JSONSerialization.data(
            withJSONObject: [
                "outcome": outcome, "directory_count": directories.count,
                "removed_directory_count": directories.filter { $0.role != "level" }.count,
                "output_sha256": outputSHA256, "directories": rows,
            ], options: [.sortedKeys])
    }
}

/// Content-bound evidence from an application's local DICOM pixel redaction.
/// Applications must obtain these measurements from their actual offline provider.
public struct SlidePixelRedactionEvidence: Sendable {
    public let pixelSHA256: String
    public let framesProcessed: Int
    public let residualVerified: Bool
    public let residualEntityCount: Int
    public init(pixelSHA256: String, framesProcessed: Int, residualVerified: Bool, residualEntityCount: Int) {
        self.pixelSHA256 = pixelSHA256
        self.framesProcessed = framesProcessed
        self.residualVerified = residualVerified
        self.residualEntityCount = residualEntityCount
    }
}

/// On-device structural slide handling; no inference, cloud fallback or diagnosis.
public enum SlideDeidentification {
    /// Copy only approved numeric imaging tags and unchanged pyramid tile bytes.
    /// Thumbnail, label and macro images and all orphan bytes are dropped.
    /// Callers review and redact pyramid pixels before consequential release.
    public static func deidentifyTIFF(_ source: Data, limits: SlideDeidLimits = SlideDeidLimits()) throws -> SlideDeidResult {
        let reader = try SlideReader(source, limits: limits)
        let directories = try reader.read()
        let output = try reader.write(directories)
        let reports = directories.map {
            SlideDirectoryReport(role: $0.role, blockCount: $0.blocks.count, blockSHA256: $0.blocks.map(hash))
        }
        return SlideDeidResult(
            data: output,
            outcome: directories.contains { $0.role != "level" } ? "deidentified" : "no_associated_images",
            directories: reports, outputSHA256: hash(output))
    }

    /// Refuse WSI LABEL/OVERVIEW unless explicitly retained after local pixel
    /// redaction, verified residual checking and reviewer confirmation. Native
    /// applications call this before writing any DICOM copy; it is not a decoder.
    public static func validateDICOMInstance(
        sopClassUID: String, imageType: [String], specimenLabelInImage: String = "NO",
        keepAssociatedImages: Bool = false, reviewerConfirmed: Bool = false,
        pixels: Data? = nil, evidence: SlidePixelRedactionEvidence? = nil
    ) throws -> String? {
        guard sopClassUID == "1.2.840.10008.5.1.4.1.1.77.1.6" else { return nil }
        guard imageType.count >= 3,
            ["VOLUME", "THUMBNAIL", "LABEL", "OVERVIEW"].contains(imageType[2])
        else { throw SlideDeidError(category: "dicom_slide_image_type_unknown") }
        let role = imageType[2].lowercased()
        if role == "volume" || role == "thumbnail" {
            guard specimenLabelInImage == "NO" else { throw SlideDeidError(category: "dicom_slide_image_type_unknown") }
            return role
        }
        guard keepAssociatedImages, reviewerConfirmed, let pixels, let evidence,
            evidence.framesProcessed > 0, evidence.residualVerified,
            evidence.residualEntityCount == 0, evidence.pixelSHA256 == hash(pixels)
        else { throw SlideDeidError(category: "dicom_slide_associated_image_refused") }
        return role
    }

    fileprivate static func hash(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

private struct SlideEntry {
    let kind: Int
    let count: Int
    let raw: Data
}
private struct SlideDirectory {
    var entries: [Int: SlideEntry]
    let blocks: [Data]
    var role = ""
}

private final class SlideReader {
    let data: Data
    let limits: SlideDeidLimits
    let little: Bool
    let big: Bool
    let word: Int
    let countSize: Int
    let entrySize: Int
    let headerSize: Int
    var ranges: [Range<Int>] = []
    var blockCount = 0
    var claimedBytes = 0
    let technical: Set<Int> = [254, 256, 257, 258, 259, 262, 266, 273, 274, 277, 278, 279, 282, 283, 284, 296, 317, 322, 323, 324, 325, 338, 339, 347, 529, 530, 531, 532]
    let sizes = [1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 7: 1, 16: 8]

    init(_ source: Data, limits: SlideDeidLimits) throws {
        guard [limits.maxFileBytes, limits.maxDirectories, limits.maxIFDEntries, limits.maxValueBytes, limits.maxBlocks].allSatisfy({ $0 > 0 }) else { throw SlideDeidError(category: "slide_limits_invalid") }
        guard source.count <= limits.maxFileBytes else { throw SlideDeidError(category: "slide_file_limit") }
        guard source.count >= 4 else { throw SlideDeidError(category: "slide_range_invalid") }
        let marker = Array(source.prefix(2))
        guard marker == [73, 73] || marker == [77, 77] else { throw SlideDeidError(category: "slide_byte_order_invalid") }
        little = marker == [73, 73]
        let magic = little ? Int(source[2]) + Int(source[3]) * 256 : Int(source[3]) + Int(source[2]) * 256
        guard magic == 42 || magic == 43 else { throw SlideDeidError(category: "slide_magic_invalid") }
        big = magic == 43
        word = big ? 8 : 4
        countSize = big ? 8 : 2
        entrySize = big ? 20 : 12
        headerSize = big ? 16 : 8
        data = Data(source)
        self.limits = limits
        if big {
            let offsetSize = try number(4, 2)
            let reserved = try number(6, 2)
            if offsetSize != 8 || reserved != 0 { try fail("header_invalid") }
        }
        _ = try claim(0, headerSize)
    }
    func fail(_ code: String) throws -> Never { throw SlideDeidError(category: "slide_" + code) }
    func slice(_ at: Int, _ size: Int) throws -> Data {
        guard at >= 0, size > 0, size <= data.count, at <= data.count - size else { try fail("range_invalid") }
        return data.subdata(in: at..<at + size)
    }
    func claim(_ at: Int, _ size: Int) throws -> Data {
        guard size > 0, size <= data.count - claimedBytes else { try fail("range_overlap") }
        let raw = try slice(at, size)
        claimedBytes += size
        ranges.append(at..<at + size)
        return raw
    }
    func number(_ at: Int, _ size: Int) throws -> Int {
        let bytes = try slice(at, size)
        var value: UInt64 = 0
        for byte in little ? Array(bytes.reversed()) : Array(bytes) { value = (value << 8) | UInt64(byte) }
        guard value <= UInt64(Int.max) else { try fail("range_invalid") }
        return Int(value)
    }
    func numbers(_ e: SlideEntry) throws -> [Int] {
        guard [1, 3, 4, 16].contains(e.kind), let size = sizes[e.kind] else { try fail("type_unsupported") }
        var result: [Int] = []
        for offset in stride(from: 0, to: e.raw.count, by: size) {
            var value: UInt64 = 0
            let bytes = Array(e.raw[offset..<offset + size])
            for byte in little ? bytes.reversed().map({ $0 }) : bytes { value = (value << 8) | UInt64(byte) }
            guard value <= UInt64(Int.max) else { try fail("range_invalid") }
            result.append(Int(value))
        }
        return result
    }
    func scalar(_ entries: [Int: SlideEntry], _ tag: Int, _ fallback: Int? = nil) throws -> Int {
        guard let entry = entries[tag] else {
            guard let fallback else { try fail("tag_missing") }
            return fallback
        }
        let values = try numbers(entry)
        guard values.count == 1 else { try fail("count_invalid") }
        return values[0]
    }
    func read() throws -> [SlideDirectory] {
        var offset = try number(big ? 8 : 4, word)
        var directories: [SlideDirectory] = []
        var seen = Set<Int>()
        while offset != 0 {
            guard seen.insert(offset).inserted else { try fail("directory_cycle") }
            guard directories.count < limits.maxDirectories else { try fail("directory_limit") }
            let count = try number(offset, countSize)
            guard count > 0, count <= limits.maxIFDEntries,
                count <= (data.count - countSize - word) / entrySize
            else { try fail("entry_limit") }
            _ = try claim(offset, countSize + count * entrySize + word)
            var entries: [Int: SlideEntry] = [:]
            var tags = Set<Int>()
            for index in 0..<count {
                let start = offset + countSize + index * entrySize
                let tag = try number(start, 2)
                let kind = try number(start + 2, 2)
                guard tags.insert(tag).inserted else { try fail("tag_duplicate") }
                if [330, 34665, 34853].contains(tag) { try fail("nested_directory_unsupported") }
                guard technical.contains(tag) || tag == 270 else { continue }
                let n = try number(start + 4, word)
                guard let size = sizes[kind], kind != 16 || big else { try fail("type_unsupported") }
                guard n > 0, n <= limits.maxValueBytes / size else { try fail("value_limit") }
                let at = start + 4 + word
                let raw = try n * size <= word ? slice(at, n * size) : claim(number(at, word), n * size)
                if (tag == 270 && kind != 2) || (tag != 270 && kind == 2) { try fail("type_unsupported") }
                if tag == 347 { try jpegTables(raw) }
                entries[tag] = SlideEntry(kind: kind, count: n, raw: raw)
            }
            let tiled = entries[324] != nil
            guard let offsetsEntry = entries[tiled ? 324 : 273], let countsEntry = entries[tiled ? 325 : 279] else { try fail("layout_unknown") }
            if tiled && [273, 278, 279].contains(where: { entries[$0] != nil }) { try fail("layout_unknown") }
            let offsets = try numbers(offsetsEntry)
            let counts = try numbers(countsEntry)
            guard offsets.count == counts.count, offsets.count <= limits.maxBlocks - blockCount else { try fail("block_limit") }
            blockCount += offsets.count
            let blocks = try zip(offsets, counts).map { try claim($0.0, $0.1) }
            directories.append(SlideDirectory(entries: entries, blocks: blocks))
            offset = try number(offset + countSize + count * entrySize, word)
        }
        guard !directories.isEmpty else { try fail("layout_unknown") }
        let ordered = ranges.sorted { $0.lowerBound < $1.lowerBound }
        for (a, b) in zip(ordered, ordered.dropFirst()) where a.upperBound > b.lowerBound { try fail("range_overlap") }
        try classify(&directories)
        return directories
    }
    func classify(_ directories: inout [SlideDirectory]) throws {
        let first = directories[0].entries
        let aperio = first[270]?.raw.starts(with: Data("Aperio".utf8)) == true
        var previous: (Int, Int)?
        var associated = false
        var roles = Set<String>()
        for index in directories.indices {
            let e = directories[index].entries
            let width = try scalar(e, 256)
            let height = try scalar(e, 257)
            guard width > 0, height > 0 else { try fail("dimensions_invalid") }
            let samples = try scalar(e, 277, 1)
            let photo = try scalar(e, 262)
            guard samples > 0, samples <= 64 else { try fail("sample_layout_unsupported") }
            let bits = try e[258].map { try numbers($0) } ?? Array(repeating: 1, count: samples)
            guard [0, 1, 2, 6].contains(photo),
                ![0, 1].contains(photo) || samples == 1,
                ![2, 6].contains(photo) || samples >= 3,
                bits.count == samples, bits.allSatisfy({ $0 > 0 && $0 <= 64 })
            else { try fail("sample_layout_unsupported") }
            let subtype = try scalar(e, 254, 0)
            var role: String
            if e[324] != nil {
                guard !associated, [0, 1].contains(subtype), index != 0 || subtype == 0 else { try fail("layout_unknown") }
                if !aperio && index > 0 && subtype != 1 { try fail("layout_unknown") }
                let tw = try scalar(e, 322)
                let th = try scalar(e, 323)
                guard tw > 0, th > 0, try scalar(e, 284, 1) == 1 else { try fail("layout_unknown") }
                let across = (width - 1) / tw + 1
                let down = (height - 1) / th + 1
                guard across <= limits.maxBlocks / down, across * down == directories[index].blocks.count else { try fail("block_count_invalid") }
                if let p = previous, width > p.0 || height > p.1 || (width == p.0 && height == p.1) { try fail("layout_unknown") }
                if aperio {
                    for tag in [259, 322, 323] where try scalar(e, tag, 1) != scalar(first, tag, 1) { try fail("layout_unknown") }
                }
                previous = (width, height)
                role = "level"
            } else if aperio && index == 1 && subtype == 0 {
                role = "thumbnail"
            } else if aperio && index > 0 && [1, 9].contains(subtype) {
                associated = true
                role = subtype == 1 ? "label" : "macro"
            } else {
                try fail("layout_unknown")
            }
            if role != "level", !roles.insert(role).inserted { try fail("associated_image_unexpected") }
            guard [1, 5, 7, 8, 32946, 33003, 33005].contains(try scalar(e, 259, 1)) else { try fail("compression_unsupported") }
            directories[index].role = role
        }
    }
    func jpegTables(_ raw: Data) throws {
        var pos = 0
        while pos < raw.count {
            guard pos + 2 <= raw.count, raw[pos] == 255 else { try fail("jpeg_tables_invalid") }
            let marker = raw[pos + 1]
            pos += 2
            if [0xD8, 0xD9].contains(marker) { continue }
            guard [0xDB, 0xC4].contains(marker), pos + 2 <= raw.count else { try fail("jpeg_tables_invalid") }
            let size = Int(raw[pos]) * 256 + Int(raw[pos + 1])
            guard size >= 2, size <= raw.count - pos else { try fail("jpeg_tables_invalid") }
            pos += size
        }
    }
    func description(_ raw: Data, aperio: Bool) throws -> Data {
        var values: [String: String] = [:]
        let text = String(decoding: raw, as: UTF8.self).trimmingCharacters(in: CharacterSet(charactersIn: "\0"))
        for field in text.components(separatedBy: "|").dropFirst() {
            let pair = field.split(separator: "=", maxSplits: 1, omittingEmptySubsequences: false)
            guard pair.count == 2 else { continue }
            let key = pair[0].trimmingCharacters(in: .whitespacesAndNewlines)
            let value = pair[1].trimmingCharacters(in: .whitespacesAndNewlines)
            guard ["MPP", "AppMag"].contains(key), value.range(of: "^[0-9]{1,6}(\\.[0-9]{1,6})?$", options: .regularExpression) != nil,
                let numeric = Double(value), numeric > 0, numeric <= 100000
            else { continue }
            guard values[key] == nil else { try fail("description_duplicate_key") }
            values[key] = value
        }
        return Data(((aperio ? "Aperio\n" : "OpenMed deidentified slide\n") + values.keys.sorted().map { "|" + $0 + "=" + (values[$0] ?? "") }.joined() + "\0").utf8)
    }
    func write(_ directories: [SlideDirectory]) throws -> Data {
        var output = data.prefix(headerSize)
        func pack(_ value: Int, _ size: Int) -> Data {
            let bytes = (0..<size).map { UInt8(truncatingIfNeeded: UInt64(value) >> ($0 * 8)) }
            return Data(little ? bytes : bytes.reversed().map({ $0 }))
        }
        func append(_ raw: Data) throws -> Int {
            let alignment = big ? 8 : 2
            let padding = (alignment - output.count % alignment) % alignment
            guard raw.count <= limits.maxFileBytes, output.count <= limits.maxFileBytes - raw.count - padding,
                big || output.count + raw.count + padding <= Int(UInt32.max)
            else { try fail("file_limit") }
            output.append(contentsOf: repeatElement(UInt8(0), count: padding))
            let offset = output.count
            output.append(raw)
            return offset
        }
        var previousPointer = big ? 8 : 4
        let aperio = directories[0].entries[270]?.raw.starts(with: Data("Aperio".utf8)) == true
        for directory in directories where directory.role == "level" {
            var entries = directory.entries
            let desc = try description(entries[270]?.raw ?? Data(), aperio: aperio)
            entries[270] = SlideEntry(kind: 2, count: desc.count, raw: desc)
            var offsets = Data()
            for block in directory.blocks { offsets.append(pack(try append(block), word)) }
            entries[324] = SlideEntry(kind: big ? 16 : 4, count: directory.blocks.count, raw: offsets)
            var encoded = Data()
            for tag in entries.keys.sorted() {
                guard let entry = entries[tag] else { continue }
                var value = entry.raw
                if value.count > word { value = pack(try append(value), word) }
                value.append(contentsOf: repeatElement(UInt8(0), count: word - value.count))
                encoded.append(pack(tag, 2))
                encoded.append(pack(entry.kind, 2))
                encoded.append(pack(entry.count, word))
                encoded.append(value)
            }
            var ifd = pack(entries.count, countSize)
            ifd.append(encoded)
            ifd.append(pack(0, word))
            let offset = try append(ifd)
            output.replaceSubrange(previousPointer..<previousPointer + word, with: pack(offset, word))
            previousPointer = offset + countSize + entries.count * entrySize
        }
        return Data(output)
    }
}
