import Foundation

/// Stable refusals without source bytes, item names or private paths.
public enum BMFFPreflightError: String, Error, CustomStringConvertible {
    case truncated = "bmff_truncated"
    case outOfBounds = "bmff_metadata_out_of_bounds"
    case depthLimit = "bmff_depth_limit"
    case boxCountLimit = "bmff_box_count_limit"
    case sizeInvalid = "bmff_box_size_invalid"
    case sizeOverflow = "bmff_size_overflow"
    case declaredSizeLimit = "bmff_declared_size_limit"
    case metadataMissing = "bmff_metadata_missing_or_duplicate"
    case unsupportedLayout = "bmff_layout_unsupported"
    case metadataInvalid = "bmff_metadata_invalid"
    case signatureInvalid = "bmff_signature_invalid"
    case pixelLimit = "bmff_pixel_limit"
    case itemCycle = "bmff_item_cycle"
    case codecUnsupported = "bmff_codec_unsupported"
    case audioTrackAmbiguous = "bmff_audio_track_ambiguous"
    case limitsInvalid = "bmff_limits_invalid"

    public var description: String { rawValue }
}

/// Declared primary image geometry and unique item-role counts, never pixels.
public struct HEIFPreflight: Equatable, Sendable {
    public let width: Int
    public let height: Int
    public let itemCount: Int
    public let thumbnailCount: Int
    public let auxiliaryCount: Int
    public let burstCount: Int
    public let hasEXIF: Bool
    public let hasXMP: Bool
    public let status = "decoder_required"
}

/// Declared sample-entry metadata and media time, not edited playback duration.
public struct M4APreflight: Equatable, Sendable {
    public let codecCategory: String
    public let channels: Int
    public let sampleRateHz: Double
    public let durationSeconds: Double
    public let status = "decoder_required"
    public let notice = "non_diagnostic"
    public let reviewerConfirmationRequired = true
}

/// The bounded reader's metadata-only result. It never invokes a decoder.
public enum BMFFPreflightResult: Equatable, Sendable {
    case heif(HEIFPreflight)
    case m4a(M4APreflight)
}

/// Local HEIF item and M4A track header inspection with Python-equivalent limits.
public enum ISOBMFFPreflight {
    /// Detect a complete first file-type box within 132 bytes. Ambiguity returns nil.
    public static func detectMediaType(_ source: Data) -> String? {
        let data = Array(source.prefix(132))
        guard data.count >= 16, number(data, 4, 4) == fourCC("ftyp") else { return nil }
        var size = number(data, 0, 4)
        var header = 8
        if size == 1 {
            size = number(data, 8, 8)
            header = 16
        }
        guard size >= UInt64(header + 8), size <= UInt64(data.count), (size - UInt64(header)) % 4 == 0 else { return nil }
        var brands: Set<UInt64> = [number(data, header, 4)]
        for offset in stride(from: header + 8, to: Int(size), by: 4) { brands.insert(number(data, offset, 4)) }
        if !brands.isDisjoint(with: [fourCC("avif"), fourCC("avis")]) { return nil }
        var image: String?
        if !brands.isDisjoint(with: [fourCC("hevc"), fourCC("hevx")]) {
            image = "image/heic-sequence"
        } else if brands.contains(fourCC("msf1")) {
            image = "image/heif-sequence"
        } else if !brands.isDisjoint(with: [fourCC("heic"), fourCC("heix"), fourCC("heim"), fourCC("heis")]) {
            image = "image/heic"
        } else if brands.contains(fourCC("mif1")) {
            image = "image/heif"
        }
        let audio = brands.contains(fourCC("M4A ")) ? "audio/mp4" : nil
        return image != nil && audio != nil ? nil : image ?? audio
    }

    /// Inspect caller-owned complete container data without retaining it or decoding media.
    /// The byte limit bounds absolute offsets, including skipped media. Unknown payloads
    /// are skipped; unsupported tracks, edits and fragments are refused. This does not
    /// scrub EXIF, validate media payloads or establish clinical evidence.
    public static func read(
        _ source: Data, maxHeaderBytes: Int = 1_048_576, maxDepth: Int = 16,
        maxBoxes: Int = 4096, maxDeclaredBoxBytes: UInt64 = 4_294_967_296,
        maxPixels: Int = 100_000_000
    ) throws -> BMFFPreflightResult {
        guard maxHeaderBytes > 0, maxDepth > 0, maxDepth <= 64, maxBoxes > 0, maxDeclaredBoxBytes > 0, maxPixels > 0 else {
            throw BMFFPreflightError.limitsInvalid
        }
        let walker = BMFFWalker(source, maxHeaderBytes, maxDepth, maxBoxes, maxDeclaredBoxBytes)
        let top = try walker.boxes(0, source.count)
        let ftyp = try one(top, "ftyp")
        guard top.first == ftyp else { throw BMFFPreflightError.signatureInvalid }
        let prefix = try walker.read(0, min(ftyp.end, 132), source.count)
        guard let media = detectMediaType(Data(prefix)) else { throw BMFFPreflightError.signatureInvalid }
        guard !top.contains(where: { $0.kind == fourCC("moof") }) else { throw BMFFPreflightError.unsupportedLayout }
        if media.hasPrefix("image/") {
            guard !top.contains(where: { $0.kind == fourCC("moov") }) else { throw BMFFPreflightError.unsupportedLayout }
            return .heif(try heif(walker, one(top, "meta"), maxPixels))
        }
        return .m4a(try m4a(walker, one(top, "moov")))
    }

    private static func heif(_ w: BMFFWalker, _ meta: BMFFBox, _ maxPixels: Int) throws -> HEIFPreflight {
        _ = try BMFFFields(w.read(meta.start, 4, meta.end)).full([0])
        let children = try w.children(meta, 4)
        let pitm = try BMFFFields(w.payload(one(children, "pitm")))
        let (pv, _) = try pitm.full([0, 1])
        let primary = try pitm.int(pv == 0 ? 2 : 4)
        try pitm.done()
        let iinf = try one(children, "iinf")
        let info = try BMFFFields(w.read(iinf.start, min(8, iinf.end - iinf.start), iinf.end))
        let (iv, _) = try info.full([0, 1])
        let count = try info.int(iv == 0 ? 2 : 4)
        let entries = try w.children(iinf, iv == 0 ? 6 : 8)
        guard count == entries.count else { throw BMFFPreflightError.metadataInvalid }
        var items: [Int: UInt64] = [:]
        var exif = false
        var xmp = false
        for entry in entries {
            guard entry.kind == fourCC("infe") else { throw BMFFPreflightError.unsupportedLayout }
            let f = try BMFFFields(w.payload(entry))
            let (version, _) = try f.full([2, 3], 1)
            let id = try f.int(version == 2 ? 2 : 4)
            let protection = try f.int(2)
            guard id != 0, items[id] == nil, protection == 0 else { throw BMFFPreflightError.metadataInvalid }
            let kind = try f.uint(4)
            _ = try f.cstring()
            items[id] = kind
            exif = exif || kind == fourCC("Exif")
            if kind == fourCC("mime") {
                let contentType = try f.cstring()
                xmp = xmp || contentType == Array("application/rdf+xml".utf8)
                if f.offset < f.data.count { _ = try f.cstring() }
            } else if kind == fourCC("uri ") {
                _ = try f.cstring()
            }
            try f.done()
        }
        guard let primaryType = items[primary], imageItems.contains(primaryType) else { throw BMFFPreflightError.unsupportedLayout }
        let iprp = try w.children(one(children, "iprp"))
        let properties = try w.children(one(iprp, "ipco"))
        var associations: [Int: [Int]] = [:]
        for association in iprp where association.kind == fourCC("ipma") {
            let f = try BMFFFields(w.payload(association))
            let (version, flags) = try f.full([0, 1], 1)
            let entryCount = try f.int(4)
            for _ in 0..<entryCount {
                let id = try f.int(version == 0 ? 2 : 4)
                guard items[id] != nil, associations[id] == nil else { throw BMFFPreflightError.metadataInvalid }
                var indices: [Int] = []
                let associationCount = try f.int(1)
                for _ in 0..<associationCount {
                    let index = try f.int(flags & 1 == 1 ? 2 : 1) & (flags & 1 == 1 ? 0x7fff : 0x7f)
                    guard index <= properties.count else { throw BMFFPreflightError.metadataInvalid }
                    if index != 0 { indices.append(index) }
                }
                associations[id] = indices
            }
            try f.done()
        }
        var sizes: [(Int, Int)] = []
        for index in associations[primary] ?? [] {
            let prop = properties[index - 1]
            if prop.kind == fourCC("ispe") {
                let f = try BMFFFields(w.payload(prop))
                _ = try f.full([0])
                sizes.append((try f.int(4), try f.int(4)))
                try f.done()
            }
        }
        guard sizes.count == 1 else { throw BMFFPreflightError.metadataMissing }
        let (width, height) = sizes[0]
        guard width > 0, height > 0 else { throw BMFFPreflightError.metadataInvalid }
        guard width <= maxPixels / height else { throw BMFFPreflightError.pixelLimit }
        var roles: [UInt64: Set<Int>] = [fourCC("thmb"): [], fourCC("auxl"): []]
        var graph = Dictionary(uniqueKeysWithValues: items.keys.map { ($0, Set<Int>()) })
        let refs = children.filter { $0.kind == fourCC("iref") }
        guard refs.count <= 1 else { throw BMFFPreflightError.metadataMissing }
        for ref in refs {
            let (version, _) = try BMFFFields(w.read(ref.start, 4, ref.end)).full([0, 1])
            for box in try w.children(ref, 4) {
                let f = try BMFFFields(w.payload(box))
                let origin = try f.int(version == 0 ? 2 : 4)
                let count = try f.int(2)
                var targets: Set<Int> = []
                for _ in 0..<count { targets.insert(try f.int(version == 0 ? 2 : 4)) }
                try f.done()
                guard items[origin] != nil, !targets.isEmpty, targets.allSatisfy({ items[$0] != nil }) else { throw BMFFPreflightError.metadataInvalid }
                graph[origin, default: []].formUnion(targets)
                if roles[box.kind] != nil {
                    guard let type = items[origin], imageItems.contains(type) else { throw BMFFPreflightError.metadataInvalid }
                    roles[box.kind, default: []].insert(origin)
                }
            }
        }
        try acyclic(graph)
        var burst: Set<Int> = []
        for group in children where group.kind == fourCC("grpl") {
            for box in try w.children(group) where box.kind == fourCC("brst") {
                let f = try BMFFFields(w.payload(box))
                _ = try f.full([0])
                _ = try f.int(4)
                let count = try f.int(4)
                var entities: Set<Int> = []
                for _ in 0..<count { entities.insert(try f.int(4)) }
                guard !entities.isEmpty, entities.allSatisfy({ id in items[id].map { imageItems.contains($0) } ?? false }) else { throw BMFFPreflightError.unsupportedLayout }
                burst.formUnion(entities)
            }
        }
        return HEIFPreflight(width: width, height: height, itemCount: items.count, thumbnailCount: roles[fourCC("thmb")]!.count, auxiliaryCount: roles[fourCC("auxl")]!.count, burstCount: burst.count, hasEXIF: exif, hasXMP: xmp)
    }

    private static func acyclic(_ graph: [Int: Set<Int>]) throws {
        var incoming = Dictionary(uniqueKeysWithValues: graph.keys.map { ($0, 0) })
        for targets in graph.values { for target in targets { incoming[target, default: 0] += 1 } }
        var queue = incoming.filter { $0.value == 0 }.map { $0.key }
        var visited = 0
        while let node = queue.popLast() {
            visited += 1
            for target in graph[node] ?? [] {
                incoming[target, default: 0] -= 1
                if incoming[target] == 0 { queue.append(target) }
            }
        }
        guard visited == graph.count else { throw BMFFPreflightError.itemCycle }
    }

    private static func m4a(_ w: BMFFWalker, _ moov: BMFFBox) throws -> M4APreflight {
        var tracks: [M4APreflight] = []
        for track in try w.children(moov) where track.kind == fourCC("trak") {
            let children = try w.children(track)
            guard !children.contains(where: { $0.kind == fourCC("edts") }) else { throw BMFFPreflightError.unsupportedLayout }
            let mdia = try w.children(one(children, "mdia"))
            let handler = try BMFFFields(w.payload(one(mdia, "hdlr")))
            _ = try handler.full([0])
            _ = try handler.take(4)
            guard try handler.uint(4) == fourCC("soun") else { continue }
            let f = try BMFFFields(w.payload(one(mdia, "mdhd")))
            let (version, _) = try f.full([0, 1])
            _ = try f.take(version == 0 ? 8 : 16)
            let timescale = try f.uint(4)
            let duration = try f.uint(version == 0 ? 4 : 8)
            _ = try f.take(4)
            try f.done()
            guard timescale != 0, duration != (version == 0 ? UInt64(UInt32.max) : UInt64.max) else { throw BMFFPreflightError.metadataInvalid }
            let minf = try w.children(one(mdia, "minf"))
            let stbl = try w.children(one(minf, "stbl"))
            let stsd = try one(stbl, "stsd")
            let countFields = try BMFFFields(w.read(stsd.start, 8, stsd.end))
            _ = try countFields.full([0])
            guard try countFields.int(4) == 1 else { throw BMFFPreflightError.unsupportedLayout }
            let entries = try w.children(stsd, 8)
            guard entries.count == 1 else { throw BMFFPreflightError.metadataInvalid }
            let entry = entries[0]
            let codec: String
            if entry.kind == fourCC("mp4a") { codec = "mpeg4_audio" } else if entry.kind == fourCC("alac") { codec = "apple_lossless" } else { throw BMFFPreflightError.codecUnsupported }
            let sample = try BMFFFields(w.read(entry.start, 28, entry.end))
            _ = try sample.take(8)
            guard try sample.int(2) == 0 else { throw BMFFPreflightError.unsupportedLayout }
            _ = try sample.take(6)
            let channels = try sample.int(2)
            _ = try sample.take(6)
            let rate = Double(try sample.uint(4)) / 65536
            guard channels != 0, rate != 0 else { throw BMFFPreflightError.metadataInvalid }
            _ = try w.children(entry, 28)
            tracks.append(M4APreflight(codecCategory: codec, channels: channels, sampleRateHz: rate, durationSeconds: Double(duration) / Double(timescale)))
        }
        guard tracks.count == 1 else { throw BMFFPreflightError.audioTrackAmbiguous }
        return tracks[0]
    }

    private static let imageItems: Set<UInt64> = ["hvc1", "av01", "jpeg", "grid", "iovl", "iden", "unci"].map(fourCC).reduce(into: []) { $0.insert($1) }
}

private func fourCC(_ value: String) -> UInt64 { value.utf8.reduce(0) { ($0 << 8) | UInt64($1) } }
private func number(_ data: [UInt8], _ start: Int, _ size: Int) -> UInt64 { data[start..<(start + size)].reduce(0) { ($0 << 8) | UInt64($1) } }
private func one(_ boxes: [BMFFBox], _ kind: String) throws -> BMFFBox {
    let matches = boxes.filter { $0.kind == fourCC(kind) }
    guard matches.count == 1 else { throw BMFFPreflightError.metadataMissing }
    return matches[0]
}

private struct BMFFBox: Equatable {
    let kind: UInt64
    let start: Int
    let end: Int
    let depth: Int
}

private final class BMFFWalker {
    let data: Data
    let limit: Int
    let maxDepth: Int
    let maxCount: Int
    let maxSize: UInt64
    var count = 0
    init(_ data: Data, _ limit: Int, _ depth: Int, _ count: Int, _ size: UInt64) {
        self.data = data
        self.limit = limit
        self.maxDepth = depth
        self.maxCount = count
        self.maxSize = size
    }
    func read(_ offset: Int, _ size: Int, _ end: Int) throws -> [UInt8] {
        guard offset >= 0, size >= 0, size <= end - offset else { throw BMFFPreflightError.truncated }
        guard size <= limit - offset else { throw BMFFPreflightError.outOfBounds }
        let start = data.startIndex + offset
        return Array(data[start..<(start + size)])
    }
    func boxes(_ begin: Int, _ end: Int, _ depth: Int = 1) throws -> [BMFFBox] {
        guard depth <= maxDepth else { throw BMFFPreflightError.depthLimit }
        var start = begin
        var result: [BMFFBox] = []
        while start < end {
            count += 1
            guard count <= maxCount else { throw BMFFPreflightError.boxCountLimit }
            let raw = try read(start, 8, end)
            var size = number(raw, 0, 4)
            var header = 8
            if size == 1 {
                size = number(try read(start + 8, 8, end), 0, 8)
                header = 16
            } else if size == 0 {
                size = UInt64(end - start)
            }
            guard size >= UInt64(header) else { throw BMFFPreflightError.sizeInvalid }
            guard size <= UInt64.max - UInt64(start) else { throw BMFFPreflightError.sizeOverflow }
            guard size <= maxSize else { throw BMFFPreflightError.declaredSizeLimit }
            guard size <= UInt64(end - start) else { throw BMFFPreflightError.truncated }
            result.append(BMFFBox(kind: number(raw, 4, 4), start: start + header, end: start + Int(size), depth: depth))
            start += Int(size)
        }
        return result
    }
    func children(_ box: BMFFBox, _ skip: Int = 0) throws -> [BMFFBox] {
        _ = try read(box.start, skip, box.end)
        return try boxes(box.start + skip, box.end, box.depth + 1)
    }
    func payload(_ box: BMFFBox) throws -> [UInt8] { try read(box.start, box.end - box.start, box.end) }
}

private final class BMFFFields {
    let data: [UInt8]
    var offset = 0
    init(_ data: [UInt8]) { self.data = data }
    func take(_ size: Int) throws -> [UInt8] {
        guard size <= data.count - offset else { throw BMFFPreflightError.truncated }
        let result = Array(data[offset..<(offset + size)])
        offset += size
        return result
    }
    func uint(_ size: Int) throws -> UInt64 { try take(size).reduce(0) { ($0 << 8) | UInt64($1) } }
    func int(_ size: Int) throws -> Int { Int(try uint(size)) }
    func cstring() throws -> [UInt8] {
        guard let end = data[offset...].firstIndex(of: 0) else { throw BMFFPreflightError.truncated }
        return Array(try take(end - offset + 1).dropLast())
    }
    func full(_ versions: [Int], _ flags: Int = 0) throws -> (Int, Int) {
        let version = try int(1)
        let actualFlags = try int(3)
        guard versions.contains(version), actualFlags & ~flags == 0 else { throw BMFFPreflightError.unsupportedLayout }
        return (version, actualFlags)
    }
    func done() throws {
        guard offset == data.count else { throw BMFFPreflightError.metadataInvalid }
    }
}
