import Foundation

#if canImport(Compression)
    import Compression
#endif

/// Strict, read-only reader of a local ZIP archive that carries one portable
/// bundle (`docs/mobile/managed-model-bundles.md`).
///
/// The reader parses the central directory, cross-checks every local header,
/// and refuses encryption, ZIP64, multi-disk archives, compression methods
/// other than stored and DEFLATE, unsafe names, symbolic links, duplicates,
/// and overlapping entries. It never extracts by archive name: bytes are read
/// only for paths the signed manifest declares. Errors carry reason codes only.
final class OpenMedBundleArchive {
    struct Entry {
        let name: String
        let isDirectory: Bool
        let method: UInt16
        let compressedSize: Int64
        let size: Int64
        let localOffset: Int64
        let hasDescriptor: Bool
        var dataOffset: Int64 = 0
    }

    struct Failure: Error {
        let reason: OpenMedManagedImportReason
    }

    let byteCount: Int64
    let entries: [Entry]
    /// Directory prefix (`""` or `"<segment>/"`) that holds the manifest.
    let root: String
    /// File entries by path relative to `root`.
    let files: [String: Entry]

    private let handle: FileHandle

    private static let maxEndSearch = 22 + 0xFFFF
    private static let unsupported = Failure(reason: .archiveFormatUnsupported)
    private static let malformed = Failure(reason: .archiveMalformed)

    init(url: URL, limits: OpenMedManagedImportLimits) throws {
        guard let handle = try? FileHandle(forReadingFrom: url) else { throw Failure(reason: .handoffUnreadable) }
        self.handle = handle
        do {
            let size = Int64(try handle.seekToEnd())
            byteCount = size
            guard size <= limits.maxArchiveBytes else { throw Failure(reason: .budgetArchiveBytes) }
            var (entries, directoryStart) = try Self.centralDirectory(handle, size: size, limits: limits)
            try Self.checkLocalHeaders(handle, &entries, directoryStart: directoryStart)
            try Self.checkBudgets(entries, limits: limits)
            self.entries = entries
            (root, files) = Self.layout(entries)
        } catch {
            try? handle.close()
            throw error
        }
    }

    deinit { try? handle.close() }

    // MARK: - Reading

    /// Returns the whole entry, or throws `archiveMalformed` when the stored or
    /// inflated length differs from the central directory.
    func read(_ entry: Entry) throws -> Data {
        let reader = try self.reader(entry, offset: 0)
        var data = Data()
        while true {
            let next = try reader.read(upToCount: 64 << 10)
            if next.isEmpty { break }
            data.append(next)
            if Int64(data.count) > entry.size { throw Self.malformed }
        }
        guard Int64(data.count) == entry.size else { throw Self.malformed }
        return data
    }

    /// A reader positioned `offset` bytes into the entry's uncompressed data.
    /// It yields at most one byte more than declared, so an overlong stream is
    /// detectable without inflating it fully.
    func reader(_ entry: Entry, offset: Int64) throws -> any OpenMedBundleRangeReader & AnyObject {
        if entry.method == 0 {
            return StoredReader(
                handle: handle, position: entry.dataOffset + offset,
                end: entry.dataOffset + entry.compressedSize)
        }
        let reader = try InflateReader(
            handle: handle, start: entry.dataOffset, end: entry.dataOffset + entry.compressedSize,
            limit: entry.size + 1)
        var skipped: Int64 = 0
        while skipped < offset {
            let step = try reader.read(upToCount: Int(min(Int64(64 << 10), offset - skipped)))
            if step.isEmpty { throw Self.malformed }
            skipped += Int64(step.count)
        }
        return reader
    }

    static func readExactly(_ handle: FileHandle, at offset: Int64, count: Int) throws -> Data {
        try handle.seek(toOffset: UInt64(offset))
        var data = Data()
        while data.count < count {
            guard let next = try handle.read(upToCount: count - data.count), !next.isEmpty else { break }
            data.append(next)
        }
        guard data.count == count else { throw malformed }
        return data
    }

    // MARK: - Structure

    private static func centralDirectory(_ handle: FileHandle, size: Int64, limits: OpenMedManagedImportLimits) throws
        -> ([Entry], Int64)
    {
        guard size >= 22 else { throw unsupported }
        let tailLength = Int(min(size, Int64(maxEndSearch)))
        let tail = [UInt8](try readExactly(handle, at: size - Int64(tailLength), count: tailLength))
        var found: Int?
        var position = tail.count - 22
        while position >= 0 {
            if le32(tail, position) == 0x0605_4B50, Int(le16(tail, position + 20)) == tail.count - position - 22 {
                found = position
                break
            }
            position -= 1
        }
        guard let end = found else { throw unsupported }
        let endOffset = size - Int64(tail.count - end)
        if end >= 20, le32(tail, end - 20) == 0x0706_4B50 { throw unsupported }
        let disk = le16(tail, end + 4)
        let directoryDisk = le16(tail, end + 6)
        let diskEntries = le16(tail, end + 8)
        let total = le16(tail, end + 10)
        let directorySize = le32(tail, end + 12)
        let directoryOffset = le32(tail, end + 16)
        guard disk == 0, directoryDisk == 0, diskEntries == total else { throw unsupported }
        guard total != 0xFFFF, directorySize != 0xFFFF_FFFF, directoryOffset != 0xFFFF_FFFF else { throw unsupported }
        guard Int(total) <= limits.maxEntries else { throw Failure(reason: .budgetFileCount) }
        guard total > 0, Int64(directoryOffset) + Int64(directorySize) == endOffset else { throw malformed }

        var entries: [Entry] = []
        var seen = Set<String>()
        var folded = Set<String>()
        var cursor = Int64(directoryOffset)
        let directoryEnd = endOffset
        for _ in 0..<Int(total) {
            guard cursor + 46 <= directoryEnd else { throw malformed }
            let header = [UInt8](try readExactly(handle, at: cursor, count: 46))
            guard le32(header, 0) == 0x0201_4B50 else { throw malformed }
            let madeBy = le16(header, 4)
            let flags = le16(header, 8)
            let method = le16(header, 10)
            let compressed = le32(header, 20)
            let uncompressed = le32(header, 24)
            let nameLength = Int(le16(header, 28))
            let extraLength = Int(le16(header, 30))
            let commentLength = Int(le16(header, 32))
            let startDisk = le16(header, 34)
            let external = le32(header, 38)
            let localOffset = le32(header, 42)
            let next = cursor + 46 + Int64(nameLength + extraLength + commentLength)
            guard next <= directoryEnd else { throw malformed }
            if flags & 0x0001 != 0 || flags & 0x0040 != 0 || flags & 0x2000 != 0 { throw unsupported }
            guard method == 0 || method == 8 else { throw unsupported }
            guard compressed != 0xFFFF_FFFF, uncompressed != 0xFFFF_FFFF, localOffset != 0xFFFF_FFFF, startDisk == 0
            else { throw unsupported }
            guard nameLength > 0, nameLength <= 513 else { throw Failure(reason: .entryUnsafe) }
            let nameBytes = try readExactly(handle, at: cursor + 46, count: nameLength)
            guard let raw = String(data: nameBytes, encoding: .utf8) else { throw Failure(reason: .entryUnsafe) }
            let isDirectory = raw.hasSuffix("/")
            let name = isDirectory ? String(raw.dropLast()) : raw
            guard isSafePath(name) else { throw Failure(reason: .entryUnsafe) }
            if madeBy >> 8 == 3 {
                let type = (external >> 16) & 0o170000
                if type != 0, type != (isDirectory ? 0o040000 : 0o100000) { throw Failure(reason: .entryUnsafe) }
            }
            if isDirectory && uncompressed != 0 { throw malformed }
            if method == 0 && compressed != uncompressed { throw malformed }
            if !isDirectory {
                guard seen.insert(name).inserted, folded.insert(name.lowercased()).inserted else {
                    throw Failure(reason: .entryDuplicate)
                }
            }
            entries.append(
                Entry(
                    name: name, isDirectory: isDirectory, method: UInt16(method), compressedSize: Int64(compressed),
                    size: Int64(uncompressed), localOffset: Int64(localOffset), hasDescriptor: flags & 0x0008 != 0))
            cursor = next
        }
        guard cursor == directoryEnd else { throw malformed }
        return (entries, Int64(directoryOffset))
    }

    private static func checkLocalHeaders(_ handle: FileHandle, _ entries: inout [Entry], directoryStart: Int64) throws {
        for index in entries.indices {
            let entry = entries[index]
            guard entry.localOffset + 30 <= directoryStart else { throw malformed }
            let header = [UInt8](try readExactly(handle, at: entry.localOffset, count: 30))
            guard le32(header, 0) == 0x0403_4B50 else { throw malformed }
            let flags = le16(header, 6)
            let method = le16(header, 8)
            let nameLength = Int(le16(header, 26))
            let extraLength = Int(le16(header, 28))
            guard flags & 0x0001 == 0, method == UInt32(entry.method) else { throw malformed }
            if !entry.hasDescriptor {
                guard Int64(le32(header, 18)) == entry.compressedSize, Int64(le32(header, 22)) == entry.size else {
                    throw malformed
                }
            }
            let rawName = entry.isDirectory ? entry.name + "/" : entry.name
            let localName = try readExactly(handle, at: entry.localOffset + 30, count: nameLength)
            guard localName == Data(rawName.utf8) else { throw malformed }
            let dataOffset = entry.localOffset + 30 + Int64(nameLength + extraLength)
            guard dataOffset + entry.compressedSize <= directoryStart else { throw malformed }
            entries[index].dataOffset = dataOffset
        }
        let ordered = entries.sorted { $0.localOffset < $1.localOffset }
        for (left, right) in zip(ordered, ordered.dropFirst())
        where left.dataOffset + left.compressedSize > right.localOffset {
            throw malformed
        }
    }

    private static func checkBudgets(_ entries: [Entry], limits: OpenMedManagedImportLimits) throws {
        let total = entries.reduce(Int64(0)) { $0 + $1.size }
        guard total <= limits.maxTotalBytes else { throw Failure(reason: .budgetTotalBytes) }
        for entry in entries where entry.method == 8 && entry.size > 0 {
            guard entry.compressedSize > 0, entry.size <= entry.compressedSize * limits.maxCompressionRatio else {
                throw Failure(reason: .budgetCompressionRatio)
            }
        }
    }

    /// Finds the manifest at the archive root or under one top-level directory.
    private static func layout(_ entries: [Entry]) -> (String, [String: Entry]) {
        let fileEntries = entries.filter { !$0.isDirectory }
        var root = ""
        if !fileEntries.contains(where: { $0.name == PortableBundle.manifestFilename }),
            let first = fileEntries.first?.name.split(separator: "/").first.map(String.init)
        {
            let prefix = first + "/"
            let contained = entries.allSatisfy { $0.name.hasPrefix(prefix) || ($0.isDirectory && $0.name == first) }
            if contained, fileEntries.contains(where: { $0.name == prefix + PortableBundle.manifestFilename }) {
                root = prefix
            }
        }
        var files: [String: Entry] = [:]
        for entry in fileEntries where entry.name.hasPrefix(root) {
            files[String(entry.name.dropFirst(root.count))] = entry
        }
        return (root, files)
    }

    /// The portable-bundle path rule: 1 to 512 bytes, at most 16 segments.
    static func isSafePath(_ path: String) -> Bool {
        let segments = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        return path.utf8.count > 0 && path.utf8.count <= 512 && segments.count <= 16
            && segments.allSatisfy { PortableBundlePattern.pathSegment.matches($0) }
    }

    private static func le16(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        UInt32(bytes[offset]) | UInt32(bytes[offset + 1]) << 8
    }

    private static func le32(_ bytes: [UInt8], _ offset: Int) -> UInt32 {
        le16(bytes, offset) | le16(bytes, offset + 2) << 16
    }

    // MARK: - Readers

    private final class StoredReader: OpenMedBundleRangeReader {
        private let handle: FileHandle
        private var position: Int64
        private let end: Int64

        init(handle: FileHandle, position: Int64, end: Int64) {
            self.handle = handle
            self.position = position
            self.end = end
        }

        func read(upToCount count: Int) throws -> Data {
            let want = Int(min(Int64(count), end - position))
            guard want > 0 else { return Data() }
            try handle.seek(toOffset: UInt64(position))
            let data = try handle.read(upToCount: want) ?? Data()
            position += Int64(data.count)
            return data
        }
    }

    /// Streams raw DEFLATE (RFC 1951) through the Compression framework.
    final class InflateReader: OpenMedBundleRangeReader {
        private let handle: FileHandle
        private var position: Int64
        private let end: Int64
        private var remaining: Int64
        private var finished = false
        #if canImport(Compression)
            private let stream = UnsafeMutablePointer<compression_stream>.allocate(capacity: 1)
            private let input = UnsafeMutablePointer<UInt8>.allocate(capacity: 64 << 10)
            private var inputLength = 0
        #endif

        init(handle: FileHandle, start: Int64, end: Int64, limit: Int64) throws {
            self.handle = handle
            position = start
            self.end = end
            remaining = limit
            #if canImport(Compression)
                guard compression_stream_init(stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB) == COMPRESSION_STATUS_OK
                else {
                    stream.deallocate()
                    input.deallocate()
                    throw OpenMedBundleArchive.unsupported
                }
                stream.pointee.src_size = 0
            #else
                throw OpenMedBundleArchive.unsupported
            #endif
        }

        deinit {
            #if canImport(Compression)
                compression_stream_destroy(stream)
                stream.deallocate()
                input.deallocate()
            #endif
        }

        func read(upToCount count: Int) throws -> Data {
            #if canImport(Compression)
                let want = Int(min(Int64(count), remaining))
                guard want > 0, !finished else { return Data() }
                var output = Data(count: want)
                var produced = 0
                while produced < want && !finished {
                    if stream.pointee.src_size == 0 && position < end {
                        try handle.seek(toOffset: UInt64(position))
                        let chunk = try handle.read(upToCount: Int(min(Int64(64 << 10), end - position))) ?? Data()
                        guard !chunk.isEmpty else { throw OpenMedBundleArchive.malformed }
                        chunk.copyBytes(to: input, count: chunk.count)
                        inputLength = chunk.count
                        position += Int64(chunk.count)
                        stream.pointee.src_ptr = UnsafePointer(input)
                        stream.pointee.src_size = inputLength
                    }
                    let flags = position >= end ? Int32(COMPRESSION_STREAM_FINALIZE.rawValue) : 0
                    let status = output.withUnsafeMutableBytes { buffer -> compression_status in
                        let base = buffer.bindMemory(to: UInt8.self).baseAddress! + produced
                        stream.pointee.dst_ptr = base
                        stream.pointee.dst_size = want - produced
                        let result = compression_stream_process(stream, flags)
                        produced = want - stream.pointee.dst_size
                        return result
                    }
                    switch status {
                    case COMPRESSION_STATUS_END:
                        finished = true
                    case COMPRESSION_STATUS_OK:
                        if position >= end && stream.pointee.src_size == 0 && produced < want
                            && stream.pointee.dst_size > 0
                        {
                            // Input exhausted without an end-of-stream marker.
                            throw OpenMedBundleArchive.malformed
                        }
                    default:
                        throw OpenMedBundleArchive.malformed
                    }
                }
                remaining -= Int64(produced)
                output.count = produced
                return output
            #else
                throw OpenMedBundleArchive.unsupported
            #endif
        }
    }
}

/// Range source over the declared files of an archive. Readers record a
/// corrupt compressed stream, which the importer reports as `archive_malformed`.
final class OpenMedArchiveBundleSource: OpenMedBundleRangeSource {
    let identity: String
    private let archive: OpenMedBundleArchive
    private let lock = NSLock()
    private var failed = false

    init(archive: OpenMedBundleArchive, identity: String) {
        self.archive = archive
        self.identity = identity
    }

    var streamFailed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return failed
    }

    fileprivate func markFailed() {
        lock.lock()
        failed = true
        lock.unlock()
    }

    private final class Reader: OpenMedBundleRangeReader {
        let inner: any OpenMedBundleRangeReader
        weak var source: OpenMedArchiveBundleSource?
        init(inner: any OpenMedBundleRangeReader, source: OpenMedArchiveBundleSource) {
            self.inner = inner
            self.source = source
        }

        func read(upToCount count: Int) throws -> Data {
            do {
                return try inner.read(upToCount: count)
            } catch {
                source?.markFailed()
                throw error
            }
        }
    }

    func open(path: String, offset: Int64) throws -> OpenMedBundleRangeResponse {
        guard let entry = archive.files[path] else { throw OpenMedBundleCacheError(.transferInterrupted) }
        do {
            let reader = try archive.reader(entry, offset: offset)
            return OpenMedBundleRangeResponse(offset: offset, reader: Reader(inner: reader, source: self))
        } catch {
            markFailed()
            throw OpenMedBundleCacheError(.transferInterrupted)
        }
    }
}
