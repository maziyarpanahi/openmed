package com.openmed.openmedkit.managed

import com.openmed.openmedkit.bundle.PortableBundle
import com.openmed.openmedkit.cache.BundleRangeResponse
import com.openmed.openmedkit.cache.BundleRangeSource
import java.io.Closeable
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.io.RandomAccessFile
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.zip.DataFormatException
import java.util.zip.Inflater
import java.util.zip.ZipException

/** Archive refusal carrying only a reason code. */
internal class ArchiveFailure(val reason: ManagedImportReason) : Exception(reason.code)

/**
 * Strict, read-only reader of a local ZIP archive that carries one portable
 * bundle (`docs/mobile/managed-model-bundles.md`).
 *
 * It parses the central directory, cross-checks every local header, and
 * refuses encryption, ZIP64, multi-disk archives, compression methods other
 * than stored and DEFLATE, unsafe names, symbolic links, duplicates, and
 * overlapping entries. It never extracts by archive name: bytes are read only
 * for paths the signed manifest declares. It mirrors `OpenMedBundleArchive`
 * in OpenMedKit for Swift.
 */
internal class BundleArchive(file: File, limits: ManagedImportLimits) : Closeable {
    class Entry(
        val name: String,
        val isDirectory: Boolean,
        val method: Int,
        val compressedSize: Long,
        val size: Long,
        val localOffset: Long,
        val hasDescriptor: Boolean,
    ) {
        var dataOffset: Long = 0L
    }

    private val handle: RandomAccessFile = try {
        RandomAccessFile(file, "r")
    } catch (error: IOException) {
        throw ArchiveFailure(ManagedImportReason.HANDOFF_UNREADABLE)
    }

    val byteCount: Long
    val entries: List<Entry>

    /** Directory prefix (`""` or `"<segment>/"`) that holds the manifest. */
    val root: String

    /** File entries by path relative to [root]. */
    val files: Map<String, Entry>

    init {
        try {
            byteCount = handle.length()
            if (byteCount > limits.maxArchiveBytes) throw ArchiveFailure(ManagedImportReason.BUDGET_ARCHIVE_BYTES)
            val (parsed, directoryStart) = centralDirectory(limits)
            checkLocalHeaders(parsed, directoryStart)
            checkBudgets(parsed, limits)
            entries = parsed
            val (prefix, byPath) = layout(parsed)
            root = prefix
            files = byPath
        } catch (error: Throwable) {
            handle.close()
            if (error is IOException) throw ArchiveFailure(ManagedImportReason.ARCHIVE_MALFORMED)
            throw error
        }
    }

    override fun close() = handle.close()

    /** Returns the whole entry, or fails with `archive_malformed` on a length mismatch. */
    fun read(entry: Entry): ByteArray {
        val output = java.io.ByteArrayOutputStream()
        try {
            open(entry, 0L).use { stream ->
                val buffer = ByteArray(64 shl 10)
                while (true) {
                    val count = stream.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    if (output.size() > entry.size) throw ArchiveFailure(ManagedImportReason.ARCHIVE_MALFORMED)
                }
            }
        } catch (error: IOException) {
            throw ArchiveFailure(ManagedImportReason.ARCHIVE_MALFORMED)
        }
        if (output.size().toLong() != entry.size) throw ArchiveFailure(ManagedImportReason.ARCHIVE_MALFORMED)
        return output.toByteArray()
    }

    /**
     * A stream positioned [offset] bytes into the entry's uncompressed data. It
     * yields at most one byte more than declared, so an overlong stream is
     * detectable without inflating it fully.
     */
    fun open(entry: Entry, offset: Long): InputStream {
        if (entry.method == 0) {
            return StoredStream(entry.dataOffset + offset, entry.dataOffset + entry.compressedSize)
        }
        val stream = InflateStream(entry.dataOffset, entry.dataOffset + entry.compressedSize, entry.size + 1)
        var skipped = 0L
        val buffer = ByteArray(64 shl 10)
        while (skipped < offset) {
            val count = stream.read(buffer, 0, minOf(buffer.size.toLong(), offset - skipped).toInt())
            if (count <= 0) throw ZipException("archive_malformed")
            skipped += count
        }
        return stream
    }

    // region Structure

    private fun readAt(offset: Long, count: Int): ByteArray {
        val data = ByteArray(count)
        synchronized(handle) {
            handle.seek(offset)
            handle.readFully(data)
        }
        return data
    }

    private fun centralDirectory(limits: ManagedImportLimits): Pair<List<Entry>, Long> {
        val size = byteCount
        if (size < 22) throw ArchiveFailure(ManagedImportReason.ARCHIVE_FORMAT_UNSUPPORTED)
        val tailLength = minOf(size, MAX_END_SEARCH.toLong()).toInt()
        val tail = readAt(size - tailLength, tailLength)
        var end = -1
        var position = tail.size - 22
        while (position >= 0) {
            if (le32(tail, position) == 0x06054B50L && le16(tail, position + 20) == tail.size - position - 22) {
                end = position
                break
            }
            position--
        }
        if (end < 0) throw ArchiveFailure(ManagedImportReason.ARCHIVE_FORMAT_UNSUPPORTED)
        val endOffset = size - (tail.size - end)
        if (end >= 20 && le32(tail, end - 20) == 0x07064B50L) unsupported()
        val disk = le16(tail, end + 4)
        val directoryDisk = le16(tail, end + 6)
        val diskEntries = le16(tail, end + 8)
        val total = le16(tail, end + 10)
        val directorySize = le32(tail, end + 12)
        val directoryOffset = le32(tail, end + 16)
        if (disk != 0 || directoryDisk != 0 || diskEntries != total) unsupported()
        if (total == 0xFFFF || directorySize == 0xFFFFFFFFL || directoryOffset == 0xFFFFFFFFL) unsupported()
        if (total > limits.maxEntries) throw ArchiveFailure(ManagedImportReason.BUDGET_FILE_COUNT)
        if (total == 0 || directoryOffset + directorySize != endOffset) malformed()

        val entries = mutableListOf<Entry>()
        val seen = HashSet<String>()
        val folded = HashSet<String>()
        var cursor = directoryOffset
        repeat(total) {
            if (cursor + 46 > endOffset) malformed()
            val header = readAt(cursor, 46)
            if (le32(header, 0) != 0x02014B50L) malformed()
            val madeBy = le16(header, 4)
            val flags = le16(header, 8)
            val method = le16(header, 10)
            val compressed = le32(header, 20)
            val uncompressed = le32(header, 24)
            val nameLength = le16(header, 28)
            val extraLength = le16(header, 30)
            val commentLength = le16(header, 32)
            val startDisk = le16(header, 34)
            val external = le32(header, 38)
            val localOffset = le32(header, 42)
            val next = cursor + 46 + nameLength + extraLength + commentLength
            if (next > endOffset) malformed()
            if (flags and 0x0001 != 0 || flags and 0x0040 != 0 || flags and 0x2000 != 0) unsupported()
            if (method != 0 && method != 8) unsupported()
            if (compressed == 0xFFFFFFFFL || uncompressed == 0xFFFFFFFFL || localOffset == 0xFFFFFFFFL ||
                startDisk != 0
            ) {
                unsupported()
            }
            if (nameLength == 0 || nameLength > 513) unsafe()
            val raw = decodeUtf8(readAt(cursor + 46, nameLength)) ?: unsafe()
            val isDirectory = raw.endsWith("/")
            val name = if (isDirectory) raw.dropLast(1) else raw
            if (!isSafePath(name)) unsafe()
            if (madeBy shr 8 == 3) {
                val type = (external shr 16) and 0xF000L
                if (type != 0L && type != (if (isDirectory) 0x4000L else 0x8000L)) unsafe()
            }
            if (isDirectory && uncompressed != 0L) malformed()
            if (method == 0 && compressed != uncompressed) malformed()
            if (!isDirectory && (!seen.add(name) || !folded.add(name.lowercase(Locale.ROOT)))) {
                throw ArchiveFailure(ManagedImportReason.ENTRY_DUPLICATE)
            }
            entries += Entry(name, isDirectory, method, compressed, uncompressed, localOffset, flags and 0x0008 != 0)
            cursor = next
        }
        if (cursor != endOffset) malformed()
        return entries to directoryOffset
    }

    private fun checkLocalHeaders(entries: List<Entry>, directoryStart: Long) {
        for (entry in entries) {
            if (entry.localOffset + 30 > directoryStart) malformed()
            val header = readAt(entry.localOffset, 30)
            if (le32(header, 0) != 0x04034B50L) malformed()
            val flags = le16(header, 6)
            val method = le16(header, 8)
            val nameLength = le16(header, 26)
            val extraLength = le16(header, 28)
            if (flags and 0x0001 != 0 || method != entry.method) malformed()
            if (!entry.hasDescriptor && (le32(header, 18) != entry.compressedSize || le32(header, 22) != entry.size)) {
                malformed()
            }
            val rawName = if (entry.isDirectory) entry.name + "/" else entry.name
            if (!readAt(entry.localOffset + 30, nameLength)
                    .contentEquals(rawName.toByteArray(StandardCharsets.UTF_8))
            ) {
                malformed()
            }
            val dataOffset = entry.localOffset + 30 + nameLength + extraLength
            if (dataOffset + entry.compressedSize > directoryStart) malformed()
            entry.dataOffset = dataOffset
        }
        val ordered = entries.sortedBy { it.localOffset }
        for (index in 0 until ordered.size - 1) {
            if (ordered[index].dataOffset + ordered[index].compressedSize > ordered[index + 1].localOffset) malformed()
        }
    }

    private fun checkBudgets(entries: List<Entry>, limits: ManagedImportLimits) {
        if (entries.sumOf { it.size } > limits.maxTotalBytes) {
            throw ArchiveFailure(ManagedImportReason.BUDGET_TOTAL_BYTES)
        }
        for (entry in entries) {
            if (entry.method != 8 || entry.size == 0L) continue
            if (entry.compressedSize == 0L || entry.size > entry.compressedSize * limits.maxCompressionRatio) {
                throw ArchiveFailure(ManagedImportReason.BUDGET_COMPRESSION_RATIO)
            }
        }
    }

    /** Finds the manifest at the archive root or under one top-level directory. */
    private fun layout(entries: List<Entry>): Pair<String, Map<String, Entry>> {
        val fileEntries = entries.filter { !it.isDirectory }
        var prefix = ""
        val first = fileEntries.firstOrNull()?.name?.substringBefore('/')
        if (fileEntries.none { it.name == PortableBundle.MANIFEST_FILENAME } && first != null) {
            val candidate = "$first/"
            val contained = entries.all { it.name.startsWith(candidate) || (it.isDirectory && it.name == first) }
            if (contained && fileEntries.any { it.name == candidate + PortableBundle.MANIFEST_FILENAME }) {
                prefix = candidate
            }
        }
        val byPath = LinkedHashMap<String, Entry>()
        for (entry in fileEntries) {
            if (entry.name.startsWith(prefix)) byPath[entry.name.substring(prefix.length)] = entry
        }
        return prefix to byPath
    }

    // endregion

    // region Streams

    private inner class StoredStream(private var position: Long, private val end: Long) : InputStream() {
        override fun read(): Int {
            val one = ByteArray(1)
            return if (read(one, 0, 1) <= 0) -1 else one[0].toInt() and 0xFF
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val want = minOf(length.toLong(), end - position).toInt()
            if (want <= 0) return -1
            val count = synchronized(handle) {
                handle.seek(position)
                handle.read(buffer, offset, want)
            }
            if (count <= 0) return -1
            position += count
            return count
        }
    }

    /** Streams raw DEFLATE (RFC 1951) through [Inflater] with `nowrap`. */
    private inner class InflateStream(
        private var position: Long,
        private val end: Long,
        private var remaining: Long,
    ) : InputStream() {
        private val inflater = Inflater(true)
        private val input = ByteArray(64 shl 10)
        private var padded = false

        override fun read(): Int {
            val one = ByteArray(1)
            return if (read(one, 0, 1) <= 0) -1 else one[0].toInt() and 0xFF
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val want = minOf(length.toLong(), remaining).toInt()
            if (want <= 0 || inflater.finished()) return -1
            while (true) {
                if (inflater.needsInput()) {
                    if (position < end) {
                        val count = synchronized(handle) {
                            handle.seek(position)
                            handle.read(input, 0, minOf(input.size.toLong(), end - position).toInt())
                        }
                        if (count <= 0) throw ZipException("archive_malformed")
                        position += count
                        inflater.setInput(input, 0, count)
                    } else if (!padded) {
                        // `nowrap` inflation may need one dummy byte after the stream.
                        padded = true
                        inflater.setInput(ByteArray(1), 0, 1)
                    } else {
                        throw ZipException("archive_malformed")
                    }
                }
                val produced = try {
                    inflater.inflate(buffer, offset, want)
                } catch (error: DataFormatException) {
                    throw ZipException("archive_malformed")
                }
                if (produced > 0) {
                    remaining -= produced
                    return produced
                }
                if (inflater.finished()) return -1
                if (inflater.needsDictionary()) throw ZipException("archive_malformed")
            }
        }

        override fun close() = inflater.end()
    }

    // endregion

    companion object {
        private const val MAX_END_SEARCH = 22 + 0xFFFF

        /** The portable-bundle path rule: 1 to 512 bytes, at most 16 segments. */
        fun isSafePath(path: String): Boolean {
            val segments = path.split("/")
            return path.isNotEmpty() && path.toByteArray(StandardCharsets.UTF_8).size <= 512 && segments.size <= 16 &&
                segments.all { PATH_SEGMENT.matches(it) }
        }

        private val PATH_SEGMENT = Regex("[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}")

        private fun decodeUtf8(bytes: ByteArray): String? = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(bytes))
                .toString()
        } catch (error: CharacterCodingException) {
            null
        }

        private fun le16(bytes: ByteArray, offset: Int): Int =
            (bytes[offset].toInt() and 0xFF) or ((bytes[offset + 1].toInt() and 0xFF) shl 8)

        private fun le32(bytes: ByteArray, offset: Int): Long =
            le16(bytes, offset).toLong() or (le16(bytes, offset + 2).toLong() shl 16)

        private fun unsupported(): Nothing = throw ArchiveFailure(ManagedImportReason.ARCHIVE_FORMAT_UNSUPPORTED)
        private fun malformed(): Nothing = throw ArchiveFailure(ManagedImportReason.ARCHIVE_MALFORMED)
        private fun unsafe(): Nothing = throw ArchiveFailure(ManagedImportReason.ENTRY_UNSAFE)
    }
}

/**
 * Range source over the declared files of a [BundleArchive]. A corrupt
 * compressed stream is recorded so the importer reports `archive_malformed`.
 */
internal class ArchiveBundleRangeSource(
    private val archive: BundleArchive,
    override val identity: String,
) : BundleRangeSource {
    @Volatile
    var streamFailed: Boolean = false
        private set

    override fun open(path: String, offset: Long): BundleRangeResponse {
        val entry = archive.files[path] ?: throw IOException("file_missing")
        val stream = try {
            archive.open(entry, offset)
        } catch (error: IOException) {
            streamFailed = true
            throw error
        }
        return BundleRangeResponse(offset, Tracking(stream))
    }

    private inner class Tracking(private val inner: InputStream) : InputStream() {
        override fun read(): Int = guard { inner.read() }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int = guard { inner.read(buffer, offset, length) }

        override fun close() = inner.close()

        private fun guard(block: () -> Int): Int = try {
            block()
        } catch (error: IOException) {
            streamFailed = true
            throw error
        }
    }

    override fun toString(): String = "ArchiveBundleRangeSource"
}
