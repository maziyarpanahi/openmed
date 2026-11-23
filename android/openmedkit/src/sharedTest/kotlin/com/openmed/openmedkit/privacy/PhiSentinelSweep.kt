package com.openmed.openmedkit.privacy

import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import java.nio.file.LinkOption
import java.nio.file.Path
import java.nio.file.attribute.BasicFileAttributes

/**
 * Sweeps app-writable directories and the process's own logs for planted sentinels.
 *
 * Run a workload with [PhiSentinel] values as inputs, then call [run]. The sweep only
 * reads: it walks each [SweepRoot] without following symbolic links, matches file
 * names and up to [SweepBounds.maxBytesPerFile] content bytes per file against every
 * [SentinelEncoding] form of every sentinel, then reads each [ProcessLogReader]. The
 * returned [PhiSentinelSweepReport] is value-free.
 *
 * This is test support compiled only into the `test` and `androidTest` source sets,
 * so it never ships in the OpenMedKit AAR.
 */
class PhiSentinelSweep(
    val sentinels: List<PhiSentinel>,
    val roots: List<SweepRoot>,
    private val logReaders: List<ProcessLogReader> = emptyList(),
    val bounds: SweepBounds = SweepBounds.STANDARD,
    val allowList: ArtifactAllowList = ArtifactAllowList.NONE,
    private val clockMillis: () -> Long = { System.nanoTime() / 1_000_000L },
) {
    internal val needles: List<SentinelNeedle>

    init {
        if (sentinels.isEmpty()) {
            throw PhiSentinelSweepException(SweepRefusal.SENTINELS_EMPTY)
        }
        if (sentinels.size > PhiSentinel.MAXIMUM_COUNT) {
            throw PhiSentinelSweepException(SweepRefusal.SENTINELS_TOO_MANY)
        }
        if (sentinels.map { it.identifier }.toSet().size != sentinels.size) {
            throw PhiSentinelSweepException(SweepRefusal.SENTINEL_IDENTIFIER_DUPLICATE)
        }
        bounds.validate()
        needles = sentinels.flatMapIndexed { index, sentinel ->
            SentinelNeedles.make(sentinel.plantedValue, index)
        }
    }

    /** Run the sweep and return its value-free report. */
    fun run(): PhiSentinelSweepReport {
        val run = SweepRun(sentinels, needles, bounds, allowList, clockMillis)
        roots.forEach(run::sweep)
        logReaders.forEach(run::sweep)
        return run.report()
    }

    internal companion object {
        /** Bytes read per file read call; matches spanning chunks are still found. */
        const val CHUNK_SIZE = 1_048_576
    }
}

private data class FindingKey(
    val sentinelIndex: Int,
    val encoding: SentinelEncoding,
    val location: SweepLocation,
    val surface: String,
    val fileKind: SweepFileKind?,
)

private class SweepRun(
    private val sentinels: List<PhiSentinel>,
    private val needles: List<SentinelNeedle>,
    private val bounds: SweepBounds,
    private val allowList: ArtifactAllowList,
    private val clockMillis: () -> Long,
) {
    private val deadline = clockMillis() + bounds.maxDurationMillis
    private val carryLength = maxOf(0, (needles.maxOfOrNull { it.bytes.size } ?: 1) - 1)
    private val boundsReached = mutableSetOf<SweepBound>()
    private val findings = linkedMapOf<FindingKey, LongArray>()
    private val rootSummaries = mutableListOf<SweepRootSummary>()
    private val logSummaries = mutableListOf<SweepLogSummary>()
    private var filesVisited = 0
    private var filesScanned = 0
    private var bytesScanned = 0L
    private var allowListedFiles = 0
    private var truncatedFiles = 0
    private var skippedSymbolicLinks = 0
    private var skippedSpecialFiles = 0
    private var unreadableItems = 0
    private var logEntriesScanned = 0
    private var logBytesScanned = 0L
    private var filesStopped = false
    private var timeExpired = false

    private fun outOfTime(): Boolean {
        if (timeExpired || clockMillis() >= deadline) {
            timeExpired = true
            boundsReached += SweepBound.MAX_DURATION
        }
        return timeExpired
    }

    fun sweep(root: SweepRoot) {
        val filesBefore = filesScanned
        val bytesBefore = bytesScanned
        val present = root.directory.isDirectory
        if (present && !filesStopped) {
            walk(root)
        }
        rootSummaries += SweepRootSummary(
            storageClass = root.storageClass,
            present = present,
            filesScanned = filesScanned - filesBefore,
            bytesScanned = bytesScanned - bytesBefore,
        )
    }

    private class Pending(val path: Path, val depth: Int, val components: List<String>)

    private fun walk(root: SweepRoot) {
        val stack = ArrayDeque<Pending>()
        stack.addLast(Pending(root.directory.toPath(), 0, emptyList()))
        while (stack.isNotEmpty()) {
            val directory = stack.removeLast()
            if (filesStopped || outOfTime()) {
                filesStopped = true
                return
            }
            val children = try {
                Files.newDirectoryStream(directory.path).use { stream ->
                    stream.sortedBy { it.fileName.toString() }
                }
            } catch (ignored: IOException) {
                unreadableItems += 1
                continue
            } catch (ignored: SecurityException) {
                unreadableItems += 1
                continue
            }
            val subdirectories = mutableListOf<Pending>()
            for (child in children) {
                val name = child.fileName.toString()
                if (directory.depth == 0 && name in root.excludedChildNames) {
                    continue
                }
                if (outOfTime()) {
                    filesStopped = true
                    return
                }
                val components = directory.components + name
                val kind = SweepFileKind.of(name)
                scanName(name, root.storageClass, kind)
                val attributes = try {
                    Files.readAttributes(
                        child,
                        BasicFileAttributes::class.java,
                        LinkOption.NOFOLLOW_LINKS,
                    )
                } catch (ignored: IOException) {
                    unreadableItems += 1
                    continue
                }
                when {
                    attributes.isSymbolicLink -> skippedSymbolicLinks += 1
                    attributes.isDirectory -> if (directory.depth + 1 > bounds.maxDepth) {
                        boundsReached += SweepBound.MAX_DEPTH
                    } else {
                        subdirectories += Pending(child, directory.depth + 1, components)
                    }
                    attributes.isRegularFile -> {
                        if (filesVisited >= bounds.maxFiles) {
                            boundsReached += SweepBound.MAX_FILES
                            filesStopped = true
                            return
                        }
                        filesVisited += 1
                        if (allowList.allows(components, root.storageClass)) {
                            allowListedFiles += 1
                        } else {
                            scanContents(child, root.storageClass, kind)
                        }
                        if (filesStopped) {
                            return
                        }
                    }
                    else -> skippedSpecialFiles += 1
                }
            }
            subdirectories.asReversed().forEach(stack::addLast)
        }
    }

    private fun scanName(name: String, storageClass: StorageClass, kind: SweepFileKind) {
        val bytes = name.toByteArray(Charsets.UTF_8)
        record(
            matches(bytes, bytes.size, emptySet()),
            SweepLocation.FILE_NAME,
            storageClass.wireName,
            kind,
            bytes.size.toLong(),
        )
    }

    private fun scanContents(path: Path, storageClass: StorageClass, kind: SweepFileKind) {
        val stream: InputStream = try {
            Files.newInputStream(path, LinkOption.NOFOLLOW_LINKS)
        } catch (ignored: IOException) {
            unreadableItems += 1
            return
        } catch (ignored: SecurityException) {
            unreadableItems += 1
            return
        }
        val matched = mutableSetOf<Int>()
        val window = ByteArray(carryLength + PhiSentinelSweep.CHUNK_SIZE)
        var carried = 0
        var fileBytes = 0L
        stream.use { input ->
            while (true) {
                if (outOfTime()) {
                    filesStopped = true
                    break
                }
                val perFileRemaining = bounds.maxBytesPerFile - fileBytes
                val totalRemaining = bounds.maxTotalBytes - bytesScanned
                if (perFileRemaining <= 0 || totalRemaining <= 0) {
                    val hasMore = try {
                        input.read() >= 0
                    } catch (ignored: IOException) {
                        false
                    }
                    if (hasMore) {
                        truncatedFiles += 1
                        if (perFileRemaining <= 0) {
                            boundsReached += SweepBound.MAX_BYTES_PER_FILE
                        } else {
                            boundsReached += SweepBound.MAX_TOTAL_BYTES
                            filesStopped = true
                        }
                    }
                    break
                }
                val want = minOf(
                    PhiSentinelSweep.CHUNK_SIZE.toLong(),
                    perFileRemaining,
                    totalRemaining,
                ).toInt()
                val read = try {
                    input.read(window, carried, want)
                } catch (ignored: IOException) {
                    unreadableItems += 1
                    break
                }
                if (read <= 0) {
                    break
                }
                fileBytes += read
                bytesScanned += read
                val filled = carried + read
                matched += matches(window, filled, matched)
                carried = minOf(carryLength, filled)
                System.arraycopy(window, filled - carried, window, 0, carried)
            }
        }
        filesScanned += 1
        record(matched, SweepLocation.FILE_CONTENT, storageClass.wireName, kind, fileBytes)
    }

    fun sweep(reader: ProcessLogReader) {
        if (outOfTime()) {
            logSummaries += SweepLogSummary(reader.source, SweepLogStatus.SKIPPED, 0, 0)
            return
        }
        var entries = 0
        var bytes = 0L
        val status = try {
            reader.readEntries { message ->
                if (outOfTime()) {
                    return@readEntries false
                }
                if (logEntriesScanned >= bounds.maxLogEntries) {
                    boundsReached += SweepBound.MAX_LOG_ENTRIES
                    return@readEntries false
                }
                var data = message.toByteArray(Charsets.UTF_8)
                val remaining = bounds.maxLogBytes - logBytesScanned
                var exhausted = false
                if (data.size > remaining) {
                    data = data.copyOf(remaining.toInt())
                    exhausted = true
                    boundsReached += SweepBound.MAX_LOG_BYTES
                }
                entries += 1
                bytes += data.size
                logEntriesScanned += 1
                logBytesScanned += data.size
                record(
                    matches(data, data.size, emptySet()),
                    SweepLocation.LOG_ENTRY,
                    reader.source.wireName,
                    null,
                    data.size.toLong(),
                )
                !exhausted
            }
            SweepLogStatus.READ
        } catch (ignored: Exception) {
            SweepLogStatus.UNREADABLE
        }
        logSummaries += SweepLogSummary(reader.source, status, entries, bytes)
    }

    private fun matches(haystack: ByteArray, length: Int, already: Set<Int>): Set<Int> {
        if (length == 0) {
            return emptySet()
        }
        return needles.indices.filterTo(mutableSetOf()) { index ->
            index !in already && haystack.containsBytes(needles[index].bytes, length)
        }
    }

    private fun record(
        matched: Set<Int>,
        location: SweepLocation,
        surface: String,
        kind: SweepFileKind?,
        bytes: Long,
    ) {
        val keys = matched.mapTo(mutableSetOf()) { index ->
            FindingKey(needles[index].sentinelIndex, needles[index].encoding, location, surface, kind)
        }
        for (key in keys) {
            val totals = findings.getOrPut(key) { LongArray(2) }
            totals[0] += 1L
            totals[1] += bytes
        }
    }

    fun report(): PhiSentinelSweepReport {
        val reported = findings.map { (key, totals) ->
            SweepFinding(
                sentinelIdentifier = sentinels[key.sentinelIndex].identifier,
                location = key.location,
                surface = key.surface,
                fileKind = key.fileKind,
                encoding = key.encoding,
                itemCount = totals[0].toInt(),
                itemBytes = totals[1],
            )
        }.sortedBy { it.sortKey }
        val exhaustive = boundsReached.isEmpty() && unreadableItems == 0 &&
            logSummaries.all { it.status == SweepLogStatus.READ }
        return PhiSentinelSweepReport(
            sentinelIdentifiers = sentinels.map { it.identifier },
            bounds = bounds,
            roots = rootSummaries.toList(),
            logs = logSummaries.toList(),
            filesScanned = filesScanned,
            bytesScanned = bytesScanned,
            allowListedFiles = allowListedFiles,
            truncatedFiles = truncatedFiles,
            skippedSymbolicLinks = skippedSymbolicLinks,
            skippedSpecialFiles = skippedSpecialFiles,
            unreadableItems = unreadableItems,
            logEntriesScanned = logEntriesScanned,
            logBytesScanned = logBytesScanned,
            boundsReached = boundsReached.sortedBy { it.wireName },
            findings = reported,
            exhaustive = exhaustive,
        )
    }
}
