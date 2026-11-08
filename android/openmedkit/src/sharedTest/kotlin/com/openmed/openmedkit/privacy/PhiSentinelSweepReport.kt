package com.openmed.openmedkit.privacy

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/** Value-free file category derived from a file name's extension. */
enum class SweepFileKind(val wireName: String) {
    PLIST("plist"),
    JSON("json"),
    XML("xml"),
    SQLITE("sqlite"),
    TEXT("text"),
    OTHER("other"),
    ;

    companion object {
        /** Classify a file by extension only; the name itself is never reported. */
        fun of(fileName: String): SweepFileKind =
            when (fileName.substringAfterLast('.', "").lowercase()) {
                "plist" -> PLIST
                "json", "jsonl", "ndjson" -> JSON
                "xml" -> XML
                "db", "db-shm", "db-wal", "sqlite", "sqlite-shm", "sqlite-wal", "sqlite3", "wal", "shm" ->
                    SQLITE
                "csv", "log", "md", "txt" -> TEXT
                else -> OTHER
            }
    }
}

/** Where a sentinel was found. */
enum class SweepLocation(val wireName: String) {
    FILE_NAME("file_name"),
    FILE_CONTENT("file_content"),
    LOG_ENTRY("log_entry"),
}

/**
 * Aggregated, value-free evidence that a sentinel survived somewhere.
 *
 * Carries the sentinel identifier, never its value, the matched bytes, the file
 * name or a path.
 */
data class SweepFinding(
    val sentinelIdentifier: String,
    val location: SweepLocation,
    /** A [StorageClass] wire name for files, a [ProcessLogSource] wire name for logs. */
    val surface: String,
    /** File category for file findings; `null` for log entries. */
    val fileKind: SweepFileKind?,
    val encoding: SentinelEncoding,
    /** Matching files, names or log entries. */
    val itemCount: Int,
    /** Scanned bytes of the matching items (name bytes for name matches). */
    val itemBytes: Long,
) {
    internal val sortKey: String
        get() = listOf(
            sentinelIdentifier,
            location.wireName,
            surface,
            fileKind?.wireName.orEmpty(),
            encoding.wireName,
        ).joinToString("\u001f")
}

/** Per-root counters, labelled only by storage class. */
data class SweepRootSummary(
    val storageClass: StorageClass,
    val present: Boolean,
    val filesScanned: Int,
    val bytesScanned: Long,
)

/** Whether a log reader was read to completion, failed or was skipped by a bound. */
enum class SweepLogStatus(val wireName: String) {
    READ("read"),
    UNREADABLE("unreadable"),
    SKIPPED("skipped"),
}

/** Per-reader counters, labelled only by log source. */
data class SweepLogSummary(
    val source: ProcessLogSource,
    val status: SweepLogStatus,
    val entriesScanned: Int,
    val bytesScanned: Long,
)

/**
 * Value-free result of one sentinel sweep.
 *
 * Contains counts, storage classes, file kinds, log sources, encodings and sentinel
 * identifiers only. It never contains sentinel values, file contents, file names or
 * absolute paths, so it is safe to attach to CI artifacts.
 */
data class PhiSentinelSweepReport(
    val sentinelIdentifiers: List<String>,
    val bounds: SweepBounds,
    val roots: List<SweepRootSummary>,
    val logs: List<SweepLogSummary>,
    val filesScanned: Int,
    val bytesScanned: Long,
    val allowListedFiles: Int,
    val truncatedFiles: Int,
    val skippedSymbolicLinks: Int,
    val skippedSpecialFiles: Int,
    val unreadableItems: Int,
    val logEntriesScanned: Int,
    val logBytesScanned: Long,
    val boundsReached: List<SweepBound>,
    val findings: List<SweepFinding>,
    /** `true` when no bound was reached, nothing was unreadable and every reader was read. */
    val exhaustive: Boolean,
) {
    /** `true` when no sentinel was found in any scanned name, file or log entry. */
    val clean: Boolean
        get() = findings.isEmpty()

    /** Sorted-key JSON with the same snake_case keys as the Swift report; durations in seconds. */
    fun toJson(): String = buildJsonObject {
        put("allow_listed_files", allowListedFiles)
        put(
            "bounds",
            buildJsonObject {
                put("max_bytes_per_file", bounds.maxBytesPerFile)
                put("max_depth", bounds.maxDepth)
                put("max_duration", bounds.maxDurationMillis / 1_000.0)
                put("max_files", bounds.maxFiles)
                put("max_log_bytes", bounds.maxLogBytes)
                put("max_log_entries", bounds.maxLogEntries)
                put("max_total_bytes", bounds.maxTotalBytes)
            },
        )
        put("bounds_reached", JsonArray(boundsReached.map { JsonPrimitive(it.wireName) }))
        put("bytes_scanned", bytesScanned)
        put("clean", clean)
        put("encodings", JsonArray(SentinelEncoding.values().map { JsonPrimitive(it.wireName) }))
        put("exhaustive", exhaustive)
        put("files_scanned", filesScanned)
        put(
            "findings",
            JsonArray(
                findings.map { finding ->
                    buildJsonObject {
                        put("encoding", finding.encoding.wireName)
                        finding.fileKind?.let { put("file_kind", it.wireName) }
                        put("item_bytes", finding.itemBytes)
                        put("item_count", finding.itemCount)
                        put("location", finding.location.wireName)
                        put("sentinel_identifier", finding.sentinelIdentifier)
                        put("surface", finding.surface)
                    }
                },
            ),
        )
        put("log_bytes_scanned", logBytesScanned)
        put("log_entries_scanned", logEntriesScanned)
        put(
            "logs",
            JsonArray(
                logs.map { log ->
                    buildJsonObject {
                        put("bytes_scanned", log.bytesScanned)
                        put("entries_scanned", log.entriesScanned)
                        put("source", log.source.wireName)
                        put("status", log.status.wireName)
                    }
                },
            ),
        )
        put(
            "roots",
            JsonArray(
                roots.map { root ->
                    buildJsonObject {
                        put("bytes_scanned", root.bytesScanned)
                        put("files_scanned", root.filesScanned)
                        put("present", root.present)
                        put("storage_class", root.storageClass.wireName)
                    }
                },
            ),
        )
        put("schema", SCHEMA)
        put("sentinel_identifiers", JsonArray(sentinelIdentifiers.map { JsonPrimitive(it) }))
        put("skipped_special_files", skippedSpecialFiles)
        put("skipped_symbolic_links", skippedSymbolicLinks)
        put("truncated_files", truncatedFiles)
        put("unreadable_items", unreadableItems)
    }.toString()

    override fun toString(): String {
        val findingText = findings.joinToString(", ") { finding ->
            val kind = finding.fileKind?.let { "/${it.wireName}" }.orEmpty()
            "${finding.sentinelIdentifier} ${finding.location.wireName} " +
                "${finding.surface}$kind ${finding.encoding.wireName} x${finding.itemCount}"
        }
        return "PhiSentinelSweepReport(clean: $clean, exhaustive: $exhaustive, " +
            "findings: [$findingText], " +
            "bounds_reached: [${boundsReached.joinToString(", ") { it.wireName }}], " +
            "files: $filesScanned, bytes: $bytesScanned, allow_listed: $allowListedFiles, " +
            "unreadable: $unreadableItems, log_entries: $logEntriesScanned)"
    }

    companion object {
        const val SCHEMA = "openmed.phi_sentinel_sweep.v1"
    }
}
