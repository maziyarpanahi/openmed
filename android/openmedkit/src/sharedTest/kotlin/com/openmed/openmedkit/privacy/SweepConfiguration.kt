package com.openmed.openmedkit.privacy

import java.io.File

/** Value-free class of an app-writable storage root. */
enum class StorageClass(val wireName: String) {
    FILES("files"),
    CACHE("cache"),
    CODE_CACHE("code_cache"),
    NO_BACKUP("no_backup"),
    SHARED_PREFS("shared_prefs"),
    DATABASES("databases"),

    /** Data-directory children not covered by another class, such as `app_*` dirs. */
    DATA_OTHER("data_other"),
    EXTERNAL_FILES("external_files"),
    EXTERNAL_CACHE("external_cache"),
}

/** One directory tree to sweep, labelled with its value-free storage class. */
class SweepRoot(
    val directory: File,
    val storageClass: StorageClass,
    /** Direct children of [directory] to skip because another root covers them. */
    val excludedChildNames: Set<String> = emptySet(),
)

/** The bound that stopped or shortened a sweep. */
enum class SweepBound(val wireName: String) {
    MAX_FILES("max_files"),
    MAX_DEPTH("max_depth"),
    MAX_BYTES_PER_FILE("max_bytes_per_file"),
    MAX_TOTAL_BYTES("max_total_bytes"),
    MAX_DURATION("max_duration"),
    MAX_LOG_ENTRIES("max_log_entries"),
    MAX_LOG_BYTES("max_log_bytes"),
}

/**
 * File, byte, depth, time and log bounds for one sweep.
 *
 * Reaching a bound is listed in [PhiSentinelSweepReport.boundsReached] and makes the
 * report non-exhaustive.
 */
data class SweepBounds(
    /** Regular files visited across all roots, including allow-listed files. */
    val maxFiles: Int = 5_000,
    /** Directory levels below a root that are entered; `0` scans only direct children. */
    val maxDepth: Int = 12,
    /** Content bytes read from one file; later bytes are not scanned. */
    val maxBytesPerFile: Long = 8_388_608L,
    /** Content bytes read across all files. */
    val maxTotalBytes: Long = 134_217_728L,
    /** Milliseconds for files and logs together, measured by the injected clock. */
    val maxDurationMillis: Long = 30_000L,
    /** Log entries read across all log readers. */
    val maxLogEntries: Int = 20_000,
    /** UTF-8 log bytes read across all log readers. */
    val maxLogBytes: Long = 16_777_216L,
) {
    internal fun validate() {
        val ceiling = CEILING
        val valid = maxFiles in 1..ceiling.maxFiles &&
            maxDepth in 0..ceiling.maxDepth &&
            maxBytesPerFile in 1..ceiling.maxBytesPerFile &&
            maxTotalBytes in 1..ceiling.maxTotalBytes &&
            maxDurationMillis in 1..ceiling.maxDurationMillis &&
            maxLogEntries in 1..ceiling.maxLogEntries &&
            maxLogBytes in 1..ceiling.maxLogBytes
        if (!valid) {
            throw PhiSentinelSweepException(SweepRefusal.BOUNDS_INVALID)
        }
    }

    companion object {
        /** Defaults for a test-host app sandbox. */
        val STANDARD = SweepBounds()

        /** Hard ceilings no configuration may exceed. */
        val CEILING = SweepBounds(
            maxFiles = 100_000,
            maxDepth = 64,
            maxBytesPerFile = 268_435_456L,
            maxTotalBytes = 1_073_741_824L,
            maxDurationMillis = 600_000L,
            maxLogEntries = 1_000_000,
            maxLogBytes = 268_435_456L,
        )
    }
}

/**
 * Explicit model-artifact allow-list: matching files keep their names swept but
 * their contents skipped, because weights and vocabularies hold arbitrary bytes.
 */
class ArtifactAllowList(
    fileExtensions: Set<String> = emptySet(),
    /** Storage-class-relative path prefixes, matched on whole `/` components. */
    val pathPrefixes: Map<StorageClass, Set<String>> = emptyMap(),
) {
    /** Lowercased extensions without a dot, matched on any path component. */
    val fileExtensions: Set<String> = fileExtensions.map { it.lowercase() }.toSet()

    internal fun allows(components: List<String>, storageClass: StorageClass): Boolean {
        if (components.any { it.substringAfterLast('.', "").lowercase() in fileExtensions }) {
            return true
        }
        val relative = components.joinToString("/")
        return pathPrefixes[storageClass].orEmpty().any { prefix ->
            relative == prefix || relative.startsWith("$prefix/")
        }
    }

    companion object {
        /** Nothing is allow-listed: every file's contents are swept. */
        val NONE = ArtifactAllowList()

        /** Model bundles and OpenMedKit's own `no_backup` model cache. */
        val MODEL_ARTIFACTS = ArtifactAllowList(
            fileExtensions = setOf(
                "gguf", "mlmodel", "mlmodelc", "mlpackage", "onnx", "ort", "safetensors", "tflite",
            ),
            pathPrefixes = mapOf(StorageClass.NO_BACKUP to setOf("openmed-model-cache")),
        )
    }
}

/** Value-free class of the log stream a reader returns. */
enum class ProcessLogSource(val wireName: String) {
    /** `logcat` filtered to the current process id. */
    LOGCAT("logcat"),

    /** Robolectric's in-memory `android.util.Log` capture. */
    ROBOLECTRIC_SHADOW_LOG("robolectric_shadow_log"),

    /** Lines captured by an injected sink or fake reader. */
    CAPTURED("captured"),
}

/** Reads the current process's own log entries for a sweep; implementations only read. */
interface ProcessLogReader {
    val source: ProcessLogSource

    /** Visit entries in order until [visit] returns `false` or entries run out. */
    fun readEntries(visit: (String) -> Boolean)
}

/** A reader over an in-memory list of log lines, for tests and captured sinks. */
class CapturedLogReader(
    private val lines: List<String>,
    override val source: ProcessLogSource = ProcessLogSource.CAPTURED,
) : ProcessLogReader {
    override fun readEntries(visit: (String) -> Boolean) {
        for (line in lines) {
            if (!visit(line)) {
                return
            }
        }
    }
}
