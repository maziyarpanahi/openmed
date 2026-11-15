package com.openmed.openmedkit.privacy

import android.content.Context
import java.io.File
import java.util.concurrent.TimeUnit

/** App-writable roots of the current Android app sandbox. */
object AndroidAppStorageRoots {
    /** Data-directory children swept under their own storage class (or never: `lib`). */
    internal val CLASSIFIED_DATA_CHILDREN = setOf(
        "cache",
        "code_cache",
        "databases",
        "files",
        "lib",
        "no_backup",
        "shared_prefs",
    )

    /**
     * Roots for [context]: `files`, `cache`, `code_cache`, `no_backup`, `shared_prefs`,
     * `databases`, the rest of the data directory, and every external app directory.
     */
    fun of(context: Context): List<SweepRoot> {
        val external = context.getExternalFilesDirs(null).filterNotNull()
        val externalCache = context.externalCacheDirs.filterNotNull()
        return roots(
            dataDirectory = context.dataDir,
            filesDirectory = context.filesDir,
            cacheDirectory = context.cacheDir,
            codeCacheDirectory = context.codeCacheDir,
            noBackupDirectory = context.noBackupFilesDir,
            externalFilesDirectories = external,
            externalCacheDirectories = externalCache,
        )
    }

    /** Non-overlapping roots from injected directories, for JVM tests. */
    fun roots(
        dataDirectory: File,
        filesDirectory: File = File(dataDirectory, "files"),
        cacheDirectory: File = File(dataDirectory, "cache"),
        codeCacheDirectory: File = File(dataDirectory, "code_cache"),
        noBackupDirectory: File = File(dataDirectory, "no_backup"),
        externalFilesDirectories: List<File> = emptyList(),
        externalCacheDirectories: List<File> = emptyList(),
    ): List<SweepRoot> = buildList {
        add(SweepRoot(filesDirectory, StorageClass.FILES))
        add(SweepRoot(cacheDirectory, StorageClass.CACHE))
        add(SweepRoot(codeCacheDirectory, StorageClass.CODE_CACHE))
        add(SweepRoot(noBackupDirectory, StorageClass.NO_BACKUP))
        add(SweepRoot(File(dataDirectory, "shared_prefs"), StorageClass.SHARED_PREFS))
        add(SweepRoot(File(dataDirectory, "databases"), StorageClass.DATABASES))
        add(SweepRoot(dataDirectory, StorageClass.DATA_OTHER, CLASSIFIED_DATA_CHILDREN))
        externalFilesDirectories.forEach { add(SweepRoot(it, StorageClass.EXTERNAL_FILES)) }
        externalCacheDirectories.forEach { add(SweepRoot(it, StorageClass.EXTERNAL_CACHE)) }
    }
}

/** Value-free failure raised when `logcat` cannot be read for this process. */
class LogcatUnavailableException : IllegalStateException("logcat_unavailable")

/**
 * Reads this process's own `logcat` entries (`logcat -d -v tag --pid=<pid>`).
 *
 * Apps can read their own process's log lines without `READ_LOGS`. `-d` dumps the
 * buffer and exits, so reading ends. The [launcher] seam lets JVM tests feed
 * synthetic output instead of starting a process.
 */
class LogcatProcessLogReader(
    private val pid: Int,
    private val launcher: (List<String>) -> Process = ::startProcess,
) : ProcessLogReader {
    override val source: ProcessLogSource = ProcessLogSource.LOGCAT

    /** The exact command, without shell interpretation. */
    val command: List<String>
        get() = listOf("logcat", "-d", "-v", "tag", "--pid=$pid")

    override fun readEntries(visit: (String) -> Boolean) {
        val process = launcher(command)
        try {
            var completed = true
            process.inputStream.bufferedReader(Charsets.UTF_8).use { reader ->
                while (true) {
                    val line = reader.readLine() ?: break
                    if (!visit(line)) {
                        completed = false
                        break
                    }
                }
            }
            if (completed) {
                val exited = process.waitFor(EXIT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                if (!exited || process.exitValue() != 0) {
                    throw LogcatUnavailableException()
                }
            }
        } finally {
            process.destroy()
        }
    }

    companion object {
        private const val EXIT_TIMEOUT_SECONDS = 10L

        /** A reader for the calling process; only callable on Android or Robolectric. */
        fun forCurrentProcess(): LogcatProcessLogReader =
            LogcatProcessLogReader(android.os.Process.myPid())

        private fun startProcess(command: List<String>): Process =
            ProcessBuilder(command).redirectErrorStream(true).start()
    }
}
