package com.openmed.openmedkit.download

/**
 * One value-free `openmed.model.lifecycle.v1` event body, as described by
 * `model_lifecycle_event.schema.json` and shared with Python and the other kits.
 *
 * Unknown values are `null`. Identifiers that do not match the schema shapes
 * (a `namespace/name` repository id, a 40- or 64-hex commit, a
 * `sha256:`/`git-sha1:` digest, a snake_case reason) are recorded as `null`, so
 * an event never carries a URL, filename, path, token or exception text.
 *
 * @property event One of `requested`, `fetching`, `verified`,
 *   `integrity_failed`, `ready`, `activated`, `rolled_back`, `evicted`, `failed`.
 */
class ModelLifecycleEvent internal constructor(
    val event: String,
    repoId: String?,
    revisionSha: String?,
    val fileCount: Int?,
    val byteCount: Long?,
    digest: String?,
    reason: String?,
) {
    val schemaVersion: String get() = SCHEMA_VERSION
    val repoId: String? = repoId?.takeIf(REPO_ID::matches)
    val revisionSha: String? = revisionSha?.takeIf(REVISION::matches)
    val digest: String? = digest?.takeIf(DIGEST::matches)
    val reason: String? = reason?.takeIf(REASON::matches)

    /** The event body as compact JSON with the schema's field order. */
    fun toJson(): String =
        "{\"schema_version\":\"$SCHEMA_VERSION\",\"event\":\"$event\",\"repo_id\":${quoted(repoId)}," +
            "\"revision_sha\":${quoted(revisionSha)},\"file_count\":$fileCount,\"byte_count\":$byteCount," +
            "\"digest\":${quoted(digest)},\"reason\":${quoted(reason)}}"

    override fun toString(): String = toJson()

    companion object {
        const val SCHEMA_VERSION: String = "openmed.model.lifecycle.v1"

        private val REPO_ID = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
        private val REVISION = Regex("[0-9a-f]{40}|[0-9a-f]{64}")
        private val DIGEST = Regex("sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40}")
        private val REASON = Regex("[a-z][a-z0-9_]{0,63}")

        private fun quoted(value: String?): String = if (value == null) "null" else "\"$value\""
    }
}

/**
 * Receives [ModelLifecycleEvent]s from [ModelDownloader]. An exception thrown by
 * the sink is ignored and never changes the download outcome.
 */
fun interface ModelLifecycleSink {
    fun emit(event: ModelLifecycleEvent)

    companion object {
        /** Drops every event. */
        val NONE: ModelLifecycleSink = ModelLifecycleSink {}
    }
}

/**
 * Byte progress of one file of a [ModelDownloader.download]. Values for a file
 * only grow and end at the file's size; files are identified by index only.
 */
fun interface ModelDownloadProgressListener {
    /**
     * @param fileIndex Zero-based index of the file in download order.
     * @param fileCount Number of files being downloaded.
     * @param bytesTransferred Bytes of this file written so far.
     * @param fileSizeBytes The published size of the file, or `null`.
     */
    fun onProgress(fileIndex: Int, fileCount: Int, bytesTransferred: Long, fileSizeBytes: Long?)

    companion object {
        /** Ignores progress. */
        val NONE: ModelDownloadProgressListener = ModelDownloadProgressListener { _, _, _, _ -> }
    }
}
