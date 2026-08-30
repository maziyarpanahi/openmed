package org.openmed.maple

import android.content.Context
import android.os.StatFs
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong
import java.util.zip.ZipInputStream

data class InstalledMapleBundle(
    val root: File,
    val manifest: MapleBundleManifest,
) {
    val prefillModel: File = manifest.resolve(root, manifest.graphs.prefillPath)
    val decodeModel: File? = manifest.graphs.decodePath?.let { manifest.resolve(root, it) }
}

data class MapleImportProgress(
    val fileName: String,
    val copiedBytes: Long,
    val totalBytes: Long,
) {
    val fraction: Float = if (totalBytes <= 0L) {
        0f
    } else {
        (copiedBytes.toDouble() / totalBytes.toDouble()).toFloat().coerceIn(0f, 1f)
    }
}

class MapleBundleInstaller(context: Context) {
    private val storageRoot = File(context.noBackupFilesDir, "openmed-maple")
    private val bundlesRoot = File(storageRoot, "bundles")
    private val activePointer = File(storageRoot, "active-bundle")

    fun install(
        source: InputStream,
        onProgress: (MapleImportProgress) -> Unit = {},
    ): InstalledMapleBundle {
        storageRoot.mkdirs()
        bundlesRoot.mkdirs()
        val staging = File(storageRoot, "staging-${UUID.randomUUID()}")
        requireBundle(staging.mkdirs(), "Could not create protected model staging storage")

        var installed = false
        try {
            val buffered = BufferedInputStream(source, COPY_BUFFER_SIZE)
            ZipInputStream(buffered).use { zip ->
                val manifestEntry = zip.nextEntry
                    ?: throw MapleBundleException("The selected bundle is empty")
                requireBundle(
                    !manifestEntry.isDirectory && manifestEntry.name == MANIFEST_NAME,
                    "$MANIFEST_NAME must be the first ZIP entry",
                )
                val manifestBytes = zip.readLimited(MAX_MANIFEST_BYTES)
                val manifestText = manifestBytes.toString(Charsets.UTF_8)
                val manifest = MapleBundleManifestParser.parse(manifestText)
                checkAvailableSpace(manifest.totalSizeBytes)

                val declared = manifest.files.associateBy { it.path }
                val seen = mutableSetOf<String>()
                val copiedTotal = AtomicLong(0L)
                while (true) {
                    val entry = zip.nextEntry ?: break
                    if (entry.isDirectory) {
                        validateRelativePath(entry.name.removeSuffix("/"))
                        continue
                    }
                    val specification = declared[entry.name]
                        ?: throw MapleBundleException("Bundle contains undeclared file: ${entry.name}")
                    requireBundle(seen.add(entry.name), "Duplicate ZIP entry: ${entry.name}")
                    val destination = resolveInside(staging, specification.path)
                    destination.parentFile?.mkdirs()
                    val digest = MessageDigest.getInstance("SHA-256")
                    var copiedFile = 0L
                    FileOutputStream(destination).buffered(COPY_BUFFER_SIZE).use { output ->
                        val buffer = ByteArray(COPY_BUFFER_SIZE)
                        while (true) {
                            val count = zip.read(buffer)
                            if (count == -1) break
                            copiedFile += count
                            val currentTotal = copiedTotal.addAndGet(count.toLong())
                            requireBundle(
                                copiedFile <= specification.sizeBytes,
                                "Bundle file is larger than declared: ${specification.path}",
                            )
                            requireBundle(
                                currentTotal <= manifest.totalSizeBytes,
                                "Bundle payload is larger than declared",
                            )
                            digest.update(buffer, 0, count)
                            output.write(buffer, 0, count)
                            onProgress(
                                MapleImportProgress(
                                    fileName = specification.path,
                                    copiedBytes = currentTotal,
                                    totalBytes = manifest.totalSizeBytes,
                                ),
                            )
                        }
                    }
                    requireBundle(
                        copiedFile == specification.sizeBytes,
                        "Bundle file size mismatch: ${specification.path}",
                    )
                    requireBundle(
                        digest.hexDigest() == specification.sha256,
                        "Bundle checksum mismatch: ${specification.path}",
                    )
                }
                requireBundle(
                    seen == declared.keys,
                    "Bundle is missing one or more declared files",
                )

                File(staging, MANIFEST_NAME).writeText(manifestText)
                val manifestDigest = MessageDigest.getInstance("SHA-256")
                    .digest(manifestBytes)
                    .toHex()
                File(staging, VERIFIED_MARKER).writeText(manifestDigest)
                val directoryName = "${manifest.sourceRevision.take(12)}-${manifestDigest.take(12)}"
                val destination = File(bundlesRoot, directoryName)
                if (destination.exists()) {
                    requireBundle(isReady(destination), "Existing bundle directory is not verified")
                    staging.deleteRecursively()
                } else {
                    requireBundle(
                        staging.renameTo(destination),
                        "Could not promote the verified Maple bundle",
                    )
                }
                writeActivePointer(directoryName)
                installed = true
                return InstalledMapleBundle(destination, manifest)
            }
        } finally {
            if (!installed && staging.exists()) {
                staging.deleteRecursively()
            }
        }
    }

    fun activeBundle(): InstalledMapleBundle? {
        if (!activePointer.isFile) return null
        val directoryName = activePointer.readText().trim()
        if (!SAFE_DIRECTORY.matches(directoryName)) return null
        val directory = File(bundlesRoot, directoryName)
        if (!isReady(directory)) return null
        return runCatching {
            val manifestText = File(directory, MANIFEST_NAME).readText()
            val manifestDigest = MessageDigest.getInstance("SHA-256")
                .digest(manifestText.toByteArray())
                .toHex()
            requireBundle(
                File(directory, VERIFIED_MARKER).readText().trim() == manifestDigest,
                "Verified marker does not match the bundle manifest",
            )
            val manifest = MapleBundleManifestParser.parse(manifestText)
            manifest.files.forEach { file ->
                val candidate = manifest.resolve(directory, file.path)
                requireBundle(
                    candidate.isFile && candidate.length() == file.sizeBytes,
                    "Installed bundle file is missing or changed",
                )
            }
            InstalledMapleBundle(directory, manifest)
        }.getOrNull()
    }

    private fun isReady(directory: File): Boolean =
        directory.isDirectory &&
            File(directory, MANIFEST_NAME).isFile &&
            File(directory, VERIFIED_MARKER).isFile

    private fun writeActivePointer(directoryName: String) {
        val temporary = File(storageRoot, "active-${UUID.randomUUID()}")
        temporary.writeText(directoryName)
        if (activePointer.exists() && !activePointer.delete()) {
            temporary.delete()
            throw MapleBundleException("Could not update the active bundle pointer")
        }
        requireBundle(temporary.renameTo(activePointer), "Could not activate the Maple bundle")
    }

    private fun checkAvailableSpace(payloadBytes: Long) {
        val available = StatFs(storageRoot.absolutePath).availableBytes
        requireBundle(
            available >= payloadBytes + STORAGE_HEADROOM_BYTES,
            "Not enough protected app storage for this bundle",
        )
    }

    private fun ZipInputStream.readLimited(limit: Int): ByteArray {
        val output = java.io.ByteArrayOutputStream()
        val buffer = ByteArray(16 * 1024)
        var total = 0
        while (true) {
            val count = read(buffer)
            if (count == -1) break
            total += count
            requireBundle(total <= limit, "$MANIFEST_NAME is too large")
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    private fun MessageDigest.hexDigest(): String = digest().toHex()

    private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it) }

    private companion object {
        const val MANIFEST_NAME = "maple-bundle.json"
        const val VERIFIED_MARKER = ".verified"
        const val MAX_MANIFEST_BYTES = 1024 * 1024
        const val COPY_BUFFER_SIZE = 1024 * 1024
        const val STORAGE_HEADROOM_BYTES = 512L * 1024L * 1024L
        val SAFE_DIRECTORY = Regex("[0-9a-f]{12}-[0-9a-f]{12}")
    }
}
