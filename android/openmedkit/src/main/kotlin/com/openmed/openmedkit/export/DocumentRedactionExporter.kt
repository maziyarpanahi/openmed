package com.openmed.openmedkit.export

import android.content.ClipData
import android.content.Context
import android.content.Intent
import android.net.Uri
import com.openmed.openmedkit.export.DocumentRedactionExportContract.fail
import com.openmed.openmedkit.intake.BitmapDocumentImageSource
import com.openmed.openmedkit.intake.DocumentBox
import com.openmed.openmedkit.intake.DocumentImageSource
import com.openmed.openmedkit.intake.DocumentInputKind
import com.openmed.openmedkit.intake.DocumentIntake
import com.openmed.openmedkit.intake.DocumentIntakeException
import com.openmed.openmedkit.intake.DocumentIntakeLimits
import com.openmed.openmedkit.intake.DocumentIntakeOutcome
import com.openmed.openmedkit.intake.DocumentPageGeometry
import com.openmed.openmedkit.intake.PdfPageEvent
import com.openmed.openmedkit.intake.PdfPageRenderer
import com.openmed.openmedkit.intake.PdfPageSource
import com.openmed.openmedkit.intake.documentSourceDigest
import com.openmed.openmedkit.ocr.OcrAdapter
import com.openmed.openmedkit.ocr.OcrImage
import com.openmed.openmedkit.ocr.OcrResult
import java.io.Closeable
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive

/**
 * Exports verified redactions as new image-only PDFs in private cache storage.
 *
 * [export] renders every source page through the same [PdfPageSource] or
 * [DocumentImageSource] used for intake, paints each mask opaque, and writes a
 * new PDF whose pages are only those pixels. Nothing from the original file is
 * copied: no text layer, annotation, attachment or metadata. The stored file is
 * then read back and verified before it can be offered: it must match the
 * profile exactly, re-intake through [DocumentIntake] with [ocrAdapter] must be
 * complete with every painted mask opaque, and the re-extracted text must not
 * contain any masked identifier.
 *
 * The exporter never shares anything itself. [shareIntent], [contentUri] and
 * [saveTo] are explicit caller actions; call [complete] when that flow ends.
 * Artifacts are deleted when the export is cancelled or fails, when they are
 * completed or invalidated, on [close], and by the next exporter created after
 * a process restart.
 *
 * Shared content URIs are served by [DocumentRedactionExportProvider], which
 * the app declares in its manifest; see the provider for the declaration.
 *
 * @param ocrAdapter On-device OCR used to re-extract the artifact.
 * @param pdfPageSource Renders PDF sources; use the one passed to intake.
 * @param imageSource Decodes image sources; use the one passed to intake.
 * @param limits Byte and page bounds for the artifact and its re-intake.
 * @param verificationPdfPageSource Reads the artifact back for verification.
 *   The default decodes the export profile; `PdfPageRenderer()` reads it with
 *   the platform renderer instead.
 */
class DocumentRedactionExporter(
    context: Context,
    private val ocrAdapter: OcrAdapter,
    private val pdfPageSource: PdfPageSource = PdfPageRenderer(),
    private val imageSource: DocumentImageSource = BitmapDocumentImageSource(),
    private val limits: DocumentIntakeLimits = DocumentIntakeLimits(),
    private val verificationPdfPageSource: PdfPageSource = DocumentRedactionPdfPageSource(),
) : Closeable {
    private val context: Context = context.applicationContext ?: context
    private val sessionName = DocumentRedactionExportContract.randomReference()
    private val session: File
    private val lock = Any()
    private val entries = mutableMapOf<String, Entry>()

    private class Entry(
        val file: File,
        val receipt: DocumentRedactionReceipt,
        val authorities: MutableSet<String> = mutableSetOf(),
    )

    init {
        val root = exportRoot(this.context)
        session = File(root, sessionName)
        if (!(root.isDirectory || root.mkdirs())) {
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        }
        DocumentRedactionSessions.register(sessionName, root)
        if (!session.mkdir()) {
            DocumentRedactionSessions.remove(sessionName)
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        }
    }

    /** Number of verified artifacts that can still be shared. */
    val artifactCount: Int
        get() = synchronized(lock) { entries.size }

    /**
     * Renders, writes and verifies a sanitized artifact for [request].
     *
     * @param source The document the report was produced from. Its digest, page
     *   count and page geometry must still match the report.
     * @throws DocumentRedactionExportException with a value-free reason; a
     *   failed export leaves no file. Coroutine cancellation is rethrown after
     *   partial output is removed.
     */
    suspend fun export(request: DocumentRedactionRequest, source: Uri): DocumentRedactionArtifact {
        currentCoroutineContext().ensureActive()
        val plan = DocumentRedactionExportContract.plan(request)
        val kind = request.report.result?.inputKind ?: fail(DocumentRedactionExportReason.INCOMPLETE_REPORT)
        val id = DocumentRedactionExportContract.randomReference()
        val pending = File(session, "pending-$id.pdf")
        val final = File(session, "$id.pdf")
        var exported = false
        try {
            if (digest(source) != request.report.evidence.sourceDigest) {
                fail(DocumentRedactionExportReason.SOURCE_CHANGED)
            }
            val rendered = when (kind) {
                DocumentInputKind.PDF -> renderPdf(plan, request.policy, source)
                DocumentInputKind.IMAGE -> renderImage(plan, request.policy, source)
            }
            val data = DocumentRedactionPdf.write(rendered.pages)
            if (data.size > limits.maxDocumentBytes) {
                fail(DocumentRedactionExportReason.RESOURCE_LIMIT)
            }
            store(pending, data)
            val stored = try {
                pending.readBytes()
            } catch (_: IOException) {
                fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
            }
            val receipt = verify(stored, pending, request, rendered)
            currentCoroutineContext().ensureActive()
            if (!pending.renameTo(final)) {
                fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
            }
            synchronized(lock) { entries[id] = Entry(final, receipt) }
            exported = true
            return DocumentRedactionArtifact(id = id, receipt = receipt)
        } catch (error: CancellationException) {
            throw error
        } catch (error: DocumentRedactionExportException) {
            throw error
        } catch (_: IOException) {
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        } catch (_: RuntimeException) {
            fail(DocumentRedactionExportReason.RENDER_FAILED)
        } finally {
            if (!exported) {
                pending.delete()
                final.delete()
            }
        }
    }

    /**
     * The explicit share action: returns a content URI for the verified file,
     * served read-only by [DocumentRedactionExportProvider] under [authority].
     * Grant it only to the target the user chooses, as [shareIntent] does.
     *
     * @param request The caller's current redaction state. When its binding
     *   differs from the artifact's, the artifact is deleted.
     * @throws DocumentRedactionExportException with
     *   [DocumentRedactionExportReason.ARTIFACT_UNAVAILABLE],
     *   [DocumentRedactionExportReason.STALE_REQUEST] or
     *   [DocumentRedactionExportReason.ARTIFACT_CHANGED]; the original document
     *   is never returned in place of the artifact.
     */
    fun contentUri(artifact: DocumentRedactionArtifact, request: DocumentRedactionRequest, authority: String): Uri {
        require(authority.isNotBlank()) { "authority must be non-blank" }
        val entry = current(artifact, request).first
        synchronized(lock) { entry.authorities += authority }
        return Uri.Builder()
            .scheme("content")
            .authority(authority)
            .appendPath(sessionName)
            .appendPath("${artifact.id}.pdf")
            .build()
    }

    /**
     * Builds a chooser for an `ACTION_SEND` of the verified PDF. The target
     * intent carries only a read grant, so the platform grants read access to
     * the one target the user picks. Nothing is sent until the caller starts
     * the returned intent.
     */
    fun shareIntent(
        artifact: DocumentRedactionArtifact,
        request: DocumentRedactionRequest,
        authority: String,
        title: CharSequence? = null,
    ): Intent {
        val uri = contentUri(artifact, request, authority)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = MIME_TYPE
            putExtra(Intent.EXTRA_STREAM, uri)
            clipData = ClipData.newRawUri(null, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(send, title)
    }

    /**
     * The explicit save action: copies the verified bytes to [destination], for
     * example a URI the user picked with `ACTION_CREATE_DOCUMENT`.
     */
    fun saveTo(artifact: DocumentRedactionArtifact, request: DocumentRedactionRequest, destination: Uri) {
        val bytes = current(artifact, request).second
        try {
            val output = context.contentResolver.openOutputStream(destination, "wt")
                ?: fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
            output.use { it.write(bytes) }
        } catch (_: IOException) {
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        } catch (_: SecurityException) {
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        }
    }

    /** Revokes grants and deletes an artifact after its share or save flow ends. */
    fun complete(artifact: DocumentRedactionArtifact) = remove(artifact.id)

    /**
     * Deletes every artifact whose binding differs from [request], for example
     * after the source, masks or policy changed.
     */
    fun invalidate(keeping: DocumentRedactionRequest) {
        val stale = synchronized(lock) {
            entries.filterValues { it.receipt.binding != keeping.binding }.keys.toList()
        }
        stale.forEach(::remove)
    }

    /** Deletes every artifact. */
    fun invalidateAll() {
        synchronized(lock) { entries.keys.toList() }.forEach(::remove)
    }

    /** Deletes every artifact and this exporter's private session directory. */
    override fun close() {
        invalidateAll()
        session.deleteRecursively()
        DocumentRedactionSessions.remove(sessionName)
    }

    private fun remove(id: String) {
        val entry = synchronized(lock) { entries.remove(id) } ?: return
        entry.authorities.forEach { authority ->
            val uri = Uri.Builder().scheme("content").authority(authority)
                .appendPath(sessionName).appendPath("$id.pdf").build()
            try {
                context.revokeUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
            } catch (_: RuntimeException) {
                // Nothing was granted under this authority.
            }
        }
        entry.file.delete()
    }

    private fun current(
        artifact: DocumentRedactionArtifact,
        request: DocumentRedactionRequest,
    ): Pair<Entry, ByteArray> {
        val entry = synchronized(lock) { entries[artifact.id] }
            ?.takeIf { it.receipt == artifact.receipt }
            ?: fail(DocumentRedactionExportReason.ARTIFACT_UNAVAILABLE)
        if (request.binding != entry.receipt.binding) {
            remove(artifact.id)
            fail(DocumentRedactionExportReason.STALE_REQUEST)
        }
        val bytes = try {
            entry.file.readBytes()
        } catch (_: IOException) {
            null
        }
        if (bytes == null || DocumentRedactionExportContract.sha256(bytes) != entry.receipt.outputDigest) {
            remove(artifact.id)
            fail(DocumentRedactionExportReason.ARTIFACT_CHANGED)
        }
        return entry to bytes
    }

    private fun digest(source: Uri): String =
        try {
            documentSourceDigest(context, source, limits.maxDocumentBytes)
        } catch (_: DocumentIntakeException) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        } catch (_: SecurityException) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        } catch (_: IOException) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        }

    private class Rendered {
        val pages = mutableListOf<DocumentRedactionPdf.Page>()

        /** Painted rectangles per page, normalized to the page image size. */
        val expectations = mutableListOf<List<DocumentBox>>()
        var painted = 0
    }

    private suspend fun renderPdf(
        plan: List<DocumentRedactionPagePlan>,
        policy: DocumentRedactionPolicy,
        source: Uri,
    ): Rendered {
        val rendered = Rendered()
        try {
            pdfPageSource.renderPages(context, source) { event ->
                currentCoroutineContext().ensureActive()
                val index = rendered.pages.size
                when (event) {
                    is PdfPageEvent.Unavailable -> fail(DocumentRedactionExportReason.RENDER_FAILED)
                    is PdfPageEvent.Rendered -> {
                        val page = event.page
                        val bitmap = page.bitmap
                        val geometry = page.geometry ?: DocumentPageGeometry(
                            width = bitmap.width.toDouble(),
                            height = bitmap.height.toDouble(),
                        )
                        if (page.page != index || index >= plan.size || geometry != plan[index].geometry) {
                            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
                        }
                        paint(rendered, plan[index], policy, pixels(bitmap), bitmap.width, bitmap.height)
                    }
                }
            }
        } catch (_: DocumentIntakeException) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        } catch (_: SecurityException) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        }
        if (rendered.pages.size != plan.size) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        }
        return rendered
    }

    private suspend fun renderImage(
        plan: List<DocumentRedactionPagePlan>,
        policy: DocumentRedactionPolicy,
        source: Uri,
    ): Rendered {
        val loaded = try {
            imageSource.loadPage(context, source)
        } catch (error: CancellationException) {
            throw error
        } catch (_: Exception) {
            fail(DocumentRedactionExportReason.SOURCE_CHANGED)
        }
        val rendered = Rendered()
        try {
            if (plan.size != 1 || loaded.geometry != plan.single().geometry) {
                fail(DocumentRedactionExportReason.SOURCE_CHANGED)
            }
            val bitmap = loaded.image.bitmap
            val upright = DocumentRedactionRaster.rotate(
                pixels(bitmap),
                bitmap.width,
                bitmap.height,
                loaded.image.rotationDegrees,
            )
            paint(rendered, plan.single(), policy, upright.pixels, upright.width, upright.height)
        } finally {
            loaded.image.bitmap.recycle()
        }
        return rendered
    }

    private fun pixels(bitmap: android.graphics.Bitmap): IntArray {
        if (bitmap.width.toLong() * bitmap.height > MAX_PIXELS) {
            fail(DocumentRedactionExportReason.RESOURCE_LIMIT)
        }
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return pixels
    }

    private fun paint(
        rendered: Rendered,
        page: DocumentRedactionPagePlan,
        policy: DocumentRedactionPolicy,
        pixels: IntArray,
        width: Int,
        height: Int,
    ) {
        val rects = page.boxes.mapNotNull { box ->
            DocumentRedactionExportContract.imageBox(box, page.geometry, width, height)?.let {
                DocumentRedactionExportContract.pixelRect(it, policy.padding, width, height)
            }
        }
        DocumentRedactionRaster.flatten(pixels)
        DocumentRedactionRaster.paint(pixels, width, rects)
        val geometry = page.geometry
        val crop = geometry.crop ?: DocumentBox(0.0, 0.0, geometry.width, geometry.height)
        val quarter = geometry.rotationDegrees == 90 || geometry.rotationDegrees == 270
        val cropWidth = crop.right - crop.left
        val cropHeight = crop.bottom - crop.top
        rendered.pages += DocumentRedactionPdf.page(
            pixels,
            width,
            height,
            if (quarter) cropHeight else cropWidth,
            if (quarter) cropWidth else cropHeight,
        )
        rendered.expectations += rects.map {
            DocumentBox(
                it[0].toDouble() / width,
                it[1].toDouble() / height,
                it[2].toDouble() / width,
                it[3].toDouble() / height,
            )
        }
        rendered.painted += rects.size
    }

    private fun store(file: File, data: ByteArray) {
        try {
            if (!file.createNewFile()) {
                fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
            }
            FileOutputStream(file).use { output ->
                output.write(data)
                output.fd.sync()
            }
        } catch (_: IOException) {
            fail(DocumentRedactionExportReason.STORAGE_UNAVAILABLE)
        }
    }

    private suspend fun verify(
        data: ByteArray,
        file: File,
        request: DocumentRedactionRequest,
        rendered: Rendered,
    ): DocumentRedactionReceipt {
        val pages = DocumentRedactionPdf.inspect(data, MAX_PIXELS)
        if (pages.size != rendered.pages.size) {
            fail(DocumentRedactionExportReason.PAGE_COUNT_MISMATCH)
        }
        if (pages != rendered.pages) {
            fail(DocumentRedactionExportReason.PROFILE_VIOLATION)
        }
        val verifier = VerifyingOcrAdapter(ocrAdapter, rendered.expectations)
        val report = DocumentIntake(
            context = context,
            ocrAdapter = verifier,
            pdfPageSource = verificationPdfPageSource,
            limits = limits,
        ).intake(Uri.fromFile(file))
        if (report.outcome != DocumentIntakeOutcome.COMPLETE) {
            fail(DocumentRedactionExportReason.VERIFICATION_INCOMPLETE)
        }
        if (report.evidence.pages.size != pages.size) {
            fail(DocumentRedactionExportReason.PAGE_COUNT_MISMATCH)
        }
        if (!verifier.allOpaque) {
            fail(DocumentRedactionExportReason.MASK_NOT_OPAQUE)
        }
        if (DocumentRedactionExportContract.containsResidual(
                DocumentRedactionExportContract.maskedTexts(request),
                report.evidence.text,
                request.policy.minimumResidualTokenScalars,
            )
        ) {
            fail(DocumentRedactionExportReason.RESIDUAL_TEXT)
        }
        return DocumentRedactionReceipt(
            schemaVersion = DocumentRedactionExportContract.SCHEMA_VERSION,
            profile = DocumentRedactionExportContract.PROFILE,
            binding = request.binding,
            sourceDigest = request.report.evidence.sourceDigest.orEmpty(),
            outputDigest = DocumentRedactionExportContract.sha256(data),
            byteCount = data.size.toLong(),
            pageCount = pages.size,
            maskCount = request.masks.size,
            paintedRegionCount = rendered.painted,
            checks = DocumentRedactionCheck.values().toList(),
        )
    }

    companion object {
        /** Folder under the app cache directory that holds export sessions. */
        const val DIRECTORY_NAME: String = "openmed-redaction-exports"

        /** Media type of every exported artifact. */
        const val MIME_TYPE: String = "application/pdf"

        internal const val MAX_PIXELS: Long = 40_000_000L

        internal fun exportRoot(context: Context): File = File(context.cacheDir, DIRECTORY_NAME)
    }
}

/**
 * Process-wide live export sessions. A new exporter sweeps every other session
 * directory, which after a process restart is every leftover one.
 */
internal object DocumentRedactionSessions {
    private val live = mutableSetOf<String>()

    @Synchronized
    fun register(session: String, root: File) {
        live += session
        root.listFiles()?.filter { it.name !in live }?.forEach { it.deleteRecursively() }
    }

    @Synchronized
    fun remove(session: String) {
        live -= session
    }
}

/** ARGB page rasters with a top-left origin, painted and rotated in memory. */
internal object DocumentRedactionRaster {
    class Upright(val pixels: IntArray, val width: Int, val height: Int)

    private const val OPAQUE_BLACK = 0xFF000000.toInt()

    /** Turns stored pixels clockwise by [degrees] so the content is upright. */
    fun rotate(pixels: IntArray, width: Int, height: Int, degrees: Int): Upright {
        if (degrees == 0) {
            return Upright(pixels, width, height)
        }
        val quarter = degrees == 90 || degrees == 270
        val targetWidth = if (quarter) height else width
        val targetHeight = if (quarter) width else height
        val target = IntArray(pixels.size)
        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val source = when (degrees) {
                    90 -> (height - 1 - x) * width + y
                    180 -> (height - 1 - y) * width + (width - 1 - x)
                    else -> x * width + (width - 1 - y)
                }
                target[y * targetWidth + x] = pixels[source]
            }
        }
        return Upright(target, targetWidth, targetHeight)
    }

    /** Composites translucent pixels over white so every pixel is opaque. */
    fun flatten(pixels: IntArray) {
        for (index in pixels.indices) {
            val pixel = pixels[index]
            val alpha = pixel ushr 24
            if (alpha == 0xFF) {
                continue
            }
            fun channel(shift: Int): Int = (((pixel shr shift) and 0xFF) * alpha + 0xFF * (0xFF - alpha) + 127) / 0xFF
            pixels[index] = OPAQUE_BLACK or (channel(16) shl 16) or (channel(8) shl 8) or channel(0)
        }
    }

    /** Paints `[left, top, right, bottom]` pixel rectangles opaque black. */
    fun paint(pixels: IntArray, width: Int, rects: List<IntArray>) {
        for (rect in rects) {
            for (y in rect[1] until rect[3]) {
                pixels.fill(OPAQUE_BLACK, y * width + rect[0], y * width + rect[2])
            }
        }
    }

    /** True when every normalized rectangle, inset by one pixel, is dark. */
    fun isOpaque(pixels: IntArray, width: Int, height: Int, rects: List<DocumentBox>): Boolean {
        for (rect in rects) {
            val left = Math.ceil(rect.left * width).toInt() + 1
            val top = Math.ceil(rect.top * height).toInt() + 1
            val right = Math.floor(rect.right * width).toInt() - 1
            val bottom = Math.floor(rect.bottom * height).toInt() - 1
            if (!(left < right && top < bottom) || left < 0 || top < 0 || right > width || bottom > height) {
                continue
            }
            for (y in top until bottom) {
                for (x in left until right) {
                    val pixel = pixels[y * width + x]
                    if ((pixel shr 16 and 0xFF) > DARK || (pixel shr 8 and 0xFF) > DARK || (pixel and 0xFF) > DARK) {
                        return false
                    }
                }
            }
        }
        return true
    }

    private const val DARK = 48
}

/** Checks painted masks on each re-read page, then defers to the caller's OCR. */
private class VerifyingOcrAdapter(
    private val inner: OcrAdapter,
    private val expectations: List<List<DocumentBox>>,
) : OcrAdapter {
    @Volatile
    private var seen = 0

    @Volatile
    private var opaque = true

    val allOpaque: Boolean
        get() = opaque && seen == expectations.size

    override suspend fun recognize(image: OcrImage): OcrResult {
        val index = seen++
        val rects = expectations.getOrNull(index)
        val painted = rects != null && image is OcrImage.BitmapImage && run {
            val bitmap = image.bitmap
            val stored = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(stored, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            val upright = DocumentRedactionRaster.rotate(stored, bitmap.width, bitmap.height, image.rotationDegrees)
            DocumentRedactionRaster.isOpaque(upright.pixels, upright.width, upright.height, rects)
        }
        if (!painted) {
            opaque = false
        }
        return inner.recognize(image)
    }
}
