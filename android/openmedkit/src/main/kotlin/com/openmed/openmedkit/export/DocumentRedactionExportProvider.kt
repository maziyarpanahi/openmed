package com.openmed.openmedkit.export

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.content.pm.ProviderInfo
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.io.File
import java.io.FileNotFoundException

/**
 * Read-only, FileProvider-style provider for verified redaction exports.
 *
 * It serves only `<session>/<artifact>.pdf` files that a
 * [DocumentRedactionExporter] verified and still holds; completed, invalidated
 * and pending files are not addressable. The core artifact does not declare it,
 * so no app exposes it by accident. An app that shares exports declares it,
 * unexported and with URI grants, and passes the same authority to the exporter:
 *
 * ```xml
 * <provider
 *     android:name="com.openmed.openmedkit.export.DocumentRedactionExportProvider"
 *     android:authorities="${applicationId}.openmed.redaction-exports"
 *     android:exported="false"
 *     android:grantUriPermissions="true" />
 * ```
 *
 * Apps that already use `androidx.core.content.FileProvider` can serve the same
 * directory with it instead; this provider avoids adding that dependency.
 */
class DocumentRedactionExportProvider : ContentProvider() {
    override fun attachInfo(context: Context, info: ProviderInfo) {
        super.attachInfo(context, info)
        if (info.exported) {
            throw SecurityException("DocumentRedactionExportProvider must not be exported")
        }
        if (!info.grantUriPermissions) {
            throw SecurityException("DocumentRedactionExportProvider must grant URI permissions")
        }
    }

    override fun onCreate(): Boolean = true

    override fun getType(uri: Uri): String? =
        resolve(uri)?.let { DocumentRedactionExporter.MIME_TYPE }

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor {
        if (mode != "r") {
            throw SecurityException("redaction exports are read-only")
        }
        val file = resolve(uri) ?: throw FileNotFoundException("artifact_unavailable")
        return ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
    }

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?,
    ): Cursor? {
        val file = resolve(uri) ?: return null
        val columns = (projection ?: arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE))
            .filter { it == OpenableColumns.DISPLAY_NAME || it == OpenableColumns.SIZE }
        return MatrixCursor(columns.toTypedArray(), 1).apply {
            addRow(
                columns.map { column ->
                    if (column == OpenableColumns.DISPLAY_NAME) file.name else file.length()
                },
            )
        }
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? =
        throw UnsupportedOperationException("redaction exports are read-only")

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?,
    ): Int = throw UnsupportedOperationException("redaction exports are read-only")

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int =
        throw UnsupportedOperationException("redaction exports are read-only")

    /** Resolves only a verified artifact file inside an export session. */
    private fun resolve(uri: Uri): File? {
        val context = context ?: return null
        val segments = uri.pathSegments
        if (uri.scheme != "content" || segments.size != 2 || !REFERENCE.matches(segments[0]) ||
            !ARTIFACT.matches(segments[1])
        ) {
            return null
        }
        val file = File(File(DocumentRedactionExporter.exportRoot(context), segments[0]), segments[1])
        return file.takeIf { it.isFile }
    }

    private companion object {
        val REFERENCE = Regex("[0-9a-f]{32}")
        val ARTIFACT = Regex("[0-9a-f]{32}\\.pdf")
    }
}
