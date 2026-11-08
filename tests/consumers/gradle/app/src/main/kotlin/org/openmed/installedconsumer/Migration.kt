package org.openmed.installedconsumer

import android.content.Context
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.OpenMedMLXModelCacheState
import com.openmed.openmedkit.OpenMedModelStore
import com.openmed.openmedkit.PolicyDeidentificationResult
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.ModelDownloadResult
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.download.ModelIntegrityException
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import kotlinx.coroutines.runBlocking

/**
 * Migration targets from docs/migration/openmedkit-installed-packages.md.
 *
 * Compiling this app against the installed AAR proves the documented
 * replacements resolve. Helpers that need a device context or real model
 * assets are compiled, not executed, by the unit checks.
 */
object MigrationExamples {
    /** Replace the Swift-shaped `OpenMed` scaffold with the ONNX facade. */
    fun onnxFacade(modelDirectory: File): OpenMedKit =
        OpenMedKit.fromDirectory(modelDirectory, variant = "int8")

    /** Construct the facade from explicit local assets. */
    fun explicitBackend(modelDirectory: File): OpenMedKit =
        OpenMedKit(OpenMedBackend(modelDirectory = modelDirectory))

    /** Replace `OpenMedModelStore.downloadMLXModel` with catalog-driven ONNX acquisition. */
    fun acquireFromCatalog(context: Context, repoId: String): ModelDownloadResult? {
        val entry = ModelCatalog.load(context).byRepoId(repoId) ?: return null
        return ModelDownloader(context).download(entry)
    }

    /** Use an already verified artifact without touching the network. */
    fun verifiedCacheDirectory(cache: ModelCache, entry: ModelCatalogEntry): File? =
        if (cache.isAvailable(entry)) cache.modelDirectory(entry.repoId) else null

    /** Policy-driven redaction with the documented default profile. */
    suspend fun redact(kit: OpenMedKit, text: String): PolicyDeidentificationResult =
        kit.deidentify(text, policy = PolicyProfiles.DEFAULT_PROFILE)

    /** Blocking bridge for Java or non-coroutine call sites. */
    fun redactBlocking(kit: OpenMedKit, text: String): PolicyDeidentificationResult =
        runBlocking { redact(kit, text) }

    /** The legacy MLX-named store only reports a marker-file state on Android. */
    fun legacyStoreState(repoId: String, cacheDirectory: File): OpenMedMLXModelCacheState =
        OpenMedModelStore.mlxModelCacheState(repoId, cacheDirectory = cacheDirectory)

    /** Map the documented typed incompatibilities to stable codes. */
    fun incompatibilityCode(error: Throwable): String? = when (error) {
        is ModelIntegrityException -> "integrity_mismatch"
        is UnsupportedOperationException -> "unsupported_on_android"
        is IllegalArgumentException -> "invalid_argument"
        else -> null
    }
}
