// Exercises the deprecated MLX-named members on purpose.
@file:Suppress("DEPRECATION")

package com.openmed.openmedkit

import com.openmed.openmedkit.artifact.ArtifactAcquisitionRequest
import com.openmed.openmedkit.artifact.ArtifactAcquisitionResult
import com.openmed.openmedkit.artifact.ArtifactBackendCapability
import com.openmed.openmedkit.artifact.ArtifactCacheState
import com.openmed.openmedkit.artifact.ArtifactFormat
import com.openmed.openmedkit.artifact.ArtifactNetworkConsent
import com.openmed.openmedkit.artifact.ArtifactRef
import com.openmed.openmedkit.artifact.ArtifactRuntime
import com.openmed.openmedkit.artifact.HuggingFaceFixtureServer
import java.io.File
import java.nio.file.Files
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import kotlinx.coroutines.runBlocking
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class OpenMedModelStoreTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun formatAwareStoreAcquiresInsteadOfThrowing() = runBlocking<Unit> {
        val cacheDirectory = temporaryFolder.newFolder("store")
        val repoId = "OpenMed/store-synthetic-onnx"
        HuggingFaceFixtureServer(
            repoId,
            mapOf(
                "model.onnx" to HuggingFaceFixtureServer.syntheticOnnx(4096),
                "config.json" to """{"synthetic":true}""".toByteArray(),
            ),
        ).use { server ->
            val ref = ArtifactRef(
                repoId = repoId,
                format = ArtifactFormat.ONNX,
                runtime = ArtifactRuntime.ONNX_RUNTIME,
                digest = server.digest(),
                revision = HuggingFaceFixtureServer.REVISION_SHA,
            )
            val request = ArtifactAcquisitionRequest(
                ref,
                ArtifactBackendCapability.onnxRuntime(),
                ArtifactNetworkConsent.GRANTED,
            )
            assertEquals(
                ArtifactCacheState.MISSING,
                OpenMedModelStore.artifactCacheState(ref, cacheDirectory),
            )

            val result = OpenMedModelStore.acquireArtifact(request, cacheDirectory, server.client())

            assertIs<ArtifactAcquisitionResult.Downloaded>(result)
            assertEquals(
                ArtifactCacheState.READY,
                OpenMedModelStore.artifactCacheState(ref, cacheDirectory),
            )
            assertEquals(
                result.artifact.directory,
                assertNotNull(OpenMedModelStore.cachedArtifact(ref, cacheDirectory)).directory,
            )
        }
    }

    @Test
    fun mlxDownloadStaysUnsupportedWithAValueFreeMessage() {
        val cacheDirectory = temporaryFolder.newFolder("mlx")

        val error = assertFailsWith<UnsupportedOperationException> {
            OpenMedModelStore.downloadMLXModel("OpenMed/tiny-mlx", "main", cacheDirectory)
        }

        assertTrue(error.message.orEmpty().contains("acquireArtifact"))
        assertFalse(error.message.orEmpty().contains(cacheDirectory.path))
    }

    @Test
    fun legacyReadyMarkerRemainsSourceCompatible() {
        val cacheDirectory = temporaryFolder.newFolder("legacy")
        val directory =
            OpenMedModelStore.cachedMLXModelDirectory("OpenMed/tiny", "main", cacheDirectory)
        directory.mkdirs()
        File(directory, ".openmed-artifact-ready").writeText("")

        assertEquals(
            directory,
            OpenMedModelStore.downloadMLXModel("OpenMed/tiny", "main", cacheDirectory),
        )
        assertTrue(OpenMedModelStore.isMLXModelCached("OpenMed/tiny", "main", cacheDirectory))
    }

    @Test
    fun scaffoldDirectoriesStayInsideTheCacheRoot() {
        val cacheDirectory = temporaryFolder.newFolder("contained")
        val root = cacheDirectory.canonicalPath + File.separator

        val unsafe = listOf(".." to "..", "." to ".", "../../etc" to "../..", "" to "")
        unsafe.forEach { (repo, revision) ->
            val directory =
                OpenMedModelStore.cachedMLXModelDirectory(repo, revision, cacheDirectory)
            assertTrue(directory.canonicalPath.startsWith(root))
        }
    }

    @Test
    fun symlinkedScaffoldDirectoriesAreRejectedAndNeverReady() {
        val cacheDirectory = temporaryFolder.newFolder("symlinked")
        val outside = temporaryFolder.newFolder("outside")
        File(outside, "main").mkdirs()
        File(outside, "main/.openmed-artifact-ready").writeText("")
        Files.createSymbolicLink(File(cacheDirectory, "OpenMed_linked").toPath(), outside.toPath())

        val error = assertFailsWith<IllegalArgumentException> {
            OpenMedModelStore.cachedMLXModelDirectory("OpenMed/linked", "main", cacheDirectory)
        }

        assertFalse(error.message.orEmpty().contains(outside.path))
        assertEquals(
            OpenMedMLXModelCacheState.missing,
            OpenMedModelStore.mlxModelCacheState("OpenMed/linked", "main", cacheDirectory),
        )
        assertFalse(OpenMedModelStore.isMLXModelCached("OpenMed/linked", "main", cacheDirectory))
    }

    @Test
    fun mlxNamedMembersAreDeprecated() {
        val deprecatedMembers = OpenMedModelStore::class.java.declaredMethods
            .filter { it.name.contains("MLX", ignoreCase = true) }
            .filter { !it.name.endsWith("\$default") }

        assertEquals(
            setOf(
                "downloadMLXModel",
                "cachedMLXModelDirectory",
                "isMLXModelCached",
                "mlxModelCacheState",
            ),
            deprecatedMembers.map { it.name }.toSet(),
        )
        assertTrue(deprecatedMembers.all { it.isAnnotationPresent(Deprecated::class.java) })
        assertTrue(
            OpenMedMLXModelCacheState::class.java.isAnnotationPresent(Deprecated::class.java),
        )
    }
}
