package org.openmed.installedconsumer

import com.openmed.openmedkit.OpenMedMLXModelCacheState
import com.openmed.openmedkit.OpenMedModelStore
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.cache.ModelCache
import com.openmed.openmedkit.catalog.ModelCatalogEntry
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelFile
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import com.openmed.openmedkit.download.ModelDownloader
import com.openmed.openmedkit.download.ModelIntegrity
import com.openmed.openmedkit.download.ModelIntegrityException
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Plain-JVM checks against the OpenMedKit AAR resolved from an isolated Maven
 * repository. Failure messages carry fixture case ids and codes, never text.
 */
class InstalledPackageTest {
    private val contract = ConsumerContract.root

    @Test
    fun contractIsSyntheticVersionOne() {
        assertEquals(1, contract.int("schema_version"))
        assertTrue(contract.string("synthetic").toBoolean(), "not_synthetic")
    }

    @Test
    fun bundledPolicyProfilesLoadFromInstalledClassesJar() {
        assertTrue(PolicyProfiles.bundledProfileNames.isNotEmpty(), "no_bundled_profiles")
        for (name in PolicyProfiles.bundledProfileNames) {
            val profile = PolicyProfiles.load(name)
            assertEquals(name, profile.name, "$name:name_mismatch")
            assertEquals(1, profile.schemaVersion, "$name:schema_version")
        }
        assertEquals("hipaa_safe_harbor", PolicyProfiles.DEFAULT_PROFILE)
        for ((alias, canonical) in contract.obj("profile_aliases")) {
            assertEquals(canonical.jsonPrimitive.content, PolicyProfiles.load(alias).name, alias)
        }
    }

    @Test
    fun unknownPolicyProfileIsATypedIncompatibility() {
        val error = assertFailsWith<IllegalArgumentException> {
            PolicyProfiles.load(contract.string("unknown_profile"))
        }
        assertEquals("invalid_argument", MigrationExamples.incompatibilityCode(error))
    }

    @Test
    fun policyLabelActionsMatchContract() {
        for (policyCase in contract.array("policy_cases")) {
            val profileName = policyCase.string("profile")
            val profile = PolicyProfiles.load(profileName)
            for ((label, action) in policyCase.obj("label_actions")) {
                assertEquals(
                    action.jsonPrimitive.content,
                    profile.actionFor(label).wireName,
                    "$profileName:$label:action_mismatch",
                )
            }
        }
    }

    @Test
    fun unicodeScalarOffsetsMatchContract() {
        assertEquals("unicode_scalar", contract.string("offset_unit"))
        for (offsetCase in contract.array("offset_cases")) {
            val id = offsetCase.string("id")
            val text = offsetCase.string("text")
            val start = offsetCase.int("start")
            val end = offsetCase.int("end")
            assertEquals(
                offsetCase.int("scalar_length"),
                UnicodeOffsetContract.scalarLength(text),
                "$id:scalar_length",
            )
            assertEquals(offsetCase.int("utf16_length"), text.length, "$id:utf16_length")
            val span = UnicodeOffsetContract.utf16Span(text, start, end)
            assertEquals(offsetCase.int("utf16_start"), span.start, "$id:utf16_start")
            assertEquals(offsetCase.int("utf16_end"), span.end, "$id:utf16_end")
            assertEquals(
                start,
                UnicodeOffsetContract.utf16ToScalarOffset(text, span.start),
                "$id:roundtrip",
            )
            assertEquals(
                offsetCase.string("expected_redacted"),
                UnicodeOffsetContract.replaceScalarSpan(
                    text,
                    start,
                    end,
                    offsetCase.string("replacement"),
                ),
                "$id:redaction",
            )
        }
    }

    @Test
    fun syntheticArtifactDownloadsVerifiesAndServesFromCacheOffline() {
        val fixture = contract.obj("model_fixture")
        val files = fixture.array("onnx_files")
        val entry = catalogEntry(fixture, files)
        withTempDirectory { root ->
            val cache = ModelCache(root)
            val first = ModelDownloader(cache, FixtureModelClient(fixture, files)).download(entry)
            assertFalse(first.fromCache, "first_download_from_cache")
            assertEquals(files.size, first.filesDownloaded, "file_count")
            assertEquals(entry.reproducibilityHash, first.integrityHash, "integrity_hash")
            assertTrue(cache.isAvailable(entry), "cache_not_ready")
            assertEquals(first.directory, MigrationExamples.verifiedCacheDirectory(cache, entry))

            val offline = ModelDownloader(cache, OfflineModelClient).download(entry)
            assertTrue(offline.fromCache, "cache_hit_used_network")
            assertEquals(0, offline.filesDownloaded, "cache_hit_files")
        }
    }

    @Test
    fun tamperedArtifactIsATypedIntegrityFailure() {
        val fixture = contract.obj("model_fixture")
        val files = fixture.array("onnx_files")
        val entry = catalogEntry(fixture, files)
        withTempDirectory { root ->
            val cache = ModelCache(root)
            val tampered = FixtureModelClient(fixture, files, tamperPath = "model.onnx")
            val error = assertFailsWith<ModelIntegrityException> {
                ModelDownloader(cache, tampered).download(entry)
            }
            assertEquals("integrity_mismatch", MigrationExamples.incompatibilityCode(error))
            assertFalse(cache.isAvailable(entry), "tampered_artifact_cached")

            val unverified = entry.copy(reproducibilityHash = "sha256:" + "0".repeat(64))
            assertFailsWith<ModelIntegrityException> {
                ModelDownloader(cache, FixtureModelClient(fixture, files)).download(unverified)
            }
            assertFalse(cache.isAvailable(unverified), "unverified_manifest_cached")
        }
    }

    @Test
    fun legacyMlxStoreIsMarkerOnlyAndDownloadIsUnsupported() {
        val repoId = contract.obj("model_fixture").string("repo_id")
        withTempDirectory { root ->
            assertEquals(
                OpenMedMLXModelCacheState.missing,
                MigrationExamples.legacyStoreState(repoId, root),
            )
            val error = assertFailsWith<UnsupportedOperationException> {
                OpenMedModelStore.downloadMLXModel(repoId, cacheDirectory = root)
            }
            assertEquals("unsupported_on_android", MigrationExamples.incompatibilityCode(error))
        }
    }

    private fun withTempDirectory(block: (File) -> Unit) {
        val root = Files.createTempDirectory("openmed-installed-consumer").toFile()
        try {
            block(root)
        } finally {
            root.deleteRecursively()
        }
    }

    private fun catalogEntry(fixture: JsonObject, files: List<JsonObject>): ModelCatalogEntry {
        val repoId = fixture.string("repo_id")
        return ModelCatalogEntry(
            repoId = repoId,
            formats = listOf("onnx"),
            tier = "synthetic",
            paramCount = 1,
            languages = listOf("en"),
            license = "apache-2.0",
            reproducibilityHash = ModelIntegrity.reproducibilityHash(
                repoId = repoId,
                revisionSha = fixture.string("revision_sha"),
                released = fixture.string("released"),
                siblings = files.map { it.string("path") },
            ),
        )
    }

    private class FixtureModelClient(
        private val fixture: JsonObject,
        private val files: List<JsonObject>,
        private val tamperPath: String? = null,
    ) : HuggingFaceModelClient {
        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo =
            HuggingFaceModelInfo(
                repoId = repoId,
                revisionSha = fixture.string("revision_sha"),
                released = fixture.string("released"),
                files = files.map {
                    HuggingFaceModelFile(
                        path = it.string("path"),
                        sizeBytes = it.int("size_bytes").toLong(),
                        sha256 = it.string("sha256"),
                    )
                },
            )

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
        ) {
            val content = files.first { it.string("path") == path }.string("content")
            destination.writeText(if (path == tamperPath) content.reversed() else content)
        }
    }

    private object OfflineModelClient : HuggingFaceModelClient {
        override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo =
            throw AssertionError("network_requested")

        override fun downloadFile(
            repoId: String,
            path: String,
            revision: String,
            destination: File,
        ) = throw AssertionError("network_requested")
    }
}
