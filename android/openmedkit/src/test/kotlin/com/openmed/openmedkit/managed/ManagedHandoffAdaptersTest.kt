package com.openmed.openmedkit.managed

import android.content.Context
import android.content.RestrictionsManager
import android.net.Uri
import android.os.Bundle
import com.openmed.openmedkit.bundle.Json
import com.openmed.openmedkit.bundle.PortableBundleContext
import com.openmed.openmedkit.bundle.PortableBundleTarget
import com.openmed.openmedkit.cache.BundleGenerationCache
import java.io.ByteArrayInputStream
import java.io.File
import java.io.FileNotFoundException
import java.util.Base64
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.Shadows.shadowOf
import org.robolectric.annotation.Config

/** Exercises the Android handoff adapters against Robolectric's platform fakes. */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ManagedHandoffAdaptersTest {
    @get:Rule
    val temporary = TemporaryFolder()

    private val vectors: Json.Obj by lazy {
        val module = File(requireNotNull(System.getProperty("openmedkit.moduleDirectory")))
        val repoRoot = requireNotNull(module.absoluteFile.parentFile?.parentFile)
        Json.parse(File(repoRoot, "tests/fixtures/managed_import/vectors.json").readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value

    private fun archive(alias: String): ByteArray =
        Base64.getDecoder().decode(vectors["archives"].obj()[alias].obj()["base64"].str())

    private fun generation(alias: String): String = vectors["bundles"].obj()[alias].obj()["generation"].str()
    private val bundleId: String get() = vectors["identities"].obj()["bundle_id"].str()

    private val context = PortableBundleContext(
        now = "2026-11-07T00:00:00Z",
        target = PortableBundleTarget(
            platform = "android",
            sdkName = "openmedkit-android",
            sdkVersion = "4.0.0",
            runtimes = listOf("onnx", "ort"),
            osVersion = "26",
            policyVersion = 1,
        ),
    )

    private fun restrictions(location: String?): Bundle {
        val defaults = vectors["configs"].obj()["default"].obj()
        return Bundle().apply {
            putInt(ManagedBundleConfiguration.VERSION_KEY, 1)
            putString(ManagedBundleConfiguration.TRUST_POLICY_KEY, defaults[ManagedBundleConfiguration.TRUST_POLICY_KEY].str())
            if (location != null) putString(ManagedBundleConfiguration.LOCATION_KEY, location)
            putString("com.example.unrelated", "kept by the app")
        }
    }

    private fun restrictionsManager(): RestrictionsManager {
        val application = RuntimeEnvironment.getApplication()
        return application.getSystemService(Context.RESTRICTIONS_SERVICE) as RestrictionsManager
    }

    @Test
    fun restrictionsManagerSourceReadsAppRestrictions() {
        val application = RuntimeEnvironment.getApplication()
        val source = RestrictionsManagerConfigurationSource(application)
        assertNull(source.managedConfiguration())

        val root = temporary.newFolder()
        val inbox = File(root, "inbox").also { it.mkdirs() }
        File(inbox, "bundle.zip").writeBytes(archive("v1"))
        shadowOf(restrictionsManager()).setApplicationRestrictions(restrictions("bundle.zip"))
        val values = requireNotNull(source.managedConfiguration())
        assertEquals(1, values[ManagedBundleConfiguration.VERSION_KEY])

        val cache = BundleGenerationCache(File(root, "cache"))
        val importer = ManagedBundleImporter(cache, File(root, "state"))
        val receipt = importer.importManaged(source, context, approvedRoots = listOf(inbox))
        assertEquals(ManagedImportOutcome.ACTIVATED, receipt.outcome, receipt.reason?.code)
        assertEquals(generation("v1"), cache.snapshot(bundleId).active)
        assertEquals(ManagedImportOutcome.ALREADY_PRESENT, importer.importManaged(source, context, listOf(inbox)).outcome)
    }

    @Test
    fun contentResolverOpenerImportsFromAContentProviderStream() {
        val application = RuntimeEnvironment.getApplication()
        val uri = "content://org.example.emm.files/bundles/model.zip"
        shadowOf(application.contentResolver).registerInputStream(Uri.parse(uri), ByteArrayInputStream(archive("v2")))
        shadowOf(restrictionsManager()).setApplicationRestrictions(restrictions(uri))

        val root = temporary.newFolder()
        val cache = BundleGenerationCache(File(root, "cache"))
        val opener = ContentResolverStreamOpener(application.contentResolver)
        val importer = ManagedBundleImporter(cache, File(root, "state"), opener)
        val source = RestrictionsManagerConfigurationSource(application)

        val unapproved = importer.importManaged(source, context, approvedAuthorities = setOf("org.example.other"))
        assertEquals(ManagedImportReason.LOCATION_UNAPPROVED, unapproved.reason)
        val receipt = importer.importManaged(source, context, approvedAuthorities = setOf("org.example.emm.files"))
        assertEquals(ManagedImportOutcome.ACTIVATED, receipt.outcome, receipt.reason?.code)
        assertEquals(ManagedHandoffKind.CONTENT_URI, receipt.handoff)
        assertEquals(generation("v2"), cache.snapshot(bundleId).active)

        assertFailsWith<FileNotFoundException> { opener.open("https://example.invalid/model.zip") }
        assertFailsWith<FileNotFoundException> { opener.open("file:///sdcard/model.zip") }
    }
}
