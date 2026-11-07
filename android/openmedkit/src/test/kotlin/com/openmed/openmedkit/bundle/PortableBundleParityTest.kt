package com.openmed.openmedkit.bundle

import java.io.File
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.security.Permission
import java.util.concurrent.CopyOnWriteArrayList
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import org.junit.Test

class PortableBundleParityTest {
    private val vectors: Json.Obj by lazy {
        val module = File(
            requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
                "openmedkit.moduleDirectory test property is missing"
            },
        )
        val repoRoot = requireNotNull(module.absoluteFile.parentFile?.parentFile) {
            "cannot locate the repository root"
        }
        val file = File(repoRoot, "tests/fixtures/portable_bundle/vectors.json")
        Json.parse(file.readBytes()) as Json.Obj
    }

    private fun Json?.obj(): Json.Obj = this as Json.Obj
    private fun Json?.str(): String = (this as Json.Str).value
    private fun Json?.long(): Long = (this as Json.Num).value
    private fun Json?.items(): List<Json> = (this as Json.Arr).items
    private fun Json?.optionalString(): String? = if (this == Json.Null) null else str()
    private fun Json?.optionalLong(): Long? = if (this == Json.Null) null else long()

    private fun manifest(key: String): ByteArray {
        val entry = vectors["manifests"].obj()[key].obj()
        entry["text"]?.let { return it.str().toByteArray(StandardCharsets.UTF_8) }
        val base = manifest(entry["base"].str())
        val at = entry["at"].long().toInt()
        val delete = entry["delete"].long().toInt()
        val insert = entry["insert"].str().toByteArray(StandardCharsets.UTF_8)
        return base.copyOfRange(0, at) + insert + base.copyOfRange(at + delete, base.size)
    }

    private fun context(raw: Json.Obj): PortableBundleContext {
        val target = raw["target"].takeUnless { it == Json.Null }?.obj()?.let { value ->
            PortableBundleTarget(
                platform = value["platform"].str(),
                sdkName = value["sdk_name"].str(),
                sdkVersion = value["sdk_version"].str(),
                runtimes = value["runtimes"].items().map { it.str() },
                osVersion = value["os_version"].optionalString(),
                policyVersion = value["policy_version"].optionalLong(),
                memoryBytes = value["memory_bytes"].optionalLong(),
                storageBytes = value["storage_bytes"].optionalLong(),
            )
        }
        return PortableBundleContext(
            now = raw["now"].str(),
            target = target,
            sequenceFloors = raw["sequence_floors"].obj().members.associate { it.first to it.second.long() },
            understoodExtensions = raw["understood_extensions"].items().map { it.str() }.toSet(),
            maxManifestBytes = raw["max_manifest_bytes"].long().toInt(),
            maxEnvelopeBytes = raw["max_envelope_bytes"].long().toInt(),
            maxTotalBytes = raw["max_total_bytes"].long(),
            maxFiles = raw["max_files"].long().toInt(),
        )
    }

    private fun policy(name: String): PortableBundleTrustPolicy {
        val raw = vectors["trust_policies"].obj()[name]!!
        return PortableBundleTrustPolicy.fromJson(String(Json.encode(raw), StandardCharsets.UTF_8))
    }

    private fun tree(item: Json.Obj): Map<String, ByteArray>? {
        val name = item["tree"].takeUnless { it == Json.Null }?.str() ?: return null
        val files = linkedMapOf<String, ByteArray>()
        for (entry in vectors["trees"].obj()[name].items()) {
            val value = entry.obj()
            files[value["path"].str()] = value["text"].str().toByteArray(StandardCharsets.UTF_8)
        }
        item["tree_remove"].items().forEach { files.remove(it.str()) }
        for (entry in item["tree_set"].items()) {
            val value = entry.obj()
            files[value["path"].str()] = value["text"].str().toByteArray(StandardCharsets.UTF_8)
        }
        return files
    }

    private fun run(item: Json.Obj): PortableBundleVerification = PortableBundle.verify(
        manifest = manifest(item["manifest"].str()),
        envelope = vectors["envelopes"].obj()[item["envelope"].str()].str().toByteArray(StandardCharsets.UTF_8),
        trustPolicy = policy(item["trust_policy"].str()),
        context = context(vectors["contexts"].obj()[item["context"].str()].obj()),
        files = tree(item),
    )

    private fun assertMatchesExpected(item: Json.Obj, result: PortableBundleVerification) {
        val id = item["id"].str()
        val expected = item["expected"].obj()
        assertEquals((expected["ok"] as Json.Bool).value, result.ok, id)
        assertEquals(expected["reason"].optionalString(), result.reason?.code, id)
        assertEquals(expected["content_id"].optionalString(), result.contentId, id)
        assertEquals(expected["valid_signatures"].long().toInt(), result.validSignatures, id)
        assertEquals(expected["eligible_runtimes"].items().map { it.long().toInt() }, result.eligibleRuntimes, id)
        assertEquals(expected["files_checked"].long().toInt(), result.filesChecked, id)
        assertEquals((expected["files_verified"] as Json.Bool).value, result.filesVerified, id)
    }

    private fun hex(text: String): ByteArray =
        text.chunked(2).map { it.toInt(16).toByte() }.toByteArray()

    @Test
    fun vectorsAreSyntheticAndShareTheReasonVocabulary() {
        assertEquals(Json.Bool(true), vectors["synthetic_only"])
        assertEquals(Json.Bool(true), vectors["test_only_keys"])
        assertEquals(PortableBundle.PAYLOAD_TYPE, vectors["payload_type"].str())
        assertEquals(PortableBundleReason.values().map { it.code }, vectors["reason_codes"].items().map { it.str() })
    }

    @Test
    fun everyGoldenCaseMatchesThePythonReference() {
        val cases = vectors["cases"].items()
        assertTrue(cases.size > 50)
        for (item in cases) assertMatchesExpected(item.obj(), run(item.obj()))
    }

    @Test
    fun goldenIdentityAndSigningInputMatch() {
        val golden = vectors["golden"].obj()
        val data = manifest("golden")
        assertEquals(golden["content_id"].str(), PortableBundle.contentId(data))
        val digest = MessageDigest.getInstance("SHA-256").digest(PortableBundle.signingInput(data))
        assertEquals(golden["signing_input_sha256"].str(), digest.joinToString("") { "%02x".format(it) })
    }

    @Test
    fun canonicalizationVectorsMatchRfc8785() {
        for (item in vectors["canonicalization"].items()) {
            val value = item.obj()
            val canonical = PortableBundle.canonicalJson(value["input"].str().toByteArray(StandardCharsets.UTF_8))
            assertEquals(value["canonical"].str(), String(canonical, StandardCharsets.UTF_8), value["id"].str())
        }
        for (item in vectors["canonicalization_rejects"].items()) {
            val value = item.obj()
            val error = assertFailsWith<PortableBundleException>(value["id"].str()) {
                PortableBundle.canonicalJson(value["input"].str().toByteArray(StandardCharsets.UTF_8))
            }
            assertEquals(PortableBundleReason.MANIFEST_NOT_CANONICAL, error.reason)
        }
    }

    @Test
    fun unknownNoncriticalExtensionsRoundTripByteForByte() {
        val golden = manifest("golden")
        PortableBundle.validateManifest(golden)
        assertContentEquals(golden, PortableBundle.canonicalJson(golden))
        val critical = manifest("valid_critical_extension_understood")
        val error = assertFailsWith<PortableBundleException> { PortableBundle.validateManifest(critical) }
        assertEquals(PortableBundleReason.UNKNOWN_CRITICAL_EXTENSION, error.reason)
        PortableBundle.validateManifest(critical, setOf("org.openmed.test.note"))
    }

    @Test
    fun rfc8032VectorsVerifyWithTheBundledVerifier() {
        for (item in vectors["ed25519_rfc8032"].items()) {
            val value = item.obj()
            val public = hex(value["public_hex"].str())
            val signature = hex(value["signature_hex"].str())
            val message = hex(value["message_hex"].str())
            assertTrue(Rfc8032Ed25519Verifier.verify(public, signature, message), value["name"].str())
            assertFalse(Rfc8032Ed25519Verifier.verify(public, signature, message + 1), value["name"].str())
            val tampered = signature.copyOf().also { it[0] = (it[0].toInt() xor 1).toByte() }
            assertFalse(Rfc8032Ed25519Verifier.verify(public, tampered, message), value["name"].str())
        }
        assertFalse(Rfc8032Ed25519Verifier.verify(ByteArray(31), ByteArray(64), ByteArray(0)))
    }

    @Test
    fun injectedVerifierIsUsedAndCanFailClosed() {
        val item = vectors["cases"].items().first { it.obj()["id"].str() == "valid_android" }.obj()
        val result = PortableBundle.verify(
            manifest = manifest(item["manifest"].str()),
            envelope = vectors["envelopes"].obj()["golden"].str().toByteArray(StandardCharsets.UTF_8),
            trustPolicy = policy("default"),
            context = context(vectors["contexts"].obj()["default"].obj()),
            verifier = { _, _, _ -> false },
        )
        assertEquals(PortableBundleReason.SIGNATURE_INVALID, result.reason)
    }

    @Suppress("DEPRECATION")
    @Test
    fun verificationNeverOpensSockets() {
        val attempts = CopyOnWriteArrayList<String>()
        val previous = System.getSecurityManager()
        System.setSecurityManager(DenyNetwork(attempts))
        try {
            for (item in vectors["cases"].items()) assertMatchesExpected(item.obj(), run(item.obj()))
        } finally {
            System.setSecurityManager(previous)
        }
        assertTrue(attempts.isEmpty())
    }

    @Test
    fun callerInputErrorsCarryOnlyReasonCodes() {
        val policyError = assertFailsWith<PortableBundleException> {
            PortableBundleTrustPolicy.fromJson("{\"schema\":\"other\"}")
        }
        assertEquals("trust_policy_invalid", policyError.message)
        val contextError = assertFailsWith<PortableBundleException> { PortableBundleContext(now = "yesterday") }
        assertEquals("context_invalid", contextError.message)
        val key = PortableBundleTrustedKey(
            "unit-test",
            ByteArray(32) { 7 },
            "2026-01-01T00:00:00Z",
            "2027-01-01T00:00:00Z",
        )
        assertFalse(key.toString().contains("BwcH"))
    }

    @Suppress("DEPRECATION")
    private class DenyNetwork(private val attempts: MutableList<String>) : SecurityManager() {
        override fun checkPermission(permission: Permission?) = Unit

        override fun checkPermission(permission: Permission?, context: Any?) = Unit

        override fun checkConnect(host: String?, port: Int) {
            attempts += "$host:$port"
            throw AssertionError("network access is forbidden during bundle verification")
        }

        override fun checkConnect(host: String?, port: Int, context: Any?) = checkConnect(host, port)
    }
}
