package com.openmed.openmedkit.bundle

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Stable, value-free reason codes shared by every portable-bundle reader, in
 * evaluation order. Readers in Python, Swift, Kotlin, and TypeScript report the
 * first failing check in this order.
 */
enum class PortableBundleReason(val code: String) {
    MANIFEST_TOO_LARGE("manifest_too_large"),
    ENVELOPE_TOO_LARGE("envelope_too_large"),
    ENVELOPE_MALFORMED("envelope_malformed"),
    ENVELOPE_VERSION_UNSUPPORTED("envelope_version_unsupported"),
    CONTENT_ID_MISMATCH("content_id_mismatch"),
    BUNDLE_REVOKED("bundle_revoked"),
    SIGNATURE_MISSING("signature_missing"),
    SIGNATURE_ALGORITHM_UNSUPPORTED("signature_algorithm_unsupported"),
    UNTRUSTED_KEY("untrusted_key"),
    KEY_REVOKED("key_revoked"),
    KEY_NOT_YET_VALID("key_not_yet_valid"),
    KEY_EXPIRED("key_expired"),
    SIGNATURE_INVALID("signature_invalid"),
    SIGNATURE_VERIFIER_UNAVAILABLE("signature_verifier_unavailable"),
    SIGNATURE_THRESHOLD_NOT_MET("signature_threshold_not_met"),
    MANIFEST_NOT_CANONICAL("manifest_not_canonical"),
    SCHEMA_UNSUPPORTED("schema_unsupported"),
    SCHEMA_VERSION_UNSUPPORTED("schema_version_unsupported"),
    UNKNOWN_CRITICAL_FIELD("unknown_critical_field"),
    MANIFEST_MALFORMED("manifest_malformed"),
    PATH_UNSAFE("path_unsafe"),
    EXECUTABLE_FILE_FORBIDDEN("executable_file_forbidden"),
    RUNTIME_COMBINATION_UNSUPPORTED("runtime_combination_unsupported"),
    DUPLICATE_PATH("duplicate_path"),
    PATH_CONFLICT("path_conflict"),
    FILE_ORDER_INVALID("file_order_invalid"),
    FILE_REFERENCE_UNDECLARED("file_reference_undeclared"),
    FALLBACK_INVALID("fallback_invalid"),
    UNKNOWN_CRITICAL_EXTENSION("unknown_critical_extension"),
    SIZE_LIMIT_EXCEEDED("size_limit_exceeded"),
    BUNDLE_NOT_YET_VALID("bundle_not_yet_valid"),
    BUNDLE_EXPIRED("bundle_expired"),
    ROLLBACK_REJECTED("rollback_rejected"),
    SDK_INCOMPATIBLE("sdk_incompatible"),
    POLICY_INCOMPATIBLE("policy_incompatible"),
    PLATFORM_INCOMPATIBLE("platform_incompatible"),
    FILE_MISSING("file_missing"),
    FILE_SIZE_MISMATCH("file_size_mismatch"),
    FILE_DIGEST_MISMATCH("file_digest_mismatch"),
    FILE_UNDECLARED("file_undeclared"),
    TRUST_POLICY_INVALID("trust_policy_invalid"),
    CONTEXT_INVALID("context_invalid"),
    SIGNING_KEY_INVALID("signing_key_invalid"),
}

/** Error that carries only a controlled reason code. */
class PortableBundleException(val reason: PortableBundleReason) : Exception(reason.code)

/** One caller-controlled Ed25519 trust anchor. The key bytes never print. */
class PortableBundleTrustedKey(
    val kid: String,
    publicKey: ByteArray,
    val notBefore: String,
    val notAfter: String,
    val revoked: Boolean = false,
) {
    private val key = publicKey.copyOf()
    val publicKey: ByteArray get() = key.copyOf()

    init {
        if (!Patterns.KID.matches(kid) || key.size != 32 || !Patterns.TIMESTAMP.matches(notBefore) ||
            !Patterns.TIMESTAMP.matches(notAfter) || notBefore >= notAfter
        ) {
            throw PortableBundleException(PortableBundleReason.TRUST_POLICY_INVALID)
        }
    }

    override fun toString(): String = "PortableBundleTrustedKey(kid=$kid, revoked=$revoked)"
}

/** Offline trust roots, rotation windows, and revocations. */
class PortableBundleTrustPolicy(
    val keys: List<PortableBundleTrustedKey>,
    val threshold: Int = 1,
    val revokedContentIds: Set<String> = emptySet(),
) {
    init {
        if (keys.isEmpty() || keys.size > 64 || threshold !in 1..PortableBundle.MAX_SIGNATURES ||
            keys.map { it.kid }.toSet().size != keys.size ||
            !revokedContentIds.all { Patterns.SHA256.matches(it) }
        ) {
            throw PortableBundleException(PortableBundleReason.TRUST_POLICY_INVALID)
        }
    }

    internal fun key(kid: String): PortableBundleTrustedKey? = keys.firstOrNull { it.kid == kid }

    companion object {
        /** Parses a JSON trust-policy document (`portable_bundle_trust_policy`). */
        fun fromJson(text: String): PortableBundleTrustPolicy {
            val invalid = PortableBundleException(PortableBundleReason.TRUST_POLICY_INVALID)
            val root = try {
                Json.parse(text.toByteArray(StandardCharsets.UTF_8))
            } catch (error: Json.Invalid) {
                throw invalid
            }
            if (root !is Json.Obj ||
                !root.hasExactKeys("schema", "schema_version", "threshold", "keys", "revoked_content_ids") ||
                root["schema"] != Json.Str("openmed.portable_bundle.trust_policy") ||
                root["schema_version"] != Json.Num(1)
            ) {
                throw invalid
            }
            val threshold = (root["threshold"] as? Json.Num)?.value ?: throw invalid
            val rawKeys = (root["keys"] as? Json.Arr)?.items ?: throw invalid
            val rawRevoked = (root["revoked_content_ids"] as? Json.Arr)?.items ?: throw invalid
            val keys = rawKeys.map { entry ->
                if (entry !is Json.Obj ||
                    !entry.hasExactKeys("kid", "kty", "crv", "x", "not_before", "not_after", "status") ||
                    entry["kty"] != Json.Str("OKP") || entry["crv"] != Json.Str("Ed25519")
                ) {
                    throw invalid
                }
                val status = (entry["status"] as? Json.Str)?.value
                if (status != "active" && status != "revoked") throw invalid
                PortableBundleTrustedKey(
                    kid = (entry["kid"] as? Json.Str)?.value ?: throw invalid,
                    publicKey = Base64Url.decode((entry["x"] as? Json.Str)?.value, 32) ?: throw invalid,
                    notBefore = (entry["not_before"] as? Json.Str)?.value ?: throw invalid,
                    notAfter = (entry["not_after"] as? Json.Str)?.value ?: throw invalid,
                    revoked = status == "revoked",
                )
            }
            val revoked = rawRevoked.map { (it as? Json.Str)?.value ?: throw invalid }.toSet()
            if (threshold > Int.MAX_VALUE) throw invalid
            return PortableBundleTrustPolicy(keys, threshold.toInt(), revoked)
        }
    }
}

/** Runtime environment a bundle must be compatible with. */
data class PortableBundleTarget(
    val platform: String,
    val sdkName: String,
    val sdkVersion: String,
    val runtimes: List<String>,
    val osVersion: String? = null,
    val policyVersion: Long? = null,
    val memoryBytes: Long? = null,
    val storageBytes: Long? = null,
) {
    init {
        val optional = { value: Long? -> value == null || value in 0..PortableBundle.MAX_SAFE_INTEGER }
        if (platform !in PortableBundle.PLATFORMS || !Patterns.TOKEN.matches(sdkName) ||
            !Patterns.VERSION.matches(sdkVersion) ||
            !runtimes.all { it in PortableBundle.FORMAT_PLATFORMS } ||
            (osVersion != null && !Patterns.VERSION.matches(osVersion)) ||
            !optional(policyVersion) || !optional(memoryBytes) || !optional(storageBytes)
        ) {
            throw PortableBundleException(PortableBundleReason.CONTEXT_INVALID)
        }
    }
}

/** Caller-controlled verification inputs. */
data class PortableBundleContext(
    val now: String,
    val target: PortableBundleTarget? = null,
    val sequenceFloors: Map<String, Long> = emptyMap(),
    val understoodExtensions: Set<String> = emptySet(),
    val maxManifestBytes: Int = 1 shl 20,
    val maxEnvelopeBytes: Int = 64 shl 10,
    val maxTotalBytes: Long = 64L shl 30,
    val maxFiles: Int = 4096,
) {
    init {
        if (!Patterns.TIMESTAMP.matches(now) || sequenceFloors.values.any { it < 0 } ||
            !understoodExtensions.all { Patterns.EXTENSION.matches(it) } ||
            maxManifestBytes <= 0 || maxEnvelopeBytes <= 0 || maxTotalBytes <= 0 || maxFiles <= 0
        ) {
            throw PortableBundleException(PortableBundleReason.CONTEXT_INVALID)
        }
    }
}

/** Bounded, value-free verification result. */
data class PortableBundleVerification(
    val ok: Boolean,
    val reason: PortableBundleReason?,
    val contentId: String?,
    val validSignatures: Int,
    val eligibleRuntimes: List<Int>,
    val filesChecked: Int,
    val filesVerified: Boolean,
) {
    /** Whether a separate activation step may consider the bundle. */
    val eligibleForActivation: Boolean get() = ok && filesVerified
}

/**
 * Offline reader for signed OpenMed portable model bundles.
 *
 * Mirrors `openmed/core/portable_bundle.py`: the same checks run in the same
 * order and produce the same result for `tests/fixtures/portable_bundle`. It
 * performs no network access and no logging, never executes bundled content,
 * and reports only reason codes, digests, counts, and indexes.
 */
object PortableBundle {
    const val SCHEMA = "openmed.portable_bundle"
    const val ENVELOPE_SCHEMA = "openmed.portable_bundle.signatures"
    const val MANIFEST_FILENAME = "openmed-bundle.json"
    const val ENVELOPE_FILENAME = "openmed-bundle.sig.json"
    const val PAYLOAD_TYPE = "application/vnd.openmed.portable-bundle.v1+json"
    const val MAX_SAFE_INTEGER = 9_007_199_254_740_991L
    internal const val MAX_SIGNATURES = 16

    internal val FORMAT_PLATFORMS = mapOf(
        "coreml" to setOf("ios", "macos", "watchos", "visionos"),
        "mlx" to setOf("ios", "macos"),
        "onnx" to setOf("android", "ios", "macos", "linux", "windows", "web"),
        "ort" to setOf("android", "ios"),
        "transformersjs" to setOf("web"),
        "pytorch" to setOf("linux", "macos", "windows"),
    )
    internal val FORMAT_PRECISIONS = mapOf(
        "coreml" to setOf("fp32", "fp16", "int8", "int4"),
        "mlx" to setOf("fp32", "fp16", "bf16", "int8", "int4", "int2"),
        "onnx" to setOf("fp32", "fp16", "int8", "int4"),
        "ort" to setOf("fp32", "fp16", "int8"),
        "transformersjs" to setOf("fp32", "fp16", "int8", "int4"),
        "pytorch" to setOf("fp32", "fp16", "bf16"),
    )
    internal val PLATFORMS = setOf("ios", "macos", "watchos", "visionos", "android", "web", "linux", "windows")
    private val PRECISIONS = setOf("fp32", "fp16", "bf16", "int8", "int4", "int2")
    private val FILE_ROLES = setOf(
        "calibration", "config", "decoder", "evidence", "graph", "labels", "license", "metadata", "policy",
        "provenance", "segmenter", "tokenizer", "weights",
    )
    private val OFFSET_UNITS = setOf("unicode_scalar", "utf16_code_unit", "utf8_byte")
    private val TOP_KEYS = listOf(
        "schema", "schema_version", "bundle", "model", "tokenizer", "labels", "offsets", "decoder", "segmenter",
        "policy", "calibration", "compatibility", "runtimes", "licenses", "provenance", "evidence", "files",
        "extensions", "critical_extensions",
    )

    /** Returns the `sha256:<hex>` content identity of canonical manifest bytes. */
    fun contentId(manifest: ByteArray): String = "sha256:" + sha256Hex(manifest)

    /** Returns the DSSE v1 pre-authentication encoding that signatures cover. */
    fun signingInput(manifest: ByteArray): ByteArray {
        val header = "DSSEv1 ${PAYLOAD_TYPE.length} $PAYLOAD_TYPE ${manifest.size} "
        return header.toByteArray(StandardCharsets.US_ASCII) + manifest
    }

    /** Returns RFC 8785 canonical bytes for JSON text in the portable subset. */
    fun canonicalJson(data: ByteArray): ByteArray = try {
        Json.encode(Json.parse(data))
    } catch (error: Json.Invalid) {
        throw PortableBundleException(PortableBundleReason.MANIFEST_NOT_CANONICAL)
    }

    /** Validates canonical manifest bytes without checking signatures. */
    fun validateManifest(manifest: ByteArray, understoodExtensions: Set<String> = emptySet()) {
        try {
            validate(parseCanonical(manifest), understoodExtensions)
        } catch (refusal: Refusal) {
            throw PortableBundleException(refusal.reason)
        }
    }

    /**
     * Verifies a portable bundle offline and returns a value-free result.
     *
     * @param manifest exact bytes of `openmed-bundle.json`.
     * @param envelope exact bytes of `openmed-bundle.sig.json`.
     * @param trustPolicy caller-controlled trust roots.
     * @param context verification instant, target, rollback floors, and limits.
     * @param files bundle files by relative path; `null` checks only the
     *   manifest and signatures.
     * @param verifier Ed25519 implementation; defaults to the bundled RFC 8032
     *   verifier because Android API 26 ships none.
     */
    fun verify(
        manifest: ByteArray,
        envelope: ByteArray,
        trustPolicy: PortableBundleTrustPolicy,
        context: PortableBundleContext,
        files: Map<String, ByteArray>? = null,
        verifier: Ed25519SignatureVerifier = Rfc8032Ed25519Verifier,
    ): PortableBundleVerification {
        var contentId: String? = null
        var validSignatures = 0
        var eligible: List<Int> = emptyList()
        var checked = 0
        try {
            if (manifest.size > context.maxManifestBytes) throw Refusal(PortableBundleReason.MANIFEST_TOO_LARGE)
            val identity = contentId(manifest)
            contentId = identity
            if (envelope.size > context.maxEnvelopeBytes) throw Refusal(PortableBundleReason.ENVELOPE_TOO_LARGE)
            val (declared, signatures) = parseEnvelope(envelope)
            if (declared != identity) throw Refusal(PortableBundleReason.CONTENT_ID_MISMATCH)
            if (identity in trustPolicy.revokedContentIds) throw Refusal(PortableBundleReason.BUNDLE_REVOKED)
            validSignatures = checkSignatures(signatures, manifest, trustPolicy, context.now, verifier)
            val validated = validate(parseCanonical(manifest), context.understoodExtensions)
            val declaredBytes = validated.files.sumOf { it.size }
            if (validated.files.size > context.maxFiles || declaredBytes > context.maxTotalBytes) {
                throw Refusal(PortableBundleReason.SIZE_LIMIT_EXCEEDED)
            }
            if (context.now < validated.issuedAt) throw Refusal(PortableBundleReason.BUNDLE_NOT_YET_VALID)
            if (context.now >= validated.expiresAt) throw Refusal(PortableBundleReason.BUNDLE_EXPIRED)
            val floor = context.sequenceFloors[validated.bundleId]
            if (floor != null && validated.sequence < floor) throw Refusal(PortableBundleReason.ROLLBACK_REJECTED)
            eligible = eligibleRuntimes(validated, context.target)
            if (files != null) {
                for (item in validated.files) {
                    val data = files[item.path] ?: throw Refusal(PortableBundleReason.FILE_MISSING)
                    if (data.size.toLong() != item.size) throw Refusal(PortableBundleReason.FILE_SIZE_MISMATCH)
                    if ("sha256:" + sha256Hex(data) != item.sha256) {
                        throw Refusal(PortableBundleReason.FILE_DIGEST_MISMATCH)
                    }
                    checked += 1
                }
                val declaredPaths = validated.files.map { it.path }.toSet()
                if (files.keys.any { it !in declaredPaths && it != MANIFEST_FILENAME && it != ENVELOPE_FILENAME }) {
                    throw Refusal(PortableBundleReason.FILE_UNDECLARED)
                }
            }
        } catch (refusal: Refusal) {
            return PortableBundleVerification(
                false, refusal.reason, contentId, validSignatures, eligible, checked, false,
            )
        }
        return PortableBundleVerification(true, null, contentId, validSignatures, eligible, checked, files != null)
    }

    internal class Refusal(val reason: PortableBundleReason) : Exception(reason.code)

    private class ValidatedFile(val path: String, val size: Long, val sha256: String)

    private class ValidatedPlatform(val name: String, val minOs: String?, val maxOs: String?)

    private class ValidatedRuntime(
        val format: String,
        val platforms: List<ValidatedPlatform>,
        val minMemory: Long,
        val minStorage: Long,
    )

    private class ValidatedSdk(val name: String, val min: String, val max: String?)

    private class ValidatedManifest(
        val bundleId: String,
        val sequence: Long,
        val issuedAt: String,
        val expiresAt: String,
        val sdks: List<ValidatedSdk>,
        val policyMin: Long,
        val policyMax: Long,
        val runtimes: List<ValidatedRuntime>,
        val files: List<ValidatedFile>,
    )

    internal fun sha256Hex(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }

    private fun parseCanonical(data: ByteArray): Json {
        val value = try {
            Json.parse(data)
        } catch (error: Json.Invalid) {
            throw Refusal(PortableBundleReason.MANIFEST_NOT_CANONICAL)
        }
        val encoded = try {
            Json.encode(value)
        } catch (error: Json.Invalid) {
            throw Refusal(PortableBundleReason.MANIFEST_NOT_CANONICAL)
        }
        if (!encoded.contentEquals(data)) throw Refusal(PortableBundleReason.MANIFEST_NOT_CANONICAL)
        return value
    }

    private fun parseEnvelope(data: ByteArray): Pair<String, List<Json.Obj>> {
        val malformed = Refusal(PortableBundleReason.ENVELOPE_MALFORMED)
        val value = try {
            parseCanonical(data)
        } catch (refusal: Refusal) {
            throw malformed
        }
        if (value !is Json.Obj || value["schema"] != Json.Str(ENVELOPE_SCHEMA)) throw malformed
        val version = value["schema_version"] as? Json.Num ?: throw malformed
        if (version.value != 1L) throw Refusal(PortableBundleReason.ENVELOPE_VERSION_UNSUPPORTED)
        if (!value.hasExactKeys("schema", "schema_version", "content_id", "signatures")) throw malformed
        val contentId = (value["content_id"] as? Json.Str)?.value ?: throw malformed
        val signatures = (value["signatures"] as? Json.Arr)?.items ?: throw malformed
        if (!Patterns.SHA256.matches(contentId) || signatures.size > MAX_SIGNATURES) throw malformed
        val entries = signatures.map { entry ->
            if (entry !is Json.Obj || !entry.hasExactKeys("kid", "alg", "sig")) throw malformed
            val kid = (entry["kid"] as? Json.Str)?.value ?: throw malformed
            if (!Patterns.KID.matches(kid) || entry["alg"] !is Json.Str || entry["sig"] !is Json.Str) throw malformed
            entry
        }
        return contentId to entries
    }

    private fun checkSignatures(
        signatures: List<Json.Obj>,
        manifest: ByteArray,
        policy: PortableBundleTrustPolicy,
        now: String,
        verifier: Ed25519SignatureVerifier,
    ): Int {
        if (signatures.isEmpty()) throw Refusal(PortableBundleReason.SIGNATURE_MISSING)
        val message = signingInput(manifest)
        val valid = linkedSetOf<String>()
        var firstFailure: PortableBundleReason? = null
        for (entry in signatures) {
            val kid = (entry["kid"] as Json.Str).value
            if (kid in valid) continue
            val failure = signatureFailure(entry, kid, message, policy, now, verifier)
            if (failure == null) valid += kid else if (firstFailure == null) firstFailure = failure
        }
        if (valid.size >= policy.threshold) return valid.size
        throw Refusal(firstFailure ?: PortableBundleReason.SIGNATURE_THRESHOLD_NOT_MET)
    }

    private fun signatureFailure(
        entry: Json.Obj,
        kid: String,
        message: ByteArray,
        policy: PortableBundleTrustPolicy,
        now: String,
        verifier: Ed25519SignatureVerifier,
    ): PortableBundleReason? {
        if ((entry["alg"] as Json.Str).value != "Ed25519") return PortableBundleReason.SIGNATURE_ALGORITHM_UNSUPPORTED
        val key = policy.key(kid) ?: return PortableBundleReason.UNTRUSTED_KEY
        if (key.revoked) return PortableBundleReason.KEY_REVOKED
        if (now < key.notBefore) return PortableBundleReason.KEY_NOT_YET_VALID
        if (now >= key.notAfter) return PortableBundleReason.KEY_EXPIRED
        val signature = Base64Url.decode((entry["sig"] as Json.Str).value, 64)
            ?: return PortableBundleReason.SIGNATURE_INVALID
        return if (verifier.verify(key.publicKey, signature, message)) null else PortableBundleReason.SIGNATURE_INVALID
    }

    private fun versionKey(value: String): List<Long> {
        val parts = value.split(".").map { it.toLong() }.toMutableList()
        while (parts.size < 4) parts += 0L
        return parts
    }

    private fun compareVersions(left: String, right: String): Int {
        for ((a, b) in versionKey(left).zip(versionKey(right))) {
            if (a != b) return if (a < b) -1 else 1
        }
        return 0
    }

    private fun eligibleRuntimes(manifest: ValidatedManifest, target: PortableBundleTarget?): List<Int> {
        if (target == null) return manifest.runtimes.indices.toList()
        val sdk = manifest.sdks.firstOrNull { it.name == target.sdkName }
        if (sdk == null || compareVersions(target.sdkVersion, sdk.min) < 0 ||
            (sdk.max != null && compareVersions(target.sdkVersion, sdk.max) >= 0)
        ) {
            throw Refusal(PortableBundleReason.SDK_INCOMPATIBLE)
        }
        val policyVersion = target.policyVersion
        if (policyVersion != null && !(manifest.policyMin <= policyVersion && policyVersion < manifest.policyMax)) {
            throw Refusal(PortableBundleReason.POLICY_INCOMPATIBLE)
        }
        val indexes = manifest.runtimes.indices.filter { runtimeFits(manifest.runtimes[it], target) }
        if (indexes.isEmpty()) throw Refusal(PortableBundleReason.PLATFORM_INCOMPATIBLE)
        return indexes
    }

    private fun runtimeFits(runtime: ValidatedRuntime, target: PortableBundleTarget): Boolean {
        if (runtime.format !in target.runtimes) return false
        if (target.memoryBytes != null && runtime.minMemory > target.memoryBytes) return false
        if (target.storageBytes != null && runtime.minStorage > target.storageBytes) return false
        val platform = runtime.platforms.firstOrNull { it.name == target.platform } ?: return false
        if (platform.minOs == null && platform.maxOs == null) return true
        val osVersion = target.osVersion ?: return false
        return (platform.minOs == null || compareVersions(osVersion, platform.minOs) >= 0) &&
            (platform.maxOs == null || compareVersions(osVersion, platform.maxOs) < 0)
    }

    // ---------------------------------------------------------------------
    // Manifest structure
    // ---------------------------------------------------------------------

    private fun malformed(): Refusal = Refusal(PortableBundleReason.MANIFEST_MALFORMED)

    private fun obj(value: Json?, vararg keys: String): Json.Obj {
        if (value !is Json.Obj) throw malformed()
        if (value.members.any { it.first !in keys }) throw Refusal(PortableBundleReason.UNKNOWN_CRITICAL_FIELD)
        if (value.members.size != keys.size) throw malformed()
        return value
    }

    private fun str(value: Json?, pattern: Regex): String {
        val text = (value as? Json.Str)?.value ?: throw malformed()
        if (!pattern.matches(text)) throw malformed()
        return text
    }

    private fun choice(value: Json?, choices: Set<String>): String {
        val text = (value as? Json.Str)?.value ?: throw malformed()
        if (text !in choices) throw malformed()
        return text
    }

    private fun uint(value: Json?, minimum: Long = 0): Long {
        val number = (value as? Json.Num)?.value ?: throw malformed()
        if (number < minimum || number > MAX_SAFE_INTEGER) throw malformed()
        return number
    }

    private fun arr(value: Json?, minimum: Int, maximum: Int): List<Json> {
        val items = (value as? Json.Arr)?.items ?: throw malformed()
        if (items.size < minimum || items.size > maximum) throw malformed()
        return items
    }

    private fun unique(items: List<String>) {
        if (items.toSet().size != items.size) throw malformed()
    }

    private fun path(value: Json?): String {
        val text = (value as? Json.Str)?.value ?: throw malformed()
        val segments = text.split("/")
        val lowered = text.lowercase()
        if (text.isEmpty() || text.length > 512 || segments.size > 16 ||
            !segments.all { Patterns.PATH_SEGMENT.matches(it) } ||
            lowered == MANIFEST_FILENAME || lowered == ENVELOPE_FILENAME
        ) {
            throw Refusal(PortableBundleReason.PATH_UNSAFE)
        }
        return text
    }

    private fun pathList(value: Json?, minimum: Int, maximum: Int): List<String> =
        arr(value, minimum, maximum).map { path(it) }.also { unique(it) }

    private fun version(value: Json?): String = str(value, Patterns.VERSION)

    private fun nullable(value: Json?): Boolean = value == Json.Null

    private fun validate(value: Json, understood: Set<String>): ValidatedManifest {
        if (value !is Json.Obj) throw malformed()
        if (value["schema"] != Json.Str(SCHEMA)) throw Refusal(PortableBundleReason.SCHEMA_UNSUPPORTED)
        val declaredVersion = value["schema_version"] as? Json.Num ?: throw malformed()
        if (declaredVersion.value != 1L) throw Refusal(PortableBundleReason.SCHEMA_VERSION_UNSUPPORTED)
        val top = obj(value, *TOP_KEYS.toTypedArray())
        val references = mutableListOf<String>()

        val bundle = obj(top["bundle"], "id", "version", "sequence", "issued_at", "expires_at")
        val bundleId = str(bundle["id"], Patterns.BUNDLE_ID)
        str(bundle["version"], Patterns.SEMVER)
        val sequence = uint(bundle["sequence"])
        val issuedAt = str(bundle["issued_at"], Patterns.TIMESTAMP)
        val expiresAt = str(bundle["expires_at"], Patterns.TIMESTAMP)
        if (issuedAt >= expiresAt) throw malformed()

        val model = obj(top["model"], "id", "revision", "family", "architecture", "task", "languages")
        str(model["id"], Patterns.MODEL_ID)
        if (!nullable(model["revision"])) str(model["revision"], Patterns.REVISION)
        str(model["family"], Patterns.TOKEN)
        str(model["architecture"], Patterns.TOKEN)
        str(model["task"], Patterns.TOKEN)
        unique(arr(model["languages"], 1, 32).map { str(it, Patterns.LANGUAGE) })

        if (!nullable(top["tokenizer"])) {
            val tokenizer = obj(top["tokenizer"], "kind", "contract_version", "files")
            str(tokenizer["kind"], Patterns.TOKEN)
            uint(tokenizer["contract_version"], 1)
            references += pathList(tokenizer["files"], 1, 64)
        }
        if (!nullable(top["labels"])) {
            val labels = obj(top["labels"], "path", "count", "scheme")
            references += path(labels["path"])
            uint(labels["count"], 1)
            str(labels["scheme"], Patterns.TOKEN)
        }
        val offsets = obj(top["offsets"], "unit", "contract_version")
        choice(offsets["unit"], OFFSET_UNITS)
        uint(offsets["contract_version"], 1)
        val decoder = obj(top["decoder"], "kind", "contract_version")
        str(decoder["kind"], Patterns.TOKEN)
        uint(decoder["contract_version"], 1)
        if (!nullable(top["segmenter"])) {
            val segmenter = obj(top["segmenter"], "id", "files")
            str(segmenter["id"], Patterns.TOKEN)
            references += pathList(segmenter["files"], 1, 64)
        }
        val policy = obj(top["policy"], "min_version", "max_version_exclusive", "path")
        val policyMin = uint(policy["min_version"], 1)
        val policyMax = uint(policy["max_version_exclusive"], 1)
        if (policyMin >= policyMax) throw malformed()
        if (!nullable(policy["path"])) references += path(policy["path"])
        if (!nullable(top["calibration"])) {
            val calibration = obj(top["calibration"], "profile_version", "path")
            uint(calibration["profile_version"], 1)
            references += path(calibration["path"])
        }

        val compatibility = obj(top["compatibility"], "sdks")
        val sdks = arr(compatibility["sdks"], 1, 16).map { entry ->
            val sdk = obj(entry, "name", "min", "max_exclusive")
            val name = str(sdk["name"], Patterns.TOKEN)
            val minimum = version(sdk["min"])
            val maximum = if (nullable(sdk["max_exclusive"])) null else version(sdk["max_exclusive"])
            if (maximum != null && compareVersions(minimum, maximum) >= 0) throw malformed()
            ValidatedSdk(name, minimum, maximum)
        }
        unique(sdks.map { it.name })

        val runtimeIds = mutableListOf<String>()
        val fallbacks = mutableListOf<String?>()
        val runtimes = arr(top["runtimes"], 1, 32).map { entry ->
            val runtime = obj(
                entry, "id", "format", "precision", "entrypoint", "files", "platforms", "min_memory_bytes",
                "min_storage_bytes", "fallback",
            )
            val runtimeId = str(runtime["id"], Patterns.TOKEN)
            val format = choice(runtime["format"], FORMAT_PLATFORMS.keys)
            val precision = choice(runtime["precision"], PRECISIONS)
            val entrypoint = path(runtime["entrypoint"])
            val runtimeFiles = pathList(runtime["files"], 1, 1024)
            if (entrypoint !in runtimeFiles) throw malformed()
            references += runtimeFiles
            val platforms = arr(runtime["platforms"], 1, 16).map { item ->
                val platform = obj(item, "platform", "min_os", "max_os_exclusive")
                val name = choice(platform["platform"], PLATFORMS)
                val minOs = if (nullable(platform["min_os"])) null else version(platform["min_os"])
                val maxOs = if (nullable(platform["max_os_exclusive"])) null else version(platform["max_os_exclusive"])
                if (minOs != null && maxOs != null && compareVersions(minOs, maxOs) >= 0) throw malformed()
                ValidatedPlatform(name, minOs, maxOs)
            }
            unique(platforms.map { it.name })
            val minMemory = uint(runtime["min_memory_bytes"])
            val minStorage = uint(runtime["min_storage_bytes"])
            val fallback = if (nullable(runtime["fallback"])) null else str(runtime["fallback"], Patterns.TOKEN)
            if (precision !in FORMAT_PRECISIONS.getValue(format) ||
                !platforms.all { it.name in FORMAT_PLATFORMS.getValue(format) }
            ) {
                throw Refusal(PortableBundleReason.RUNTIME_COMBINATION_UNSUPPORTED)
            }
            runtimeIds += runtimeId
            fallbacks += fallback
            ValidatedRuntime(format, platforms, minMemory, minStorage)
        }

        for (entry in arr(top["licenses"], 1, 16)) {
            val license = obj(entry, "id", "files")
            str(license["id"], Patterns.LICENSE)
            references += pathList(license["files"], 0, 64)
        }
        val provenance = obj(top["provenance"], "source_digest", "export_record")
        if (!nullable(provenance["source_digest"])) str(provenance["source_digest"], Patterns.SHA256)
        if (!nullable(provenance["export_record"])) references += path(provenance["export_record"])
        for (entry in arr(top["evidence"], 0, 64)) {
            val evidence = obj(entry, "kind", "path")
            str(evidence["kind"], Patterns.TOKEN)
            references += path(evidence["path"])
        }

        val files = arr(top["files"], 1, 65536).map { entry ->
            val item = obj(entry, "path", "media_type", "size_bytes", "sha256", "role", "executable")
            val filePath = path(item["path"])
            str(item["media_type"], Patterns.MEDIA_TYPE)
            val size = uint(item["size_bytes"])
            val digest = str(item["sha256"], Patterns.SHA256)
            choice(item["role"], FILE_ROLES)
            val executable = (item["executable"] as? Json.Bool)?.value ?: throw malformed()
            if (executable) throw Refusal(PortableBundleReason.EXECUTABLE_FILE_FORBIDDEN)
            ValidatedFile(filePath, size, digest)
        }

        val extensions = top["extensions"] as? Json.Obj ?: throw malformed()
        if (extensions.members.size > 64) throw malformed()
        extensions.members.forEach { str(Json.Str(it.first), Patterns.EXTENSION) }
        val critical = arr(top["critical_extensions"], 0, 64).map { str(it, Patterns.EXTENSION) }
        unique(critical)
        val extensionNames = extensions.members.map { it.first }.toSet()
        if (!critical.all { it in extensionNames }) throw malformed()

        val paths = files.map { it.path }
        val folded = paths.map { it.lowercase() }
        if (folded.toSet().size != folded.size) throw Refusal(PortableBundleReason.DUPLICATE_PATH)
        val foldedSet = folded.toSet()
        for (item in folded) {
            val segments = item.split("/")
            for (index in 1 until segments.size) {
                if (segments.subList(0, index).joinToString("/") in foldedSet) {
                    throw Refusal(PortableBundleReason.PATH_CONFLICT)
                }
            }
        }
        for (index in 0 until paths.size - 1) {
            if (paths[index] >= paths[index + 1]) throw Refusal(PortableBundleReason.FILE_ORDER_INVALID)
        }
        val declared = paths.toSet()
        if (!references.all { it in declared }) throw Refusal(PortableBundleReason.FILE_REFERENCE_UNDECLARED)

        unique(runtimeIds)
        fallbacks.forEachIndexed { index, fallback ->
            var current = fallback ?: return@forEachIndexed
            if (current !in runtimeIds || current == runtimeIds[index]) {
                throw Refusal(PortableBundleReason.FALLBACK_INVALID)
            }
            val seen = mutableSetOf(runtimeIds[index])
            while (true) {
                val position = runtimeIds.indexOf(current)
                if (current in seen || position < 0) throw Refusal(PortableBundleReason.FALLBACK_INVALID)
                seen += current
                current = fallbacks[position] ?: break
            }
        }

        if (!critical.all { it in understood }) throw Refusal(PortableBundleReason.UNKNOWN_CRITICAL_EXTENSION)

        return ValidatedManifest(
            bundleId, sequence, issuedAt, expiresAt, sdks, policyMin, policyMax, runtimes, files,
        )
    }
}

internal object Patterns {
    val TOKEN = Regex("[a-z0-9][a-z0-9._-]{0,63}")
    val BUNDLE_ID = Regex("[a-z0-9][a-z0-9._-]{0,127}")
    val MODEL_ID = Regex("[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
    val REVISION = Regex("[0-9a-f]{40}")
    val LANGUAGE = Regex("[a-z]{2,3}(?:-[A-Za-z0-9]{1,8}){0,4}")
    val LICENSE = Regex("[a-z0-9][a-z0-9.+-]{0,63}")
    val SHA256 = Regex("sha256:[0-9a-f]{64}")
    val MEDIA_TYPE = Regex("[a-z]+/[a-z0-9][a-z0-9.+-]{0,126}")
    val EXTENSION = Regex("[a-z0-9][a-z0-9-]{0,31}(?:\\.[a-z0-9][a-z0-9_-]{0,31}){1,7}")
    val SEMVER = Regex("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){2}")
    val VERSION = Regex("(?:0|[1-9][0-9]{0,8})(?:\\.(?:0|[1-9][0-9]{0,8})){0,3}")
    val TIMESTAMP = Regex(
        "[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z",
    )
    val KID = Regex("[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    val PATH_SEGMENT = Regex("[A-Za-z0-9_][A-Za-z0-9._+-]{0,127}")
}

internal object Base64Url {
    private const val ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

    fun encode(data: ByteArray): String {
        val output = StringBuilder()
        var index = 0
        while (index < data.size) {
            val remaining = data.size - index
            val a = data[index].toInt() and 0xff
            val b = if (remaining > 1) data[index + 1].toInt() and 0xff else 0
            val c = if (remaining > 2) data[index + 2].toInt() and 0xff else 0
            val triple = (a shl 16) or (b shl 8) or c
            output.append(ALPHABET[(triple shr 18) and 63]).append(ALPHABET[(triple shr 12) and 63])
            if (remaining > 1) output.append(ALPHABET[(triple shr 6) and 63])
            if (remaining > 2) output.append(ALPHABET[triple and 63])
            index += 3
        }
        return output.toString()
    }

    fun decode(text: String?, length: Int): ByteArray? {
        if (text == null || text.isEmpty() || text.length % 4 == 1) return null
        val bytes = ArrayList<Byte>()
        var accumulator = 0
        var bits = 0
        for (char in text) {
            val value = ALPHABET.indexOf(char)
            if (value < 0) return null
            accumulator = (accumulator shl 6) or value
            bits += 6
            if (bits >= 8) {
                bits -= 8
                bytes += ((accumulator shr bits) and 0xff).toByte()
            }
        }
        val raw = bytes.toByteArray()
        return if (raw.size == length && encode(raw) == text) raw else null
    }
}

/**
 * Minimal strict JSON value for canonical checks. Object members keep their
 * exact keys and order; duplicates are kept and then rejected on encode.
 */
internal sealed class Json {
    object Null : Json()
    data class Bool(val value: Boolean) : Json()
    data class Num(val value: Long) : Json()
    data class Str(val value: String) : Json()
    data class Arr(val items: List<Json>) : Json()
    data class Obj(val members: List<Pair<String, Json>>) : Json() {
        operator fun get(key: String): Json? = members.firstOrNull { it.first == key }?.second

        fun hasExactKeys(vararg keys: String): Boolean =
            members.size == keys.size && keys.all { key -> members.any { it.first == key } }
    }

    class Invalid : Exception("invalid")

    companion object {
        private const val MAX_DEPTH = 32

        fun parse(data: ByteArray): Json {
            val parser = Parser(data)
            parser.skipWhitespace()
            val value = parser.value(0)
            parser.skipWhitespace()
            if (parser.index != data.size) throw Invalid()
            return value
        }

        fun encode(value: Json): ByteArray {
            val output = StringBuilder()
            encode(value, output, 0)
            return output.toString().toByteArray(StandardCharsets.UTF_8)
        }

        private fun encode(value: Json, output: StringBuilder, depth: Int) {
            if (depth > MAX_DEPTH) throw Invalid()
            when (value) {
                Null -> output.append("null")
                is Bool -> output.append(if (value.value) "true" else "false")
                is Num -> {
                    if (value.value !in -PortableBundle.MAX_SAFE_INTEGER..PortableBundle.MAX_SAFE_INTEGER) {
                        throw Invalid()
                    }
                    output.append(value.value)
                }
                is Str -> encodeString(value.value, output)
                is Arr -> {
                    output.append('[')
                    value.items.forEachIndexed { index, item ->
                        if (index > 0) output.append(',')
                        encode(item, output, depth + 1)
                    }
                    output.append(']')
                }
                is Obj -> {
                    // String.compareTo orders by UTF-16 code units, as RFC 8785 requires.
                    val sorted = value.members.sortedWith { a, b -> a.first.compareTo(b.first) }
                    for (index in 1 until sorted.size) {
                        if (sorted[index - 1].first == sorted[index].first) throw Invalid()
                    }
                    output.append('{')
                    sorted.forEachIndexed { index, member ->
                        if (index > 0) output.append(',')
                        encodeString(member.first, output)
                        output.append(':')
                        encode(member.second, output, depth + 1)
                    }
                    output.append('}')
                }
            }
        }

        private fun encodeString(text: String, output: StringBuilder) {
            output.append('"')
            var index = 0
            while (index < text.length) {
                val char = text[index]
                when {
                    char == '"' -> output.append("\\\"")
                    char == '\\' -> output.append("\\\\")
                    char == '\b' -> output.append("\\b")
                    char == '\t' -> output.append("\\t")
                    char == '\n' -> output.append("\\n")
                    char == '\u000C' -> output.append("\\f")
                    char == '\r' -> output.append("\\r")
                    char < ' ' -> output.append("\\u%04x".format(char.code))
                    Character.isHighSurrogate(char) -> {
                        val next = text.getOrNull(index + 1)
                        if (next == null || !Character.isLowSurrogate(next)) throw Invalid()
                        output.append(char).append(next)
                        index += 1
                    }
                    Character.isLowSurrogate(char) -> throw Invalid()
                    else -> output.append(char)
                }
                index += 1
            }
            output.append('"')
        }
    }

    private class Parser(private val bytes: ByteArray) {
        var index = 0
        private val whitespace = setOf(0x20, 0x09, 0x0A, 0x0D)

        fun skipWhitespace() {
            while (index < bytes.size && bytes[index].toInt() in whitespace) index += 1
        }

        private fun peek(): Int = if (index < bytes.size) bytes[index].toInt() and 0xff else throw Invalid()

        private fun expect(literal: String) {
            for (char in literal) {
                if (peek() != char.code) throw Invalid()
                index += 1
            }
        }

        fun value(depth: Int): Json {
            if (depth > MAX_DEPTH) throw Invalid()
            return when (peek()) {
                'n'.code -> Null.also { expect("null") }
                't'.code -> Bool(true).also { expect("true") }
                'f'.code -> Bool(false).also { expect("false") }
                '"'.code -> Str(string())
                '['.code -> array(depth)
                '{'.code -> obj(depth)
                else -> Num(integer())
            }
        }

        private fun array(depth: Int): Json {
            index += 1
            val items = mutableListOf<Json>()
            skipWhitespace()
            if (peek() == ']'.code) {
                index += 1
                return Arr(items)
            }
            while (true) {
                skipWhitespace()
                items += value(depth + 1)
                skipWhitespace()
                if (peek() == ']'.code) {
                    index += 1
                    return Arr(items)
                }
                expect(",")
            }
        }

        private fun obj(depth: Int): Json {
            index += 1
            val members = mutableListOf<Pair<String, Json>>()
            skipWhitespace()
            if (peek() == '}'.code) {
                index += 1
                return Obj(members)
            }
            while (true) {
                skipWhitespace()
                if (peek() != '"'.code) throw Invalid()
                val key = string()
                skipWhitespace()
                expect(":")
                skipWhitespace()
                members += key to value(depth + 1)
                skipWhitespace()
                if (peek() == '}'.code) {
                    index += 1
                    return Obj(members)
                }
                expect(",")
            }
        }

        private fun integer(): Long {
            val start = index
            if (index < bytes.size && bytes[index] == '-'.code.toByte()) index += 1
            val digitsStart = index
            while (index < bytes.size && bytes[index] in '0'.code.toByte()..'9'.code.toByte()) index += 1
            val digits = index - digitsStart
            if (digits == 0 || digits > 16 || (digits > 1 && bytes[digitsStart] == '0'.code.toByte())) throw Invalid()
            if (index < bytes.size && bytes[index].toInt().let { it == '.'.code || it == 'e'.code || it == 'E'.code }) {
                throw Invalid()
            }
            val number = String(bytes, start, index - start, StandardCharsets.US_ASCII).toLongOrNull()
                ?: throw Invalid()
            if (number < -PortableBundle.MAX_SAFE_INTEGER || number > PortableBundle.MAX_SAFE_INTEGER) throw Invalid()
            return number
        }

        private fun hexQuad(): Int {
            if (index + 4 > bytes.size) throw Invalid()
            val text = String(bytes, index, 4, StandardCharsets.US_ASCII)
            if (!text.all { it in '0'..'9' || it in 'a'..'f' || it in 'A'..'F' }) throw Invalid()
            index += 4
            return text.toInt(16)
        }

        private fun string(): String {
            expect("\"")
            val output = java.io.ByteArrayOutputStream()
            while (true) {
                val byte = peek()
                index += 1
                when {
                    byte == '"'.code -> return decodeUtf8(output.toByteArray())
                    byte == '\\'.code -> {
                        val escape = peek()
                        index += 1
                        when (escape) {
                            '"'.code -> output.write(0x22)
                            '\\'.code -> output.write(0x5C)
                            '/'.code -> output.write(0x2F)
                            'b'.code -> output.write(0x08)
                            'f'.code -> output.write(0x0C)
                            'n'.code -> output.write(0x0A)
                            'r'.code -> output.write(0x0D)
                            't'.code -> output.write(0x09)
                            'u'.code -> {
                                var code = hexQuad()
                                if (code in 0xD800..0xDBFF) {
                                    expect("\\u")
                                    val low = hexQuad()
                                    if (low !in 0xDC00..0xDFFF) throw Invalid()
                                    code = 0x10000 + ((code - 0xD800) shl 10) + (low - 0xDC00)
                                } else if (code in 0xDC00..0xDFFF) {
                                    throw Invalid()
                                }
                                output.write(String(Character.toChars(code)).toByteArray(StandardCharsets.UTF_8))
                            }
                            else -> throw Invalid()
                        }
                    }
                    byte < 0x20 -> throw Invalid()
                    else -> output.write(byte)
                }
            }
        }

        private fun decodeUtf8(data: ByteArray): String = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(data))
                .toString()
        } catch (error: java.nio.charset.CharacterCodingException) {
            throw Invalid()
        }
    }
}
