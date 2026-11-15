package com.openmed.openmedkit.parityprobe

import android.content.pm.ApplicationInfo
import android.os.Build
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.module.annotations.ReactModule
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.policy.PolicyProfiles
import java.io.File
import java.security.MessageDigest
import java.security.SecureRandom
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject

/**
 * Example-only native module that calls OpenMedKit Android directly.
 *
 * The React Native parity harness compares `OpenMedKitRN` bridge results with
 * the results of this module. It never goes through `OpenMedKitRN`: it builds
 * its own [OpenMedKit] runtime from the same verified local bundle and returns
 * results as a JSON string. Rejections and error records carry fixed codes
 * only, never input text, file paths, or exception messages.
 */
@ReactModule(name = OpenMedKitParityProbeModule.NAME)
class OpenMedKitParityProbeModule(
    reactContext: ReactApplicationContext,
) : ReactContextBaseJavaModule(reactContext) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var runtime: OpenMedKit? = null
    private var runtimeKey: String? = null
    private var fixture: VerifiedFixture? = null

    override fun getName(): String = NAME

    override fun invalidate() {
        executor.shutdownNow()
        runtime?.close()
        runtime = null
        super.invalidate()
    }

    @ReactMethod
    fun describeEnvironment(promise: Promise) {
        val debuggable =
            (reactApplicationContext.applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE) != 0
        promise.resolve(
            Arguments.createMap().apply {
                putString("platform", "android")
                putString("runtime_backend", "onnx")
                putString("device_class", if (isEmulator()) "emulator" else "device")
                putString("os_version", "${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT})")
                putString("build_configuration", if (debuggable) "debug" else "release")
            },
        )
    }

    @ReactMethod
    fun prepareModelFixture(promise: Promise) {
        executor.execute {
            try {
                val verified = FixtureVerifier.verify(fixtureRoot())
                fixture = verified
                promise.resolve(verified.toWritableMap())
            } catch (error: ProbeException) {
                promise.reject(error.code, "model fixture verification failed")
            } catch (error: Throwable) {
                promise.reject("model_fixture_unverified", "model fixture verification failed")
            }
        }
    }

    @ReactMethod
    fun createEphemeralHashSecret(promise: Promise) {
        val bytes = ByteArray(32).also { SecureRandom().nextBytes(it) }
        promise.resolve(bytes.toHex())
    }

    @ReactMethod
    fun runDirect(
        requestJson: String,
        promise: Promise,
    ) {
        executor.execute {
            var operation = "unknown"
            val response = try {
                val request = DirectRequest.parse(requestJson)
                operation = request.operation
                val verified = fixture ?: throw ProbeException("direct_model_unverified")
                if (!request.model.matches(verified)) {
                    throw ProbeException("direct_model_unverified")
                }
                JSONObject()
                    .put("ok", true)
                    .put("result", run(request, runtimeFor(request.model)))
            } catch (error: ProbeException) {
                failure(error.code)
            } catch (error: Throwable) {
                failure("direct_${operation}_failed")
            }
            promise.resolve(response.toString())
        }
    }

    @ReactMethod
    fun writeReport(
        reportJson: String,
        promise: Promise,
    ) {
        executor.execute {
            try {
                val directory = externalFilesDirectory()
                val target = File(directory, REPORT_FILE_NAME)
                val staging = File(directory, "$REPORT_FILE_NAME.tmp")
                staging.writeText(reportJson, Charsets.UTF_8)
                if (!staging.renameTo(target)) {
                    throw ProbeException("report_write_failed")
                }
                promise.resolve(true)
            } catch (error: Throwable) {
                promise.reject("report_write_failed", "parity report could not be written")
            }
        }
    }

    private fun runtimeFor(model: DirectModel): OpenMedKit {
        val current = runtime
        if (current != null && runtimeKey == model.cacheKey) {
            return current
        }
        current?.close()
        runtime = null
        runtimeKey = null
        if (model.backend != "onnx") {
            throw ProbeException("direct_backend_unsupported")
        }
        // Same construction as the OpenMedKitRN bridge: model directory plus
        // an explicit id2label map read from the verified bundle.
        val loaded = OpenMedKit(
            OpenMedBackend(
                modelDirectory = File(model.modelPath),
                id2Label = readId2Label(model.id2LabelPath),
            ),
        )
        runtime = loaded
        runtimeKey = model.cacheKey
        return loaded
    }

    private fun run(
        request: DirectRequest,
        kit: OpenMedKit,
    ): JSONObject = runBlocking {
        val options = request.options
        val text = request.text
        when (request.operation) {
            "analyzeText" -> JSONObject().put(
                "spans",
                entitySpans(
                    kit.analyzeText(text, options.confidenceThreshold),
                    text,
                    options,
                ),
            )
            "extractPii" -> JSONObject().put(
                "spans",
                entitySpans(
                    kit.extractPii(text, options.confidenceThreshold, options.useSmartMerging),
                    text,
                    options,
                ),
            )
            "deidentify" -> {
                val result = kit.deidentify(
                    text = text,
                    policy = options.policy ?: PolicyProfiles.DEFAULT_PROFILE,
                    confidenceThreshold = options.confidenceThreshold,
                    useSmartMerging = options.useSmartMerging,
                )
                val spans = JSONArray()
                result.actions.forEach { action ->
                    spans.put(
                        span(
                            label = action.label,
                            canonicalLabel = action.canonicalLabel,
                            start = action.start,
                            end = action.end,
                            confidence = action.confidence,
                            action = action.action.wireName,
                            replacement = action.replacement,
                            text = text,
                            options = options,
                        ),
                    )
                }
                JSONObject()
                    .put("text", text)
                    .put("deidentifiedText", result.redactedText)
                    .put("spans", spans)
            }
            else -> throw ProbeException("direct_request_invalid")
        }
    }

    private fun entitySpans(
        entities: List<EntityPrediction>,
        text: String,
        options: DirectOptions,
    ): JSONArray {
        val spans = JSONArray()
        entities.forEach { entity ->
            spans.put(
                span(
                    label = entity.label,
                    canonicalLabel = PolicyProfiles.canonicalLabel(entity.label),
                    start = entity.start,
                    end = entity.end,
                    confidence = entity.confidence,
                    action = "keep",
                    replacement = null,
                    text = text,
                    options = options,
                ),
            )
        }
        return spans
    }

    /** The documented span schema v1, built from direct native values. */
    private fun span(
        label: String,
        canonicalLabel: String,
        start: Int,
        end: Int,
        confidence: Float,
        action: String,
        replacement: String?,
        text: String,
        options: DirectOptions,
    ): JSONObject = JSONObject()
        .put("schema_version", 1)
        .put("doc_id", options.docId)
        .put("start", start)
        .put("end", end)
        .put("text_hash", hmacTextHash(scalarSubstring(text, start, end), options.hashSecret))
        .put("entity_type", label)
        .put("canonical_label", canonicalLabel)
        .put("policy_label", policyLabel(canonicalLabel))
        .put("regulatory_tags", JSONArray())
        .put("score", confidence.toDouble())
        .put("detector", "openmedkit-android")
        .put(
            "evidence",
            JSONObject().put("bridge", "react-native").put("runtime", "OpenMedKit"),
        )
        .put("action", action)
        .put("replacement", replacement ?: JSONObject.NULL)
        .put("reversible_id", JSONObject.NULL)
        .put("section", JSONObject.NULL)
        .put("metadata", JSONObject(options.metadata))

    /** Slice by Unicode scalar offsets, the OpenMed offset unit. */
    private fun scalarSubstring(
        text: String,
        start: Int,
        end: Int,
    ): String = try {
        UnicodeOffsetContract.substring(text, start, end)
    } catch (error: IllegalArgumentException) {
        ""
    }

    private fun hmacTextHash(
        surface: String,
        secret: ByteArray,
    ): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret, "HmacSHA256"))
        return "hmac-sha256:" + mac.doFinal(surface.toByteArray(Charsets.UTF_8)).toHex()
    }

    private fun readId2Label(path: String?): Map<Int, String> {
        if (path.isNullOrBlank()) {
            return emptyMap()
        }
        val root = JSONObject(File(path).readText(Charsets.UTF_8))
        return root.keys().asSequence().associate { key -> key.toInt() to root.getString(key) }
    }

    private fun failure(code: String): JSONObject =
        JSONObject().put("ok", false).put("error", JSONObject().put("code", code))

    private fun externalFilesDirectory(): File =
        reactApplicationContext.getExternalFilesDir(null)
            ?: throw ProbeException("model_fixture_missing")

    private fun fixtureRoot(): File = File(externalFilesDirectory(), FIXTURE_DIRECTORY_NAME)

    private fun isEmulator(): Boolean =
        Build.FINGERPRINT.startsWith("generic") ||
            Build.FINGERPRINT.contains("emulator") ||
            Build.HARDWARE == "ranchu" ||
            Build.HARDWARE == "goldfish" ||
            Build.PRODUCT.contains("sdk")

    companion object {
        const val NAME = "OpenMedKitParityProbe"
        const val FIXTURE_DIRECTORY_NAME = "openmed-parity-fixture"
        const val REPORT_FILE_NAME = "openmed-rn-parity-report.json"

        /** Mirrors the private policy-label table in OpenMedKit `PolicyProfiles.kt`. */
        private val CLINICAL_CONCEPT_LABELS = setOf(
            "MICROORGANISM",
            "ANTIBIOTIC",
            "SUSCEPTIBILITY",
            "CONDITION",
            "MEDICATION",
            "LAB_TEST",
            "PROCEDURE",
            "BODY_SITE",
            "DIET_TYPE",
            "NUTRITION_TARGET",
            "FEEDING_ROUTE",
            "NUTRITIONAL_STATUS",
            "OTHER",
        )

        private val QUASI_IDENTIFIER_LABELS = setOf(
            "LOCATION",
            "ZIPCODE",
            "ORDINAL_DIRECTION",
            "DATE",
            "TIME",
            "AGE",
            "CREDIT_CARD_ISSUER",
            "AMOUNT",
            "CURRENCY",
            "GENDER",
            "EYE_COLOR",
            "HEIGHT",
            "ORGANIZATION",
            "JOB_TITLE",
            "JOB_DEPARTMENT",
            "OCCUPATION",
        )

        private fun policyLabel(canonicalLabel: String): String = when (canonicalLabel) {
            in CLINICAL_CONCEPT_LABELS -> "CLINICAL_CONCEPT"
            in QUASI_IDENTIFIER_LABELS -> "QUASI_IDENTIFIER"
            else -> "DIRECT_IDENTIFIER"
        }
    }
}

private class ProbeException(val code: String) : Exception(code)

private class DirectOptions(
    val docId: String,
    val hashSecret: ByteArray,
    val metadata: Map<String, Any?>,
    val confidenceThreshold: Float,
    val useSmartMerging: Boolean,
    val policy: String?,
)

private class DirectModel(
    val cacheKey: String,
    val modelPath: String,
    val backend: String,
    val id2LabelPath: String?,
    val tokenizerFolderPath: String?,
) {
    fun matches(fixture: VerifiedFixture): Boolean =
        modelPath == fixture.modelPath &&
            backend == fixture.backend &&
            id2LabelPath == fixture.id2LabelPath &&
            tokenizerFolderPath == fixture.tokenizerFolderPath
}

private class DirectRequest(
    val operation: String,
    val text: String,
    val options: DirectOptions,
    val model: DirectModel,
) {
    companion object {
        fun parse(json: String): DirectRequest {
            try {
                val root = JSONObject(json)
                val options = root.getJSONObject("options")
                val model = root.getJSONObject("model")
                val secret = options.getString("hashSecret")
                if (secret.isEmpty()) {
                    throw ProbeException("direct_request_invalid")
                }
                val metadata = options.optJSONObject("metadata")
                return DirectRequest(
                    operation = root.getString("operation"),
                    text = root.getString("text"),
                    options = DirectOptions(
                        docId = options.getString("docId"),
                        hashSecret = secret.toByteArray(Charsets.UTF_8),
                        metadata = metadata?.keys()?.asSequence()
                            ?.associateWith { metadata.get(it) }
                            ?: emptyMap(),
                        confidenceThreshold = options.getDouble("confidenceThreshold").toFloat(),
                        useSmartMerging = options.getBoolean("useSmartMerging"),
                        policy = options.optString("policy").takeIf { options.has("policy") },
                    ),
                    model = DirectModel(
                        cacheKey = model.getString("cacheKey"),
                        modelPath = model.getString("modelPath"),
                        backend = model.getString("backend"),
                        id2LabelPath = model.optString("id2LabelPath").takeIf { model.has("id2LabelPath") },
                        tokenizerFolderPath = model.optString("tokenizerFolderPath")
                            .takeIf { model.has("tokenizerFolderPath") },
                    ),
                )
            } catch (error: ProbeException) {
                throw error
            } catch (error: Throwable) {
                throw ProbeException("direct_request_invalid")
            }
        }
    }
}

private class VerifiedFixture(
    val modelPath: String,
    val backend: String,
    val id2LabelPath: String,
    val tokenizerFolderPath: String?,
    val bundleDigest: String,
    val fileCount: Int,
) {
    fun toWritableMap() = Arguments.createMap().apply {
        putString("modelPath", modelPath)
        putString("backend", backend)
        putString("id2LabelPath", id2LabelPath)
        tokenizerFolderPath?.let { putString("tokenizerFolderPath", it) }
        putString("bundleDigest", bundleDigest)
        putInt("fileCount", fileCount)
    }
}

/** Verifies a staged fixture exactly as `example/src/parity/fixture.ts` defines it. */
private object FixtureVerifier {
    private val SAFE_SEGMENT = Regex("^[A-Za-z0-9._-]+$")

    fun verify(root: File): VerifiedFixture {
        val manifestFile = File(root, "openmed-parity-fixture.json")
        if (!manifestFile.isFile) {
            throw ProbeException("model_fixture_missing")
        }
        val manifest = try {
            JSONObject(manifestFile.readText(Charsets.UTF_8))
        } catch (error: Throwable) {
            throw ProbeException("model_fixture_manifest_invalid")
        }
        val files = manifest.optJSONArray("files")
            ?: throw ProbeException("model_fixture_manifest_invalid")
        if (
            manifest.optString("schema") != "openmed.rn.parity.model_fixture" ||
            manifest.optInt("version") != 1 ||
            manifest.optString("platform") != "android" ||
            files.length() == 0
        ) {
            throw ProbeException("model_fixture_manifest_invalid")
        }

        data class Entry(val path: String, val sha256: String, val bytes: Long)

        val entries = (0 until files.length()).map { index ->
            val file = files.getJSONObject(index)
            Entry(file.getString("path"), file.getString("sha256"), file.getLong("bytes"))
        }
        val modelPath = manifest.getString("model_path")
        val id2LabelPath = manifest.getString("id2label_path")
        val tokenizerFolderPath =
            if (manifest.isNull("tokenizer_folder_path")) null else manifest.getString("tokenizer_folder_path")
        if (
            (entries.map { it.path } + listOfNotNull(modelPath, id2LabelPath, tokenizerFolderPath))
                .any { !isSafeRelativePath(it) } ||
            entries.map { it.path }.toSet().size != entries.size
        ) {
            throw ProbeException("model_fixture_manifest_invalid")
        }

        val canonicalRoot = root.canonicalFile
        val modelRoot = File(canonicalRoot, "model")
        val found = mutableSetOf<String>()
        modelRoot.walkTopDown().forEach { file ->
            if (file.canonicalPath != file.absolutePath) {
                throw ProbeException("model_fixture_symlink")
            }
            if (file.isFile) {
                found += "model/" + file.relativeTo(modelRoot).invariantSeparatorsPath
            }
        }
        if (found != entries.map { it.path }.toSet()) {
            throw ProbeException("model_fixture_file_set_mismatch")
        }
        for (entry in entries) {
            val file = File(canonicalRoot, entry.path)
            val (digest, bytes) = sha256(file)
            if (digest != entry.sha256 || bytes != entry.bytes) {
                throw ProbeException("model_fixture_digest_mismatch")
            }
        }
        val lines = entries
            .sortedWith { left, right -> compareUtf8(left.path, right.path) }
            .joinToString("") { "${it.path}\t${it.sha256}\t${it.bytes}\n" }
        val digest = "sha256:" + MessageDigest.getInstance("SHA-256")
            .digest(lines.toByteArray(Charsets.UTF_8))
            .toHex()
        if (digest != manifest.getString("bundle_digest")) {
            throw ProbeException("model_fixture_bundle_digest_mismatch")
        }

        return VerifiedFixture(
            modelPath = File(canonicalRoot, modelPath).path,
            backend = manifest.getString("backend"),
            id2LabelPath = File(canonicalRoot, id2LabelPath).path,
            tokenizerFolderPath = tokenizerFolderPath?.let { File(canonicalRoot, it).path },
            bundleDigest = digest,
            fileCount = entries.size,
        )
    }

    private fun isSafeRelativePath(path: String): Boolean {
        val segments = path.split("/")
        return segments.first() == "model" &&
            segments.all { it.isNotEmpty() && it != "." && it != ".." && SAFE_SEGMENT.matches(it) }
    }

    private fun compareUtf8(
        left: String,
        right: String,
    ): Int {
        val leftBytes = left.toByteArray(Charsets.UTF_8)
        val rightBytes = right.toByteArray(Charsets.UTF_8)
        for (index in 0 until minOf(leftBytes.size, rightBytes.size)) {
            val difference = (leftBytes[index].toInt() and 0xff) - (rightBytes[index].toInt() and 0xff)
            if (difference != 0) {
                return difference
            }
        }
        return leftBytes.size - rightBytes.size
    }

    private fun sha256(file: File): Pair<String, Long> {
        val digest = MessageDigest.getInstance("SHA-256")
        var total = 0L
        file.inputStream().buffered().use { stream ->
            val buffer = ByteArray(1 shl 20)
            while (true) {
                val read = stream.read(buffer)
                if (read < 0) {
                    break
                }
                digest.update(buffer, 0, read)
                total += read
            }
        }
        return digest.digest().toHex() to total
    }
}

private fun ByteArray.toHex(): String = joinToString("") { "%02x".format(it.toInt() and 0xff) }
