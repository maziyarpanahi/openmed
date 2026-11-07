package com.openmed.openmedkit.reactnative

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.Arguments
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.Promise
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.bridge.ReactContextBaseJavaModule
import com.facebook.react.bridge.ReactMethod
import com.facebook.react.bridge.ReadableMap
import com.facebook.react.bridge.WritableMap
import com.facebook.react.module.annotations.ReactModule
import com.facebook.react.uimanager.ViewManager
import com.openmed.openmedkit.DeidentifiedSpanAction
import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMedBackend
import com.openmed.openmedkit.OpenMedHostScope
import com.openmed.openmedkit.OpenMedKit
import com.openmed.openmedkit.OpenMedRuntimeException
import com.openmed.openmedkit.OpenMedSessionFactory
import com.openmed.openmedkit.OpenMedTrimMemorySource
import java.io.File
import java.security.SecureRandom
import java.util.concurrent.atomic.AtomicBoolean
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import org.json.JSONObject

@ReactModule(name = OpenMedKitRnModule.NAME)
class OpenMedKitRnModule(
    reactContext: ReactApplicationContext,
) : ReactContextBaseJavaModule(reactContext) {
    // Scope, runtime, and trim-memory registration live exactly as long as
    // this module; invalidate() cancels the scope and releases sessions.
    private val host = OpenMedHostScope(
        lifecycleSources = listOf(OpenMedTrimMemorySource(reactContext)),
    )

    override fun getName(): String = NAME

    override fun invalidate() {
        host.invalidate()
        super.invalidate()
    }

    @ReactMethod
    fun loadModel(
        options: ReadableMap,
        promise: Promise,
    ) {
        launchOperation(promise, "openmedkit_load_failed") {
            val modelPath = options.requiredString("modelPath")
            val backendName = options.string("backend") ?: "onnx"
            val cacheKey = options.string("cacheKey") ?: "$backendName:$modelPath"
            require(backendName == "onnx") { "unsupported OpenMedKit Android backend" }
            val id2Label = readId2Label(options.string("id2LabelPath"))

            val loaded = host.load(cacheKey) {
                OpenMedKit(
                    OpenMedBackend(
                        modelDirectory = File(modelPath),
                        id2Label = id2Label,
                    ),
                )
            }
            Arguments.createMap().apply {
                putString("cacheKey", cacheKey)
                putString("modelPath", modelPath)
                putString("backend", backendName)
                putString("platform", "android")
                putBoolean("loaded", loaded)
            }
        }
    }

    @ReactMethod
    fun analyzeText(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        launchOperation(promise, "openmedkit_analyze_failed") {
            val bridgeOptions = BridgeOptions(options)
            val entities = host.requireRuntime().analyzeText(
                text = text,
                confidenceThreshold = bridgeOptions.confidenceThreshold,
            )
            currentCoroutineContext().ensureActive()
            entityArray(
                entities = entities,
                text = text,
                options = bridgeOptions,
                action = "keep",
                replacement = null,
            )
        }
    }

    @ReactMethod
    fun extractPii(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        launchOperation(promise, "openmedkit_extract_failed") {
            val bridgeOptions = BridgeOptions(options)
            val entities = host.requireRuntime().extractPii(
                text = text,
                confidenceThreshold = bridgeOptions.confidenceThreshold,
                useSmartMerging = bridgeOptions.useSmartMerging,
            )
            currentCoroutineContext().ensureActive()
            entityArray(
                entities = entities,
                text = text,
                options = bridgeOptions,
                action = "keep",
                replacement = null,
            )
        }
    }

    @ReactMethod
    fun deidentify(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        launchOperation(promise, "openmedkit_deidentify_failed") {
            val bridgeOptions = BridgeOptions(options)
            val result = host.requireRuntime().deidentify(
                text = text,
                policy = bridgeOptions.policy,
                confidenceThreshold = bridgeOptions.confidenceThreshold,
                useSmartMerging = bridgeOptions.useSmartMerging,
            )
            currentCoroutineContext().ensureActive()
            Arguments.createMap().apply {
                putString("text", text)
                putString("deidentifiedText", result.redactedText)
                putArray(
                    "spans",
                    actionArray(
                        actions = result.actions,
                        text = text,
                        options = bridgeOptions,
                    ),
                )
            }
        }
    }

    /**
     * Run [operation] in the module-owned scope and settle [promise] once.
     *
     * Results are resolved only while the scope is active, so a cancelled or
     * invalidated operation never resolves with a late result. Failures are
     * rejected with value-free messages; the underlying throwable is never
     * forwarded because its message may contain paths or input text.
     */
    private fun launchOperation(
        promise: Promise,
        failureCode: String,
        operation: suspend () -> Any?,
    ) {
        val settled = AtomicBoolean(false)
        val job = host.launch {
            try {
                val value = operation()
                currentCoroutineContext().ensureActive()
                if (settled.compareAndSet(false, true)) {
                    promise.resolve(value)
                }
            } catch (error: CancellationException) {
                throw error
            } catch (error: Throwable) {
                if (settled.compareAndSet(false, true)) {
                    rejectValueFree(promise, failureCode, error)
                }
            }
        }
        job.invokeOnCompletion { cause ->
            if (cause != null && settled.compareAndSet(false, true)) {
                try {
                    promise.reject(
                        CANCELLED_CODE,
                        "OpenMedKit operation was cancelled",
                        failureInfo(reason = "cancelled", retry = null),
                    )
                } catch (_: RuntimeException) {
                    // The bridge may already be torn down; nothing to report.
                }
            }
        }
    }

    private fun rejectValueFree(
        promise: Promise,
        failureCode: String,
        error: Throwable,
    ) {
        if (error is OpenMedRuntimeException) {
            promise.reject(
                failureCode,
                "OpenMedKit runtime: ${error.code}",
                failureInfo(
                    reason = error.code,
                    retry = error.retryGuidance.name,
                    retryAfterMillis = error.retryAfterMillis,
                ),
            )
        } else {
            promise.reject(
                failureCode,
                "OpenMedKit operation failed",
                failureInfo(reason = "operation_failed", retry = null),
            )
        }
    }

    private fun failureInfo(
        reason: String,
        retry: String?,
        retryAfterMillis: Long? = null,
    ): WritableMap = Arguments.createMap().apply {
        putString("reason", reason)
        if (retry == null) {
            putNull("retry")
        } else {
            putString("retry", retry)
        }
        if (retryAfterMillis == null) {
            putNull("retryAfterMillis")
        } else {
            putDouble("retryAfterMillis", retryAfterMillis.toDouble())
        }
    }

    private fun entityArray(
        entities: List<EntityPrediction>,
        text: String,
        options: BridgeOptions,
        action: String,
        replacement: String?,
    ) = Arguments.createArray().apply {
        entities.forEach { entity ->
            pushMap(
                spanMap(
                    label = entity.label,
                    canonicalLabel = canonicalLabel(entity.label),
                    start = entity.start,
                    end = entity.end,
                    confidence = entity.confidence,
                    action = action,
                    replacement = replacement,
                    text = text,
                    options = options,
                ),
            )
        }
    }

    private fun actionArray(
        actions: List<DeidentifiedSpanAction>,
        text: String,
        options: BridgeOptions,
    ) = Arguments.createArray().apply {
        actions.forEach { action ->
            pushMap(
                spanMap(
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
    }

    private fun spanMap(
        label: String,
        canonicalLabel: String,
        start: Int,
        end: Int,
        confidence: Float,
        action: String,
        replacement: String?,
        text: String,
        options: BridgeOptions,
    ): WritableMap {
        val lowerBound = start.coerceIn(0, text.length)
        val upperBound = end.coerceIn(lowerBound, text.length)
        val surface = text.substring(lowerBound, upperBound)
        return Arguments.createMap().apply {
            putInt("schema_version", 1)
            putString("doc_id", options.docId)
            putInt("start", start)
            putInt("end", end)
            putString("text_hash", hmacTextHash(surface, options.hashSecret))
            putString("entity_type", label)
            putString("canonical_label", canonicalLabel)
            putString("policy_label", policyLabel(canonicalLabel))
            putArray("regulatory_tags", Arguments.createArray())
            putDouble("score", confidence.toDouble())
            putString("detector", options.detector ?: "openmedkit-android")
            putMap(
                "evidence",
                Arguments.createMap().apply {
                    putString("bridge", "react-native")
                    putString("runtime", "OpenMedKit")
                },
            )
            putString("action", action)
            if (replacement == null) {
                putNull("replacement")
            } else {
                putString("replacement", replacement)
            }
            putNull("reversible_id")
            putNull("section")
            // putMap consumes a native map, so every span gets its own instance;
            // the JS layer merges caller metadata into each span.
            putMap("metadata", Arguments.createMap())
        }
    }

    private fun readId2Label(path: String?): Map<Int, String> {
        if (path.isNullOrBlank()) {
            return emptyMap()
        }
        val root = JSONObject(File(path).readText(Charsets.UTF_8))
        return root.keys().asSequence().associate { key ->
            key.toInt() to root.getString(key)
        }
    }

    private fun hmacTextHash(surface: String, secret: ByteArray): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret, "HmacSHA256"))
        return "hmac-sha256:" + mac.doFinal(surface.toByteArray(Charsets.UTF_8))
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
    }

    private fun canonicalLabel(label: String): String {
        val key = label
            .removePrefix("B-")
            .removePrefix("I-")
            .removePrefix("E-")
            .removePrefix("S-")
            .uppercase()
            .replace("-", "_")
            .replace(" ", "_")
            .filter { it.isLetterOrDigit() || it == '_' }
        return if (key in canonicalLabels) key else "OTHER"
    }

    private fun policyLabel(canonicalLabel: String): String = when (canonicalLabel) {
        in clinicalConceptLabels -> "CLINICAL_CONCEPT"
        in quasiIdentifierLabels -> "QUASI_IDENTIFIER"
        else -> "DIRECT_IDENTIFIER"
    }

    private fun ReadableMap.requiredString(key: String): String =
        requireNotNull(string(key)) { "missing required OpenMedKit bridge option: $key" }
            .also { require(it.isNotBlank()) { "$key must not be blank" } }

    private fun ReadableMap.string(key: String): String? =
        if (hasKey(key) && !isNull(key)) getString(key) else null

    companion object {
        const val NAME = "OpenMedKitRN"
        private const val CANCELLED_CODE = "openmedkit_cancelled"

        private val clinicalConceptLabels = setOf(
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

        private val quasiIdentifierLabels = setOf(
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

        private val canonicalLabels = setOf(
            "PERSON",
            "FIRST_NAME",
            "LAST_NAME",
            "EMAIL",
            "PHONE",
            "DATE",
            "DATE_OF_BIRTH",
            "SSN",
            "ID_NUM",
            "LOCATION",
            "STREET_ADDRESS",
            "ZIPCODE",
            "CONDITION",
            "MEDICATION",
            "LAB_TEST",
            "PROCEDURE",
            "OTHER",
        )
    }
}

class OpenMedKitRnPackage : ReactPackage {
    override fun createNativeModules(
        reactContext: ReactApplicationContext,
    ): List<NativeModule> = listOf(OpenMedKitRnModule(reactContext))

    override fun createViewManagers(
        reactContext: ReactApplicationContext,
    ): List<ViewManager<*, *>> = emptyList()
}

private class BridgeOptions(options: ReadableMap?) {
    val confidenceThreshold: Float =
        if (options?.hasKey("confidenceThreshold") == true && !options.isNull("confidenceThreshold")) {
            options.getDouble("confidenceThreshold").toFloat()
        } else {
            0.5f
        }
    val useSmartMerging: Boolean =
        if (options?.hasKey("useSmartMerging") == true && !options.isNull("useSmartMerging")) {
            options.getBoolean("useSmartMerging")
        } else {
            true
        }
    val docId: String =
        if (options?.hasKey("docId") == true && !options.isNull("docId")) {
            options.getString("docId") ?: "document"
        } else {
            "document"
        }
    val hashSecret: ByteArray =
        if (options?.hasKey("hashSecret") == true && !options.isNull("hashSecret")) {
            val secret = requireNotNull(options.getString("hashSecret"))
            require(secret.isNotEmpty()) { "hashSecret must not be empty" }
            secret.toByteArray(Charsets.UTF_8)
        } else {
            ByteArray(32).also { SecureRandom().nextBytes(it) }
        }
    val detector: String? =
        if (options?.hasKey("detector") == true && !options.isNull("detector")) {
            options.getString("detector")
        } else {
            null
        }
    val policy: String =
        if (options?.hasKey("policy") == true && !options.isNull("policy")) {
            options.getString("policy") ?: "hipaa_safe_harbor"
        } else {
            "hipaa_safe_harbor"
        }
}
