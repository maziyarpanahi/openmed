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
import com.openmed.openmedkit.OpenMedKit
import java.io.File
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject

@ReactModule(name = OpenMedKitRnModule.NAME)
class OpenMedKitRnModule(
    reactContext: ReactApplicationContext,
) : ReactContextBaseJavaModule(reactContext) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var runtime: OpenMedKit? = null
    private var loadedCacheKey: String? = null

    override fun getName(): String = NAME

    @ReactMethod
    fun loadModel(
        options: ReadableMap,
        promise: Promise,
    ) {
        scope.launch {
            try {
                val modelPath = options.requiredString("modelPath")
                val backendName = options.string("backend") ?: "onnx"
                val cacheKey = options.string("cacheKey") ?: "$backendName:$modelPath"

                if (loadedCacheKey == cacheKey && runtime != null) {
                    promise.resolve(
                        Arguments.createMap().apply {
                            putString("cacheKey", cacheKey)
                            putString("modelPath", modelPath)
                            putString("backend", backendName)
                            putString("platform", "android")
                            putBoolean("loaded", false)
                        },
                    )
                    return@launch
                }

                require(backendName == "onnx") {
                    "unsupported OpenMedKit Android backend: $backendName"
                }
                runtime = OpenMedKit(
                    OpenMedBackend(
                        modelDirectory = File(modelPath),
                        id2Label = readId2Label(options.string("id2LabelPath")),
                    ),
                )
                loadedCacheKey = cacheKey
                promise.resolve(
                    Arguments.createMap().apply {
                        putString("cacheKey", cacheKey)
                        putString("modelPath", modelPath)
                        putString("backend", backendName)
                        putString("platform", "android")
                        putBoolean("loaded", true)
                    },
                )
            } catch (error: Throwable) {
                promise.reject("openmedkit_load_failed", error.message, error)
            }
        }
    }

    @ReactMethod
    fun analyzeText(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        scope.launch {
            try {
                val bridgeOptions = BridgeOptions(options)
                val entities = requireRuntime().analyzeText(
                    text = text,
                    confidenceThreshold = bridgeOptions.confidenceThreshold,
                )
                promise.resolve(
                    entityArray(
                        entities = entities,
                        text = text,
                        options = bridgeOptions,
                        action = "keep",
                        replacement = null,
                    ),
                )
            } catch (error: Throwable) {
                promise.reject("openmedkit_analyze_failed", error.message, error)
            }
        }
    }

    @ReactMethod
    fun extractPii(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        scope.launch {
            try {
                val bridgeOptions = BridgeOptions(options)
                val entities = requireRuntime().extractPii(
                    text = text,
                    confidenceThreshold = bridgeOptions.confidenceThreshold,
                    useSmartMerging = bridgeOptions.useSmartMerging,
                )
                promise.resolve(
                    entityArray(
                        entities = entities,
                        text = text,
                        options = bridgeOptions,
                        action = "keep",
                        replacement = null,
                    ),
                )
            } catch (error: Throwable) {
                promise.reject("openmedkit_extract_failed", error.message, error)
            }
        }
    }

    @ReactMethod
    fun deidentify(
        text: String,
        options: ReadableMap?,
        promise: Promise,
    ) {
        scope.launch {
            try {
                val bridgeOptions = BridgeOptions(options)
                val result = requireRuntime().deidentify(
                    text = text,
                    policy = bridgeOptions.policy,
                    confidenceThreshold = bridgeOptions.confidenceThreshold,
                    useSmartMerging = bridgeOptions.useSmartMerging,
                )
                promise.resolve(
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
                    },
                )
            } catch (error: Throwable) {
                promise.reject("openmedkit_deidentify_failed", error.message, error)
            }
        }
    }

    private fun requireRuntime(): OpenMedKit =
        runtime ?: throw IllegalStateException("OpenMedKit model is not loaded")

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
            putMap("metadata", options.metadata)
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

    private fun hmacTextHash(surface: String, secret: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(secret.toByteArray(Charsets.UTF_8), "HmacSHA256"))
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
    val hashSecret: String =
        if (options?.hasKey("hashSecret") == true && !options.isNull("hashSecret")) {
            options.getString("hashSecret") ?: "openmedkit-react-native"
        } else {
            "openmedkit-react-native"
        }
    val detector: String? =
        if (options?.hasKey("detector") == true && !options.isNull("detector")) {
            options.getString("detector")
        } else {
            null
        }
    val metadata: WritableMap = Arguments.createMap()
    val policy: String =
        if (options?.hasKey("policy") == true && !options.isNull("policy")) {
            options.getString("policy") ?: "hipaa_safe_harbor"
        } else {
            "hipaa_safe_harbor"
        }
}
