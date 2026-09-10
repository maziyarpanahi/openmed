package org.openmed.maple

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

data class MapleBundleFile(
    val path: String,
    val sizeBytes: Long,
    val sha256: String,
)

data class MapleGraphContract(
    val prefillPath: String,
    val decodePath: String?,
    val inputIdsName: String,
    val attentionMaskName: String,
    val positionIdsName: String,
    val logitsName: String,
)

data class MapleCacheContract(
    val pastInputPrefix: String,
    val presentOutputPrefix: String,
)

data class MapleGenerationDefaults(
    val eosTokenIds: Set<Long>,
    val maxContextTokens: Int,
    val maxInputTokens: Int,
)

data class MapleBundleManifest(
    val schemaVersion: Int,
    val sourceModel: String,
    val sourceRevision: String,
    val architecture: String,
    val quantization: String,
    val runtime: String,
    val tokenizerPath: String,
    val graphs: MapleGraphContract,
    val cache: MapleCacheContract?,
    val generation: MapleGenerationDefaults,
    val files: List<MapleBundleFile>,
) {
    val totalSizeBytes: Long = files.sumOf { it.sizeBytes }

    fun resolve(root: File, relativePath: String): File = resolveInside(root, relativePath)
}

object MapleBundleManifestParser {
    private val json = Json { ignoreUnknownKeys = false }
    private val revisionPattern = Regex("[0-9a-f]{40,64}")
    private val checksumPattern = Regex("[0-9a-f]{64}")

    fun parse(value: String): MapleBundleManifest {
        val root = try {
            json.parseToJsonElement(value).jsonObject
        } catch (error: Exception) {
            throw MapleBundleException("maple-bundle.json must be valid JSON", error)
        }

        val manifest = MapleBundleManifest(
            schemaVersion = root.requiredInt("schema_version"),
            sourceModel = root.requiredString("source_model"),
            sourceRevision = root.requiredString("source_revision").lowercase(),
            architecture = root.requiredString("architecture"),
            quantization = root.requiredString("quantization"),
            runtime = root.requiredString("runtime"),
            tokenizerPath = root.requiredString("tokenizer_path"),
            graphs = root.requiredObject("graphs").toGraphContract(),
            cache = root.optionalObject("cache")?.toCacheContract(),
            generation = root.requiredObject("generation").toGenerationDefaults(),
            files = root.requiredArray("files").map { it.jsonObject.toBundleFile() },
        )
        validate(manifest)
        return manifest
    }

    private fun validate(manifest: MapleBundleManifest) {
        requireBundle(manifest.schemaVersion == 1, "Unsupported Maple bundle schema")
        requireBundle(
            manifest.sourceModel == MAPLE_SOURCE_MODEL,
            "Bundle source_model must be $MAPLE_SOURCE_MODEL",
        )
        requireBundle(
            revisionPattern.matches(manifest.sourceRevision),
            "source_revision must be an immutable commit SHA",
        )
        requireBundle(
            manifest.architecture == MAPLE_ARCHITECTURE,
            "Bundle architecture must be $MAPLE_ARCHITECTURE",
        )
        requireBundle(
            manifest.runtime == MAPLE_RUNTIME,
            "Bundle runtime must be $MAPLE_RUNTIME",
        )
        requireBundle(
            manifest.quantization == MAPLE_QUANTIZATION,
            "Bundle quantization must be $MAPLE_QUANTIZATION",
        )
        listOf(
            manifest.graphs.inputIdsName,
            manifest.graphs.attentionMaskName,
            manifest.graphs.positionIdsName,
            manifest.graphs.logitsName,
        ).forEach { tensorName ->
            requireBundle(tensorName.isNotBlank(), "Graph tensor names must not be blank")
        }
        requireBundle(
            manifest.graphs.prefillPath.endsWith(".onnx") ||
                manifest.graphs.prefillPath.endsWith(".ort"),
            "The prefill graph must be an ONNX or ORT model",
        )
        manifest.graphs.decodePath?.let { decodePath ->
            requireBundle(
                decodePath.endsWith(".onnx") || decodePath.endsWith(".ort"),
                "The decode graph must be an ONNX or ORT model",
            )
        }
        requireBundle(manifest.files.isNotEmpty(), "Bundle files must not be empty")
        requireBundle(manifest.files.size <= MAX_BUNDLE_FILES, "Bundle declares too many files")
        requireBundle(
            manifest.totalSizeBytes in 1..MAX_BUNDLE_BYTES,
            "Bundle exceeds the 12 GiB import limit",
        )
        requireBundle(
            manifest.generation.eosTokenIds.isNotEmpty(),
            "At least one EOS token is required",
        )
        requireBundle(
            manifest.generation.maxContextTokens in 64..MAPLE_MAX_CONTEXT,
            "max_context_tokens is outside Maple's supported range",
        )
        requireBundle(
            manifest.generation.maxInputTokens in 32 until manifest.generation.maxContextTokens,
            "max_input_tokens must leave room for generation",
        )

        val paths = mutableSetOf<String>()
        manifest.files.forEach { file ->
            validateRelativePath(file.path)
            requireBundle(paths.add(file.path), "Duplicate bundle path: ${file.path}")
            requireBundle(file.sizeBytes > 0, "Bundle file sizes must be positive")
            requireBundle(
                checksumPattern.matches(file.sha256) && file.sha256.any { it != '0' },
                "Bundle file ${file.path} needs a non-placeholder SHA-256",
            )
        }
        val requiredPaths = buildSet {
            add(manifest.tokenizerPath)
            add(manifest.graphs.prefillPath)
            manifest.graphs.decodePath?.let(::add)
        }
        requiredPaths.forEach { path ->
            validateRelativePath(path)
            requireBundle(path in paths, "Required file is not declared: $path")
        }
        if (manifest.graphs.decodePath != null) {
            requireBundle(manifest.cache != null, "A cached decode graph needs a cache contract")
        } else {
            requireBundle(manifest.cache == null, "A cache contract needs a decode graph")
        }
    }

    private fun JsonObject.toGraphContract() = MapleGraphContract(
        prefillPath = requiredString("prefill_path"),
        decodePath = optionalString("decode_path"),
        inputIdsName = optionalString("input_ids_name") ?: "input_ids",
        attentionMaskName = optionalString("attention_mask_name") ?: "attention_mask",
        positionIdsName = optionalString("position_ids_name") ?: "position_ids",
        logitsName = optionalString("logits_name") ?: "logits",
    )

    private fun JsonObject.toCacheContract() = MapleCacheContract(
        pastInputPrefix = requiredString("past_input_prefix"),
        presentOutputPrefix = requiredString("present_output_prefix"),
    ).also {
        requireBundle(it.pastInputPrefix.isNotBlank(), "past_input_prefix must not be blank")
        requireBundle(it.presentOutputPrefix.isNotBlank(), "present_output_prefix must not be blank")
    }

    private fun JsonObject.toGenerationDefaults() = MapleGenerationDefaults(
        eosTokenIds = requiredArray("eos_token_ids").map {
            it.jsonPrimitive.longOrNull
                ?: throw MapleBundleException("eos_token_ids must contain integers")
        }.toSet(),
        maxContextTokens = requiredInt("max_context_tokens"),
        maxInputTokens = requiredInt("max_input_tokens"),
    )

    private fun JsonObject.toBundleFile() = MapleBundleFile(
        path = requiredString("path"),
        sizeBytes = this["size_bytes"]?.jsonPrimitive?.longOrNull
            ?: throw MapleBundleException("Bundle file size_bytes must be an integer"),
        sha256 = requiredString("sha256").lowercase(),
    )

    private fun JsonObject.requiredObject(key: String): JsonObject =
        this[key]?.let {
            runCatching { it.jsonObject }.getOrNull()
        } ?: throw MapleBundleException("Missing object: $key")

    private fun JsonObject.optionalObject(key: String): JsonObject? {
        val value = this[key] ?: return null
        if (value is JsonNull) return null
        return runCatching { value.jsonObject }.getOrNull()
            ?: throw MapleBundleException("Expected object or null: $key")
    }

    private fun JsonObject.requiredArray(key: String): JsonArray =
        this[key]?.let {
            runCatching { it.jsonArray }.getOrNull()
        } ?: throw MapleBundleException("Missing array: $key")

    private fun JsonObject.requiredString(key: String): String =
        optionalString(key) ?: throw MapleBundleException("Missing string: $key")

    private fun JsonObject.optionalString(key: String): String? =
        this[key]?.jsonPrimitive?.contentOrNull

    private fun JsonObject.requiredInt(key: String): Int =
        this[key]?.jsonPrimitive?.intOrNull
            ?: throw MapleBundleException("Missing integer: $key")
}

class MapleBundleException(message: String, cause: Throwable? = null) :
    IllegalArgumentException(message, cause)

internal fun validateRelativePath(path: String) {
    val pieces = path.split('/')
    requireBundle(path.isNotBlank(), "Bundle paths must not be blank")
    requireBundle(!path.startsWith('/'), "Bundle paths must be relative")
    requireBundle('\\' !in path, "Bundle paths must use forward slashes")
    requireBundle(
        pieces.none { it.isBlank() || it == "." || it == ".." },
        "Bundle path contains an unsafe segment",
    )
}

internal fun resolveInside(root: File, relativePath: String): File {
    validateRelativePath(relativePath)
    val canonicalRoot = root.canonicalFile
    val candidate = File(canonicalRoot, relativePath).canonicalFile
    requireBundle(
        candidate.path.startsWith(canonicalRoot.path + File.separator),
        "Bundle path escapes its destination",
    )
    return candidate
}

internal fun requireBundle(condition: Boolean, message: String) {
    if (!condition) {
        throw MapleBundleException(message)
    }
}

const val MAPLE_SOURCE_MODEL = "deepgrove/maple-preview"
const val MAPLE_ARCHITECTURE = "MapleForCausalLM"
const val MAPLE_RUNTIME = "onnxruntime-mobile"
const val MAPLE_QUANTIZATION = "qmoe-4bit-blockwise-128"
const val MAPLE_MAX_CONTEXT = 131_072
private const val MAX_BUNDLE_FILES = 512
private const val MAX_BUNDLE_BYTES = 12L * 1024L * 1024L * 1024L
