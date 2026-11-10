package com.openmed.openmedkit.catalog

import android.content.Context
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.BufferedReader
import java.io.InputStream
import java.io.InputStreamReader

/**
 * Manifest-backed catalog entry for Android-runnable OpenMed models.
 */
data class ModelCatalogEntry(
    val repoId: String,
    val formats: List<String>,
    val tier: String?,
    val paramCount: Long?,
    val languages: List<String>,
    val license: String?,
    val reproducibilityHash: String,
)

/**
 * Read-only on-device model catalog bundled with OpenMedKit.
 */
class ModelCatalog private constructor(
    entries: List<ModelCatalogEntry>,
) {
    val entries: List<ModelCatalogEntry> = entries.toList()

    fun filter(
        format: String? = null,
        tier: String? = null,
        maxParamCount: Long? = null,
        language: String? = null,
        license: String? = null,
    ): List<ModelCatalogEntry> = entries.filter { entry ->
        matchesFormat(entry, format) &&
            matchesToken(entry.tier, tier) &&
            matchesMaxParamCount(entry.paramCount, maxParamCount) &&
            matchesLanguage(entry, language) &&
            matchesToken(entry.license, license)
    }

    fun byRepoId(repoId: String): ModelCatalogEntry? =
        entries.firstOrNull { it.repoId == repoId }

    companion object {
        const val ASSET_NAME = "openmed_model_catalog.jsonl"

        private val json = Json {
            ignoreUnknownKeys = true
        }

        fun load(context: Context, assetName: String = ASSET_NAME): ModelCatalog =
            context.assets.open(assetName).use { parse(it) }

        fun parse(inputStream: InputStream): ModelCatalog =
            BufferedReader(InputStreamReader(inputStream, Charsets.UTF_8)).use { reader ->
                fromLines(reader.lineSequence())
            }

        fun fromJsonLines(jsonLines: String): ModelCatalog =
            fromLines(jsonLines.lineSequence())

        private fun fromLines(lines: Sequence<String>): ModelCatalog {
            val entries = lines
                .mapIndexedNotNull { index, line ->
                    val trimmed = line.trim()
                    if (trimmed.isEmpty()) {
                        null
                    } else {
                        parseEntry(trimmed, index + 1)
                    }
                }
                .toList()
            return ModelCatalog(entries)
        }

        private fun parseEntry(line: String, lineNumber: Int): ModelCatalogEntry {
            val jsonObject = json.parseToJsonElement(line).jsonObject
            val repoId = jsonObject.stringValue("repo_id")
                ?: throw IllegalArgumentException("Catalog entry missing repo_id on line $lineNumber")
            return ModelCatalogEntry(
                repoId = repoId,
                formats = jsonObject.stringList("formats"),
                tier = jsonObject.stringValue("tier"),
                paramCount = jsonObject.longValue("param_count"),
                languages = jsonObject.stringList("languages"),
                license = jsonObject.stringValue("license"),
                reproducibilityHash = jsonObject.stringValue("reproducibility_hash")
                    ?: throw IllegalArgumentException(
                        "Catalog entry missing reproducibility_hash on line $lineNumber",
                    ),
            )
        }

        private fun JsonObject.stringValue(key: String): String? =
            this[key]?.jsonPrimitive?.contentOrNull

        private fun JsonObject.longValue(key: String): Long? =
            this[key]?.jsonPrimitive?.contentOrNull?.toLongOrNull()

        private fun JsonObject.stringList(key: String): List<String> =
            this[key]
                ?.jsonArray
                ?.mapNotNull { it.jsonPrimitive.contentOrNull }
                ?: emptyList()
    }
}

private fun matchesFormat(entry: ModelCatalogEntry, format: String?): Boolean =
    format == null || entry.formats.any { it.equals(format, ignoreCase = true) }

private fun matchesToken(actual: String?, expected: String?): Boolean =
    expected == null || actual?.equals(expected, ignoreCase = true) == true

private fun matchesMaxParamCount(actual: Long?, maxParamCount: Long?): Boolean =
    maxParamCount == null || actual?.let { it <= maxParamCount } == true

private fun matchesLanguage(entry: ModelCatalogEntry, language: String?): Boolean =
    language == null || entry.languages.any { it.equals(language, ignoreCase = true) }
