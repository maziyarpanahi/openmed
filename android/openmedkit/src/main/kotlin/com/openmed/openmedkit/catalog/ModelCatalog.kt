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
 *
 * @property family Model family from the manifest, such as `PII` or `NER`, or
 *   null when the catalog asset predates the field.
 * @property task Manifest task, such as `token-classification`, or null.
 * @property canonicalLabels Canonical entity labels the model emits, in manifest
 *   order; empty when the asset predates the field.
 */
data class ModelCatalogEntry(
    val repoId: String,
    val formats: List<String>,
    val tier: String?,
    val paramCount: Long?,
    val languages: List<String>,
    val license: String?,
    val reproducibilityHash: String,
    val family: String? = null,
    val task: String? = null,
    val canonicalLabels: List<String> = emptyList(),
) {
    /** Builds an entry without family, task or labels, as catalogs before 4.1 did. */
    constructor(
        repoId: String,
        formats: List<String>,
        tier: String?,
        paramCount: Long?,
        languages: List<String>,
        license: String?,
        reproducibilityHash: String,
    ) : this(repoId, formats, tier, paramCount, languages, license, reproducibilityHash, null, null, emptyList())

    /** Keeps the 4.0 `copy` signature for callers compiled against it. */
    @Deprecated("Kept for binary compatibility.", level = DeprecationLevel.HIDDEN)
    fun copy(
        repoId: String = this.repoId,
        formats: List<String> = this.formats,
        tier: String? = this.tier,
        paramCount: Long? = this.paramCount,
        languages: List<String> = this.languages,
        license: String? = this.license,
        reproducibilityHash: String = this.reproducibilityHash,
    ): ModelCatalogEntry = copy(
        repoId, formats, tier, paramCount, languages, license, reproducibilityHash, family, task, canonicalLabels,
    )
}

/**
 * Read-only on-device model catalog bundled with OpenMedKit.
 */
class ModelCatalog private constructor(
    entries: List<ModelCatalogEntry>,
) {
    val entries: List<ModelCatalogEntry> = entries.toList()

    /**
     * Returns the entries that match every non-null predicate.
     *
     * Token predicates, including [family], compare case-insensitively. An entry
     * without a family never matches a [family] predicate.
     */
    fun filter(
        format: String? = null,
        tier: String? = null,
        maxParamCount: Long? = null,
        language: String? = null,
        license: String? = null,
        family: String? = null,
    ): List<ModelCatalogEntry> = entries.filter { entry ->
        matchesFormat(entry, format) &&
            matchesToken(entry.tier, tier) &&
            matchesMaxParamCount(entry.paramCount, maxParamCount) &&
            matchesLanguage(entry, language) &&
            matchesToken(entry.license, license) &&
            matchesToken(entry.family, family)
    }

    /** Keeps the 4.0 `filter` signature for callers compiled against it. */
    @Deprecated("Kept for binary compatibility.", level = DeprecationLevel.HIDDEN)
    fun filter(
        format: String? = null,
        tier: String? = null,
        maxParamCount: Long? = null,
        language: String? = null,
        license: String? = null,
    ): List<ModelCatalogEntry> = filter(format, tier, maxParamCount, language, license, family = null)

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
                // Optional since 4.1; older assets omit them and still load.
                family = jsonObject.stringValue("family"),
                task = jsonObject.stringValue("task"),
                canonicalLabels = jsonObject.stringList("canonical_labels"),
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
