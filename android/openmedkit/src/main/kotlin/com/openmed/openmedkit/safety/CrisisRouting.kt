package com.openmed.openmedkit.safety

import java.net.URI
import java.time.LocalDate
import java.util.Locale
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/** Display data and provenance. Opening a resource requires a host UI action. */
data class CrisisResource(
    val id: String,
    val label: String,
    val url: String,
    val sourceUrl: String,
    val verifiedOn: String,
)

/** Every positive signal yields this notice, including when no locale entry exists. */
class CrisisNotice internal constructor(val matchedLocale: String?, val resources: List<CrisisResource>) {
    val text: String = "If someone is in immediate danger, contact local emergency services now."
}

/** Resolves an explicit signal offline. A local override replaces only its declared locales. */
class CrisisRouting(overrideJson: String? = null) {
    private val entries: Map<String, List<CrisisResource>> = try {
        val bundled = requireNotNull(javaClass.getResourceAsStream("/crisis_resources.json"))
            .bufferedReader(Charsets.UTF_8).use { parseTable(it.readText()) }
        if (overrideJson == null) bundled else bundled + parseTable(overrideJson)
    } catch (_: Exception) {
        throw IllegalArgumentException("invalid_crisis_table")
    }

    fun route(locale: String, signal: Boolean): CrisisNotice? {
        if (!signal) return null
        for (key in localeKeys(locale)) {
            val resources = entries[key] ?: continue
            return CrisisNotice(key, resources.toList())
        }
        return CrisisNotice(null, emptyList())
    }

    private fun parseTable(raw: String): Map<String, List<CrisisResource>> {
        val table = Json.parseToJsonElement(raw).jsonObject
        require(table.keys == setOf("schema_version", "locales") && table["schema_version"] == JsonPrimitive(1))
        return buildMap {
            for ((tag, rows) in table.getValue("locales").jsonObject) {
                val key = localeKeys(tag).first()
                val resources = rows.jsonArray.map { resource(it.jsonObject) }
                require(resources.isNotEmpty() && resources.distinctBy { it.id }.size == resources.size)
                require(put(key, resources) == null)
            }
        }
    }

    private fun resource(row: JsonObject): CrisisResource {
        require(row.keys == setOf("id", "label", "url", "source_url", "verified_on"))
        fun string(key: String): String {
            val value = row.getValue(key).jsonPrimitive
            require(value.isString)
            return value.content
        }
        fun link(key: String): String {
            val raw = string(key)
            require(raw.length <= 2048 && raw.all { it.code in 33..126 && it != '\\' })
            val url = URI(raw)
            require(url.scheme.equals("https", ignoreCase = true) && !url.host.isNullOrEmpty() &&
                url.rawUserInfo == null && (url.port == -1 || url.port in 1..65535))
            return raw
        }
        val id = string("id")
        val label = string("label")
        val date = string("verified_on")
        require(Regex("[a-z][a-z0-9_]{0,63}").matches(id))
        require(label.isNotBlank() && label.codePointCount(0, label.length) <= 160 && label.none { it.isISOControl() })
        require(Regex("[0-9]{4}-[0-9]{2}-[0-9]{2}").matches(date) && LocalDate.parse(date).year in 1..9999)
        return CrisisResource(id, label, link("url"), link("source_url"), date)
    }

    private fun localeKeys(tag: String): List<String> {
        val match = LOCALE.matchEntire(tag.replace('_', '-').lowercase(Locale.ROOT)) ?: return emptyList()
        val (language, rawScript, rawRegion) = match.destructured
        val script = rawScript.takeIf { it.isNotEmpty() }?.replaceFirstChar { it.uppercaseChar() }
        val region = rawRegion.takeIf { it.isNotEmpty() }?.uppercase(Locale.ROOT)
        return buildList {
            add(listOfNotNull(language, script, region).joinToString("-"))
            if (script != null && region != null) {
                add("$language-$region")
                add("$language-$script")
            }
            add(language)
        }.distinct()
    }

    private companion object {
        val LOCALE = Regex("([a-z]{2,3})(?:-([a-z]{4}))?(?:-([a-z]{2}|[0-9]{3}))?")
    }
}
