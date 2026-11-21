package com.openmed.openmedkit.safety

import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.time.format.DateTimeParseException
import java.util.Locale
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put

/** In-memory record of explicit acknowledgement, in the Python wire format. */
data class OpenMedNoticeAcknowledgement(
    val noticeId: String,
    val versionDigest: String,
    val acknowledgedAt: String,
) {
    fun toJson(): JsonObject = buildJsonObject {
        put("notice_id", noticeId)
        put("version_digest", versionDigest)
        put("acknowledged_at", acknowledgedAt)
    }
}

/** Reviewed notice from the kit's offline catalog. The host owns rendering. */
class OpenMedDisclaimer private constructor(row: JsonObject) {
    val id = row.getValue("id").jsonPrimitive.content
    val tier = row.getValue("tier").jsonPrimitive.content
    val version = row.getValue("version").jsonPrimitive.content
    val versionDigest = row.getValue("version_digest").jsonPrimitive.content
    val locale = row.getValue("locale").jsonPrimitive.content
    val text = row.getValue("text").jsonPrimitive.content
    val requiresAcknowledgement = row.getValue("requires_acknowledgement").jsonPrimitive.boolean

    fun acknowledge(at: Instant): OpenMedNoticeAcknowledgement? {
        if (!requiresAcknowledgement) return null
        val micros = at.nano / 1_000
        val fraction = if (micros == 0) "" else String.format(Locale.ROOT, ".%06d", micros)
        return OpenMedNoticeAcknowledgement(id, versionDigest, "${utcSeconds.format(at)}${fraction}+00:00")
    }

    fun permitsResults(receipt: OpenMedNoticeAcknowledgement?): Boolean {
        if (!requiresAcknowledgement) return true
        if (receipt == null || receipt.noticeId != id || receipt.versionDigest != versionDigest) return false
        return try {
            Instant.parse(receipt.acknowledgedAt)
            true
        } catch (_: DateTimeParseException) {
            false
        }
    }

    companion object {
        private val utcSeconds = DateTimeFormatter.ofPattern("uuuu-MM-dd'T'HH:mm:ss", Locale.ROOT)
            .withZone(ZoneOffset.UTC)
        private val catalog by lazy {
            val stream = requireNotNull(OpenMedDisclaimer::class.java.getResourceAsStream("/safety_notices.json"))
            stream.bufferedReader(Charsets.UTF_8).use { Json.parseToJsonElement(it.readText()).jsonObject }
        }
        private val notices by lazy {
            catalog.getValue("notices").jsonObject.mapValues { OpenMedDisclaimer(it.value.jsonObject) }
        }

        fun byId(id: String): OpenMedDisclaimer? = notices[id]

        /** Future or missing tiers receive the restricted notice. */
        fun forTier(tier: String?): OpenMedDisclaimer {
            val tiers = catalog.getValue("tiers").jsonObject
            val id = tier?.let(tiers::get) ?: tiers.getValue("restricted")
            return notices.getValue(id.jsonPrimitive.content)
        }

        fun forCapability(code: String): OpenMedDisclaimer? {
            val id = catalog.getValue("overrides").jsonObject[code]
                ?: catalog.getValue("capabilities").jsonObject[code]
            return id?.let { notices.getValue(it.jsonPrimitive.content) }
        }
    }
}
