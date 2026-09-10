package org.openmed.maple

import java.util.Locale
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject

data class MapleResultRow(
    val headline: String,
    val supporting: String,
    val badge: String,
)

data class MaplePresentation(
    val body: String,
    val rows: List<MapleResultRow> = emptyList(),
    val isStructured: Boolean = false,
)

class MapleOutputException(message: String) : IllegalArgumentException(message)

object MapleOutputParser {
    private val json = Json { ignoreUnknownKeys = true }

    fun visibleText(raw: String): String {
        completeDirectJsonObject(raw)?.let { return it }
        val finalAnswerStart = raw.lastIndexOf("</think>")
        if (finalAnswerStart >= 0) {
            return raw.substring(finalAnswerStart + "</think>".length).trim()
        }
        return ""
    }

    fun parse(task: MapleTask, raw: String, sourceText: String): MaplePresentation {
        val visible = visibleText(raw)
        if (visible.isBlank()) {
            throw MapleOutputException(
                "Maple returned incomplete hidden reasoning with no reviewable final answer. " +
                    "No output was applied; review the source and retry.",
            )
        }
        if (task == MapleTask.CHAT) return MaplePresentation(visible)

        val objectText = firstJsonObject(visible)
            ?: throw MapleOutputException(
                "Maple returned an incomplete structured result. " +
                    "No output was applied; review the source and retry.",
            )
        val root = runCatching { json.parseToJsonElement(objectText).jsonObject }.getOrNull()
            ?: throw MapleOutputException(
                "Maple returned invalid structured JSON. " +
                    "No output was applied; review the source and retry.",
            )
        return when (task) {
            MapleTask.REDACT -> parseRedaction(root, sourceText)
            MapleTask.ENTITIES -> parseRows(root, "entities", visible) { item ->
                MapleResultRow(
                    headline = item.requiredString("text"),
                    supporting = item.requiredString("evidence"),
                    badge = item.requiredString("label"),
                )
            }
            MapleTask.RELATIONS -> parseRows(root, "relations", visible) { item ->
                val relation = item.requiredString("relation")
                MapleResultRow(
                    headline = listOf(
                        item.requiredString("subject"),
                        "→ $relation →",
                        item.requiredString("object"),
                    ).joinToString(" "),
                    supporting = item.requiredString("evidence"),
                    badge = relation,
                )
            }
            MapleTask.CHAT -> error("handled before structured parsing")
        }
    }

    private fun parseRedaction(root: JsonObject, sourceText: String): MaplePresentation {
        val entities = root.requiredObjectArray("entities")
            .mapIndexed { index, item ->
                val surface = item.requiredString("text")
                val label = item.requiredString("label").uppercase(Locale.ROOT)
                if (!PII_LABEL.matches(label)) {
                    throw MapleOutputException(
                        "Redaction entity ${index + 1} has an unsafe label. " +
                            "No redaction was applied; review the source.",
                    )
                }
                RedactionEntity(surface, label, index)
            }
        entities.groupBy { it.surface }.forEach { (_, matches) ->
            if (matches.map { it.label }.distinct().size > 1) {
                throw MapleOutputException(
                    "Maple assigned conflicting labels to one source surface. " +
                        "No redaction was applied; review the source.",
                )
            }
        }
        val spans = entities
            .distinctBy { it.surface to it.label }
            .map { entity -> entity.resolveExactlyOnce(sourceText) }
            .sortedBy { it.start }
        spans.zipWithNext().forEach { (left, right) ->
            if (left.endExclusive > right.start) {
                throw MapleOutputException(
                    "Maple returned overlapping identifier surfaces. " +
                        "No redaction was applied; review the source.",
                )
            }
        }

        val redacted = StringBuilder(sourceText)
        spans.asReversed().forEach { span ->
            redacted.replace(span.start, span.endExclusive, "[${span.entity.label}]")
        }
        val rows = spans.map { span ->
            MapleResultRow(
                headline = span.entity.surface,
                supporting = "Exact source span ${span.start}–${span.endExclusive} replaced",
                badge = span.entity.label,
            )
        }
        return MaplePresentation(body = redacted.toString(), rows = rows, isStructured = true)
    }

    private fun parseRows(
        root: JsonObject,
        key: String,
        body: String,
        convert: (JsonObject) -> MapleResultRow,
    ): MaplePresentation {
        val rows = root.requiredObjectArray(key).map(convert)
        return MaplePresentation(body = body, rows = rows, isStructured = true)
    }

    private fun JsonObject.requiredObjectArray(key: String): List<JsonObject> {
        val array = this[key] as? JsonArray
            ?: throw MapleOutputException(
                "Maple output is missing the required $key array. " +
                    "No output was applied; review the source.",
            )
        return array.mapIndexed { index, value ->
            value as? JsonObject
                ?: throw MapleOutputException(
                    "Maple output $key item ${index + 1} is not an object. " +
                        "No output was applied; review the source.",
                )
        }
    }

    private fun JsonObject.requiredString(key: String): String {
        val primitive = this[key] as? JsonPrimitive
        val value = primitive?.takeIf { it.isString }?.contentOrNull?.trim()
        return value?.takeIf { it.isNotEmpty() }
            ?: throw MapleOutputException(
                "Maple output is missing required text field $key. " +
                    "No output was applied; review the source.",
            )
    }

    private fun completeDirectJsonObject(value: String): String? {
        val trimmed = value.trim()
        if (trimmed.isEmpty() || firstJsonObject(trimmed) != trimmed) return null
        return runCatching { json.parseToJsonElement(trimmed).jsonObject }
            .getOrNull()
            ?.let { trimmed }
    }

    internal fun firstJsonObject(value: String): String? {
        var start = -1
        var depth = 0
        var inString = false
        var escaped = false
        value.forEachIndexed { index, character ->
            if (start < 0) {
                if (character == '{') {
                    start = index
                    depth = 1
                }
                return@forEachIndexed
            }
            if (inString) {
                if (escaped) {
                    escaped = false
                } else if (character == '\\') {
                    escaped = true
                } else if (character == '"') {
                    inString = false
                }
                return@forEachIndexed
            }
            when (character) {
                '"' -> inString = true
                '{' -> depth += 1
                '}' -> {
                    depth -= 1
                    if (depth == 0) {
                        return value.substring(start, index + 1)
                    }
                }
            }
        }
        return null
    }

    private fun RedactionEntity.resolveExactlyOnce(sourceText: String): RedactionSpan {
        val first = sourceText.indexOf(surface)
        if (first < 0) {
            throw MapleOutputException(
                "Redaction entity ${originalIndex + 1} is absent from the source. " +
                    "No redaction was applied; review the source.",
            )
        }
        if (sourceText.indexOf(surface, first + 1) >= 0) {
            throw MapleOutputException(
                "Redaction entity ${originalIndex + 1} is ambiguous in the source. " +
                    "No redaction was applied; review each occurrence.",
            )
        }
        return RedactionSpan(first, first + surface.length, this)
    }

    private data class RedactionEntity(
        val surface: String,
        val label: String,
        val originalIndex: Int,
    )

    private data class RedactionSpan(
        val start: Int,
        val endExclusive: Int,
        val entity: RedactionEntity,
    )

    private val PII_LABEL = Regex("[A-Z][A-Z0-9_]{0,31}")
}

object SyntheticPreviewResults {
    fun forTask(task: MapleTask): MaplePresentation = when (task) {
        MapleTask.REDACT -> MaplePresentation(
            body = """SYNTHETIC TRAINING NOTE — NOT A REAL PATIENT
Patient: [NAME]
DOB: [DATE]  MRN: [MRN]
Phone: [PHONE]  Email: [EMAIL]

[NAME] reports a three-day frontal headache with nausea. Blood pressure was
148/92 mmHg. Neurologic examination was non-focal. The clinician documented
migraine without aura and prescribed sumatriptan 50 mg by mouth as needed.""",
            rows = listOf(
                MapleResultRow("Alex Morgan", "Marked for redaction", "NAME"),
                MapleResultRow("1986-04-12", "Marked for redaction", "DATE"),
                MapleResultRow("DEMO-48291", "Marked for redaction", "MRN"),
            ),
            isStructured = true,
        )
        MapleTask.ENTITIES -> MaplePresentation(
            body = "Six source-grounded entities found in the synthetic note.",
            rows = listOf(
                MapleResultRow("frontal headache", "Exact source phrase", "PROBLEM"),
                MapleResultRow("sumatriptan", "prescribed sumatriptan 50 mg", "MEDICATION"),
                MapleResultRow("50 mg", "sumatriptan 50 mg", "DOSAGE"),
                MapleResultRow("148/92 mmHg", "Blood pressure was 148/92 mmHg", "MEASUREMENT"),
            ),
            isStructured = true,
        )
        MapleTask.RELATIONS -> MaplePresentation(
            body = "Source-grounded links from the synthetic note.",
            rows = listOf(
                MapleResultRow("sumatriptan → TREATS → migraine", "prescribed sumatriptan", "TREATS"),
                MapleResultRow("sumatriptan → HAS_DOSAGE → 50 mg", "sumatriptan 50 mg", "HAS_DOSAGE"),
                MapleResultRow("chest pain → NEGATED_FOR → patient", "No chest pain", "NEGATED_FOR"),
            ),
            isStructured = true,
        )
        MapleTask.CHAT -> MaplePresentation(
            body = """The note describes a three-day headache with nausea, a non-focal neurologic examination, and a documented migraine plan that includes as-needed sumatriptan.

Source evidence
• “three-day frontal headache with nausea”
• “Neurologic examination was non-focal”
• “sumatriptan 50 mg by mouth as needed”

Uncertainty
This synthetic excerpt is incomplete and cannot establish a diagnosis. A clinician should review the original record and the person’s current condition.""",
        )
    }
}
