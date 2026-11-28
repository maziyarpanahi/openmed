package com.openmed.openmedkit.clinical

import com.openmed.openmedkit.ScalarSpan
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.Utf16Span
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * ConText assertion axes of one span, as the string codes of the Python
 * `openmed.clinical.context.assert_context`.
 *
 * The axes are advisory annotations for review, not clinical decisions.
 *
 * @property negation `affirmed` or `negated`.
 * @property uncertainty `certain` or `uncertain`.
 * @property temporality `recent`, `historical` or `hypothetical`.
 * @property experiencer `patient`, `family` or `other`.
 */
class ClinicalAssertion internal constructor(
    val negation: String,
    val uncertainty: String,
    val temporality: String,
    val experiencer: String,
)

/**
 * Kotlin port of the English path of the Python `assert_context`, without
 * section priors.
 *
 * Cues come from the bundled `clinical/context_cues_en_v1.json`, a
 * byte-identical copy of `tests/fixtures/parity/context_cues_en_v1.json`.
 * Matching is case-insensitive, on token boundaries and longest cue first.
 * Pseudo-negation cues are masked before negation cues are counted, and an
 * odd count negates. A cue reaches a target only inside the target's
 * sentence, in its own direction and up to a scope terminator. Experiencer
 * cues apply inside the target's clause. The cases in
 * `tests/fixtures/parity/clinical_assertion_v1.json` pin the output,
 * including the recorded Python gaps where `and` and `or` end a negation
 * scope.
 *
 * Offsets in and out are Unicode scalar offsets; they are converted to UTF-16
 * only to slice JVM strings. Without caller sentence bounds, a regex splitter
 * is used, like the Python fallback when pySBD is not installed.
 */
object ClinicalContext {
    /** Reason recorded when no assertion is computed for a non-English input. */
    const val UNSUPPORTED_LANGUAGE: String = "unsupported_language"

    private const val WORD = "[\\p{L}\\p{N}_]"
    private const val SPACE = "[\\s\\p{Z}\\u0085\\u001c-\\u001f]"
    private val whitespace = Regex("$SPACE+")
    private val sentenceEnd = Regex("[.!?]+(?:$SPACE+|$)|\\n+")
    private val clauseBoundary =
        Regex("[.!?;]|(?<!$WORD)(?:but|however|whereas)(?!$WORD)", RegexOption.IGNORE_CASE)
    private val cues: Cues by lazy { loadCues(bundledCueTable()) }

    /** Bytes of the bundled English cue table, or `null` when it is not packaged. */
    internal fun bundledCueTable(): ByteArray? = javaClass.classLoader
        ?.getResourceAsStream("clinical/context_cues_en_v1.json")?.use { it.readBytes() }

    /**
     * Return the assertion of each span in [spans], in order.
     *
     * @param sentences Sentence bounds that limit cue reach; a regex splitter
     *   is used when `null`.
     * @param language `null` or an `en` tag selects the English cues. Any
     *   other language returns `null`: no assertion is computed, see
     *   [UNSUPPORTED_LANGUAGE].
     * @throws IllegalArgumentException when a span or sentence lies outside
     *   [text].
     */
    fun assertContext(
        text: String,
        spans: List<ScalarSpan>,
        sentences: List<ScalarSpan>? = null,
        language: String? = null,
    ): List<ClinicalAssertion>? {
        val tag = language?.lowercase()?.replace('_', '-')?.substringBefore('-')
        if (tag != null && tag != "en") return null
        val cues = cues
        val windows = ArrayList<Utf16Span>()
        if (sentences == null) {
            var start = 0
            var match = sentenceEnd.find(text)
            while (match != null) {
                addTrimmed(windows, text, start, match.range.last + 1)
                start = match.range.last + 1
                match = match.next()
            }
            addTrimmed(windows, text, start, text.length)
        } else {
            for (sentence in sentences) windows += utf16(text, sentence)
        }
        if (windows.isEmpty() && text.isNotEmpty()) windows += Utf16Span(0, text.length)
        val hits = cues.context.findAll(text).toList()
        val assertions = ArrayList<ClinicalAssertion>(spans.size)
        for (span in spans) {
            val target = utf16(text, span)
            val sentence = windowOf(windows, target.start, target.end)
            val reaching = ArrayList<MatchResult>()
            for (hit in hits) {
                if (sentence >= 0 && reaches(cues, text, windows, sentence, hit, target)) {
                    reaching += hit
                }
            }
            assertions += assertion(cues, text, target, reaching)
        }
        return assertions
    }

    /** Python `_cue_reaches_span`: same sentence, cue direction, no conjunction. */
    private fun reaches(
        cues: Cues,
        text: String,
        windows: List<Utf16Span>,
        sentence: Int,
        hit: MatchResult,
        target: Utf16Span,
    ): Boolean {
        val start = hit.range.first
        val end = hit.range.last + 1
        if (windowOf(windows, start, end) != sentence) return false
        val between = if (normalized(hit.value) in cues.backward) {
            if (target.end > start) return false
            text.substring(target.end, start)
        } else {
            if (end > target.start) return false
            text.substring(end, target.start)
        }
        return !cues.conjunction.containsMatchIn(between)
    }

    /** Python `_text_parts` and the axis resolvers, plus `resolve_experiencer`. */
    private fun assertion(
        cues: Cues,
        text: String,
        target: Utf16Span,
        hits: List<MatchResult>,
    ): ClinicalAssertion {
        val prefix = text.substring(0, target.start)
        val scopeEnd = cues.scope.find(text, target.end)?.range?.first ?: text.length
        val clause = text.substring(lastEnd(cues.scope, prefix, 0).coerceAtLeast(0), scopeEnd)
        val folded = clause.lowercase()
        val parts = ArrayList<String>()
        if (clause.isNotEmpty()) parts += clause
        for (hit in hits) {
            val start = hit.range.first
            val end = hit.range.last + 1
            val between = when {
                end <= target.start -> text.substring(end, target.start)
                target.end <= start -> text.substring(target.end, start)
                else -> ""
            }
            val inClause = clause.isNotEmpty() && hit.value.lowercase() in folded
            if (!inClause && !cues.scope.containsMatchIn(between)) parts += hit.value
        }
        var negations = 0
        var uncertain = false
        var hypothetical = false
        var historical = false
        for (part in parts) {
            val masked = cues.pseudoNegation.replace(part) { " ".repeat(it.value.length) }
            negations += cues.negation.findAll(masked).count()
            uncertain = uncertain || cues.uncertainty.containsMatchIn(part)
            hypothetical = hypothetical || cues.hypothetical.containsMatchIn(part)
            historical = historical || cues.historical.containsMatchIn(part)
        }
        val clauseStart = lastEnd(clauseBoundary, prefix, 0).coerceAtLeast(0)
        val other = lastEnd(cues.other, prefix, clauseStart)
        val family = lastEnd(cues.family, prefix, clauseStart)
        return ClinicalAssertion(
            negation = if (negations % 2 == 1) "negated" else "affirmed",
            uncertainty = if (uncertain) "uncertain" else "certain",
            temporality = if (hypothetical) {
                "hypothetical"
            } else if (historical) {
                "historical"
            } else {
                "recent"
            },
            experiencer = if (other >= 0 && other >= family) {
                "other"
            } else if (family >= 0) {
                "family"
            } else {
                "patient"
            },
        )
    }

    /** End of the last match of [regex] in [text] at or after [from], or -1. */
    private fun lastEnd(regex: Regex, text: String, from: Int): Int {
        var end = -1
        var match = regex.find(text, from)
        while (match != null) {
            end = match.range.last + 1
            match = match.next()
        }
        return end
    }

    private fun addTrimmed(windows: MutableList<Utf16Span>, text: String, from: Int, to: Int) {
        var start = from
        var end = to
        while (start < end && text[start].isWhitespace()) start++
        while (end > start && text[end - 1].isWhitespace()) end--
        if (start < end) windows += Utf16Span(start, end)
    }

    private fun windowOf(windows: List<Utf16Span>, start: Int, end: Int): Int {
        for (index in windows.indices) {
            if (start >= windows[index].start && end <= windows[index].end) return index
        }
        return -1
    }

    private fun utf16(text: String, span: ScalarSpan): Utf16Span {
        require(span.end <= UnicodeOffsetContract.scalarLength(text)) { "span_out_of_range" }
        return UnicodeOffsetContract.utf16Span(text, span.start, span.end)
    }

    private fun normalized(cue: String): String =
        cue.lowercase().trim().split(whitespace).joinToString(" ")

    private fun loadCues(bytes: ByteArray?): Cues {
        val root = Json.parseToJsonElement(String(checkNotNull(bytes), Charsets.UTF_8)).jsonObject
        check(text(root, "schema") == "openmed.tests.context_cues")
        check(text(root, "language") == "en")
        val lexicon = root.getValue("lexicon").jsonObject
        val experiencer = root.getValue("experiencer").jsonObject
        val bounded = lexicon.getValue("token_boundaries").jsonPrimitive.boolean
        val historical = strings(lexicon, "historical")
        val hypothetical = strings(lexicon, "hypothetical")
        val uncertainty = strings(lexicon, "uncertainty")
        val negation = strings(lexicon, "negation")
        val backward = HashSet<String>()
        for (cue in strings(lexicon, "backward")) backward += normalized(cue)
        return Cues(
            negation = cuePattern(negation, bounded),
            pseudoNegation = cuePattern(strings(lexicon, "pseudo_negation"), bounded),
            uncertainty = cuePattern(uncertainty, bounded),
            historical = cuePattern(historical, bounded),
            hypothetical = cuePattern(hypothetical, bounded),
            context = cuePattern(historical + hypothetical + uncertainty + negation, bounded),
            scope = Regex(
                "(?:[.!?;。！？；]|" +
                    cuePattern(strings(lexicon, "scope_terminators"), bounded).pattern + ")",
                RegexOption.IGNORE_CASE,
            ),
            conjunction = cuePattern(strings(lexicon, "conjunction_terminators"), bounded),
            family = cuePattern(strings(experiencer, "family_cues"), true),
            other = cuePattern(strings(experiencer, "other_cues"), true),
            backward = backward,
        )
    }

    private fun text(source: JsonObject, key: String): String =
        source.getValue(key).jsonPrimitive.content

    private fun strings(source: JsonObject, key: String): List<String> {
        val values = ArrayList<String>()
        for (element in source.getValue(key).jsonArray) values += element.jsonPrimitive.content
        return values
    }

    /** Python `_cue_pattern`: distinct cues, longest first, whitespace runs. */
    private fun cuePattern(cues: List<String>, bounded: Boolean): Regex {
        val alternation = StringBuilder()
        val longestFirst = cues.toSortedSet { left, right ->
            if (left.length != right.length) right.length - left.length else left.compareTo(right)
        }
        for (cue in longestFirst) {
            if (alternation.isNotEmpty()) alternation.append('|')
            val words = cue.trim().split(whitespace)
            alternation.append(words.joinToString("$SPACE+") { Regex.escape(it) })
        }
        val body = when {
            alternation.isEmpty() -> "(?!x)x"
            bounded -> "(?<!$WORD)(?:$alternation)(?!$WORD)"
            else -> "(?:$alternation)"
        }
        return Regex(body, RegexOption.IGNORE_CASE)
    }

    private class Cues(
        val negation: Regex,
        val pseudoNegation: Regex,
        val uncertainty: Regex,
        val historical: Regex,
        val hypothetical: Regex,
        val context: Regex,
        val scope: Regex,
        val conjunction: Regex,
        val family: Regex,
        val other: Regex,
        val backward: Set<String>,
    )
}
