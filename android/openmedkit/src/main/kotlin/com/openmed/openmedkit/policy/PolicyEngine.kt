package com.openmed.openmedkit.policy

import com.openmed.openmedkit.KitParity
import com.openmed.openmedkit.KitParityResult
import com.openmed.openmedkit.KitParitySpan
import com.openmed.openmedkit.UnicodeOffsetContract
import java.security.MessageDigest
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * One normalized detector span offered to [PolicyEngine.decide].
 *
 * Offsets are half-open Unicode scalar offsets into the source text.
 * [source] is `model` for model spans or `semantic` for pattern units;
 * [validated] reports whether a semantic unit's pattern validator passed.
 */
data class PolicyDetection(
    val label: String,
    val start: Int,
    val end: Int,
    val confidencePpm: Int,
    val source: String = "model",
    val validated: Boolean = true,
)

/** Value-free decision record for one candidate span. */
data class PolicyDecision(
    val inputs: List<Int>,
    /** `model`, `merged`, or `semantic_only`. */
    val kind: String,
    val label: String,
    val canonicalLabel: String,
    val recognized: Boolean,
    val start: Int,
    val end: Int,
    val confidencePpm: Int,
    val keepFloorPpm: Int,
    val escalate: Boolean,
    val action: String,
    val strictUpgrade: Boolean,
    /** `applied`, `overlap_loser`, `duplicate`, or `below_threshold`. */
    val outcome: String,
    val span: Int?,
)

/** Output of [PolicyEngine.decide]. */
data class PolicyDecisionResult(
    val policyName: String,
    val policyDigest: String,
    val result: KitParityResult,
    val decisions: List<PolicyDecision>,
)

/**
 * The portable policy decision engine (`openmed.portable_merge` version 1).
 *
 * Mirrors `openmed.core.portable_policy.decide`: validation, label
 * normalization, semantic-unit merging, integer parts-per-million threshold
 * decisions, action lookup with caller overrides and the strict no-leak
 * upgrade, exact-duplicate collapse, overlap clustering into union spans, and
 * right-to-left rendering at scalar offsets. Shared vectors in
 * `tests/fixtures/parity/portable_policy_vectors.json` pin the behavior.
 */
object PolicyEngine {
    const val SURROGATE_KEY_MIN_BYTES = 16
    const val SURROGATE_KEY_MAX_BYTES = 64
    private const val SURROGATE_DOMAIN = "openmed.portable_surrogate.v1"
    private val labelPattern = Regex("^[A-Za-z][A-Za-z0-9_:-]{0,79}$")

    /**
     * Apply [policy] to normalized [detections].
     *
     * Without a [surrogateKey], `replace` renders `[LABEL_REPLACED]`; with a
     * 16-to-64-byte caller key it renders a deterministic keyed surrogate.
     */
    fun decide(
        policy: PolicyProfile,
        text: String,
        detections: List<PolicyDetection>,
        overrides: Map<String, String> = emptyMap(),
        surrogateKey: ByteArray? = null,
    ): PolicyDecisionResult {
        validate(detections, UnicodeOffsetContract.scalarLength(text))
        val resolvedOverrides = validate(overrides, policy.taxonomy)
        if (surrogateKey != null && surrogateKey.size !in SURROGATE_KEY_MIN_BYTES..SURROGATE_KEY_MAX_BYTES) {
            throw PolicyProfileException("surrogate_key_invalid", "$.surrogate_key")
        }

        val candidates = buildCandidates(policy.taxonomy, detections)
        for (candidate in candidates) {
            val threshold = policy.thresholdFor(candidate.canonicalLabel)
            candidate.keepFloorPpm = threshold.keepFloorPpm
            val configured = resolvedOverrides[candidate.canonicalLabel]
                ?: policy.configuredAction(candidate.canonicalLabel)
            if (policy.strictNoLeak && configured == PolicyAction.KEEP) {
                candidate.action = PolicyAction.MASK
                candidate.strictUpgrade = true
            } else {
                candidate.action = configured
            }
            if (candidate.confidencePpm >= threshold.keepFloorPpm) {
                candidate.outcome = "retained"
                candidate.escalate = candidate.confidencePpm < threshold.escalateBelowPpm
            }
        }

        val retained = candidates.filter { it.outcome == "retained" }
        val unique = collapseDuplicates(retained, policy)
        val spans = mutableListOf<KitParitySpan>()
        for (cluster in clusters(unique)) {
            val representative = representative(cluster, policy)
            val start = cluster.minOf { it.start }
            val end = cluster.maxOf { it.end }
            for (member in cluster) {
                member.outcome = if (member === representative) "applied" else "overlap_loser"
                member.span = spans.size
            }
            val surface = UnicodeOffsetContract.substring(text, start, end)
            spans += KitParitySpan(
                label = representative.label,
                canonicalLabel = representative.canonicalLabel,
                action = representative.action.wireName,
                start = start,
                end = end,
                confidencePpm = representative.confidencePpm,
                replacement = replacement(
                    representative.action,
                    representative.canonicalLabel,
                    surface,
                    surrogateKey,
                ),
            )
        }
        for (item in retained) {
            if (item.outcome == "duplicate") {
                item.span = unique.first { duplicateKey(it) == duplicateKey(item) }.span
            }
        }

        val output = KitParity.applySpans(text, spans)
        val result = KitParityResult(output, spans).also(KitParity::validate)
        val decisions = candidates.map {
            PolicyDecision(
                inputs = it.inputs,
                kind = it.kind,
                label = it.label,
                canonicalLabel = it.canonicalLabel,
                recognized = it.recognized,
                start = it.start,
                end = it.end,
                confidencePpm = it.confidencePpm,
                keepFloorPpm = it.keepFloorPpm,
                escalate = it.escalate,
                action = it.action.wireName,
                strictUpgrade = it.strictUpgrade,
                outcome = it.outcome,
                span = it.span,
            )
        }
        return PolicyDecisionResult(policy.name, policy.digest, result, decisions)
    }

    /** Replacement text for one output span; `null` for `keep`. */
    fun replacement(
        action: PolicyAction,
        canonicalLabel: String,
        surface: String,
        surrogateKey: ByteArray? = null,
    ): String? = when (action) {
        PolicyAction.KEEP -> null
        PolicyAction.MASK, PolicyAction.REDACT -> "[$canonicalLabel]"
        PolicyAction.REMOVE -> ""
        PolicyAction.REPLACE -> if (surrogateKey == null) {
            "[${canonicalLabel}_REPLACED]"
        } else {
            "[${canonicalLabel}_${surrogateToken(surrogateKey, canonicalLabel, surface)}]"
        }
        PolicyAction.HASH -> {
            val digest = MessageDigest.getInstance("SHA-256").digest(surface.toByteArray(Charsets.UTF_8))
            "${canonicalLabel}_${hex(digest.copyOf(12))}"
        }
    }

    /**
     * Deterministic keyed surrogate token: HMAC-SHA256 over
     * `"openmed.portable_surrogate.v1" 0x00 label 0x00 surface`, first 8 bytes
     * as lowercase hex. The key never leaves the caller.
     */
    fun surrogateToken(key: ByteArray, canonicalLabel: String, surface: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(key, "HmacSHA256"))
        mac.update(SURROGATE_DOMAIN.toByteArray(Charsets.UTF_8))
        mac.update(0)
        mac.update(canonicalLabel.toByteArray(Charsets.UTF_8))
        mac.update(0)
        mac.update(surface.toByteArray(Charsets.UTF_8))
        return hex(mac.doFinal().copyOf(8))
    }

    private class Candidate(
        val inputs: List<Int>,
        val kind: String,
        val label: String,
        val canonicalLabel: String,
        val recognized: Boolean,
        val start: Int,
        val end: Int,
        val confidencePpm: Int,
    ) {
        var keepFloorPpm = 0
        var escalate = false
        var action = PolicyAction.KEEP
        var strictUpgrade = false
        var outcome = "below_threshold"
        var span: Int? = null
    }

    private class Normalized(
        val index: Int,
        val detection: PolicyDetection,
        val canonicalLabel: String,
        val recognized: Boolean,
    )

    private fun validate(detections: List<PolicyDetection>, length: Int) {
        detections.forEachIndexed { index, detection ->
            val path = "$.detections[$index]"
            if (!labelPattern.matches(detection.label)) {
                throw PolicyProfileException("detection_label_invalid", "$path.label")
            }
            if (detection.start < 0) {
                throw PolicyProfileException("detection_offset_invalid", "$path.start")
            }
            if (detection.end <= detection.start) {
                throw PolicyProfileException("detection_offset_invalid", "$path.end")
            }
            if (detection.end > length) {
                throw PolicyProfileException("detection_offset_out_of_bounds", "$path.end")
            }
            if (detection.confidencePpm !in 0..KitParity.CONFIDENCE_SCALE) {
                throw PolicyProfileException("detection_confidence_invalid", "$path.confidence_ppm")
            }
            if (detection.source != "model" && detection.source != "semantic") {
                throw PolicyProfileException("detection_source_invalid", "$path.source")
            }
        }
    }

    private fun validate(overrides: Map<String, String>, taxonomy: PolicyTaxonomy): Map<String, PolicyAction> {
        val resolved = mutableMapOf<String, PolicyAction>()
        for (label in overrides.keys.sorted()) {
            if (label !in taxonomy.policyLabels) {
                throw PolicyProfileException("override_label_unknown", "$.overrides")
            }
            resolved[label] = PolicyAction.fromWireNameOrNull(overrides[label])
                ?: throw PolicyProfileException("override_action_unknown", "$.overrides")
        }
        return resolved
    }

    private val detectionOrder = Comparator<Normalized> { left, right ->
        compareValuesBy(
            left,
            right,
            { it.detection.start },
            { it.detection.end },
            { it.canonicalLabel },
            { it.detection.label },
            { it.index },
        )
    }

    private val candidateOrder = Comparator<Candidate> { left, right ->
        compareValuesBy(
            left,
            right,
            { it.start },
            { it.end },
            { it.canonicalLabel },
            { it.label },
            { it.inputs[0] },
        )
    }

    private fun buildCandidates(taxonomy: PolicyTaxonomy, detections: List<PolicyDetection>): MutableList<Candidate> {
        val normalized = detections.mapIndexed { index, detection ->
            val (canonical, recognized) = taxonomy.normalize(detection.label)
            Normalized(index, detection, canonical, recognized)
        }
        val models = normalized.filter { it.detection.source == "model" }.sortedWith(detectionOrder)
        val units = normalized.filter { it.detection.source == "semantic" }.sortedWith(detectionOrder)
        val consumed = mutableSetOf<Int>()
        val candidates = mutableListOf<Candidate>()
        for (unit in units) {
            val overlapping = models.filter {
                it.detection.start < unit.detection.end && it.detection.end > unit.detection.start
            }
            if (overlapping.isEmpty()) {
                candidates += Candidate(
                    inputs = listOf(unit.index),
                    kind = "semantic_only",
                    label = unit.detection.label,
                    canonicalLabel = unit.canonicalLabel,
                    recognized = unit.recognized,
                    start = unit.detection.start,
                    end = unit.detection.end,
                    confidencePpm = unit.detection.confidencePpm,
                )
                continue
            }
            val dominant = dominantLabel(overlapping)
            val modelLabels = overlapping.map { it.canonicalLabel }.toSet()
            val unitRank = taxonomy.specificity[unit.canonicalLabel] ?: 1
            val dominantRank = taxonomy.specificity[dominant] ?: 1
            val final = if (unit.canonicalLabel in modelLabels || dominantRank > unitRank) {
                dominant
            } else {
                unit.canonicalLabel
            }
            var carrier: Normalized? = null
            for (entry in overlapping) {
                if (entry.canonicalLabel == final &&
                    (carrier == null || entry.detection.confidencePpm > carrier.detection.confidencePpm)
                ) {
                    carrier = entry
                }
            }
            val average = overlapping.sumOf { it.detection.confidencePpm.toLong() } / overlapping.size
            val (modelWeight, unitWeight) = if (unit.detection.validated) 6L to 4L else 9L to 1L
            val confidence = (modelWeight * average + unitWeight * unit.detection.confidencePpm + 5) / 10
            overlapping.forEach { consumed += it.index }
            candidates += Candidate(
                inputs = (listOf(unit.index) + overlapping.map { it.index }).sorted(),
                kind = "merged",
                label = carrier?.detection?.label ?: unit.detection.label,
                canonicalLabel = final,
                recognized = carrier?.recognized ?: unit.recognized,
                start = minOf(unit.detection.start, overlapping.minOf { it.detection.start }),
                end = maxOf(unit.detection.end, overlapping.maxOf { it.detection.end }),
                confidencePpm = confidence.toInt(),
            )
        }
        for (entry in models) {
            if (entry.index in consumed) {
                continue
            }
            candidates += Candidate(
                inputs = listOf(entry.index),
                kind = "model",
                label = entry.detection.label,
                canonicalLabel = entry.canonicalLabel,
                recognized = entry.recognized,
                start = entry.detection.start,
                end = entry.detection.end,
                confidencePpm = entry.detection.confidencePpm,
            )
        }
        candidates.sortWith(candidateOrder)
        return candidates
    }

    private fun dominantLabel(entries: List<Normalized>): String {
        val counts = mutableMapOf<String, Long>()
        val sums = mutableMapOf<String, Long>()
        for (entry in entries) {
            counts[entry.canonicalLabel] = (counts[entry.canonicalLabel] ?: 0L) + 1L
            sums[entry.canonicalLabel] = (sums[entry.canonicalLabel] ?: 0L) + entry.detection.confidencePpm
        }
        var best = ""
        for (label in counts.keys.sorted()) {
            if (best.isEmpty()) {
                best = label
                continue
            }
            val count = counts.getValue(label)
            val bestCount = counts.getValue(best)
            if (count != bestCount) {
                if (count > bestCount) {
                    best = label
                }
                continue
            }
            if (sums.getValue(label) * bestCount > sums.getValue(best) * count) {
                best = label
            }
        }
        return best
    }

    /** Winner order: protective action, specificity, length, confidence, start, labels. */
    private fun winnerOrder(policy: PolicyProfile) = Comparator<Candidate> { left, right ->
        compareValuesBy(
            left,
            right,
            { if (it.action != PolicyAction.KEEP) 0 else 1 },
            { -(policy.taxonomy.specificity[it.canonicalLabel] ?: 1) },
            { -(it.end - it.start) },
            { -it.confidencePpm },
            { it.start },
            { it.canonicalLabel },
            { it.label },
            { it.inputs[0] },
        )
    }

    private fun best(members: List<Candidate>, policy: PolicyProfile): Candidate {
        val order = winnerOrder(policy)
        var winner = members.first()
        for (member in members.drop(1)) {
            if (order.compare(member, winner) < 0) {
                winner = member
            }
        }
        return winner
    }

    private fun duplicateKey(candidate: Candidate): Triple<Int, Int, String> =
        Triple(candidate.start, candidate.end, candidate.canonicalLabel)

    private fun collapseDuplicates(retained: List<Candidate>, policy: PolicyProfile): List<Candidate> {
        val groups = LinkedHashMap<Triple<Int, Int, String>, MutableList<Candidate>>()
        for (item in retained) {
            groups.getOrPut(duplicateKey(item)) { mutableListOf() } += item
        }
        val unique = mutableListOf<Candidate>()
        for (members in groups.values) {
            val winner = best(members, policy)
            for (member in members) {
                if (member !== winner) {
                    member.outcome = "duplicate"
                }
            }
            unique += winner
        }
        return unique.sortedWith(candidateOrder)
    }

    private fun clusters(items: List<Candidate>): List<List<Candidate>> {
        val result = mutableListOf<List<Candidate>>()
        var current = mutableListOf<Candidate>()
        var currentEnd = -1
        for (item in items) {
            if (current.isNotEmpty() && item.start < currentEnd) {
                current += item
                currentEnd = maxOf(currentEnd, item.end)
                continue
            }
            if (current.isNotEmpty()) {
                result += current
            }
            current = mutableListOf(item)
            currentEnd = item.end
        }
        if (current.isNotEmpty()) {
            result += current
        }
        return result
    }

    private fun representative(cluster: List<Candidate>, policy: PolicyProfile): Candidate {
        var members = cluster
        if (policy.arbitrationMode == "high_recall_union") {
            val strongest = members.maxOf { it.action.strength }
            members = members.filter { it.action.strength == strongest }
        }
        return best(members, policy)
    }

    private fun hex(bytes: ByteArray): String =
        bytes.joinToString("") { "%02x".format(it.toInt() and 0xff) }
}
