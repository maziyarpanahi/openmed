package com.openmed.openmedkit.decision

import java.security.MessageDigest
import kotlin.math.exp

/**
 * Tokenizer seam for an encoder decision checkpoint.
 *
 * [encode] must return the ids the checkpoint was exported with, including its
 * special tokens, and must never truncate or pad: the runtime refuses inputs
 * over the token budget instead of scoring a truncated text.
 */
public interface DecisionTokenizer {
    public fun encode(text: String): IntArray

    /** The id of a control token such as `<<LABEL>>`, when the vocabulary has it. */
    public fun tokenId(token: String): Int? = null
}

/**
 * A GLiClass-style uni-encoder: `input_ids` and `attention_mask` of shape
 * `[1, tokens]` in, the first row of `logits` (one value per `<<LABEL>>` slot,
 * in label order) out. [OnnxDecisionEncoder] runs an ONNX export.
 */
public fun interface DecisionEncoder {
    public fun logits(inputIds: IntArray, attentionMask: IntArray): DoubleArray
}

/**
 * Value-free encoder adapter failure. [reason] is `encoder_output_missing`,
 * `encoder_output_invalid` or `encoder_input_invalid`.
 */
public class DecisionEncoderException(public val reason: String) : Exception(reason)

/** One synthetic few-shot example rendered as GLiClass example tokens. */
public class GLiClassExample(text: String, labels: List<String>) {
    internal val text: String
    internal val labels: List<String>

    init {
        val reason = "gliclass_example_invalid"
        if (!GLiClassText.isValid(text, 2048, 8192, multiline = true) ||
            labels.size !in 1..OpenMedDecision.MAX_OPTIONS
        ) {
            throw DecisionRuntimeError.InvalidConfiguration(reason)
        }
        this.text = text
        this.labels = labels.map { GLiClassText.label(it, reason) }
    }

    override fun toString(): String = "GLiClassExample(labels=${labels.size})"
}

/**
 * Backend settings of a GLiClass v4 uni-encoder decision runtime.
 *
 * The prompt, few-shot examples and score levels belong to the runtime, as in
 * the Python `GLiClassDecisionBackend`; requests that carry per-question
 * instructions, option descriptions or levels are reported as `unsupported`
 * rather than silently ignoring their wording.
 *
 * @property promptFirst `true` for GLiClass v4 (labels and prompt before the text).
 * @property maxLength Token budget; longer inputs are refused, never truncated.
 * @property labelTokenId The `<<LABEL>>` id (the export's `class_token_index`);
 *   when `null` the tokenizer's [DecisionTokenizer.tokenId] is used.
 */
public class GLiClassDecisionConfiguration(
    public val prompt: String? = null,
    public val examples: List<GLiClassExample> = emptyList(),
    scoreLevels: List<String> = DEFAULT_SCORE_LEVELS,
    public val promptFirst: Boolean = true,
    public val maxLength: Int = DEFAULT_MAX_LENGTH,
    public val labelTokenId: Int? = null,
) {
    /** Ordered levels, lowest first, scored as labels for `scalar_score`. */
    public val scoreLevels: List<String>

    init {
        val reason = "gliclass_configuration_invalid"
        if (prompt != null && !GLiClassText.isValid(prompt, 4096, 16_384, multiline = true)) {
            throw DecisionRuntimeError.InvalidConfiguration(reason)
        }
        val levels = scoreLevels.map { GLiClassText.label(it, reason) }
        if (examples.size > 8 || levels.size !in 2..16 ||
            levels.map { it.uppercase().lowercase() }.toSet().size != levels.size ||
            maxLength < 1 || (labelTokenId ?: 0) < 0
        ) {
            throw DecisionRuntimeError.InvalidConfiguration(reason)
        }
        this.scoreLevels = levels
    }

    /** SHA-256 of the instruction prompt, safe to record in audit metadata. */
    public val promptDigest: String?
        get() {
            val value = prompt ?: return null
            val digest = MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
            val hex = StringBuilder("sha256:")
            for (byte in digest) hex.append(HEX[(byte.toInt() shr 4) and 15]).append(HEX[byte.toInt() and 15])
            return hex.toString()
        }

    /**
     * Renders the uni-encoder input used by `gliclass` 0.1.21 and the Python
     * reference. The result contains the input text and must never be logged.
     */
    public fun formatInput(text: String, labels: List<String>): String {
        val head = StringBuilder()
        for (label in labels) head.append(LABEL_TOKEN).append(label)
        head.append(SEPARATOR_TOKEN)
        if (!prompt.isNullOrEmpty()) head.append(prompt)
        val rendered = StringBuilder()
        for (example in examples) {
            rendered.append(EXAMPLE_TOKEN).append(example.text).append(" \nLabels:\n ")
            example.labels.forEachIndexed { index, label ->
                if (index > 0) rendered.append(", ")
                rendered.append(label)
            }
        }
        if (examples.isNotEmpty()) rendered.append(SEPARATOR_TOKEN)
        return if (promptFirst) "$head$text$rendered" else "$text$head$rendered"
    }

    public companion object {
        public val DEFAULT_SCORE_LEVELS: List<String> =
            listOf("very low", "low", "moderate", "high", "very high")
        public const val DEFAULT_MAX_LENGTH: Int = 2048
        public const val LABEL_TOKEN: String = "<<LABEL>>"
        public const val SEPARATOR_TOKEN: String = "<<SEP>>"
        public const val EXAMPLE_TOKEN: String = "<<EXAMPLE>>"
        private const val HEX = "0123456789abcdef"
    }
}

/**
 * On-device fixed-option decisions over a GLiClass-style encoder.
 *
 * The runtime validates a decision request (schema 1.0.0, 1.1.0 or 1.2.0),
 * applies the access policy and the supplied calibration profile, scores the
 * options through the injected tokenizer and encoder, and returns a result with
 * the same fields, states, abstention codes and review flags as the Python
 * `openmed.structured.decision.decide`. It is text-only: media requests return
 * `unsupported`. It makes no network call and never logs or returns the input,
 * prompt or examples.
 *
 * @param calibrationProfiles Profiles available to requests by `calibration_id`.
 * @param clock Monotonic nanoseconds used for the request timeout.
 */
public class EncoderDecisionRuntime(
    private val encoder: DecisionEncoder,
    private val tokenizer: DecisionTokenizer,
    public val identity: DecisionBackendIdentity,
    public val configuration: GLiClassDecisionConfiguration = GLiClassDecisionConfiguration(),
    public val policy: DecisionAccessPolicy = DecisionAccessPolicy.STANDARD,
    calibrationProfiles: List<DecisionCalibrationProfile> = listOf(DecisionCalibrationProfile.STANDARD),
    private val clock: () -> Long = System::nanoTime,
) {
    private val profiles: Map<String, DecisionCalibrationProfile>

    init {
        val identifiers = calibrationProfiles.map { it.calibrationId }
        if (identifiers.isEmpty() || identifiers.toSet().size != identifiers.size) {
            throw DecisionRuntimeError.InvalidConfiguration("calibration_profiles_invalid")
        }
        profiles = calibrationProfiles.associateBy { it.calibrationId }
    }

    /**
     * Parses, validates and decides a request mapping.
     *
     * @throws DecisionRuntimeError for requests that violate the contract.
     *   Backend refusals and failures are result states instead.
     */
    public fun decide(requestJson: String): DecisionResult = decide(DecisionRequest.parse(requestJson))

    /**
     * Decides a validated request. Abstentions, denials, refusals and failures
     * are typed result states, never fabricated selections.
     */
    public fun decide(request: DecisionRequest): DecisionResult {
        val access = policy.decision(request.namespace, request.purpose, identity.kind)
        val profile = profiles[request.calibrationId]
            ?: return terminal(
                request,
                DecisionState.UNSUPPORTED,
                "calibration_unavailable",
                DecisionCalibrationProfile.STANDARD,
                access,
            )
        if (!access.allowed) {
            return terminal(request, DecisionState.DENIED, access.code ?: "access_denied", profile, access)
        }
        if (request.mediaCount > 0) {
            return terminal(request, DecisionState.UNSUPPORTED, "backend_unsupported", profile, access)
        }
        val started = clock()
        val answers = request.questions.map { question ->
            if (request.mode == DecisionMode.QUESTION_SET && elapsedMs(started) > request.timeoutMs) {
                DecisionAnswer.terminal(DecisionState.FAILURE, "backend_timeout")
            } else {
                score(question, request, profile, started)
            }
        }
        return assemble(request, answers, profile, access)
    }

    private fun elapsedMs(started: Long): Double = (clock() - started) / 1_000_000.0

    private fun score(
        question: DecisionQuestion,
        request: DecisionRequest,
        profile: DecisionCalibrationProfile,
        started: Long,
    ): DecisionAnswer {
        val output = try {
            scores(question, request.inputText)
        } catch (refusal: DecisionRefusal) {
            return DecisionAnswer.terminal(refusal.state, refusal.code)
        } catch (error: Exception) {
            return DecisionAnswer.terminal(DecisionState.FAILURE, "backend_failure")
        }
        if (elapsedMs(started) > request.timeoutMs) {
            return DecisionAnswer.terminal(DecisionState.FAILURE, "backend_timeout")
        }
        return DecisionAnswer.from(output, question, profile)
    }

    private fun scores(question: DecisionQuestion, text: String): DecisionScores {
        if (question.instructions != null ||
            question.optionDescriptions.orEmpty().any { it != null } ||
            question.levels.isNotEmpty()
        ) {
            throw DecisionRefusal.unsupported()
        }
        val labels =
            if (question.mode == DecisionMode.SCALAR_SCORE) configuration.scoreLevels else question.options
        if (GLiClassText.hasMarker(text) || labels.any(GLiClassText::hasMarker)) {
            throw DecisionRefusal.unsupported()
        }
        val inputIds = tokenizer.encode(configuration.formatInput(text, labels))
        if (inputIds.size > configuration.maxLength) throw DecisionRefusal.unsupported()
        val labelId = configuration.labelTokenId
            ?: tokenizer.tokenId(GLiClassDecisionConfiguration.LABEL_TOKEN)
        if (labelId != null && inputIds.count { it == labelId } != labels.size) {
            throw DecisionRefusal.conflict()
        }
        val raw = try {
            encoder.logits(inputIds, IntArray(inputIds.size) { 1 })
        } catch (error: DecisionEncoderException) {
            if (error.reason == "encoder_input_invalid") throw error
            throw DecisionRefusal.conflict()
        }
        if (raw.size < labels.size) throw DecisionRefusal.conflict()
        val logits = raw.copyOf(labels.size)
        if (!logits.all { it.isFinite() }) throw DecisionRefusal.conflict()
        val probabilities = if (question.mode == DecisionMode.MULTI_LABEL) {
            logits.map(DecisionMath::sigmoid)
        } else {
            DecisionMath.softmax(logits)
        }
        return if (question.mode == DecisionMode.SCALAR_SCORE) {
            DecisionScores(
                scalar = DecisionMath.expectedLevel(probabilities),
                confidence = probabilities.max(),
            )
        } else {
            DecisionScores(options = probabilities)
        }
    }

    private fun terminal(
        request: DecisionRequest,
        state: DecisionState,
        code: String,
        profile: DecisionCalibrationProfile,
        access: DecisionAccess,
    ): DecisionResult = assemble(
        request,
        request.questions.map { DecisionAnswer.terminal(state, code) },
        profile,
        access,
    )

    private fun assemble(
        request: DecisionRequest,
        answers: List<DecisionAnswer>,
        profile: DecisionCalibrationProfile,
        access: DecisionAccess,
    ): DecisionResult {
        var outer = answers.firstOrNull() ?: DecisionAnswer.terminal(DecisionState.FAILURE, "backend_failure")
        var questions = emptyList<DecisionQuestionResult>()
        if (request.mode == DecisionMode.QUESTION_SET) {
            questions = request.questions.zip(answers) { question, answer ->
                DecisionQuestionResult(
                    questionId = question.questionId,
                    mode = question.mode,
                    state = answer.state,
                    code = answer.code,
                    optionScores = answer.optionScores,
                    choice = answer.choice,
                    choices = answer.choices,
                    ranking = answer.ranking,
                    scalarScore = answer.scalarScore,
                    confidence = answer.confidence,
                    margin = answer.margin,
                    warnings = DecisionAnswer.warnings(answer.code),
                    reviewReasons = DecisionAnswer.reviewReasons(answer.code),
                )
            }
            outer = DecisionAnswer.aggregate(answers)
        }
        return DecisionResult(
            mode = request.mode,
            state = outer.state,
            code = outer.code,
            optionScores = outer.optionScores,
            choice = outer.choice,
            choices = outer.choices,
            ranking = outer.ranking,
            scalarScore = outer.scalarScore,
            confidence = outer.confidence,
            margin = outer.margin,
            questions = questions,
            calibration = profile,
            backend = identity,
            access = access,
            warnings = DecisionAnswer.warnings(outer.code),
            reviewReasons = DecisionAnswer.reviewReasons(outer.code),
            schemaVersion = request.contractVersion,
        )
    }
}

internal class DecisionScores(
    val options: List<Double>? = null,
    val scalar: Double? = null,
    val confidence: Double? = null,
)

internal class DecisionRefusal private constructor(val state: DecisionState, val code: String) :
    Exception(code) {
    companion object {
        fun unsupported() = DecisionRefusal(DecisionState.UNSUPPORTED, "backend_unsupported")

        fun conflict() = DecisionRefusal(DecisionState.CONFLICT, "backend_conflict")
    }
}

internal class DecisionAnswer(
    val state: DecisionState,
    val code: String?,
    val optionScores: List<DecisionOptionScore> = emptyList(),
    val choice: String? = null,
    val choices: List<String> = emptyList(),
    val ranking: List<String> = emptyList(),
    val scalarScore: Double? = null,
    val confidence: Double? = null,
    val margin: Double? = null,
) {
    companion object {
        fun terminal(state: DecisionState, code: String) = DecisionAnswer(state, code)

        /** The Python `_decision_from_scores` policy for encoder outputs. */
        fun from(
            output: DecisionScores,
            question: DecisionQuestion,
            profile: DecisionCalibrationProfile,
        ): DecisionAnswer {
            val scores = output.options
            if (scores == null) {
                val confidence = output.confidence ?: 0.0
                val success = confidence >= profile.minimumConfidence
                return DecisionAnswer(
                    state = if (success) DecisionState.SUCCESS else DecisionState.ABSTAINED,
                    code = if (success) null else "low_confidence",
                    scalarScore = if (success) output.scalar else null,
                    confidence = confidence,
                )
            }
            val options = question.options
            val ranked = scores.indices.sortedWith { left, right ->
                if (scores[left] != scores[right]) scores[right].compareTo(scores[left]) else left - right
            }
            val confidence = scores[ranked[0]]
            val margin = confidence - scores[ranked[1]]
            val ranking = ranked.map { options[it] }
            val optionScores = scores.indices.map { DecisionOptionScore(it, options[it], scores[it]) }
            if (question.mode == DecisionMode.MULTI_LABEL) {
                val selected = scores.indices.filter { scores[it] >= profile.multiLabelThreshold }
                    .map { options[it] }
                return DecisionAnswer(
                    state = if (selected.isEmpty()) DecisionState.ABSTAINED else DecisionState.SUCCESS,
                    code = if (selected.isEmpty()) "no_label_above_threshold" else null,
                    optionScores = optionScores,
                    choice = selected.firstOrNull(),
                    choices = selected,
                    ranking = ranking,
                    confidence = confidence,
                    margin = margin,
                )
            }
            val clearsConfidence = confidence >= profile.minimumConfidence
            val clearsMargin = margin >= profile.minimumMargin
            val success = clearsConfidence && clearsMargin
            return DecisionAnswer(
                state = if (success) DecisionState.SUCCESS else DecisionState.ABSTAINED,
                code = when {
                    !clearsConfidence -> "low_confidence"
                    !clearsMargin -> "ambiguous_scores"
                    else -> null
                },
                optionScores = optionScores,
                choice = if (success) ranking[0] else null,
                choices = if (success) listOf(ranking[0]) else emptyList(),
                ranking = ranking,
                confidence = confidence,
                margin = margin,
            )
        }

        /**
         * Question-set state: success, the shared state and code, the shared
         * state with `question_set_<state>`, or `partial`.
         */
        fun aggregate(answers: List<DecisionAnswer>): DecisionAnswer {
            val states = answers.map { it.state }.toSet()
            if (states == setOf(DecisionState.SUCCESS)) return DecisionAnswer(DecisionState.SUCCESS, null)
            val outcomes = answers.map { it.state to it.code }.toSet()
            if (outcomes.size == 1) {
                val (state, code) = outcomes.first()
                return DecisionAnswer(state, code)
            }
            if (states.size == 1) {
                val state = states.first()
                return terminal(state, "question_set_${state.wire}")
            }
            return terminal(DecisionState.PARTIAL, "question_set_partial")
        }

        fun warnings(code: String?): List<String> =
            listOf("human_review_required") + listOfNotNull(code?.takeIf { it != "human_review_required" })

        fun reviewReasons(code: String?): List<String> =
            listOf("clinical_use_requires_review") + listOfNotNull(code)
    }
}

internal object DecisionMath {
    private fun unit(value: Double): Double = value.coerceIn(0.0, 1.0)

    fun softmax(logits: DoubleArray): List<Double> {
        val peak = logits.maxOrNull() ?: 0.0
        val exps = logits.map { exp(it - peak) }
        var total = 0.0
        for (value in exps) total += value
        return exps.map { unit(it / total) }
    }

    fun sigmoid(value: Double): Double {
        if (value >= 0) return unit(1.0 / (1.0 + exp(-value)))
        val exp = exp(value)
        return unit(exp / (1.0 + exp))
    }

    fun expectedLevel(probabilities: List<Double>): Double {
        var expected = 0.0
        probabilities.forEachIndexed { index, value -> expected += index * value }
        return unit(expected / (probabilities.size - 1))
    }
}

internal object GLiClassText {
    private val MARKERS = listOf(
        GLiClassDecisionConfiguration.LABEL_TOKEN,
        GLiClassDecisionConfiguration.SEPARATOR_TOKEN,
        GLiClassDecisionConfiguration.EXAMPLE_TOKEN,
    )

    fun hasMarker(value: String): Boolean = MARKERS.any { value.contains(it) }

    fun isValid(value: String, maximumChars: Int, maximumBytes: Int, multiline: Boolean): Boolean {
        val allowed = if (multiline) "\n\r\t" else ""
        return !value.isBlank() && DecisionText.codePoints(value) <= maximumChars &&
            !DecisionText.hasSurrogate(value) &&
            value.none { (it.code < 32 && it !in allowed) || it.code == 127 } &&
            value.utf8Size() <= maximumBytes && !hasMarker(value)
    }

    fun label(value: String, reason: String): String {
        val stripped = value.trim()
        if (!isValid(stripped, 256, 1024, multiline = false)) {
            throw DecisionRuntimeError.InvalidConfiguration(reason)
        }
        return stripped
    }
}
