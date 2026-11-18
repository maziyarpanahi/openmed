package com.openmed.openmedkit.emulator

import android.content.Context
import java.io.File
import java.security.MessageDigest
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.long

/**
 * One reference token from the Python ONNX Runtime run.
 *
 * Offsets are half-open Unicode scalar offsets; special tokens are `[0, 0)`
 * and carry no label or score.
 */
internal data class LaneToken(
    val id: Long,
    val start: Int,
    val end: Int,
    val special: Boolean,
    val labelId: Int?,
    val score: Float?,
)

/** A value-free span signature: a label and half-open Unicode scalar offsets. */
internal data class LaneSpan(
    val label: String,
    val start: Int,
    val end: Int,
) {
    override fun toString(): String = "$label[$start,$end)"
}

/** A reference span and the confidence Python derived for it. */
internal data class ExpectedSpan(
    val span: LaneSpan,
    val confidence: Float,
)

/** One synthetic case of the emulator lane fixture. */
internal data class LaneCase(
    val id: String,
    val text: String,
    val tokens: List<LaneToken>,
    val spans: List<ExpectedSpan>,
)

/** Comparison tolerances recorded by the fixture generator. */
internal data class LaneContract(
    val confidenceThreshold: Float,
    val tokenScoreTolerance: Float,
    val spanConfidenceTolerance: Float,
)

/**
 * The synthetic model and Python reference output packaged in the test APK.
 *
 * `scripts/android/generate_emulator_lane_fixture.py` writes the assets under
 * `assets/emulator-lane/`. Every case text is synthetic; spans and tokens hold
 * only labels, offsets and scores.
 */
internal class LaneFixture private constructor(
    val contract: LaneContract,
    val cases: List<LaneCase>,
    private val modelDirectory: String,
    private val modelDigests: Map<String, String>,
) {
    /**
     * Copy the model directory out of the APK into app-private storage.
     *
     * `OpenMedKit.fromDirectory` needs real files. The copy is checked against
     * the SHA-256 digests the generator recorded.
     */
    fun installModel(context: Context): File {
        val target = File(context.noBackupFilesDir, "openmed-emulator-lane/$modelDirectory")
        target.deleteRecursively()
        check(target.mkdirs()) { "lane_model_directory_unavailable" }
        modelDigests.forEach { (name, sha256) ->
            val file = File(target, name)
            context.assets.open("$ASSET_ROOT/$modelDirectory/$name").use { input ->
                file.outputStream().use { output -> input.copyTo(output) }
            }
            check(file.sha256() == sha256) { "lane_model_digest_mismatch:$name" }
        }
        return target
    }

    companion object {
        const val ASSET_ROOT = "emulator-lane"
        private const val FIXTURE_FORMAT = "openmed-android-emulator-lane"
        private val CASE_ID = Regex("synth-lane-\\d{3}")

        /** Parse `assets/emulator-lane/expected.json` from the test APK. */
        fun load(context: Context): LaneFixture {
            val root = context.assets.open("$ASSET_ROOT/expected.json").use { input ->
                Json.parseToJsonElement(input.readBytes().decodeToString()).jsonObject
            }
            check(root.string("format") == FIXTURE_FORMAT) { "lane_fixture_format" }
            val model = root.getValue("model").jsonObject
            val contract = root.getValue("contract").jsonObject
            return LaneFixture(
                contract = LaneContract(
                    confidenceThreshold = contract.float("confidence_threshold"),
                    tokenScoreTolerance = contract.float("token_score_tolerance"),
                    spanConfidenceTolerance = contract.float("span_confidence_tolerance"),
                ),
                cases = root.getValue("cases").jsonArray.map { parseCase(it.jsonObject) },
                modelDirectory = model.string("directory"),
                modelDigests = model.getValue("files").jsonObject.mapValues { (_, value) ->
                    value.jsonObject.string("sha256")
                },
            )
        }

        private fun parseCase(case: JsonObject): LaneCase {
            val id = case.string("id")
            check(CASE_ID.matches(id)) { "lane_fixture_case_id" }
            check(case.getValue("synthetic").jsonPrimitive.boolean) { "lane_fixture_not_synthetic" }
            return LaneCase(
                id = id,
                text = case.string("text"),
                tokens = case.getValue("tokens").jsonArray.map { element ->
                    val token = element.jsonObject
                    val special = token.getValue("special").jsonPrimitive.boolean
                    LaneToken(
                        id = token.getValue("id").jsonPrimitive.long,
                        start = token.int("start"),
                        end = token.int("end"),
                        special = special,
                        labelId = if (special) null else token.int("label_id"),
                        score = if (special) null else token.float("score"),
                    )
                },
                spans = case.getValue("spans").jsonArray.map { element ->
                    val span = element.jsonObject
                    ExpectedSpan(
                        span = LaneSpan(
                            label = span.string("label"),
                            start = span.int("start"),
                            end = span.int("end"),
                        ),
                        confidence = span.float("confidence"),
                    )
                },
            )
        }

        private fun JsonObject.string(key: String): String =
            getValue(key).jsonPrimitive.content

        private fun JsonObject.int(key: String): Int = getValue(key).jsonPrimitive.int

        private fun JsonObject.float(key: String): Float = getValue(key).jsonPrimitive.float

        private fun File.sha256(): String {
            val digest = MessageDigest.getInstance("SHA-256")
            inputStream().use { input ->
                val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
                while (true) {
                    val read = input.read(buffer)
                    if (read < 0) break
                    digest.update(buffer, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }
    }
}
