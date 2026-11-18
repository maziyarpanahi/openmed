package com.openmed.openmedkit.parity

import ai.djl.huggingface.tokenizers.Encoding
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer
import com.openmed.openmedkit.toTokenOffsets
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.BeforeClass
import org.junit.Test
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.security.MessageDigest

/**
 * Runs DJL Hugging Face tokenizers and the OpenMedKit offset adapter against
 * the shared synthetic vectors in `tests/fixtures/parity/tokenizer` before any
 * inference comparison. Diagnostics carry case ids, field names, indices and
 * reason codes only.
 */
class TokenizerConformanceTest {
    @Test
    fun conformanceAssetsMatchVectorsAndManifestDigests() {
        val vectors = readJson(VECTORS)
        val manifest = readJson(MANIFEST)
        val tokenizerDirectory = vectors.getJSONObject("tokenizer").getString("directory")

        assertEquals(CONTRACT_VERSION, manifest.getString("contract_version"))
        assertEquals(
            vectors.getJSONObject("tokenizer").getString("sha256"),
            sha256(Files.readAllBytes(fixtureDirectory().resolve(tokenizerDirectory).resolve("tokenizer.json"))),
        )
        val files = manifest.getJSONArray("files")
        for (index in 0 until files.length()) {
            val file = files.getJSONObject(index)
            assertEquals(
                "TOKENIZER_ASSET_DIGEST_MISMATCH",
                file.getString("sha256"),
                sha256(Files.readAllBytes(fixtureDirectory().resolve(file.getString("path")))),
            )
        }
    }

    @Test
    fun djlTokenizersMatchSharedVectorsBeforeInference() {
        requireNativeTokenizer()
        val vectors = readJson(VECTORS)
        val support = readJson(SUPPORT)
        val directory = fixtureDirectory().resolve(
            vectors.getJSONObject("tokenizer").getString("directory"),
        )

        val report = withTokenizers(directory) { tokenizer, windowed ->
            verify(vectors, support, runner(tokenizer, windowed))
        }
        println(
            "TOKENIZER_CONFORMANCE runtime=$RUNTIME " +
                "status=${if (report.mismatches.isEmpty()) "pass" else "fail"} " +
                "checked=${report.checked} " +
                "unsupported=${report.unsupported.joinToString(",")} " +
                "declared_divergences=${report.declared.size}",
        )

        assertTrue(report.diagnostics(), report.mismatches.isEmpty())
        assertTrue(report.checked > 0)
        assertEquals(
            listOf(
                "normalized:normalizer_not_exposed",
                "window_ranges:overflow_encodings_without_sequence_indices",
            ),
            report.unsupported,
        )
    }

    @Test
    fun specialTokenAndNormalizationChangesFailBeforeModelExecution() {
        requireNativeTokenizer()
        val vectors = readJson(VECTORS)
        val support = readJson(SUPPORT)
        val source = fixtureDirectory().resolve(
            vectors.getJSONObject("tokenizer").getString("directory"),
        )
        var modelCalls = 0

        val withoutCls = mutatedTokenizerDirectory(source) { tokenizer ->
            val postProcessor = tokenizer.getJSONObject("post_processor")
            val single = postProcessor.getJSONArray("single")
            val kept = JSONArray()
            for (index in 0 until single.length()) {
                val item = single.getJSONObject(index)
                if (item.optJSONObject("SpecialToken")?.optString("id") != "[CLS]") {
                    kept.put(item)
                }
            }
            postProcessor.put("single", kept)
        }
        val cased = mutatedTokenizerDirectory(source) { tokenizer ->
            tokenizer.getJSONObject("normalizer")
                .put("lowercase", false)
                .put("strip_accents", false)
        }

        for ((directory, index) in listOf(withoutCls to 0, cased to 1)) {
            val report = withTokenizers(directory) { tokenizer, windowed ->
                verify(vectors, support, runner(tokenizer, windowed))
            }
            if (report.mismatches.isEmpty()) {
                modelCalls += 1
            }
            assertTrue(
                report.diagnostics(),
                report.mismatches.any {
                    it.caseId == "ascii-basic-01" && it.field == "input_ids" && it.index == index
                },
            )
            val cases = vectors.getJSONArray("cases")
            for (caseIndex in 0 until cases.length()) {
                val text = cases.getJSONObject(caseIndex).getString("text")
                if (text.trim().length > 3) {
                    assertFalse(report.diagnostics().contains(text))
                }
            }
        }
        assertEquals(0, modelCalls)
    }

    private data class Mismatch(
        val caseId: String,
        val field: String,
        val index: Int?,
        val code: String,
        val feature: String? = null,
    ) {
        override fun toString(): String =
            "$code case=$caseId field=$field index=${index ?: "-"}" +
                (feature?.let { " feature=$it" } ?: "")
    }

    private class Report {
        var checked = 0
        val unsupported = mutableListOf<String>()
        val declared = mutableListOf<Triple<String, String, String>>()
        val mismatches = mutableListOf<Mismatch>()

        fun diagnostics(): String = mismatches.joinToString("\n")
    }

    private fun runner(
        tokenizer: HuggingFaceTokenizer,
        windowed: (Int, Int, Boolean) -> HuggingFaceTokenizer,
    ): (JSONObject) -> Map<String, List<String>> = { vectorCase ->
        val text = vectorCase.getString("text")
        val encoding = tokenizer.encode(text)
        val produced = mutableMapOf(
            "tokens" to encoding.tokens.toList(),
            "input_ids" to encoding.ids.map(Long::toString),
            "attention_mask" to encoding.attentionMask.map(Long::toString),
            "token_type_ids" to encoding.typeIds.map(Long::toString),
            "special_tokens_mask" to encoding.specialTokenMask.map(Long::toString),
            "offsets" to encoding.toTokenOffsets().map { "${it.startOffset},${it.endOffset}" },
        )
        vectorCase.optJSONObject("window")?.let { window ->
            val maxLength = window.getInt("max_length")
            val stride = window.getInt("stride")
            produced["truncated_input_ids"] = windowed(maxLength, 0, false)
                .encode(text)
                .ids
                .map(Long::toString)
            val windows = windowed(maxLength, stride, true)
                .encode(text)
                .let { first -> listOf(first) + first.overflowing.toList() }
            produced["window_input_ids"] = windows.map { item: Encoding ->
                item.ids.joinToString(",")
            }
        }
        produced
    }

    private fun verify(
        vectors: JSONObject,
        support: JSONObject,
        runner: (JSONObject) -> Map<String, List<String>>,
    ): Report {
        val report = Report()
        if (
            vectors.getString("contract_version") != CONTRACT_VERSION ||
            support.getString("contract_version") != CONTRACT_VERSION
        ) {
            report.mismatches += Mismatch("*", "runtime", null, "TOKENIZER_CONTRACT_VERSION_MISMATCH")
            return report
        }
        val entry = support.getJSONObject("runtimes").optJSONObject(RUNTIME)
        if (entry == null) {
            report.mismatches += Mismatch("*", "runtime", null, "TOKENIZER_RUNTIME_UNDECLARED")
            return report
        }
        val statuses = entry.getJSONObject("fields")
        val unsupported = mutableSetOf<String>()
        for (field in FIELDS) {
            val status = statuses.optJSONObject(field)
            if (status?.optString("status") != "supported") {
                unsupported += field
                report.unsupported += "$field:${status?.optString("reason") ?: "undeclared"}"
            }
        }
        val divergences = entry.getJSONArray("divergences").objects()
        val observed = mutableSetOf<String>()
        for (vectorCase in vectors.getJSONArray("cases").objects()) {
            val caseId = vectorCase.getString("id")
            val produced = try {
                runner(vectorCase)
            } catch (error: Exception) {
                report.mismatches += Mismatch(caseId, "runtime", null, "TOKENIZER_RUNTIME_ERROR")
                continue
            }
            val expected = vectorCase.getJSONObject("expected")
            val features = vectorCase.getJSONArray("features").strings().toSet()
            for (field in FIELDS) {
                if (!expected.has(field) || field in unsupported) {
                    continue
                }
                val actual = produced[field]
                if (actual == null) {
                    report.mismatches += Mismatch(caseId, field, null, "TOKENIZER_FIELD_MISSING")
                    continue
                }
                val finding = compare(canonical(expected, field), actual)
                if (finding == null) {
                    report.checked += 1
                    continue
                }
                val matching = divergences.filter {
                    it.getString("feature") in features && field in it.getJSONArray("fields").strings()
                }
                if (matching.isNotEmpty()) {
                    matching.forEach { observed += "${it.getString("feature")}|$field" }
                    report.declared += Triple(caseId, field, matching.first().getString("reason"))
                    continue
                }
                report.checked += 1
                report.mismatches += Mismatch(caseId, field, finding.second, finding.first)
            }
        }
        for (item in divergences) {
            for (field in item.getJSONArray("fields").strings()) {
                if ("${item.getString("feature")}|$field" !in observed) {
                    report.mismatches += Mismatch(
                        "*",
                        field,
                        null,
                        "TOKENIZER_DECLARED_DIVERGENCE_STALE",
                        item.getString("feature"),
                    )
                }
            }
        }
        return report
    }

    private fun canonical(expected: JSONObject, field: String): List<String> {
        if (field == "normalized") {
            val text = expected.getString(field)
            return text.codePoints().toArray().map { String(Character.toChars(it)) }
        }
        val values = expected.getJSONArray(field)
        return (0 until values.length()).map { index ->
            when (val value = values.get(index)) {
                is JSONArray -> (0 until value.length()).joinToString(",") { value.get(it).toString() }
                else -> value.toString()
            }
        }
    }

    private fun compare(expected: List<String>, actual: List<String>): Pair<String, Int>? {
        val shared = minOf(expected.size, actual.size)
        for (index in 0 until shared) {
            if (expected[index] != actual[index]) {
                return "TOKENIZER_FIELD_VALUE_MISMATCH" to index
            }
        }
        if (expected.size != actual.size) {
            return "TOKENIZER_FIELD_LENGTH_MISMATCH" to shared
        }
        return null
    }

    private fun <T> withTokenizers(
        directory: Path,
        block: (HuggingFaceTokenizer, (Int, Int, Boolean) -> HuggingFaceTokenizer) -> T,
    ): T {
        val opened = mutableListOf<HuggingFaceTokenizer>()
        try {
            // Same offline entry point as BackendOnnxTokenClassifier.loadTokenizer.
            val tokenizer = HuggingFaceTokenizer.newInstance(directory).also { opened += it }
            // DJL pads overflow windows to the longest window by default; the
            // contract compares unpadded windows, so padding is disabled.
            val windowed = { maxLength: Int, stride: Int, overflow: Boolean ->
                HuggingFaceTokenizer.builder()
                    .optTokenizerPath(directory)
                    .optAddSpecialTokens(true)
                    .optPadding(false)
                    .optTruncation(true)
                    .optMaxLength(maxLength)
                    .optStride(stride)
                    .optWithOverflowingTokens(overflow)
                    .build()
                    .also { opened += it }
            }
            return block(tokenizer, windowed)
        } finally {
            opened.forEach(HuggingFaceTokenizer::close)
        }
    }

    private fun mutatedTokenizerDirectory(source: Path, mutate: (JSONObject) -> Unit): Path {
        val directory = Files.createTempDirectory("openmed-tokenizer-conformance")
        directory.toFile().deleteOnExit()
        val tokenizer = JSONObject(String(Files.readAllBytes(source.resolve("tokenizer.json")), Charsets.UTF_8))
        mutate(tokenizer)
        Files.write(directory.resolve("tokenizer.json"), tokenizer.toString().toByteArray(Charsets.UTF_8))
        for (name in listOf("tokenizer_config.json", "config.json")) {
            Files.copy(source.resolve(name), directory.resolve(name))
        }
        directory.toFile().listFiles()?.forEach { it.deleteOnExit() }
        return directory
    }

    private fun readJson(name: String): JSONObject =
        JSONObject(String(Files.readAllBytes(fixtureDirectory().resolve(name)), Charsets.UTF_8))

    private fun sha256(data: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(data).joinToString("") { "%02x".format(it) }

    private fun JSONArray.objects(): List<JSONObject> = (0 until length()).map(::getJSONObject)

    private fun JSONArray.strings(): List<String> = (0 until length()).map(::getString)

    companion object {
        private const val RUNTIME = "djl-tokenizers"
        private const val CONTRACT_VERSION = "openmed.tokenizer-conformance.v1"
        private const val VECTORS = "tokenizer_conformance_v1.json"
        private const val SUPPORT = "tokenizer_runtime_support_v1.json"
        private const val MANIFEST = "manifest.json"
        private val FIELDS = listOf(
            "normalized",
            "tokens",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "special_tokens_mask",
            "offsets",
            "truncated_input_ids",
            "window_ranges",
            "window_input_ids",
        )

        private fun fixtureDirectory(): Path {
            val resource = TokenizerConformanceTest::class.java.classLoader
                ?.getResource("tokenizer/$VECTORS")
            assertTrue("Missing tokenizer conformance fixture resource", resource != null)
            return Paths.get(resource!!.toURI()).parent
        }

        private val nativeTokenizerLoads: Boolean by lazy {
            try {
                HuggingFaceTokenizer.newInstance(
                    fixtureDirectory().resolve("synthetic-wordpiece-v1"),
                ).close()
                true
            } catch (error: Throwable) {
                false
            }
        }

        /** Keep DJL offline and its extracted native library out of the user cache. */
        @BeforeClass
        @JvmStatic
        fun configureOfflineNativeTokenizer() {
            System.setProperty("ai.djl.offline", "true")
            val cache = Files.createTempDirectory("openmed-djl-cache")
            cache.toFile().deleteOnExit()
            System.setProperty("DJL_CACHE_DIR", cache.toString())
            System.setProperty("ENGINE_CACHE_DIR", cache.toString())
        }

        /**
         * Skips with an explicit unsupported status when the platform has no
         * bundled DJL native library, for example an x86_64 JVM on Apple silicon.
         */
        private fun requireNativeTokenizer() {
            if (!nativeTokenizerLoads) {
                println(
                    "TOKENIZER_CONFORMANCE runtime=$RUNTIME status=unsupported " +
                        "reason=native_library_unavailable_offline",
                )
            }
            assumeTrue("native_library_unavailable_offline", nativeTokenizerLoads)
        }
    }
}
