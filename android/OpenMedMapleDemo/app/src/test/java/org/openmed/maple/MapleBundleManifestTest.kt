package org.openmed.maple

import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import org.junit.Test

class MapleBundleManifestTest {
    @Test
    fun parsesPinnedCachedDecoderContract() {
        val manifest = MapleBundleManifestParser.parse(validManifest())

        assertEquals(MAPLE_SOURCE_MODEL, manifest.sourceModel)
        assertEquals("decoder_model.ort", manifest.graphs.prefillPath)
        assertEquals("decoder_with_past_model.ort", manifest.graphs.decodePath)
        assertEquals(setOf(151645L), manifest.generation.eosTokenIds)
        assertEquals(60L, manifest.totalSizeBytes)
    }

    @Test
    fun parsesStatelessMobileContractProducedByPythonBuilder() {
        val manifest = MapleBundleManifestParser.parse(statelessManifest())

        assertEquals("onnxruntime-mobile", manifest.runtime)
        assertEquals(null, manifest.graphs.decodePath)
        assertEquals(null, manifest.cache)
        assertEquals(40L, manifest.totalSizeBytes)
    }

    @Test
    fun acceptsOneDeclaredGraphForPrefillAndCachedDecode() {
        val manifest = MapleBundleManifestParser.parse(unifiedCachedManifest())

        assertEquals(manifest.graphs.prefillPath, manifest.graphs.decodePath)
        assertEquals(2, manifest.files.size)
        assertEquals(40L, manifest.totalSizeBytes)
        assertTrue(manifest.cache != null)
    }

    @Test
    fun rejectsFloatingRevisionAndPlaceholderChecksum() {
        assertFailsWith<MapleBundleException> {
            MapleBundleManifestParser.parse(validManifest().replace(REVISION, "main"))
        }
        assertFailsWith<MapleBundleException> {
            MapleBundleManifestParser.parse(validManifest().replace(CHECKSUM_A, "0".repeat(64)))
        }
        assertFailsWith<MapleBundleException> {
            MapleBundleManifestParser.parse(
                validManifest().replace("qmoe-4bit-blockwise-128", "ternary-2bit-packed"),
            )
        }
    }

    @Test
    fun rejectsTraversalBeforeResolvingAFile() {
        assertFailsWith<MapleBundleException> {
            MapleBundleManifestParser.parse(
                validManifest().replace("decoder_model.ort", "../decoder_model.ort"),
            )
        }
        assertFailsWith<MapleBundleException> {
            resolveInside(File("build/safe-root"), "../../private/model.onnx")
        }
    }

    @Test
    fun requiresEveryRuntimeFileToBeDeclared() {
        val missingDecode = validManifest().replace(
            "\"decode_path\":\"decoder_with_past_model.ort\"",
            "\"decode_path\":\"missing.ort\"",
        )
        val error = assertFailsWith<MapleBundleException> {
            MapleBundleManifestParser.parse(missingDecode)
        }
        assertTrue(error.message.orEmpty().contains("not declared"))
    }

    private fun validManifest() = """{
      "schema_version":1,
      "source_model":"deepgrove/maple-preview",
      "source_revision":"$REVISION",
      "architecture":"MapleForCausalLM",
      "quantization":"qmoe-4bit-blockwise-128",
      "runtime":"onnxruntime-mobile",
      "tokenizer_path":"tokenizer.json",
      "graphs":{
        "prefill_path":"decoder_model.ort",
        "decode_path":"decoder_with_past_model.ort",
        "input_ids_name":"input_ids",
        "attention_mask_name":"attention_mask",
        "position_ids_name":"position_ids",
        "logits_name":"logits"
      },
      "cache":{"past_input_prefix":"past_key_values.","present_output_prefix":"present."},
      "generation":{"eos_token_ids":[151645],"max_context_tokens":4096,"max_input_tokens":3072},
      "files":[
        {"path":"decoder_model.ort","size_bytes":10,"sha256":"$CHECKSUM_A"},
        {"path":"decoder_with_past_model.ort","size_bytes":20,"sha256":"$CHECKSUM_B"},
        {"path":"tokenizer.json","size_bytes":30,"sha256":"$CHECKSUM_C"}
      ]
    }"""

    private fun statelessManifest() = validManifest()
        .replace(
            "\"decode_path\":\"decoder_with_past_model.ort\"",
            "\"decode_path\":null",
        )
        .replace(
            "\"cache\":{\"past_input_prefix\":\"past_key_values.\",\"present_output_prefix\":\"present.\"}",
            "\"cache\":null",
        )
        .replace(
            "        {\"path\":\"decoder_with_past_model.ort\",\"size_bytes\":20,\"sha256\":\"$CHECKSUM_B\"},\n",
            "",
        )

    private fun unifiedCachedManifest() = validManifest()
        .replace(
            "\"decode_path\":\"decoder_with_past_model.ort\"",
            "\"decode_path\":\"decoder_model.ort\"",
        )
        .replace(
            "        {\"path\":\"decoder_with_past_model.ort\",\"size_bytes\":20,\"sha256\":\"$CHECKSUM_B\"},\n",
            "",
        )

    private companion object {
        const val REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
        val CHECKSUM_A = "1".repeat(64)
        val CHECKSUM_B = "2".repeat(64)
        val CHECKSUM_C = "3".repeat(64)
    }
}
