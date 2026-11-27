package com.openmed.openmedkit.packs

import com.openmed.openmedkit.OnnxTokenClassifier
import com.openmed.openmedkit.TokenClassificationPrediction
import com.openmed.openmedkit.UnicodeOffsetContract
import com.openmed.openmedkit.catalog.ModelCatalog
import com.openmed.openmedkit.download.HuggingFaceModelClient
import com.openmed.openmedkit.download.HuggingFaceModelFile
import com.openmed.openmedkit.download.HuggingFaceModelInfo
import com.openmed.openmedkit.download.ModelIntegrity
import java.io.File
import java.security.MessageDigest

/** Synthetic files of a fake ONNX repository; nothing here is a real model. */
internal val FAKE_MODEL_FILES = mapOf(
    "model_int8.onnx" to "synthetic-weights",
    "tokenizer.json" to "{}",
    "id2label.json" to "{\"0\":\"O\"}",
)

internal const val FAKE_REVISION = "0123456789abcdef"
internal const val FAKE_RELEASED = "2026-01-01"

/** The repository root, from the module directory the Gradle test task passes. */
internal fun repositoryRoot(): File = File(
    requireNotNull(System.getProperty("openmedkit.moduleDirectory")) {
        "openmedkit.moduleDirectory test property is missing"
    },
).resolve("../..").canonicalFile

/** A catalog with one integrity-pinned ONNX entry per repository id. */
internal fun fakeCatalog(vararg repoIds: String): ModelCatalog =
    ModelCatalog.fromJsonLines(
        repoIds.joinToString("\n") { repoId ->
            val hash = ModelIntegrity.reproducibilityHash(
                repoId = repoId,
                revisionSha = FAKE_REVISION,
                released = FAKE_RELEASED,
                siblings = FAKE_MODEL_FILES.keys.toList(),
            )
            """{"repo_id":"$repoId","formats":["onnx"],"tier":null,""" +
                """"param_count":33000000,"languages":["en"],"license":"apache-2.0",""" +
                """"reproducibility_hash":"$hash"}"""
        },
    )

/**
 * Hugging Face client double that serves [FAKE_MODEL_FILES] and records every
 * call. With [corrupt], downloaded bytes fail the published checksum.
 */
internal class FakeHubClient(private val corrupt: Boolean = false) : HuggingFaceModelClient {
    val calls = mutableListOf<String>()

    override fun fetchModelInfo(repoId: String, revision: String?): HuggingFaceModelInfo {
        calls += "info:$repoId"
        return HuggingFaceModelInfo(
            repoId = repoId,
            revisionSha = FAKE_REVISION,
            released = FAKE_RELEASED,
            files = FAKE_MODEL_FILES.map { (path, content) ->
                HuggingFaceModelFile(path, content.length.toLong(), sha256(content))
            },
        )
    }

    override fun downloadFile(repoId: String, path: String, revision: String, destination: File) {
        calls += "file:$repoId/$path"
        val content = FAKE_MODEL_FILES.getValue(path)
        destination.writeText(if (corrupt) content.reversed() else content)
    }

    private fun sha256(content: String): String =
        MessageDigest.getInstance("SHA-256")
            .digest(content.toByteArray())
            .joinToString("") { "%02x".format(it) }
}

/**
 * Token classifier double: emits each `(surface, label)` pair it finds in the
 * input with Unicode scalar offsets, like an aggregated ONNX classifier.
 */
internal class FakeLabelClassifier(
    private vararg val mentions: Pair<String, String>,
) : OnnxTokenClassifier {
    var closed = false
        private set

    override suspend fun predict(text: String): List<TokenClassificationPrediction> =
        mentions.mapNotNull { (surface, label) ->
            val index = text.indexOf(surface).takeIf { it >= 0 } ?: return@mapNotNull null
            TokenClassificationPrediction(
                label = label,
                text = surface,
                confidence = 0.97f,
                start = UnicodeOffsetContract.utf16ToScalarOffset(text, index),
                end = UnicodeOffsetContract.utf16ToScalarOffset(text, index + surface.length),
            )
        }

    override fun close() {
        closed = true
    }
}
