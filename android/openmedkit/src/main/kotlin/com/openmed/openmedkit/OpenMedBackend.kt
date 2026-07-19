package com.openmed.openmedkit

import com.openmed.openmedkit.onnx.AcceleratorConfig
import java.io.File

/**
 * Offline backend configuration for constructing OpenMedKit from local assets.
 *
 * The Android facade is intentionally local-first: callers provide an on-device
 * model directory, tokenizer assets, and an optional id-to-label map. No network
 * access is performed by this configuration type.
 */
data class OpenMedBackend(
    val modelDirectory: File,
    val modelFile: File = File(modelDirectory, "model.onnx"),
    val tokenizerJson: File = File(modelDirectory, "tokenizer.json"),
    val tokenizerConfig: File? = File(modelDirectory, "tokenizer_config.json"),
    val id2LabelFile: File = File(modelDirectory, "id2label.json"),
    val id2Label: Map<Int, String> = emptyMap(),
    val acceleratorConfig: AcceleratorConfig = AcceleratorConfig(),
) {
    init {
        require(modelDirectory.path.isNotBlank()) {
            "modelDirectory must not be blank"
        }
    }
}
