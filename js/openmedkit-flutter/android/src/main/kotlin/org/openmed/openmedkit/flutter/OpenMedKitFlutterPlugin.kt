package org.openmed.openmedkit.flutter

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import java.io.File

/** Registers the offline model-path channel and bundled FFI library. */
class OpenMedKitFlutterPlugin : FlutterPlugin, MethodChannel.MethodCallHandler {
    private lateinit var channel: MethodChannel

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        System.loadLibrary("openmed_ffi")
        channel = MethodChannel(
            binding.binaryMessenger,
            "org.openmed.openmedkit_flutter/platform",
        )
        channel.setMethodCallHandler(this)
    }

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        if (call.method != "prepareModel") {
            result.notImplemented()
            return
        }
        val modelDirectory = call.argument<String>("modelDirectory")
        if (modelDirectory.isNullOrBlank() || modelDirectory.contains("://")) {
            result.error("invalid_model_directory", "A local model directory is required.", null)
            return
        }
        val directory = File(modelDirectory)
        if (!directory.isDirectory) {
            result.error("model_directory_missing", "The local model directory is unavailable.", null)
            return
        }
        result.success(directory.absolutePath)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }
}
