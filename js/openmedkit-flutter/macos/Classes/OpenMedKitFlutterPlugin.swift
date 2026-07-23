import Cocoa
import FlutterMacOS

public final class OpenMedKitFlutterPlugin: NSObject, FlutterPlugin {
    public static func register(with registrar: FlutterPluginRegistrar) {
        let channel = FlutterMethodChannel(
            name: "org.openmed.openmedkit_flutter/platform",
            binaryMessenger: registrar.messenger
        )
        let instance = OpenMedKitFlutterPlugin()
        registrar.addMethodCallDelegate(instance, channel: channel)
    }

    public func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
        guard call.method == "prepareModel" else {
            result(FlutterMethodNotImplemented)
            return
        }
        guard
            let arguments = call.arguments as? [String: Any],
            let path = arguments["modelDirectory"] as? String,
            !path.isEmpty,
            !path.contains("://")
        else {
            result(FlutterError(
                code: "invalid_model_directory",
                message: "A local model directory is required.",
                details: nil
            ))
            return
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory),
              isDirectory.boolValue else {
            result(FlutterError(
                code: "model_directory_missing",
                message: "The local model directory is unavailable.",
                details: nil
            ))
            return
        }
        result(URL(fileURLWithPath: path).standardizedFileURL.path)
    }
}
