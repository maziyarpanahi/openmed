#include "include/openmedkit_flutter/open_med_kit_flutter_plugin.h"

#include <flutter/standard_method_codec.h>

#include <filesystem>
#include <string>

namespace openmedkit_flutter {

namespace {

constexpr char kChannelName[] = "org.openmed.openmedkit_flutter/platform";

}  // namespace

OpenMedKitFlutterPlugin::OpenMedKitFlutterPlugin() = default;

OpenMedKitFlutterPlugin::~OpenMedKitFlutterPlugin() = default;

void OpenMedKitFlutterPlugin::RegisterWithRegistrar(
    flutter::PluginRegistrarWindows* registrar) {
  auto channel = std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
      registrar->messenger(),
      kChannelName,
      &flutter::StandardMethodCodec::GetInstance());
  auto plugin = std::make_unique<OpenMedKitFlutterPlugin>();
  channel->SetMethodCallHandler(
      [plugin_pointer = plugin.get()](const auto& call, auto result) {
        plugin_pointer->HandleMethodCall(call, std::move(result));
      });
  registrar->AddPlugin(std::move(plugin));
}

void OpenMedKitFlutterPlugin::HandleMethodCall(
    const flutter::MethodCall<flutter::EncodableValue>& method_call,
    std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result) {
  if (method_call.method_name() != "prepareModel") {
    result->NotImplemented();
    return;
  }
  const auto* arguments = std::get_if<flutter::EncodableMap>(
      method_call.arguments());
  if (arguments == nullptr) {
    result->Error(
        "invalid_model_directory",
        "A local model directory is required.");
    return;
  }
  const auto iterator = arguments->find(
      flutter::EncodableValue("modelDirectory"));
  const auto* path_value = iterator == arguments->end()
      ? nullptr
      : std::get_if<std::string>(&iterator->second);
  if (path_value == nullptr || path_value->empty() ||
      path_value->find("://") != std::string::npos) {
    result->Error(
        "invalid_model_directory",
        "A local model directory is required.");
    return;
  }
  const std::filesystem::path path = std::filesystem::absolute(*path_value);
  if (!std::filesystem::is_directory(path)) {
    result->Error(
        "model_directory_missing",
        "The local model directory is unavailable.");
    return;
  }
  result->Success(flutter::EncodableValue(path.u8string()));
}

}  // namespace openmedkit_flutter

void OpenMedKitFlutterPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  openmedkit_flutter::OpenMedKitFlutterPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
