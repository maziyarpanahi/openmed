#ifndef FLUTTER_PLUGIN_OPEN_MED_KIT_FLUTTER_PLUGIN_H_
#define FLUTTER_PLUGIN_OPEN_MED_KIT_FLUTTER_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

#if defined(FLUTTER_PLUGIN_IMPL)
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

namespace openmedkit_flutter {

class OpenMedKitFlutterPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

  OpenMedKitFlutterPlugin();
  ~OpenMedKitFlutterPlugin() override;

  OpenMedKitFlutterPlugin(const OpenMedKitFlutterPlugin&) = delete;
  OpenMedKitFlutterPlugin& operator=(const OpenMedKitFlutterPlugin&) = delete;

 private:
  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue>& method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace openmedkit_flutter

extern "C" FLUTTER_PLUGIN_EXPORT void
OpenMedKitFlutterPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#endif
