#include "include/openmedkit_flutter/open_med_kit_flutter_plugin.h"

#include <glib.h>

namespace {

constexpr char kChannelName[] = "org.openmed.openmedkit_flutter/platform";

void HandleMethodCall(
    FlMethodChannel* channel,
    FlMethodCall* method_call,
    gpointer user_data) {
  (void)channel;
  (void)user_data;
  const gchar* method = fl_method_call_get_name(method_call);
  if (g_strcmp0(method, "prepareModel") != 0) {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(
        fl_method_not_implemented_response_new());
    fl_method_call_respond(method_call, response, nullptr);
    return;
  }

  FlValue* arguments = fl_method_call_get_args(method_call);
  FlValue* value = arguments == nullptr
      ? nullptr
      : fl_value_lookup_string(arguments, "modelDirectory");
  const gchar* path = value != nullptr && fl_value_get_type(value) == FL_VALUE_TYPE_STRING
      ? fl_value_get_string(value)
      : nullptr;
  if (path == nullptr || path[0] == '\0' || g_strstr_len(path, -1, "://") != nullptr) {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(
        fl_method_error_response_new(
            "invalid_model_directory",
            "A local model directory is required.",
            nullptr));
    fl_method_call_respond(method_call, response, nullptr);
    return;
  }
  if (!g_path_is_absolute(path) || !g_file_test(path, G_FILE_TEST_IS_DIR)) {
    g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(
        fl_method_error_response_new(
            "model_directory_missing",
            "The local model directory is unavailable.",
            nullptr));
    fl_method_call_respond(method_call, response, nullptr);
    return;
  }

  g_autofree gchar* canonical = g_canonicalize_filename(path, nullptr);
  g_autoptr(FlValue) result = fl_value_new_string(canonical);
  g_autoptr(FlMethodResponse) response = FL_METHOD_RESPONSE(
      fl_method_success_response_new(result));
  fl_method_call_respond(method_call, response, nullptr);
}

}  // namespace

void open_med_kit_flutter_plugin_register_with_registrar(
    FlPluginRegistrar* registrar) {
  FlBinaryMessenger* messenger = fl_plugin_registrar_get_messenger(registrar);
  g_autoptr(FlStandardMethodCodec) codec = fl_standard_method_codec_new();
  g_autoptr(FlMethodChannel) channel = fl_method_channel_new(
      messenger,
      kChannelName,
      FL_METHOD_CODEC(codec));
  fl_method_channel_set_method_call_handler(
      channel,
      HandleMethodCall,
      nullptr,
      nullptr);
}
