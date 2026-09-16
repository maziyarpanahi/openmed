#ifndef FLUTTER_PLUGIN_OPEN_MED_KIT_FLUTTER_PLUGIN_H_
#define FLUTTER_PLUGIN_OPEN_MED_KIT_FLUTTER_PLUGIN_H_

#include <flutter_linux/flutter_linux.h>

G_BEGIN_DECLS

#if defined(FLUTTER_PLUGIN_IMPL)
#define FLUTTER_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define FLUTTER_PLUGIN_EXPORT
#endif

FLUTTER_PLUGIN_EXPORT void open_med_kit_flutter_plugin_register_with_registrar(
    FlPluginRegistrar *registrar);

G_END_DECLS

#endif
