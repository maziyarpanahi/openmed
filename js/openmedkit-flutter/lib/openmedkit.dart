/// Local-first OpenMed token classification for Flutter.
///
/// Models load only through an [OpenMedVerifiedBundle]: a signed portable
/// bundle whose manifest signature and every file digest were checked before
/// any native session exists. Platforms without `dart:io` and `dart:ffi`, such
/// as the web, get the same API with explicit `platform_unsupported` results.
library openmedkit_flutter;

export 'src/errors.dart';
export 'src/kit_unsupported.dart' if (dart.library.io) 'src/kit_native.dart';
export 'src/portable_bundle_types.dart';
export 'src/types.dart';
