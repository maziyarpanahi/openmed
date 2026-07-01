# OpenMedKit Android

This directory contains the Gradle build for the `:openmedkit` Android library.
The project is intentionally small until the Android API surface lands in later
tasks.

## Build Locally

Install JDK 11 and Android SDK Platform 33, then run:

```bash
cd android
./gradlew test
```

Dependency resolution is limited to Google Maven and Maven Central. Dependency
and plugin versions are declared in `gradle/libs.versions.toml`; avoid hardcoding
library versions in module build files.

The library currently targets SDK 33 and sets `minSdk` to 26. Android 8.0 is the
baseline for on-device inference work because it preserves broad device support
while keeping runtime, storage, and execution APIs modern enough for the planned
local-first pipeline.
