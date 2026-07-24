# OpenMedKit Android

`android/openmedkit/` is the Kotlin Android library module for OpenMedKit.
It is a Gradle `com.android.library` module under the top-level `android/`
build and uses the Android namespace `org.openmed.openmedkit`.

Public applications consume the immutable `v1.9.1` release through JitPack. See the
[Android installation guide](../README.md#install-openmedkit-190) for the
repository and dependency declarations.

## Layout

- `src/main/kotlin/com/openmed/openmedkit/` contains the current public Kotlin API.
- `src/main/AndroidManifest.xml` is intentionally minimal for the library.
- `src/test/kotlin/com/openmed/openmedkit/` contains local JVM unit tests
  running with JUnit and Robolectric.

## Platform Baseline

The module sets `minSdk` to 26. Android 8.0 is a practical baseline for an
on-device clinical utility library because it keeps support broad while
allowing later work to rely on modern platform security, storage, and runtime
behavior. The module currently compiles against Android API 33 for compatibility
with the repository's Java 11 local development baseline.

## Privacy And Safety

OpenMedKit Android is intended for local-first, on-device processing. Library
code must not send protected health information to network services by default,
must not require telemetry, and must not write raw PHI to logs, caches,
temporary files, analytics events, or audit artifacts.

This module is on-device only by design. It is not a medical device and does not
provide diagnosis, treatment decisions, or emergency clinical guidance. Any
future clinical workflow integration must keep human review and appropriate
regulatory evaluation outside this scaffold.

## ICU Boundary Segmentation

Production word and grapheme boundaries use `android.icu.text.BreakIterator`
from the Android platform, so no dictionary or additional runtime is bundled in
the AAR. Local JVM parity tests use `com.ibm.icu:icu4j:78.3` as a test-only
dependency.

ICU4J is distributed under the permissive
[Unicode Data Files and Software License](https://www.unicode.org/license.txt)
(`Unicode-3.0`, also known as the ICU/Unicode License). The dependency is used
only to verify that JVM results match Android platform ICU; it is not a runtime
dependency and does not process text off device.
