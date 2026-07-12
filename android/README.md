# OpenMedKit Android

This directory contains the Gradle build for the `:openmedkit` Android library.

## Install From A GitHub Tag

OpenMedKit Android is built directly from OpenMed GitHub tags through JitPack.
Add the repository in the consumer application's `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
        maven {
            url = uri("https://jitpack.io")
            content { includeGroup("com.github.maziyarpanahi") }
        }
    }
}
```

Then add the tagged OpenMed release:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v1.8.2")
}
```

JitPack checks out that immutable Git tag and publishes the `openmedkit` Android
release component as an AAR. Public consumers do not need GitHub credentials.

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

## Maven Central Publishing

The optional Maven Central path publishes from `.github/workflows/android-publish.yml`
on `v*` tags or manual dispatch. Tag releases skip Central publication when the
Android artifact inputs have not changed since the previous release. When
publication is needed, the workflow assembles the library, runs its unit tests,
builds a signed Central Portal bundle from the Gradle `release` Maven publication,
and uploads it to Sonatype. Manual dispatch always runs the full validated
publication path after the explicit upload confirmation.

Required repository secrets:

- `ANDROID_SIGNING_KEY`: ASCII-armored GPG private key for artifact signing.
- `ANDROID_SIGNING_KEY_PASSWORD`: passphrase for the signing key.
- `SONATYPE_USERNAME`: Central Portal user-token username.
- `SONATYPE_PASSWORD`: Central Portal user-token password.
