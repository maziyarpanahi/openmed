# OpenMedKit Android

This directory contains the Gradle build for the `:openmedkit` Android library.

## Install OpenMedKit 1.9.0

OpenMedKit Android is published from immutable OpenMed release tags through JitPack.
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

Then add the `v1.9.0` coordinate:

```kotlin
dependencies {
    implementation("com.github.maziyarpanahi:openmed:v1.9.0")
}
```

JitPack resolves the immutable `v1.9.0` tag and publishes the `openmedkit`
Android release component as an AAR. Public consumers do not need GitHub
credentials. Use a commit coordinate only when intentionally testing an
unreleased build.

## Build Locally

Install JDK 11 and Android SDK Platform 33, then run:

```bash
cd android
./gradlew test
```

Dependency resolution is limited to Google Maven, Maven Central, and the scoped
OpenMed group on JitPack. Dependency and plugin versions are declared in
`gradle/libs.versions.toml`; avoid hardcoding library versions in module build
files.

The library currently targets SDK 33 and sets `minSdk` to 26. Android 8.0 is the
baseline for on-device inference work because it preserves broad device support
while keeping runtime, storage, and execution APIs modern enough for the planned
local-first pipeline.

## Optional Maven Central Publishing

JitPack is the public installation path documented above. The separate, optional
Maven Central path runs from `.github/workflows/android-publish.yml` on `v*` tags
or manual dispatch. Tag releases with Android artifact changes always assemble
the library and run its unit tests. They build and upload a signed Central Portal
bundle only when all signing and Sonatype credentials are configured; otherwise
the validated JitPack release remains the supported public artifact. Manual
dispatch always requires the credentials and runs the full publication path after
the explicit upload confirmation.

Repository secrets required only for Central Portal publication:

- `ANDROID_SIGNING_KEY`: ASCII-armored GPG private key for artifact signing.
- `ANDROID_SIGNING_KEY_PASSWORD`: passphrase for the signing key.
- `SONATYPE_USERNAME`: Central Portal user-token username.
- `SONATYPE_PASSWORD`: Central Portal user-token password.
