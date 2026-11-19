// Installed-package consumer for the OpenMedKit Android AAR.
//
// scripts/release/verify_installed_consumers.py publishes the library with
// publishReleasePublicationToMavenLocal into an isolated Maven repository and
// passes -Dmaven.repo.local plus the toolchain versions from
// android/gradle/libs.versions.toml. This build never includes the library
// project; it resolves org.openmed:openmedkit like any downstream app.
pluginManagement {
    val consumerAgpVersion: String by settings
    val consumerKotlinVersion: String by settings

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
    resolutionStrategy {
        eachPlugin {
            when (requested.id.id) {
                "com.android.application" ->
                    useModule("com.android.tools.build:gradle:$consumerAgpVersion")
                "org.jetbrains.kotlin.android" ->
                    useModule("org.jetbrains.kotlin:kotlin-gradle-plugin:$consumerKotlinVersion")
            }
        }
    }
    plugins {
        id("com.android.application") version consumerAgpVersion
        id("org.jetbrains.kotlin.android") version consumerKotlinVersion
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        mavenLocal {
            content {
                includeGroup("org.openmed")
            }
        }
        google()
        mavenCentral()
    }
}

rootProject.name = "openmedkit-installed-consumer"
include(":app")
