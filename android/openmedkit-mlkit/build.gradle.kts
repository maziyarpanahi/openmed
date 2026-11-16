// Optional ML Kit OCR adapter for OpenMedKit Android.
//
// This module is the only OpenMedKit artifact that depends on Google ML Kit.
// ML Kit is distributed under Google's ML Kit terms, auto-initializes through
// a manifest provider, and pulls in the datatransport usage-metrics stack with
// the INTERNET and ACCESS_NETWORK_STATE permissions. Keep it out of the core
// :openmedkit module.

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    `maven-publish`
    signing
}

val publicationGroup = providers.gradleProperty("openmedAndroidGroup")
    .orElse("org.openmed")
    .get()
val coreArtifact = providers.gradleProperty("openmedAndroidArtifact")
    .orElse("openmedkit")
    .get()
val publicationArtifact = providers.gradleProperty("openmedAndroidMlKitArtifact")
    .orElse("$coreArtifact-mlkit")
    .get()

group = publicationGroup
version = providers.gradleProperty("openmedAndroidVersion")
    .orElse(providers.environmentVariable("OPENMED_ANDROID_VERSION"))
    .orElse("0.0.0-SNAPSHOT")
    .get()

val centralStagingRepository = layout.buildDirectory.dir("central-staging")
val centralPortalBundle = layout.buildDirectory.dir("central-portal")
val releaseVersionPattern = Regex("""\d+\.\d+\.\d+([.-][A-Za-z0-9][A-Za-z0-9.-]*)?""")

fun isSnapshotVersion(): Boolean {
    return version.toString().contains("SNAPSHOT", ignoreCase = true)
}

android {
    namespace = "org.openmed.openmedkit.mlkit"
    compileSdk = 33

    defaultConfig {
        minSdk = 26
        targetSdk = 33
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
            withJavadocJar()
        }
    }
}

kotlin {
    jvmToolchain(11)
}

// Negative control: this module is where the ML Kit stack is expected to appear.
extra["openmedkitDependencyBoundary"] = "optional-vendor"
apply(from = rootProject.file("gradle/dependency-boundary.gradle.kts"))

tasks.matching { it.name.startsWith("test") }.configureEach {
    dependsOn("verifyDependencyBoundary")
}

val verifyReleasePublicationVersion by tasks.registering {
    group = "publishing"
    description = "Fails when the ML Kit adapter publication version is a SNAPSHOT or malformed."

    doLast {
        val releaseVersion = version.toString()
        if (isSnapshotVersion()) {
            throw GradleException("OpenMedKit ML Kit release version cannot be a SNAPSHOT.")
        }
        if (!releaseVersionPattern.matches(releaseVersion)) {
            throw GradleException(
                "OpenMedKit ML Kit release version must look like 1.2.3 or 1.2.3-rc.1.",
            )
        }
    }
}

val cleanCentralStagingRepository by tasks.registering(Delete::class) {
    group = "publishing"
    delete(centralStagingRepository)
}

val bundleCentralPortalPublication by tasks.registering(Zip::class) {
    group = "publishing"
    description = "Bundles the signed ML Kit adapter publication in Central Portal layout."
    dependsOn("publishReleasePublicationToCentralStagingRepository")

    archiveFileName.set("$publicationArtifact-${project.version}.zip")
    destinationDirectory.set(centralPortalBundle)
    from(centralStagingRepository)
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])

                groupId = publicationGroup
                artifactId = publicationArtifact
                version = project.version.toString()

                pom {
                    name.set("OpenMedKit Android ML Kit OCR")
                    description.set(
                        "Optional ML Kit Text Recognition adapter for OpenMedKit Android. " +
                            "Depends on Google ML Kit, which is distributed under Google's " +
                            "ML Kit terms and adds Google's usage-metrics transport.",
                    )
                    url.set("https://github.com/maziyarpanahi/openmed")

                    licenses {
                        license {
                            name.set("Apache License, Version 2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                            distribution.set("repo")
                        }
                    }
                    developers {
                        developer {
                            id.set("maziyarpanahi")
                            name.set("Maziyar Panahi")
                            url.set("https://github.com/maziyarpanahi")
                        }
                    }
                    scm {
                        connection.set("scm:git:https://github.com/maziyarpanahi/openmed.git")
                        developerConnection.set(
                            "scm:git:ssh://git@github.com/maziyarpanahi/openmed.git",
                        )
                        url.set("https://github.com/maziyarpanahi/openmed")
                    }
                }
            }
        }

        repositories {
            maven {
                name = "centralStaging"
                url = centralStagingRepository.get().asFile.toURI()
            }
        }
    }

    tasks.named("publishReleasePublicationToCentralStagingRepository") {
        dependsOn(cleanCentralStagingRepository)
        dependsOn(verifyReleasePublicationVersion)
    }

    signing {
        val signingKey = providers.gradleProperty("signingInMemoryKey")
            .orElse(providers.environmentVariable("ANDROID_SIGNING_KEY"))
        val signingPassword = providers.gradleProperty("signingInMemoryKeyPassword")
            .orElse(providers.environmentVariable("ANDROID_SIGNING_KEY_PASSWORD"))

        setRequired {
            gradle.taskGraph.allTasks.any {
                it.project == project && (
                    it.name == "publishReleasePublicationToCentralStagingRepository" ||
                        it.name == "bundleCentralPortalPublication"
                    )
            } && !isSnapshotVersion()
        }

        if (signingKey.isPresent && signingPassword.isPresent) {
            useInMemoryPgpKeys(signingKey.get(), signingPassword.get())
        }
        sign(publishing.publications["release"])
    }
}

dependencies {
    api(project(":openmedkit"))
    api(libs.mlkit.text.recognition)
    implementation(libs.kotlinx.coroutines.core)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
}
