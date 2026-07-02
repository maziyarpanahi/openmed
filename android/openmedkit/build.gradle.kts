plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    `maven-publish`
    signing
}

group = "org.openmed"
version = providers.gradleProperty("openmedAndroidVersion")
    .orElse(providers.environmentVariable("OPENMED_ANDROID_VERSION"))
    .orElse("0.0.0-SNAPSHOT")
    .get()

val repoRoot = rootProject.projectDir.parentFile
val generatedCatalogAssetsDir = layout.buildDirectory.dir("generated/assets/openmedCatalog")
val generatedCatalogAsset = generatedCatalogAssetsDir.map {
    it.file("openmed_model_catalog.jsonl")
}
val centralStagingRepository = layout.buildDirectory.dir("central-staging")
val centralPortalBundle = layout.buildDirectory.dir("central-portal")
val pythonExecutable = providers.gradleProperty("openmedPython")
    .orElse(providers.environmentVariable("PYTHON"))
    .orElse("python3")
val releaseVersionPattern = Regex("""\d+\.\d+\.\d+([.-][A-Za-z0-9][A-Za-z0-9.-]*)?""")

fun isSnapshotVersion(): Boolean {
    return version.toString().contains("SNAPSHOT", ignoreCase = true)
}

val generateAndroidModelCatalog by tasks.registering(Exec::class) {
    val manifestFile = repoRoot.resolve("models.jsonl")
    val scriptFile = repoRoot.resolve("scripts/android/build_android_catalog.py")

    inputs.file(manifestFile)
    inputs.file(scriptFile)
    outputs.file(generatedCatalogAsset)

    doFirst {
        generatedCatalogAsset.get().asFile.parentFile.mkdirs()
    }

    commandLine(
        pythonExecutable.get(),
        scriptFile.absolutePath,
        "--manifest",
        manifestFile.absolutePath,
        "--output",
        generatedCatalogAsset.get().asFile.absolutePath,
    )
}

android {
    namespace = "org.openmed.openmedkit"
    compileSdk = 33

    defaultConfig {
        // Android 8.0+ keeps on-device inference support aligned with current
        // runtime and filesystem APIs without forcing recent-only devices.
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

    sourceSets {
        getByName("main") {
            assets.srcDir(generatedCatalogAssetsDir)
        }
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

tasks.matching { it.name.endsWith("Assets") }.configureEach {
    dependsOn(generateAndroidModelCatalog)
}

tasks.matching { it.name.startsWith("test") }.configureEach {
    dependsOn(generateAndroidModelCatalog)
}

val verifyReleasePublicationVersion by tasks.registering {
    group = "publishing"
    description = "Fails when the OpenMedKit publication version is a SNAPSHOT or malformed."

    doLast {
        val releaseVersion = version.toString()
        if (isSnapshotVersion()) {
            throw GradleException("OpenMedKit release publication version cannot be a SNAPSHOT.")
        }
        if (!releaseVersionPattern.matches(releaseVersion)) {
            throw GradleException(
                "OpenMedKit release publication version must look like 1.2.3 or 1.2.3-rc.1.",
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
    description = "Bundles the signed Maven publication in Central Portal upload layout."
    dependsOn("publishReleasePublicationToCentralStagingRepository")

    archiveFileName.set("openmedkit-${project.version}.zip")
    destinationDirectory.set(centralPortalBundle)
    from(centralStagingRepository)
}

afterEvaluate {
    publishing {
        publications {
            create<MavenPublication>("release") {
                from(components["release"])

                groupId = "org.openmed"
                artifactId = "openmedkit"
                version = project.version.toString()

                pom {
                    name.set("OpenMedKit Android")
                    description.set(
                        "Local-first Android library for OpenMed clinical AI workflows.",
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
                it.name.startsWith("publish") || it.name == "bundleCentralPortalPublication"
            } && !isSnapshotVersion()
        }

        if (signingKey.isPresent && signingPassword.isPresent) {
            useInMemoryPgpKeys(signingKey.get(), signingPassword.get())
        }
        sign(publishing.publications["release"])
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.robolectric)
}
