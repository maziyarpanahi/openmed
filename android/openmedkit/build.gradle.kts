plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

val repoRoot = rootProject.projectDir.parentFile
val generatedCatalogAssetsDir = layout.buildDirectory.dir("generated/assets/openmedCatalog")
val generatedCatalogAsset = generatedCatalogAssetsDir.map {
    it.file("openmed_model_catalog.jsonl")
}
val pythonExecutable = providers.gradleProperty("openmedPython")
    .orElse(providers.environmentVariable("PYTHON"))
    .orElse("python3")

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
            resources.srcDir("src/main/assets")
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

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
}
