import java.util.zip.ZipFile

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

fun consumerProperty(name: String): String = providers.gradleProperty(name).get()

val openmedkitCoordinates =
    "${consumerProperty("openmedkitGroup")}:openmedkit:${consumerProperty("openmedkitVersion")}"
val consumerContract = rootProject.layout.projectDirectory.file("consumer_contract.json")
val debugApk = layout.buildDirectory.file("outputs/apk/debug/app-debug.apk")

// Entries the built APK must receive from the installed AAR and its declared
// runtime dependencies: packaged assets, classloader policy resources and the
// native libraries the ONNX and tokenizer runtimes load on device.
val requiredApkEntries = listOf(
    "assets/openmed_model_catalog.jsonl",
    "assets/policies/hipaa_safe_harbor.json",
    "policies/hipaa_safe_harbor.json",
    "lib/arm64-v8a/libonnxruntime.so",
    "lib/arm64-v8a/libonnxruntime4j_jni.so",
    "lib/arm64-v8a/libdjl_tokenizer.so",
)

android {
    namespace = "org.openmed.installedconsumer"
    compileSdk = consumerProperty("consumerCompileSdk").toInt()

    defaultConfig {
        applicationId = "org.openmed.installedconsumer"
        minSdk = 26
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    testOptions {
        // Robolectric reads the merged assets, including the installed AAR's.
        unitTests.isIncludeAndroidResources = true
    }
}

// DJL 0.33 pulls Java-17-era logging and annotation jars that the D8 shipped
// with AGP 7.4 cannot dex. Downstream apps on that toolchain need the same
// API-compatible pins the in-repo Maple demo uses; the migration guide
// documents them.
val d8CompatiblePins = mapOf(
    ("org.slf4j" to "slf4j-api") to consumerProperty("consumerSlf4jVersion"),
    ("com.google.errorprone" to "error_prone_annotations") to
        consumerProperty("consumerErrorproneVersion"),
)
configurations.configureEach {
    resolutionStrategy.eachDependency {
        d8CompatiblePins[requested.group to requested.name]?.let { useVersion(it) }
    }
}

dependencies {
    implementation(openmedkitCoordinates)
    // OpenMedKit's facade exposes suspend functions; the AAR declares
    // kotlinx-coroutines as a runtime dependency, so callers add it to compile.
    implementation(
        "org.jetbrains.kotlinx:kotlinx-coroutines-core:${consumerProperty("consumerCoroutinesVersion")}",
    )

    testImplementation("junit:junit:${consumerProperty("consumerJunitVersion")}")
    testImplementation(
        "org.jetbrains.kotlin:kotlin-test-junit:${consumerProperty("consumerKotlinVersion")}",
    )
    testImplementation(
        "org.jetbrains.kotlinx:kotlinx-serialization-json:" +
            consumerProperty("consumerSerializationVersion"),
    )
    testImplementation("org.robolectric:robolectric:${consumerProperty("consumerRobolectricVersion")}")
}

tasks.withType<Test>().configureEach {
    inputs.file(consumerContract)
    systemProperty("openmed.consumer.contract", consumerContract.asFile.absolutePath)
}

val verifyInstalledApk by tasks.registering {
    description = "Fails when the consumer APK lacks packaged OpenMedKit resources."
    dependsOn("assembleDebug")
    inputs.file(debugApk)

    doLast {
        val apk = debugApk.get().asFile
        if (!apk.isFile) {
            throw GradleException("Consumer APK was not produced.")
        }
        val missing = ZipFile(apk).use { archive ->
            requiredApkEntries.filter { archive.getEntry(it) == null }
        }
        if (missing.isNotEmpty()) {
            throw GradleException("Consumer APK is missing: ${missing.joinToString()}")
        }
        logger.lifecycle(
            "installed-consumer: APK contains ${requiredApkEntries.size} required entries",
        )
    }
}

tasks.register("verifyInstalledConsumer") {
    group = "verification"
    description = "Runs the installed-package unit checks and APK packaging gate."
    dependsOn("testDebugUnitTest", verifyInstalledApk)
}
