import com.android.build.api.artifact.SingleArtifact
import com.android.build.api.variant.ApplicationVariant
import javax.inject.Inject
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

// A minimal host app that consumes the OpenMedKit release AAR with R8 full mode
// and `isMinifyEnabled = true`, the way integrators ship it. Three flavors:
//   openmedkit       the AAR under test, unchanged
//   baseline         the same app shell without OpenMedKit (size reference)
//   missingKeepRule  the AAR with its ONNX Runtime keep rule removed (negative
//                    control that must fail the R8 gate)
// `verifyR8Consumer` runs every gate; see android/README.md.

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// AGP 8 runs R8 in full mode unless android.enableR8.fullMode=false. This app
// verifies OpenMedKit under full mode, so refuse to configure without it.
if (!providers.gradleProperty("android.enableR8.fullMode").getOrElse("true").toBoolean()) {
    throw GradleException("consumer-r8-smoke requires R8 full mode.")
}

val repoRoot = rootProject.projectDir.parentFile
val pythonExecutable = providers.gradleProperty("openmedPython")
    .orElse(providers.environmentVariable("PYTHON"))
    .orElse("python3")
val gateScript = repoRoot.resolve("scripts/android/check_r8_consumer.py")
val modelScript = repoRoot.resolve("scripts/android/generate_r8_smoke_model.py")
val budgetFile = rootProject.layout.projectDirectory.file("gradle/budgets.properties")
val consumerRules = rootProject.layout.projectDirectory.file("openmedkit/consumer-rules.pro")
val dontwarnAllowlist = layout.projectDirectory.file("r8-dontwarn-allowlist.txt")
val reportsDirectory = layout.buildDirectory.dir("reports/r8-consumer")

val splitAbis = listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")

// Desktop JNI binaries that dependencies ship as Java resources. Android never
// loads them (DJL loads libdjl_tokenizer.so from lib/<abi>/), but without these
// excludes every APK carries them. Host apps should use the same excludes.
val desktopNativeResourceExcludes = listOf(
    // DJL tokenizers: libtokenizers for Linux, macOS and Windows.
    "native/lib/**",
    // JNA, pulled in by DJL's api module: desktop libjnidispatch builds.
    "com/sun/jna/*/*jnidispatch*",
)

// JNI libraries every ABI split must carry for OpenMedKit inference.
val requiredNativeLibraries = listOf("onnxruntime", "onnxruntime4j_jni", "djl_tokenizer")
// Packages whose classes bundled JNI libraries look up by name.
val jniPackages = listOf("ai/onnxruntime", "ai/djl", "com/openmed")

// The OpenMedKit release AAR under test: built from this checkout by default, or
// any AAR passed with -PopenmedkitAar=/absolute/path/to/openmedkit.aar.
val externalAar = providers.gradleProperty("openmedkitAar")
val openmedkitAar: Provider<RegularFile> = externalAar
    .map { rootProject.layout.projectDirectory.file(it) }
    .orElse(
        rootProject.layout.projectDirectory.file(
            "openmedkit/build/outputs/aar/openmedkit-release.aar",
        ),
    )
val openmedkitAarFiles = files(openmedkitAar).apply {
    if (!externalAar.isPresent) {
        builtBy(":openmedkit:bundleReleaseAar")
    }
}

// Negative control: the same AAR without the ONNX Runtime keep rule.
val negativeControlRule = "-keep class ai.onnxruntime.** { *; }"
val negativeControlExpectation = "jni_class_not_kept:ai.onnxruntime."
val missingKeepRuleAar = layout.buildDirectory.file(
    "intermediates/r8-negative-control/openmedkit-missing-keep-rule.aar",
)
val createMissingKeepRuleAar by tasks.registering(Exec::class) {
    group = "verification"
    description = "Copies the OpenMedKit AAR without its ONNX Runtime keep rule."
    if (!externalAar.isPresent) {
        dependsOn(":openmedkit:bundleReleaseAar")
    }
    inputs.file(openmedkitAar)
    inputs.file(gateScript)
    outputs.file(missingKeepRuleAar)
    executable(pythonExecutable.get())
    args(gateScript.absolutePath, "drop-keep-rule", "--rule", negativeControlRule)
    argumentProviders.add(
        CommandLineArgumentProvider {
            listOf(
                "--aar",
                openmedkitAar.get().asFile.absolutePath,
                "--output",
                missingKeepRuleAar.get().asFile.absolutePath,
            )
        },
    )
}

/** Writes the synthetic model, tokenizer and smoke case into generated assets. */
abstract class GenerateR8SmokeModelTask : DefaultTask() {
    @get:InputFile
    abstract val script: RegularFileProperty

    @get:Input
    abstract val python: Property<String>

    @get:OutputDirectory
    abstract val outputDirectory: DirectoryProperty

    @get:Inject
    abstract val execOperations: ExecOperations

    @TaskAction
    fun generate() {
        val output = outputDirectory.get().asFile
        output.deleteRecursively()
        execOperations.exec {
            commandLine(
                python.get(),
                script.get().asFile.absolutePath,
                "--output-dir",
                output.resolve("r8-smoke-model").absolutePath,
            )
        }
    }
}

android {
    namespace = "org.openmed.r8smoke"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.openmed.r8smoke"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
    }

    flavorDimensions += "openmedkit"
    productFlavors {
        create("openmedkit") {
            dimension = "openmedkit"
        }
        create("baseline") {
            dimension = "openmedkit"
            applicationIdSuffix = ".baseline"
        }
        create("missingKeepRule") {
            dimension = "openmedkit"
            applicationIdSuffix = ".negative"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
            // Debug-signed so an emulator lane can install the release APKs.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    splits {
        abi {
            isEnable = true
            reset()
            include(*splitAbis.toTypedArray())
            isUniversalApk = false
        }
    }

    packaging {
        resources {
            excludes += desktopNativeResourceExcludes
        }
    }

    sourceSets {
        getByName("missingKeepRule") {
            kotlin.srcDir("src/openmedkit/kotlin")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    lint {
        // A verification fixture, not a shipped app: skip lint-vital in
        // assembleRelease so the gate stays focused on R8 and packaging.
        checkReleaseBuilds = false
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

// OpenMedKit's POM runtime dependencies. A file dependency carries no POM, so
// they are repeated here; tests/unit/android/test_r8_consumer_gate.py keeps the
// list identical to the implementation dependencies in openmedkit/build.gradle.kts.
val openmedkitRuntimeDependencies = listOf(
    libs.kotlinx.coroutines.core,
    libs.kotlinx.serialization.json,
    libs.mlkit.text.recognition,
    libs.onnxruntime.android,
    libs.djl.tokenizers,
    libs.djl.tokenizer.native.android,
)

dependencies {
    "openmedkitImplementation"(openmedkitAarFiles)
    "missingKeepRuleImplementation"(files(missingKeepRuleAar).builtBy(createMissingKeepRuleAar))
    for (configuration in listOf("openmedkitImplementation", "missingKeepRuleImplementation")) {
        openmedkitRuntimeDependencies.forEach { addProvider(configuration, it) }
    }

    testImplementation(libs.junit)
}

val openmedkitApks = objects.directoryProperty()
val baselineApks = objects.directoryProperty()

fun registerR8Gate(
    variant: ApplicationVariant,
    taskName: String,
    expectations: List<String>,
) = tasks.register<Exec>(taskName) {
    group = "verification"
    description = if (expectations.isEmpty()) {
        "Verifies the R8-minified ${variant.name} APKs against the OpenMedKit gate."
    } else {
        "Negative control: the ${variant.name} APKs must fail the R8 gate."
    }
    val apkDirectory = variant.artifacts.get(SingleArtifact.APK)
    val mappingFile = variant.artifacts.get(SingleArtifact.OBFUSCATION_MAPPING_FILE)
    val mergedManifest = variant.artifacts.get(SingleArtifact.MERGED_MANIFEST)
    val report = reportsDirectory.map { it.file("${variant.name}-r8-gate.properties") }
    inputs.dir(apkDirectory)
    inputs.file(mappingFile)
    inputs.file(mergedManifest)
    inputs.files(gateScript, consumerRules, dontwarnAllowlist)
    outputs.file(report)
    executable(pythonExecutable.get())
    args(gateScript.absolutePath, "verify")
    jniPackages.forEach { args("--jni-package", it) }
    requiredNativeLibraries.forEach { args("--require-native-library", it) }
    expectations.forEach { args("--expect-failure", it) }
    argumentProviders.add(
        CommandLineArgumentProvider {
            listOf(
                "--apk-dir", apkDirectory.get().asFile.absolutePath,
                "--mapping", mappingFile.get().asFile.absolutePath,
                "--merged-manifest", mergedManifest.get().asFile.absolutePath,
                "--consumer-rules", consumerRules.asFile.absolutePath,
                "--dontwarn-allowlist", dontwarnAllowlist.asFile.absolutePath,
                "--report", report.get().asFile.absolutePath,
            )
        },
    )
}

androidComponents {
    // Only minified release builds matter here.
    beforeVariants(selector().withBuildType("debug")) { it.enable = false }
    onVariants { variant ->
        val generateModel = tasks.register<GenerateR8SmokeModelTask>(
            "generate${variant.name.replaceFirstChar(Char::uppercaseChar)}R8SmokeModel",
        ) {
            script.set(modelScript)
            python.set(pythonExecutable)
        }
        variant.sources.assets?.addGeneratedSourceDirectory(
            generateModel,
            GenerateR8SmokeModelTask::outputDirectory,
        )
        when (variant.flavorName) {
            "openmedkit" -> {
                openmedkitApks.set(variant.artifacts.get(SingleArtifact.APK))
                registerR8Gate(variant, "verifyOpenmedkitR8Gate", emptyList())
            }
            "baseline" -> baselineApks.set(variant.artifacts.get(SingleArtifact.APK))
            "missingKeepRule" -> registerR8Gate(
                variant,
                "verifyMissingKeepRuleNegativeControl",
                listOf(negativeControlExpectation),
            )
        }
    }
}

val verifyConsumerApkSize by tasks.registering(Exec::class) {
    group = "verification"
    description = "Fails when OpenMedKit adds more bytes to a release ABI split " +
        "than its committed budget."
    val report = reportsDirectory.map { it.file("consumer-apk-size.properties") }
    inputs.dir(openmedkitApks)
    inputs.dir(baselineApks)
    inputs.file(budgetFile)
    inputs.file(gateScript)
    outputs.file(report)
    executable(pythonExecutable.get())
    args(gateScript.absolutePath, "size")
    argumentProviders.add(
        CommandLineArgumentProvider {
            listOf(
                "--apk-dir", openmedkitApks.get().asFile.absolutePath,
                "--baseline-apk-dir", baselineApks.get().asFile.absolutePath,
                "--budgets", budgetFile.asFile.absolutePath,
                "--report", report.get().asFile.absolutePath,
            )
        },
    )
}

tasks.register("verifyR8Consumer") {
    group = "verification"
    description = "Runs the R8 consumer gate, its negative control, the per-ABI size " +
        "budget, 16 KB zip alignment and the smoke-report JVM tests."
    dependsOn(
        "verifyOpenmedkitR8Gate",
        "verifyMissingKeepRuleNegativeControl",
        verifyConsumerApkSize,
        "verifyOpenmedkitReleaseApkZipAlignment",
        "testOpenmedkitReleaseUnitTest",
    )
}
