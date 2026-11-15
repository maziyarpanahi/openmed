import java.util.Properties
import java.util.zip.ZipFile
import org.gradle.api.artifacts.type.ArtifactTypeDefinition
import org.gradle.api.tasks.testing.Test
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    // Public API baseline: `apiDump` writes api/openmedkit.api, `apiCheck` gates it.
    alias(libs.plugins.kotlinx.binary.compatibility.validator)
    `maven-publish`
    signing
}

val publicationGroup = providers.gradleProperty("openmedAndroidGroup")
    .orElse("org.openmed")
    .get()
val publicationArtifact = providers.gradleProperty("openmedAndroidArtifact")
    .orElse("openmedkit")
    .get()

group = publicationGroup
version = providers.gradleProperty("openmedAndroidVersion")
    .orElse(providers.environmentVariable("OPENMED_ANDROID_VERSION"))
    .orElse("0.0.0-SNAPSHOT")
    .get()

val repoRoot = rootProject.projectDir.parentFile
val generatedCatalogAssetsDir = layout.buildDirectory.dir("generated/assets/openmedCatalog")
val generatedCatalogAsset = generatedCatalogAssetsDir.map {
    it.file("openmed_model_catalog.jsonl")
}
val budgetPropertiesFile = rootProject.layout.projectDirectory.file("gradle/budgets.properties")
val consumerRulesFile = layout.projectDirectory.file("consumer-rules.pro")
val releaseAar = layout.buildDirectory.file("outputs/aar/${project.name}-release.aar")
val aarSizeBudgetReport = layout.buildDirectory.file(
    "reports/budgets/release-aar-size.properties",
)
val centralStagingRepository = layout.buildDirectory.dir("central-staging")
val centralPortalBundle = layout.buildDirectory.dir("central-portal")
val pythonExecutable = providers.gradleProperty("openmedPython")
    .orElse(providers.environmentVariable("PYTHON"))
    .orElse("python3")
val releaseVersionPattern = Regex("""\d+\.\d+\.\d+([.-][A-Za-z0-9][A-Za-z0-9.-]*)?""")

fun readPositiveBudget(key: String): Long {
    val budgetFile = budgetPropertiesFile.asFile
    if (!budgetFile.isFile) {
        throw GradleException("Android budget file does not exist: ${budgetFile.path}")
    }

    val properties = Properties()
    budgetFile.inputStream().use(properties::load)
    val rawValue = properties.getProperty(key)
        ?: throw GradleException("Missing Android budget '$key' in ${budgetFile.path}.")
    return rawValue.toLongOrNull()?.takeIf { it > 0 }
        ?: throw GradleException(
            "Android budget '$key' must be a positive integer, got '$rawValue'.",
        )
}

fun isSnapshotVersion(): Boolean {
    return version.toString().contains("SNAPSHOT", ignoreCase = true)
}

// ai.djl.android:tokenizer-native ships libdjl_tokenizer.so without the C++
// runtime it links against (DT_NEEDED libc++_shared.so), so OpenMedKit packages
// the NDK's copy for every ABI. The NDK revision is pinned through
// android.ndkVersion, and AGP strips the copy with that NDK, so local and CI
// builds package the same bytes; nothing binary is committed. The kit license
// gate (scripts/licenses/kit_inventory.py) reads this list: every entry needs an
// `ndk` row in docs/security/kit-license-inventory.md.
val bundledNdkLibraries = listOf("libc++_shared.so")
val ndkSysrootTriples = mapOf(
    "arm64-v8a" to "aarch64-linux-android",
    "armeabi-v7a" to "arm-linux-androideabi",
    "x86" to "i686-linux-android",
    "x86_64" to "x86_64-linux-android",
)

/** Copies the pinned NDK's runtime libraries into a generated `jniLibs` directory. */
abstract class CopyNdkRuntimeLibrariesTask : DefaultTask() {
    /** NDK revision that `android.ndkVersion` pins; the installed NDK must match. */
    @get:Input
    abstract val ndkVersion: Property<String>

    /** Library file names copied for every ABI. */
    @get:Input
    abstract val libraries: ListProperty<String>

    /** ABI directory name mapped to its NDK sysroot target triple. */
    @get:Input
    abstract val sysrootTriples: MapProperty<String, String>

    /** `<sdk>/ndk/<ndkVersion>`, where AGP also finds the strip tool. */
    @get:Internal
    abstract val ndkDirectory: DirectoryProperty

    /** The copied libraries only, so the rest of the NDK is never snapshotted. */
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.RELATIVE)
    abstract val sourceLibraries: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDirectory: DirectoryProperty

    @TaskAction
    fun copy() {
        val ndk = ndkDirectory.get().asFile
        val pinned = ndkVersion.get()
        val installed = ndk.resolve("source.properties").takeIf { it.isFile }?.let { file ->
            Properties().apply { file.inputStream().use(::load) }
                .getProperty("Pkg.Revision")
                ?.trim()
        }
        if (installed == null) {
            throw GradleException(
                "OpenMedKit packages ${libraries.get().joinToString()} from Android NDK " +
                    "$pinned (android.ndkVersion), which is not installed in the Android " +
                    "SDK's ndk/$pinned directory. Install it with " +
                    "`sdkmanager --install \"ndk;$pinned\"`.",
            )
        }
        if (installed != pinned) {
            throw GradleException(
                "The NDK in the Android SDK's ndk/$pinned directory reports revision " +
                    "$installed; OpenMedKit pins $pinned.",
            )
        }
        val hosts = ndk.resolve("toolchains/llvm/prebuilt")
            .listFiles { file -> file.isDirectory }
            .orEmpty()
        val host = hosts.singleOrNull()
            ?: throw GradleException(
                "NDK $pinned must contain exactly one prebuilt host toolchain, " +
                    "found ${hosts.size}.",
            )

        val output = outputDirectory.get().asFile
        output.deleteRecursively()
        for ((abi, triple) in sysrootTriples.get().toSortedMap()) {
            for (library in libraries.get()) {
                val source = host.resolve("sysroot/usr/lib/$triple/$library")
                if (!source.isFile) {
                    throw GradleException("NDK $pinned has no $library for $abi ($triple).")
                }
                source.copyTo(output.resolve("$abi/$library"), overwrite = true)
            }
        }
    }
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
    compileSdk = 35
    // Source of the packaged libc++_shared.so and the tool that strips it.
    ndkVersion = libs.versions.android.ndk.get()

    defaultConfig {
        // Android 8.0+ keeps on-device inference support aligned with current
        // runtime and filesystem APIs without forcing recent-only devices.
        minSdk = 26
        consumerProguardFiles("consumer-rules.pro")
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // The build runs on JDK 17 (required by AGP 8), while the published
    // bytecode stays at Java 11 so consumer apps keep the same floor.
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    testOptions {
        // AGP 8 deprecates targetSdk in a library's defaultConfig; it only ever
        // applied to the library's own test APK and lint, never to consumers.
        targetSdk = 35
        unitTests.isIncludeAndroidResources = true
    }

    lint {
        targetSdk = 35
    }

    sourceSets {
        getByName("main") {
            assets.srcDir(generatedCatalogAssetsDir)
            resources.srcDir("src/main/assets")
        }
        getByName("test") {
            // PHI sentinel sweep test support, shared with instrumented tests and
            // never compiled into the release AAR.
            java.srcDir("src/sharedTest/kotlin")
            resources.srcDir(repoRoot.resolve("fixtures/parity"))
            resources.srcDir(repoRoot.resolve("tests/fixtures/parity"))
            resources.srcDir(repoRoot.resolve("tests/fixtures/processing"))
        }
        getByName("androidTest") {
            java.srcDir("src/sharedTest/kotlin")
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
    jvmToolchain(17)
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_11)
    }
}

// Deliberate packaging fault for the emulator lane's negative control
// (scripts/android/run_emulator_lane.sh). `missing-native-libraries` strips the
// ONNX Runtime, tokenizer and bundled NDK runtime shared libraries from every
// ABI of the instrumented test APK, so no ABI the device supports can load them
// and the lane must fail with `native_library_unloadable`. It never changes the
// library or its AAR.
val nativePackagingFault = providers.gradleProperty("openmedNativePackagingFault")
    .orElse("none")
    .get()
val faultyNativeLibraries = when (nativePackagingFault) {
    "none" -> emptyList()
    "missing-native-libraries" -> listOf(
        "**/libonnxruntime.so",
        "**/libonnxruntime4j_jni.so",
        "**/libdjl_tokenizer.so",
    ) + bundledNdkLibraries.map { "**/$it" }
    else -> throw GradleException(
        "Unknown openmedNativePackagingFault '$nativePackagingFault'; " +
            "expected none or missing-native-libraries.",
    )
}

androidComponents {
    onVariants { variant ->
        // Plain values, so the providers below capture no script or project state.
        val ndkRevision = android.ndkVersion
        val ndkLibraryPatterns = ndkSysrootTriples.values.flatMap { triple ->
            bundledNdkLibraries.map { library ->
                "toolchains/llvm/prebuilt/*/sysroot/usr/lib/$triple/$library"
            }
        }
        val copyNdkRuntimeLibraries = tasks.register<CopyNdkRuntimeLibrariesTask>(
            "copy${variant.name.replaceFirstChar(Char::uppercaseChar)}NdkRuntimeLibraries",
        ) {
            ndkVersion.set(ndkRevision)
            libraries.set(bundledNdkLibraries)
            sysrootTriples.set(ndkSysrootTriples)
            // The side-by-side NDK that ndkVersion selects. sdkComponents.ndkDirectory
            // has no value in AGP 8.7 unless android.ndkPath is also set.
            ndkDirectory.set(
                androidComponents.sdkComponents.sdkDirectory.map { sdk ->
                    sdk.dir("ndk/$ndkRevision")
                },
            )
            sourceLibraries.from(
                ndkDirectory.map { ndk -> ndk.asFileTree.matching { include(ndkLibraryPatterns) } },
            )
        }
        variant.sources.jniLibs?.addGeneratedSourceDirectory(
            copyNdkRuntimeLibraries,
            CopyNdkRuntimeLibrariesTask::outputDirectory,
        )
        variant.androidTest?.packaging?.jniLibs?.excludes?.addAll(faultyNativeLibraries)
    }
}

// The core AAR must stay free of vendor OCR, Play services, and telemetry
// artifacts; ML Kit OCR lives in the optional :openmedkit-mlkit module.
extra["openmedkitDependencyBoundary"] = "core"
apply(from = rootProject.file("gradle/dependency-boundary.gradle.kts"))

tasks.matching { it.name.endsWith("Assets") }.configureEach {
    dependsOn(generateAndroidModelCatalog)
}

val verifyReleaseConsumerRules by tasks.registering {
    group = "verification"
    description = "Verifies that the release AAR packages every consumer rule."
    dependsOn("bundleReleaseAar")
    inputs.file(consumerRulesFile)
    inputs.file(releaseAar)

    doLast {
        val aarFile = releaseAar.get().asFile
        if (!aarFile.isFile) {
            throw GradleException("Release AAR does not exist: ${aarFile.path}")
        }

        val packagedRules = ZipFile(aarFile).use { archive ->
            val entry = archive.getEntry("proguard.txt")
                ?: throw GradleException("Release AAR does not contain proguard.txt")
            archive.getInputStream(entry).bufferedReader().use { reader ->
                reader.readLines()
                    .map { it.trim() }
                    .filter { it.isNotEmpty() && !it.startsWith("#") }
                    .toSet()
            }
        }
        val requiredRules = consumerRulesFile.asFile.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") }
        val missingRules = requiredRules.filterNot(packagedRules::contains)
        if (missingRules.isNotEmpty()) {
            throw GradleException(
                "Release AAR is missing consumer rules: ${missingRules.joinToString()}",
            )
        }
    }
}

tasks.matching { it.name.startsWith("test") }.configureEach {
    dependsOn(generateAndroidModelCatalog)
    dependsOn(verifyReleaseConsumerRules)
    dependsOn("verifyDependencyBoundary")
}

tasks.withType<Test>().configureEach {
    val coldStartBudgetReport = layout.buildDirectory.file(
        "reports/budgets/${name}-cold-start.properties",
    )

    inputs.file(budgetPropertiesFile)
    // EmulatorLaneFixtureContractTest reads the instrumented-test fixture by path.
    inputs.dir(layout.projectDirectory.dir("src/androidTest/assets/emulator-lane"))
        .withPropertyName("emulatorLaneFixture")
        .withPathSensitivity(PathSensitivity.RELATIVE)
    outputs.file(coldStartBudgetReport)

    doFirst {
        systemProperty(
            "openmed.coldStart.maxMillis",
            readPositiveBudget("coldStart.maxMillis"),
        )
        systemProperty(
            "openmed.coldStart.measurementFile",
            coldStartBudgetReport.get().asFile.absolutePath,
        )
    }

    systemProperty("openmedkit.moduleDirectory", projectDir.absolutePath)
}

val verifyReleaseAarSize by tasks.registering {
    group = "verification"
    description = "Fails when the release OpenMedKit AAR exceeds its committed size budget."
    dependsOn("bundleReleaseAar")

    inputs.file(budgetPropertiesFile)
    inputs.file(releaseAar)
    outputs.file(aarSizeBudgetReport)

    doLast {
        val artifact = releaseAar.get().asFile
        if (!artifact.isFile) {
            throw GradleException("Release AAR was not produced at ${artifact.absolutePath}.")
        }

        val measuredBytes = artifact.length()
        val maximumBytes = readPositiveBudget("aar.maxBytes")
        val passed = measuredBytes <= maximumBytes
        val report = aarSizeBudgetReport.get().asFile
        report.parentFile.mkdirs()
        report.writeText(
            """
            measuredBytes=$measuredBytes
            maxBytes=$maximumBytes
            status=${if (passed) "PASS" else "FAIL"}
            """.trimIndent() + "\n",
        )

        logger.lifecycle(
            "OpenMedKit release AAR size: $measuredBytes bytes (budget: $maximumBytes bytes)",
        )
        if (!passed) {
            throw GradleException(
                "OpenMedKit release AAR is $measuredBytes bytes, exceeding the " +
                    "$maximumBytes-byte budget.",
            )
        }
    }
}

// Android 15+ devices with 16 KB memory pages can only map 64-bit native
// libraries whose ELF LOAD segments are aligned to at least 16 KB. OpenMedKit
// ships no native code of its own, so the gate reads the release AAR together
// with every resolved runtime dependency archive that packages `.so` files.
// Reviewed, expiring exceptions live in gradle/native-page-alignment-allowlist.toml.
val nativeAlignmentScript = repoRoot.resolve("scripts/android/check_elf_page_alignment.py")
val nativeAlignmentAllowlist = rootProject.layout.projectDirectory.file(
    "gradle/native-page-alignment-allowlist.toml",
)
val releaseRuntimeNativeArchives = objects.fileCollection()

androidComponents {
    onVariants(selector().withName("release")) { variant ->
        releaseRuntimeNativeArchives.from(
            variant.runtimeConfiguration.incoming.artifactView {
                attributes {
                    attribute(ArtifactTypeDefinition.ARTIFACT_TYPE_ATTRIBUTE, "aar")
                }
            }.files,
        )
    }
}

val verifyReleaseNativeLibraryAlignment by tasks.registering(Exec::class) {
    group = "verification"
    description = "Fails when a 64-bit native library in the release AAR or its " +
        "runtime dependencies has ELF LOAD alignment below 16 KB."
    dependsOn("bundleReleaseAar")
    inputs.file(nativeAlignmentScript)
    inputs.file(nativeAlignmentAllowlist)
    inputs.file(releaseAar)
    inputs.files(releaseRuntimeNativeArchives)
    executable(pythonExecutable.get())
    args(
        nativeAlignmentScript.absolutePath,
        "--min-alignment-kb",
        "16",
        "--allowlist",
        nativeAlignmentAllowlist.asFile.absolutePath,
        "--strict-allowlist",
        "--require-native-libraries",
    )
    argumentProviders.add(
        CommandLineArgumentProvider {
            listOf(releaseAar.get().asFile.absolutePath) +
                releaseRuntimeNativeArchives.files.map { it.absolutePath }.sorted()
        },
    )
}

tasks.register("verifyAndroidBudgets") {
    group = "verification"
    description = "Runs the release AAR size, 16 KB native alignment and offline " +
        "cold-start budget gates."
    dependsOn(verifyReleaseAarSize, "testDebugUnitTest")
    dependsOn(verifyReleaseNativeLibraryAlignment)
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

                groupId = publicationGroup
                artifactId = publicationArtifact
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
                it.name == "publishReleasePublicationToCentralStagingRepository" ||
                    it.name == "bundleCentralPortalPublication"
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
    implementation(libs.onnxruntime.android)
    implementation(libs.djl.tokenizers)
    implementation(libs.djl.tokenizer.native.android)

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.icu4j)
    testImplementation("org.json:json:20240303")
    testImplementation(libs.robolectric)

    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.junit)
}
