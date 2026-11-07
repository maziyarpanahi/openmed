import com.android.build.api.artifact.SingleArtifact
import com.android.build.api.variant.ApplicationAndroidComponentsExtension
import com.android.build.api.variant.LibraryAndroidComponentsExtension

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlinx.binary.compatibility.validator) apply false
}

// Uncompressed native libraries must start on a 16 KB boundary inside an APK
// so 16 KB page-size devices can map them in place. AGP 8.5.1+ packages them
// that way; these tasks verify every demo APK and the OpenMedKit instrumented
// test APK with the same rules as `zipalign -c -P 16 4`.
val apkZipAlignmentScript = rootProject.projectDir.parentFile
    .resolve("scripts/android/check_apk_zip_alignment.py")
val apkZipAlignmentPython = providers.gradleProperty("openmedPython")
    .orElse(providers.environmentVariable("PYTHON"))
    .orElse("python3")

val verifyApkZipAlignment by tasks.registering {
    group = "verification"
    description = "Verifies 16 KB zip alignment of every debug demo APK and the " +
        "OpenMedKit instrumented test APK."
}

fun Project.registerApkZipAlignmentCheck(
    componentName: String,
    apkDirectory: Provider<Directory>,
) {
    val taskName = "verify${componentName.replaceFirstChar { it.uppercaseChar() }}" +
        "ApkZipAlignment"
    val check = tasks.register<Exec>(taskName) {
        group = "verification"
        description = "Verifies 16 KB zip alignment of native libraries in the " +
            "$componentName APK."
        inputs.file(apkZipAlignmentScript)
        inputs.dir(apkDirectory)
        executable(apkZipAlignmentPython.get())
        args(
            apkZipAlignmentScript.absolutePath,
            "--page-size-kb",
            "16",
            "--require-native-libraries",
        )
        argumentProviders.add(
            CommandLineArgumentProvider { listOf(apkDirectory.get().asFile.absolutePath) },
        )
    }
    if (componentName.startsWith("debug")) {
        verifyApkZipAlignment.configure { dependsOn(check) }
    }
}

subprojects {
    val module = this
    pluginManager.withPlugin("com.android.application") {
        extensions.configure<ApplicationAndroidComponentsExtension> {
            onVariants { variant ->
                module.registerApkZipAlignmentCheck(
                    variant.name,
                    variant.artifacts.get(SingleArtifact.APK),
                )
            }
        }
    }
    pluginManager.withPlugin("com.android.library") {
        extensions.configure<LibraryAndroidComponentsExtension> {
            onVariants { variant ->
                variant.androidTest?.let { testComponent ->
                    module.registerApkZipAlignmentCheck(
                        testComponent.name,
                        testComponent.artifacts.get(SingleArtifact.APK),
                    )
                }
            }
        }
    }
}
