plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
}

subprojects {
    apply(from = rootProject.file("gradle/dependency-locking.gradle.kts"))
}
