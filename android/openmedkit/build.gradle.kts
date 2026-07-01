plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
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
}

kotlin {
    jvmToolchain(11)
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.mlkit.text.recognition)
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit)
    testImplementation(libs.robolectric)
}
