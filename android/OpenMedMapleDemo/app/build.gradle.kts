plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "org.openmed.maple"
    compileSdk = 33

    defaultConfig {
        applicationId = "org.openmed.maple"
        minSdk = 26
        targetSdk = 33
        versionCode = 1
        versionName = "0.1.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.4.8"
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

// DJL 0.33 pulls Java-17-era annotation/logging jars that AGP 7.4's D8 cannot
// transform. These API-compatible pins keep the demo aligned with this repo's
// JDK 11 / AGP 7.4 Android baseline.
configurations.configureEach {
    resolutionStrategy.eachDependency {
        when (requested.group to requested.name) {
            "org.slf4j" to "slf4j-api" ->
                useVersion(libs.versions.slf4j.android.compatible.get())
            "com.google.errorprone" to "error_prone_annotations" ->
                useVersion(libs.versions.errorprone.android.compatible.get())
        }
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2023.06.01"))
    implementation("androidx.activity:activity-compose:1.7.2")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.core:core-ktx:1.10.1")
    implementation(libs.djl.tokenizer.native.android)
    implementation(libs.djl.tokenizers)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.onnxruntime.android)

    debugImplementation("androidx.compose.ui:ui-tooling")

    testImplementation(libs.junit)
    testImplementation(libs.kotlin.test.junit)
}
