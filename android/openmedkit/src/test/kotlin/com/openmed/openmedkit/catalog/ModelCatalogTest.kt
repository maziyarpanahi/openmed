package com.openmed.openmedkit.catalog

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RuntimeEnvironment
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [26])
class ModelCatalogTest {
    @Test
    fun parsesCatalogEntries() {
        val catalog = ModelCatalog.fromJsonLines(SAMPLE_CATALOG)

        assertEquals(3, catalog.entries.size)
        assertEquals(
            ModelCatalogEntry(
                repoId = "OpenMed/tiny-onnx",
                formats = listOf("onnx", "onnx-int8"),
                tier = "tiny",
                paramCount = 33_000_000,
                languages = listOf("en", "es"),
                license = "apache-2.0",
                reproducibilityHash =
                    "sha256:1111111111111111111111111111111111111111111111111111111111111111",
            ),
            catalog.byRepoId("OpenMed/tiny-onnx"),
        )
    }

    @Test
    fun filtersByFormatTierParamCountLanguageAndLicense() {
        val catalog = ModelCatalog.fromJsonLines(SAMPLE_CATALOG)

        assertEquals(
            listOf("OpenMed/tiny-onnx", "OpenMed/untiered-onnx"),
            catalog.filter(format = "ONNX").map { it.repoId },
        )
        assertEquals(
            listOf("OpenMed/tiny-onnx"),
            catalog.filter(tier = "TINY").map { it.repoId },
        )
        assertEquals(
            listOf("OpenMed/tiny-onnx"),
            catalog.filter(maxParamCount = 50_000_000).map { it.repoId },
        )
        assertEquals(
            listOf("OpenMed/tiny-onnx"),
            catalog.filter(language = "ES").map { it.repoId },
        )
        assertEquals(
            listOf("OpenMed/base-tflite"),
            catalog.filter(license = "MIT").map { it.repoId },
        )
    }

    @Test
    fun supportsCombinedPredicates() {
        val catalog = ModelCatalog.fromJsonLines(SAMPLE_CATALOG)

        assertEquals(
            listOf("OpenMed/base-tflite"),
            catalog.filter(
                format = "tflite-int8",
                tier = "base",
                maxParamCount = 200_000_000,
                language = "fr",
                license = "mit",
            ).map { it.repoId },
        )
    }

    @Test
    fun loadsBundledCatalogAsset() {
        val catalog = ModelCatalog.load(RuntimeEnvironment.getApplication())

        assertNotNull(catalog)
        assertTrue(catalog.entries.all { it.repoId.isNotBlank() })
    }

    private companion object {
        private val SAMPLE_CATALOG = """
            {"repo_id":"OpenMed/tiny-onnx","formats":["onnx","onnx-int8"],"tier":"tiny","param_count":33000000,"languages":["en","es"],"license":"apache-2.0","reproducibility_hash":"sha256:1111111111111111111111111111111111111111111111111111111111111111"}
            {"repo_id":"OpenMed/base-tflite","formats":["tflite-int8"],"tier":"base","param_count":125000000,"languages":["fr"],"license":"mit","reproducibility_hash":"sha256:2222222222222222222222222222222222222222222222222222222222222222"}
            {"repo_id":"OpenMed/untiered-onnx","formats":["onnx"],"tier":null,"param_count":null,"languages":[],"license":"bsd-3-clause","reproducibility_hash":"sha256:3333333333333333333333333333333333333333333333333333333333333333"}
        """.trimIndent()
    }
}
