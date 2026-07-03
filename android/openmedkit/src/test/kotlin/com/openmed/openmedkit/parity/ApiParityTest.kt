package com.openmed.openmedkit.parity

import com.openmed.openmedkit.EntityPrediction
import com.openmed.openmedkit.OpenMed
import com.openmed.openmedkit.OpenMedMLXModelCacheState
import com.openmed.openmedkit.OpenMedModelStore
import org.junit.Assert.assertTrue
import org.junit.Test

class ApiParityTest {
    @Test
    fun openMedExposesSwiftParityMethods() {
        val methods = OpenMed::class.java.methods.map { it.name }.toSet()

        assertTrue(methods.contains("analyzeText"))
        assertTrue(methods.contains("extractPII"))
        assertTrue(methods.contains("extractPIIChunked"))
    }

    @Test
    fun entityPredictionExposesSwiftParityFields() {
        val fields = EntityPrediction::class.java.declaredFields.map { it.name }.toSet()

        assertTrue(fields.contains("label"))
        assertTrue(fields.contains("text"))
        assertTrue(fields.contains("confidence"))
        assertTrue(fields.contains("start"))
        assertTrue(fields.contains("end"))
    }

    @Test
    fun modelStoreExposesSwiftParityMethods() {
        val methods = OpenMedModelStore::class.java.methods.map { it.name }.toSet()

        assertTrue(methods.contains("downloadMLXModel"))
        assertTrue(methods.contains("cachedMLXModelDirectory"))
        assertTrue(methods.contains("isMLXModelCached"))
        assertTrue(methods.contains("mlxModelCacheState"))
    }

    @Test
    fun modelCacheStateExposesSwiftParityCases() {
        val states = OpenMedMLXModelCacheState.values().map { it.name }.toSet()

        assertTrue(states.contains("missing"))
        assertTrue(states.contains("partial"))
        assertTrue(states.contains("ready"))
    }
}
