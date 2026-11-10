package org.openmed.maple

import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import org.junit.Test

class MapleOnnxContractTest {
    @Test
    fun createsZeroLengthPrefillCacheShapeFromGraphMetadata() {
        val shape = mapleInitialCacheShape(
            inputName = "past_key_values.0.key",
            declaredShape = longArrayOf(-1L, 4L, -1L, 128L),
        )

        assertContentEquals(longArrayOf(1L, 4L, 0L, 128L), shape)
    }

    @Test
    fun rejectsCacheMetadataThatCannotRepresentEmptyPrefill() {
        assertFailsWith<MapleBundleException> {
            mapleInitialCacheShape(
                inputName = "past_key_values.0.key",
                declaredShape = longArrayOf(1L, 4L, 16L, 128L),
            )
        }
        assertFailsWith<MapleBundleException> {
            mapleInitialCacheShape(
                inputName = "past_key_values.0.key",
                declaredShape = longArrayOf(1L, -1L, -1L, 128L),
            )
        }
    }
}
