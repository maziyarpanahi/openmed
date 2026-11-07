"""Offline runtime seam with real NumPy float32 rounding, no model or media I/O."""

from dataclasses import replace

import pytest

from openmed.multimodal.preprocessing import (
    PreprocessingProfile,
    execute_profile,
    qualify_profile,
    verify_preprocessing,
)


@pytest.mark.integration
def test_local_array_adapter_evidence():
    np = pytest.importorskip("numpy")
    p = PreprocessingProfile(
        (2, 2), (0, 0, 2, 2), (3, 3), interpolation="bilinear", layout="CHW"
    )

    def adapter(profile, pixels, mask):
        result = execute_profile(profile, pixels, mask)
        # This tests adapter conversion only, not independent NumPy operator qualification.
        return replace(
            result,
            tensor=tuple(
                np.asarray(result.tensor, dtype=np.float32).astype(float).tolist()
            ),
        )

    runtime = "b" * 64
    q = qualify_profile(p, adapter, runtime_digest=runtime)
    pixels = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
    mask = (True, False, False, False)
    output = adapter(p, pixels, mask)
    evidence = verify_preprocessing(p, pixels, mask, output, q, runtime_digest=runtime)
    evidence.require_current(p, output, q, runtime_digest=runtime)
