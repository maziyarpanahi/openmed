"""Synthetic operator vectors with independent pixel/geometry negative controls."""

import json
import math
import struct
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from openmed.multimodal.preprocessing import (
    NOTICE,
    PreprocessingError,
    PreprocessingOutput,
    PreprocessingProfile,
    execute_profile,
    qualify_profile,
    verify_preprocessing,
)

RUNTIME = "a" * 64
PIXELS = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255))
MASK = (True, False, False, False)


def profile(**kw):
    return PreprocessingProfile((2, 2), (0, 0, 2, 2), (2, 2), **kw)


def test_color_alpha_and_bilinear_vector():
    p = replace(
        profile(interpolation="bilinear", background=(255, 255, 255)),
        target_size=(3, 3),
    )
    pixels = ((255, 0, 0, 255), (0, 255, 0, 128), (0, 0, 255, 0), (255, 255, 255, 255))
    out = execute_profile(p, pixels, MASK)
    # Independent arithmetic: alpha=0 produces white, alpha=128 preserves green.
    assert out.tensor[:3] == (1, 0, 0)
    assert out.tensor[6:9] == pytest.approx((127 / 255, 1, 127 / 255), abs=p.atol)
    assert out.tensor[12:15] == pytest.approx(
        (223 / 255, 0.75, 159.25 / 255), abs=p.atol
    )
    assert out.tensor[18:21] == (1, 1, 1)
    assert out.geometry == ("3/2", "0", "0", "0", "3/2", "0")
    assert out.protected_mask == (
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        False,
        False,
    )


@pytest.mark.parametrize(
    "orientation,indices,geometry",
    [
        (1, [0, 1, 2, 3, 4, 5], ("1", "0", "0", "0", "1", "0")),
        (2, [1, 0, 3, 2, 5, 4], ("-1", "0", "2", "0", "1", "0")),
        (3, [5, 4, 3, 2, 1, 0], ("-1", "0", "2", "0", "-1", "3")),
        (4, [4, 5, 2, 3, 0, 1], ("1", "0", "0", "0", "-1", "3")),
        (5, [0, 2, 4, 1, 3, 5], ("0", "1", "0", "1", "0", "0")),
        (6, [4, 2, 0, 5, 3, 1], ("0", "-1", "3", "1", "0", "0")),
        (7, [5, 3, 1, 4, 2, 0], ("0", "-1", "3", "-1", "0", "2")),
        (8, [1, 3, 5, 0, 2, 4], ("0", "1", "0", "-1", "0", "2")),
    ],
)
def test_all_exif_orientations_exact(orientation, indices, geometry):
    size = (3, 2) if orientation >= 5 else (2, 3)
    p = PreprocessingProfile(
        (2, 3), (0, 0, *size), size, orientation=orientation, scale=1
    )
    pixels = tuple((i, i + 10, i + 20) for i in range(6))
    out = execute_profile(p, pixels, (True, False, False, False, False, False))
    assert out.tensor[::3] == tuple(indices)
    assert out.geometry == geometry
    assert out.protected_mask == tuple(i == 0 for i in indices)


def test_crop_resize_and_mask_support_are_separate_from_geometry():
    p = PreprocessingProfile(
        (3, 2), (1, 0, 3, 2), (1, 1), scale=1, interpolation="bilinear"
    )
    pixels = tuple((i, i, i) for i in range(6))
    out = execute_profile(p, pixels, (False, True, False, False, False, False))
    assert out.tensor == (3, 3, 3)
    assert out.geometry == ("1/2", "0", "-1/2", "0", "1/2", "0")
    assert out.protected_mask == (True,)
    nearest = execute_profile(
        replace(p, interpolation="nearest"),
        pixels,
        (False, True, False, False, False, False),
    )
    assert nearest.tensor == (5, 5, 5)  # ties round up at center of 2x2 crop
    assert nearest.protected_mask == (False,)
    cropped_out = execute_profile(p, pixels, (True, False, False, False, False, False))
    assert cropped_out.protected_mask == (False,)  # never claims redaction coverage


def test_bgr_chw_and_normalization_order():
    p = profile(color_order="BGR", layout="CHW", scale=1, mean=(1, 2, 3), std=(1, 2, 4))
    out = execute_profile(p, PIXELS, MASK)
    assert out.shape == (3, 2, 2)
    assert out.tensor[:4] == (-1, -1, 254, 254)
    assert out.tensor[4:8] == (-1, 126.5, -1, 126.5)
    assert out.tensor[8:] == (63, -0.75, -0.75, 63)


@pytest.mark.parametrize(
    "change",
    [
        dict(interpolation="bilinear"),
        dict(color_order="BGR"),
        dict(mean=(0.001, 0, 0)),
        dict(std=(1, 1, 2)),
        dict(atol=0),
        dict(dtype="float64"),
    ],
)
def test_profile_change_invalidates_qualification_and_coordinate_evidence(change):
    p = profile()
    q = qualify_profile(p, execute_profile, runtime_digest=RUNTIME)
    out = execute_profile(p, PIXELS, MASK)
    evidence = verify_preprocessing(p, PIXELS, MASK, out, q, runtime_digest=RUNTIME)
    changed = replace(p, **change)
    changed_out = execute_profile(changed, PIXELS, MASK)
    assert changed.digest != p.digest
    assert changed_out.coordinate_digest != out.coordinate_digest
    with pytest.raises(PreprocessingError, match="qualification_stale"):
        evidence.require_current(changed, changed_out, q, runtime_digest=RUNTIME)
    with pytest.raises(PreprocessingError, match="qualification_stale"):
        evidence.require_current(p, out, q, runtime_digest="b" * 64)


def test_tolerated_tensor_rounding_does_not_tolerate_coordinate_drift():
    p = profile(dtype="float64")
    q = qualify_profile(p, execute_profile, runtime_digest=RUNTIME)
    out = execute_profile(p, PIXELS, MASK)
    rounded = replace(out, tensor=(out.tensor[0] + p.atol / 2, *out.tensor[1:]))
    e = verify_preprocessing(p, PIXELS, MASK, rounded, q, runtime_digest=RUNTIME)
    assert rounded.tensor_digest != out.tensor_digest
    e.require_current(p, rounded, q, runtime_digest=RUNTIME)
    with pytest.raises(PreprocessingError, match="evidence_stale"):
        e.require_current(p, out, q, runtime_digest=RUNTIME)
    for invalid in (
        replace(out, geometry=("1000001/1000000", *out.geometry[1:])),
        replace(out, protected_mask=(False, *MASK[1:])),
    ):
        with pytest.raises(PreprocessingError, match="coordinate_mismatch"):
            verify_preprocessing(p, PIXELS, MASK, invalid, q, runtime_digest=RUNTIME)
    invalid = replace(out, tensor=(out.tensor[0] + p.atol * 2, *out.tensor[1:]))
    with pytest.raises(PreprocessingError, match="tensor_mismatch"):
        verify_preprocessing(p, PIXELS, MASK, invalid, q, runtime_digest=RUNTIME)


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("interpolation", "bicubic", "operator_unsupported"),
        ("orientation", 9, "orientation_unsupported"),
        ("version", 2, "version_unsupported"),
        ("source_size", (100000, 100000), "dimensions_invalid"),
        ("crop", (-1, 0, 2, 2), "crop_invalid"),
        ("std", (0, 1, 1), "numeric_behavior_unsupported"),
        ("scale", math.inf, "numeric_behavior_unsupported"),
        ("atol", 1, "numeric_behavior_unsupported"),
        ("background", (True, 0, 0), "background_invalid"),
        ("mean", (math.nan, 0, 0), "normalization_invalid"),
    ],
)
def test_unsupported_behavior_rejected_without_payload(field, value, code):
    with pytest.raises(PreprocessingError, match=code):
        replace(profile(), **{field: value})


def test_runtime_negative_controls_and_privacy():
    p = profile()
    for change, code in (
        (dict(color_order="BGR"), "tensor_mismatch"),
        (dict(mean=(1, 0, 0)), "tensor_mismatch"),
        (dict(interpolation="bilinear", target_size=(3, 3)), "coordinate_mismatch"),
    ):

        def lying_adapter(plan, pixels, mask):
            out = execute_profile(replace(plan, **change), pixels, mask)
            return replace(out, profile_digest=plan.digest)

        with pytest.raises(PreprocessingError, match=code):
            qualify_profile(p, lying_adapter, runtime_digest=RUNTIME)

    def failed(*args):
        raise RuntimeError("PRIVATE_PATH_AND_PAYLOAD")

    with pytest.raises(PreprocessingError) as error:
        qualify_profile(p, failed, runtime_digest=RUNTIME)
    assert str(error.value) == "adapter_failed"
    q = qualify_profile(p, execute_profile, runtime_digest=RUNTIME)
    out = execute_profile(p, PIXELS, MASK)
    evidence = verify_preprocessing(p, PIXELS, MASK, out, q, runtime_digest=RUNTIME)
    assert evidence.notice == q.notice == NOTICE
    assert evidence.reviewer_confirmation_required
    assert set(asdict(evidence)) == {
        "qualification_digest",
        "tensor_digest",
        "coordinate_digest",
        "notice",
        "reviewer_confirmation_required",
    }
    assert "255" not in repr(out)


def test_output_validation_rejects_nonfinite_unrounded_and_boolean_masks():
    out = execute_profile(profile(), PIXELS, MASK)
    for kw in (
        dict(tensor=(math.nan, *out.tensor[1:])),
        dict(protected_mask=(1, False, False, False)),
        dict(tensor=(0.1, *out.tensor[1:])),
        dict(geometry=("1.0", *out.geometry[1:])),
        dict(geometry=("1e999999999999999", *out.geometry[1:])),
    ):
        with pytest.raises(PreprocessingError):
            replace(out, **kw)
    for bad in (((True, 0, 0),) * 4, ((1, 2, 3, 4, 5),) * 4):
        with pytest.raises(PreprocessingError):
            execute_profile(profile(), bad, MASK)


def test_shared_openmedkit_vectors():
    fixture = Path(__file__).parents[2] / "fixtures/multimodal/preprocessing-v1.json"
    for v in json.loads(fixture.read_text())["vectors"]:
        raw = v["profile"]
        for key in ("source_size", "crop", "target_size", "background", "mean", "std"):
            raw[key] = tuple(raw[key])
        p = PreprocessingProfile(**raw)
        out = execute_profile(p, tuple(tuple(x) for x in v["pixels"]), tuple(v["mask"]))
        assert out.tensor == tuple(v["tensor"])
        assert out.geometry == tuple(v["geometry"])
        assert out.protected_mask == tuple(v["protected_mask"])
        assert p.digest == v["profile_digest"]
        assert out.tensor_digest == v["tensor_digest"]
        assert out.coordinate_digest == v["coordinate_digest"]
        assert (
            qualify_profile(p, execute_profile, runtime_digest=RUNTIME).digest
            == v["qualification_digest"]
        )


def test_lying_interpolation_rejected_with_same_shape_and_geometry():
    p = replace(profile(), target_size=(3, 3))

    def adapter(plan, pixels, mask):
        actual = execute_profile(replace(plan, interpolation="bilinear"), pixels, mask)
        expected = execute_profile(plan, pixels, mask)
        assert actual.shape == expected.shape and actual.geometry == expected.geometry
        # Keep geometry/masks correct to isolate the pixel negative control.
        return replace(
            actual, profile_digest=plan.digest, protected_mask=expected.protected_mask
        )

    with pytest.raises(PreprocessingError, match="tensor_mismatch"):
        qualify_profile(p, adapter, runtime_digest=RUNTIME)
