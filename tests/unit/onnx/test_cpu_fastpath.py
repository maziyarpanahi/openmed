"""Tests for the SIMD-aware INT8 token-classification fast path."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

import openmed.onnx as onnx
from openmed.onnx import cpu_fastpath, cpu_features


def _synthetic_case():
    hidden_states = np.array(
        [
            [4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    bias = np.zeros(3, dtype=np.float32)
    offsets = np.array([[0, 0], [0, 5], [6, 12], [0, 0]], dtype=np.int64)
    id2label = {0: "O", 1: "B-PERSON", 2: "E-PERSON"}
    return hidden_states, weights, bias, offsets, id2label


def _features(tier: str) -> cpu_features.CpuFeatures:
    if tier == "avx512":
        return cpu_features.detect_cpu_features(
            machine="x86_64",
            flags="avx2 avx512f avx512bw",
        )
    if tier == "avx2":
        return cpu_features.detect_cpu_features(machine="x86_64", flags="avx2")
    if tier == "neon":
        return cpu_features.detect_cpu_features(machine="aarch64", flags="asimd")
    return cpu_features.detect_cpu_features(machine="x86_64", flags="")


@pytest.mark.parametrize("tier", ["scalar", "avx2", "avx512", "neon"])
def test_fastpath_matches_reference_spans_bit_for_decision(tier: str) -> None:
    hidden_states, weights, bias, offsets, id2label = _synthetic_case()
    head = cpu_fastpath.quantize_classification_head(weights, bias)

    result = cpu_fastpath.run_cpu_fastpath(
        hidden_states,
        head,
        offsets,
        id2label,
        features=_features(tier),
    )
    reference = cpu_fastpath.run_reference_classification_head(
        hidden_states,
        weights,
        bias,
        offsets,
        id2label,
    )

    assert result.kernel == tier
    assert np.array_equal(result.label_ids, reference.label_ids)
    assert [(span.label, span.start, span.end) for span in result.spans] == [
        ("PERSON", 0, 12)
    ]
    assert [
        (span.label, span.token_start, span.token_end, span.start, span.end)
        for span in result.spans
    ] == [
        (span.label, span.token_start, span.token_end, span.start, span.end)
        for span in reference.spans
    ]


def test_numerical_equivalence_guard_accepts_synthetic_note() -> None:
    hidden_states, weights, bias, offsets, id2label = _synthetic_case()
    head = cpu_fastpath.quantize_classification_head(weights, bias)

    verification = cpu_fastpath.verify_cpu_fastpath(
        hidden_states,
        weights,
        bias,
        head,
        offsets,
        id2label,
        features=_features("avx2"),
    )

    assert verification.passed is True
    assert verification.label_ids_match is True
    assert verification.spans_match is True
    assert verification.max_abs_logit_delta == pytest.approx(0.0, abs=1e-6)
    assert verification.to_dict()["tolerance"] == (cpu_fastpath.DEFAULT_LOGIT_TOLERANCE)


def test_numerical_equivalence_guard_fails_closed_on_decision_drift() -> None:
    hidden_states, weights, bias, offsets, id2label = _synthetic_case()
    drifted_head = cpu_fastpath.quantize_classification_head(
        weights[[1, 0, 2]],
        bias,
    )

    with pytest.raises(
        cpu_fastpath.CpuFastPathVerificationError,
        match="label_ids_match=False",
    ):
        cpu_fastpath.verify_cpu_fastpath(
            hidden_states,
            weights,
            bias,
            drifted_head,
            offsets,
            id2label,
            features=_features("neon"),
        )


def test_feature_detection_selects_scalar_deterministically_without_simd() -> None:
    features = cpu_features.detect_cpu_features(
        machine="x86_64",
        flags="sse sse2 sse4_2",
    )

    assert features.tier == "scalar"
    assert features.avx2 is False
    assert features.avx512 is False
    assert features.neon is False
    assert cpu_features.select_cpu_kernel(features) == "scalar"


def test_feature_detection_prefers_avx512_and_gates_flags_by_architecture() -> None:
    avx512 = cpu_features.detect_cpu_features(
        machine="AMD64",
        flags="AVX2 AVX512F AVX512BW",
    )
    arm = cpu_features.detect_cpu_features(
        machine="aarch64",
        flags="avx2 avx512f avx512bw neon",
    )

    assert avx512.architecture == "x86_64"
    assert avx512.tier == "avx512"
    assert arm.architecture == "arm64"
    assert arm.tier == "neon"
    assert arm.avx2 is False


def test_linux_feature_detection_intersects_all_processor_flags(monkeypatch) -> None:
    cpuinfo = """processor: 0
flags: sse2 avx2
processor: 1
flags: sse2
"""
    monkeypatch.setattr(
        cpu_features.Path,
        "read_text",
        lambda self, **kwargs: cpuinfo,
    )

    assert cpu_features._linux_cpu_flags() == frozenset({"sse2"})


def test_fastpath_public_api_is_available_from_onnx_package() -> None:
    assert onnx.CpuFeatures is cpu_features.CpuFeatures
    assert onnx.CpuFastPathResult is cpu_fastpath.CpuFastPathResult
    assert onnx.quantize_classification_head is (
        cpu_fastpath.quantize_classification_head
    )
    assert onnx.verify_cpu_fastpath is cpu_fastpath.verify_cpu_fastpath


def test_quantization_uses_positive_scale_for_zero_rows() -> None:
    head = cpu_fastpath.quantize_classification_head(np.zeros((2, 4), dtype=np.float32))

    assert head.weights.dtype == np.int8
    assert np.array_equal(head.weights, np.zeros((2, 4), dtype=np.int8))
    assert np.array_equal(head.scales, np.ones(2, dtype=np.float32))


def test_bio_decode_honors_special_offsets_boundaries_and_threshold() -> None:
    spans = cpu_fastpath.decode_bio_spans(
        np.array([0, 1, 2, 3, 0]),
        np.array([1.0, 0.9, 0.8, 0.5, 1.0]),
        np.array([[0, 0], [0, 4], [5, 9], [10, 12], [0, 0]]),
        {0: "O", 1: "B-NAME", 2: "E-NAME", 3: "S-ID"},
        threshold=0.6,
    )

    assert [span.to_dict() for span in spans] == [
        {
            "label": "NAME",
            "score": pytest.approx(0.85),
            "token_start": 1,
            "token_end": 3,
            "start": 0,
            "end": 9,
        }
    ]


def test_benchmark_record_captures_avx2_speedup_with_deterministic_clock() -> None:
    hidden_states, weights, bias, offsets, id2label = _synthetic_case()
    head = cpu_fastpath.quantize_classification_head(weights, bias)
    times: Iterator[float] = iter(
        [
            0.000,
            0.010,
            0.010,
            0.020,
            0.020,
            0.030,
            0.030,
            0.032,
            0.032,
            0.034,
            0.034,
            0.036,
        ]
    )

    record = cpu_fastpath.benchmark_cpu_fastpath(
        hidden_states,
        head,
        offsets,
        id2label,
        features=_features("avx2"),
        iterations=3,
        warmup=0,
        clock=lambda: next(times),
    )

    assert record.cpu.tier == "avx2"
    assert record.scalar_latency_ms == pytest.approx(10.0)
    assert record.fastpath_latency_ms == pytest.approx(2.0)
    assert record.speedup == pytest.approx(5.0)
    assert record.passed is True
    assert record.to_dict()["shape"] == {
        "sequence_length": 4,
        "hidden_size": 4,
        "label_count": 3,
    }


def test_avx2_runner_records_nontrivial_latency_reduction() -> None:
    detected = cpu_features.detect_cpu_features()
    if detected.tier != "avx2":
        pytest.skip("requires a runner whose strongest detected tier is AVX2")

    rng = np.random.default_rng(479)
    hidden_states = rng.normal(size=(48, 256)).astype(np.float32)
    weights = rng.normal(scale=0.05, size=(9, 256)).astype(np.float32)
    offsets = np.column_stack(
        (
            np.arange(48, dtype=np.int64) * 2,
            np.arange(48, dtype=np.int64) * 2 + 1,
        )
    )
    id2label = {0: "O", **{index: f"S-PHI_{index}" for index in range(1, 9)}}
    record = cpu_fastpath.benchmark_cpu_fastpath(
        hidden_states,
        cpu_fastpath.quantize_classification_head(weights),
        offsets,
        id2label,
        features=detected,
        iterations=5,
        warmup=2,
    )

    assert record.cpu.tier == "avx2"
    assert record.speedup >= cpu_fastpath.DEFAULT_MIN_SPEEDUP
    assert record.passed is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weights": np.zeros((0, 4))}, "non-empty"),
        ({"weights": np.zeros((2, 4)), "bias": np.zeros(3)}, "bias"),
        ({"weights": np.array([[np.nan, 0.0]])}, "finite"),
    ],
)
def test_quantize_classification_head_rejects_invalid_inputs(
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        cpu_fastpath.quantize_classification_head(**kwargs)
