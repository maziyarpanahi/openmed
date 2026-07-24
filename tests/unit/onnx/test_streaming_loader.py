"""Tests for deterministic low-RAM ONNX and Safetensors loading."""

from __future__ import annotations

import ctypes
import json
import struct
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest

from openmed.onnx import ram_budget as ram_budget_module
from openmed.onnx.ram_budget import (
    PeakRamProbe,
    RamBudget,
    RamBudgetExceeded,
    RamProbeUnavailable,
)
from openmed.onnx.streaming_loader import (
    LocalWeightsRequired,
    ShardFormatError,
    StreamingWeightLoader,
)


def test_streamed_safetensors_matches_full_inference_spans(
    tmp_path: Path,
) -> None:
    note = "Dr Alice Smith"
    token_features = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    weights = {
        "embeddings.scale": np.asarray(1.0, dtype=np.float32),
        "encoder.layer.0.weight": np.eye(3, dtype=np.float32),
        "encoder.layer.0.bias": np.zeros(3, dtype=np.float32),
        "encoder.layer.1.weight": np.eye(3, dtype=np.float32),
        "encoder.layer.1.bias": np.zeros(3, dtype=np.float32),
        "classifier.weight": np.eye(3, dtype=np.float32),
        "classifier.bias": np.zeros(3, dtype=np.float32),
    }
    index_path = _write_sharded_safetensors(tmp_path, weights)

    full_logits = _run_full(token_features, weights)
    full_spans = _decode_synthetic_spans(note, full_logits)

    loader = StreamingWeightLoader(
        index_path,
        ram_budget_bytes=16 * 1024,
        rss_sampler=lambda: 10_000,
    )
    state = token_features.copy()
    released_groups = []

    def apply_group(group) -> None:
        nonlocal state
        tensors = group.tensors
        if group.name == "embeddings":
            state = state * tensors["embeddings.scale"]
        elif group.name.startswith("layer-"):
            prefix = (
                "encoder.layer.0" if group.name == "layer-0000" else "encoder.layer.1"
            )
            state = state @ tensors[f"{prefix}.weight"] + tensors[f"{prefix}.bias"]
        elif group.name == "classifier":
            state = state @ tensors["classifier.weight"] + tensors["classifier.bias"]
        released_groups.append(group)

    report = loader.stream(apply_group)
    streamed_spans = _decode_synthetic_spans(note, state)

    assert streamed_spans == full_spans == [("NAME", 3, 14, "Alice Smith")]
    assert [group.name for group in released_groups] == [
        "embeddings",
        "layer-0000",
        "layer-0001",
        "classifier",
    ]
    assert all(group.released for group in released_groups)
    assert report.source_format == "safetensors"
    assert report.groups_loaded == 4
    assert report.tensors_loaded == len(weights)
    assert report.bytes_mapped == sum(array.nbytes for array in weights.values())
    assert report.peak_ram.within_budget is True


def test_safetensors_index_and_group_order_are_deterministic(tmp_path: Path) -> None:
    weights = {
        "encoder.layer.10.weight": np.asarray([10.0], dtype=np.float32),
        "encoder.layer.2.weight": np.asarray([2.0], dtype=np.float32),
        "encoder.layer.1.weight": np.asarray([1.0], dtype=np.float32),
        "classifier.bias": np.asarray([0.0], dtype=np.float32),
    }
    index_path = _write_sharded_safetensors(tmp_path, weights)
    loader = StreamingWeightLoader(
        index_path,
        ram_budget_bytes=16 * 1024,
        layers_per_group=2,
        rss_sampler=lambda: 1_000,
    )

    first = loader.layer_groups
    second = loader.layer_groups

    assert first == second
    assert [group.name for group in first] == [
        "layer-0001..layer-0002",
        "layer-0010..classifier",
    ]
    assert first[0].tensor_names == (
        "encoder.layer.1.weight",
        "encoder.layer.2.weight",
    )


def test_large_group_is_rejected_before_it_is_mapped(tmp_path: Path) -> None:
    weights = {
        "encoder.layer.0.weight": np.zeros(32 * 1024, dtype=np.float32),
    }
    index_path = _write_sharded_safetensors(tmp_path, weights)
    loader = StreamingWeightLoader(
        index_path,
        ram_budget_bytes=64 * 1024,
        rss_sampler=lambda: 5_000,
    )

    with pytest.raises(RamBudgetExceeded, match="mapping layer group") as exc:
        loader.stream(lambda group: pytest.fail("oversized group must not be mapped"))

    assert exc.value.budget_bytes == 64 * 1024
    assert exc.value.requested_bytes == weights["encoder.layer.0.weight"].nbytes
    assert loader.last_report is not None
    assert loader.last_report.groups_loaded == 0


def test_peak_ram_probe_measures_under_budget_and_aborts_on_spike() -> None:
    values = iter([1_000, 1_100, 1_250, 1_200])
    probe = PeakRamProbe(
        300,
        rss_sampler=lambda: next(values),
        poll_interval_seconds=None,
    )

    with probe:
        probe.reserve(100, "mapping a small layer")
        probe.checkpoint("applying a small layer")

    assert probe.report.peak_incremental_bytes == 250
    assert probe.report.within_budget is True

    spike_values = iter([1_000, 1_101, 1_101])
    with pytest.raises(RamBudgetExceeded, match="synthetic allocation"):
        with PeakRamProbe(
            100,
            rss_sampler=lambda: next(spike_values),
            poll_interval_seconds=None,
        ) as exceeded:
            exceeded.checkpoint("synthetic allocation")


def test_peak_ram_probe_uses_current_process_rss() -> None:
    with PeakRamProbe(RamBudget.from_mib(64)) as probe:
        payload = bytearray(512 * 1024)
        payload[::4096] = b"x" * len(payload[::4096])
        probe.checkpoint("measuring a synthetic mapped buffer")

    assert probe.report.baseline_rss_bytes > 0
    assert probe.report.peak_rss_bytes >= probe.report.baseline_rss_bytes
    assert probe.report.within_budget is True


def test_windows_rss_probe_configures_64_bit_process_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWindowsFunction:
        def __init__(self, callback):
            self.callback = callback
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.callback(*args)

    get_current_process = FakeWindowsFunction(lambda: 12_345)

    def fill_memory_counters(_process, counters_pointer, _size):
        counters = ctypes.cast(
            counters_pointer,
            ctypes.POINTER(ram_budget_module._ProcessMemoryCounters),
        ).contents
        counters.working_set_size = 987_654
        return 1

    get_process_memory_info = FakeWindowsFunction(fill_memory_counters)
    windll = SimpleNamespace(
        kernel32=SimpleNamespace(GetCurrentProcess=get_current_process),
        psapi=SimpleNamespace(GetProcessMemoryInfo=get_process_memory_info),
    )
    monkeypatch.setattr(ram_budget_module.ctypes, "windll", windll, raising=False)

    assert ram_budget_module._windows_current_rss_bytes() == 987_654
    assert get_current_process.argtypes == ()
    assert get_current_process.restype is ctypes.c_void_p
    assert get_process_memory_info.argtypes == (
        ctypes.c_void_p,
        ctypes.POINTER(ram_budget_module._ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    assert get_process_memory_info.restype is ctypes.c_int


def test_peak_ram_probe_rejects_a_transient_sampled_spike() -> None:
    rss = {"value": 1_000}

    with pytest.raises(RamBudgetExceeded, match="sampled transient spike"):
        with PeakRamProbe(
            100,
            rss_sampler=lambda: rss["value"],
            poll_interval_seconds=0.001,
        ) as probe:
            rss["value"] = 1_250
            deadline = time.monotonic() + 0.5
            while probe.report.peak_rss_bytes < 1_250 and time.monotonic() < deadline:
                time.sleep(0.002)
            assert probe.report.peak_rss_bytes == 1_250
            rss["value"] = 1_000
            probe.checkpoint("checking a sampled transient spike")


def test_peak_ram_probe_fails_closed_without_rss_measurement() -> None:
    with pytest.raises(RamProbeUnavailable, match="refusing to load weights"):
        with PeakRamProbe(1_024, rss_sampler=lambda: None):
            pass


def test_loader_rejects_network_and_escaping_shard_paths(tmp_path: Path) -> None:
    with pytest.raises(LocalWeightsRequired, match="local path"):
        StreamingWeightLoader("https://example.test/model.safetensors", ram_budget=1)

    outside = tmp_path / "outside.safetensors"
    _write_safetensors(outside, {"layer.weight": np.asarray([1.0], dtype=np.float32)})
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    index_path = artifact / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"weight_map": {"layer.weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    loader = StreamingWeightLoader(
        index_path,
        ram_budget_bytes=4_096,
        rss_sampler=lambda: 1_000,
    )

    with pytest.raises(LocalWeightsRequired, match="escapes"):
        loader.layer_groups


def test_onnx_external_data_is_streamed_by_layer(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    initializers = [
        onnx.numpy_helper.from_array(
            np.eye(2, dtype=np.float32),
            name="encoder.layer.0.weight",
        ),
        onnx.numpy_helper.from_array(
            np.asarray([0.25, -0.25], dtype=np.float32),
            name="classifier.bias",
        ),
    ]
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "streaming-test", [], [], initializers)
    )
    model_path = tmp_path / "model.onnx"
    onnx.save_model(
        model,
        str(model_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    loader = StreamingWeightLoader(
        model_path,
        ram_budget_bytes=64 * 1024,
        rss_sampler=lambda: 1_000,
    )
    copied: dict[str, np.ndarray] = {}

    report = loader.stream(
        lambda group: copied.update(
            {name: np.array(value, copy=True) for name, value in group.tensors.items()}
        )
    )

    np.testing.assert_array_equal(
        copied["encoder.layer.0.weight"],
        np.eye(2, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        copied["classifier.bias"],
        np.asarray([0.25, -0.25], dtype=np.float32),
    )
    assert report.source_format == "onnx-external-data"
    assert [group.name for group in loader.layer_groups] == [
        "layer-0000",
        "classifier",
    ]
    assert report.peak_ram.within_budget is True


def test_inline_onnx_weights_are_rejected_as_non_streamable(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    initializer = onnx.numpy_helper.from_array(
        np.asarray([1.0], dtype=np.float32),
        name="encoder.layer.0.weight",
    )
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "inline-test", [], [], [initializer])
    )
    model_path = tmp_path / "model.onnx"
    onnx.save(model, str(model_path))
    loader = StreamingWeightLoader(
        model_path,
        ram_budget_bytes=64 * 1024,
        rss_sampler=lambda: 1_000,
    )

    with pytest.raises(ShardFormatError, match="inline"):
        loader.layer_groups


def _run_full(
    token_features: np.ndarray,
    weights: Mapping[str, np.ndarray],
) -> np.ndarray:
    state = token_features * weights["embeddings.scale"]
    for layer in range(2):
        prefix = f"encoder.layer.{layer}"
        state = state @ weights[f"{prefix}.weight"] + weights[f"{prefix}.bias"]
    return state @ weights["classifier.weight"] + weights["classifier.bias"]


def _decode_synthetic_spans(
    note: str,
    logits: np.ndarray,
) -> list[tuple[str, int, int, str]]:
    labels = ("O", "B-NAME", "I-NAME")
    offsets = ((0, 2), (3, 8), (9, 14))
    spans: list[tuple[str, int, int, str]] = []
    active_start: int | None = None
    active_end = 0
    for label_id, (start, end) in zip(logits.argmax(axis=-1), offsets):
        label = labels[int(label_id)]
        if label == "B-NAME":
            if active_start is not None:
                spans.append(
                    ("NAME", active_start, active_end, note[active_start:active_end])
                )
            active_start, active_end = start, end
        elif label == "I-NAME" and active_start is not None:
            active_end = end
        elif active_start is not None:
            spans.append(
                ("NAME", active_start, active_end, note[active_start:active_end])
            )
            active_start = None
    if active_start is not None:
        spans.append(("NAME", active_start, active_end, note[active_start:active_end]))
    return spans


def _write_sharded_safetensors(
    root: Path,
    tensors: Mapping[str, np.ndarray],
) -> Path:
    names = sorted(tensors, reverse=True)
    split = max(len(names) // 2, 1)
    shards = (
        ("model-00001-of-00002.safetensors", names[:split]),
        ("model-00002-of-00002.safetensors", names[split:]),
    )
    weight_map: dict[str, str] = {}
    for filename, shard_names in shards:
        if not shard_names:
            continue
        _write_safetensors(
            root / filename,
            {name: tensors[name] for name in shard_names},
        )
        for name in shard_names:
            weight_map[name] = filename
    index_path = root / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )
    return index_path


def _write_safetensors(
    path: Path,
    tensors: Mapping[str, np.ndarray],
) -> None:
    header: dict[str, Any] = {}
    chunks: list[bytes] = []
    offset = 0
    for name, value in tensors.items():
        array = np.asarray(value)
        data = array.tobytes(order="C")
        header[name] = {
            "dtype": _safetensors_dtype(array.dtype),
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(
        struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(chunks)
    )


def _safetensors_dtype(dtype: np.dtype[Any]) -> str:
    mapping = {
        np.dtype("float32"): "F32",
        np.dtype("float64"): "F64",
        np.dtype("float16"): "F16",
        np.dtype("int64"): "I64",
        np.dtype("int32"): "I32",
        np.dtype("uint8"): "U8",
        np.dtype("bool"): "BOOL",
    }
    return mapping[np.dtype(dtype)]
