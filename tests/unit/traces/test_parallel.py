"""Synthetic offline tests for deterministic parallel trace execution."""

from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path

import pytest

import openmed.traces.parallel as parallel
from openmed.traces import (
    TraceProcessingError,
    partition_trace_files,
    run_trace_files,
)

SYNTHETIC_SECRET = "synthetic-sensitive-value-0007"


def _fresh_store() -> dict[str, int]:
    return {}


def _ordered_handler(path: Path, store: dict[str, int]) -> dict[str, int | str]:
    """Delay higher-priority inputs so completion order differs from input order."""
    index = int(path.read_text(encoding="utf-8"))
    time.sleep((5 - index) * 0.01)
    store["records"] = store.get("records", 0) + 1
    return {"input": index, "store_records": store["records"]}


def _raising_handler(path: Path, store: dict[str, int]) -> None:
    del store
    raise RuntimeError(f"synthetic detail {SYNTHETIC_SECRET} from {path.read_text()}")


def _write_inputs(root: Path, count: int = 6) -> list[Path]:
    paths = []
    for index in range(count):
        path = root / f"trace-{index:03d}.jsonl"
        path.write_text(str(index), encoding="utf-8")
        paths.append(path)
    return paths


def test_partition_preserves_order_and_bounds_shards() -> None:
    paths = tuple(Path(f"synthetic-{index:03d}.jsonl") for index in range(5))

    shards = partition_trace_files(paths, shard_size=2)

    assert [shard.input_indices for shard in shards] == [
        (0, 1),
        (2, 3),
        (4,),
    ]
    assert [path for shard in shards for path in shard.files] == list(paths)
    assert all(shard.file_count <= 2 for shard in shards)
    assert "synthetic-000" not in repr(shards[0])
    assert "files" not in json.dumps(shards[0].to_dict())


def test_partition_sanitizes_input_iterator_failures() -> None:
    def failing_paths():
        yield Path("synthetic-000.jsonl")
        raise RuntimeError(SYNTHETIC_SECRET)

    with pytest.raises(parallel.TraceInputError) as raised:
        partition_trace_files(failing_paths())

    assert SYNTHETIC_SECRET not in str(raised.value)


def test_sequential_execution_uses_a_fresh_store_and_input_order(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path)

    result = run_trace_files(
        paths,
        _ordered_handler,
        store_factory=_fresh_store,
        shard_size=2,
        max_workers=1,
        use_processes=False,
    )

    assert result.execution_mode == "sequential"
    assert result.fallback_reason is None
    assert [item.value["input"] for item in result.items] == list(range(6))
    assert [item.value["store_records"] for item in result.items] == [1] * 6
    assert result.is_complete


@pytest.mark.skipif(
    "spawn" not in multiprocessing.get_all_start_methods(),
    reason="requires the spawn multiprocessing start method",
)
def test_process_execution_merges_slow_shards_in_input_order(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path)

    result = run_trace_files(
        paths,
        _ordered_handler,
        store_factory=_fresh_store,
        shard_size=1,
        max_workers=3,
    )

    assert result.execution_mode == "processes"
    assert result.worker_count == 3
    assert [item.value["input"] for item in result.items] == list(range(6))
    assert [item.value["store_records"] for item in result.items] == [1] * 6


def test_unpicklable_handler_falls_back_without_changing_results(
    tmp_path: Path,
) -> None:
    paths = _write_inputs(tmp_path, count=2)
    unsafe_handler = lambda path, store: (path.stem, len(store))

    result = run_trace_files(paths, unsafe_handler, max_workers=2)

    assert result.execution_mode == "sequential"
    assert result.fallback_reason == "unsafe"
    assert result.ordered_values == (("trace-000", 0), ("trace-001", 0))


def test_unavailable_spawn_falls_back_to_one_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path, count=2)
    monkeypatch.setattr(parallel.multiprocessing, "get_all_start_methods", lambda: [])

    result = run_trace_files(
        paths,
        _ordered_handler,
        store_factory=_fresh_store,
        max_workers=2,
    )

    assert result.execution_mode == "sequential"
    assert result.worker_count == 1
    assert result.fallback_reason == "unavailable"
    assert result.is_complete


def test_pool_startup_failure_falls_back_without_leaking_exception_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_inputs(tmp_path, count=2)

    def unavailable_pool(**kwargs: object) -> None:
        del kwargs
        raise OSError(f"synthetic pool failure {SYNTHETIC_SECRET}")

    monkeypatch.setattr(parallel, "ProcessPoolExecutor", unavailable_pool)

    result = run_trace_files(
        paths,
        _ordered_handler,
        store_factory=_fresh_store,
        max_workers=2,
    )

    assert result.fallback_reason == "startup_failed"
    assert result.is_complete
    assert SYNTHETIC_SECRET not in json.dumps(result.to_dict())
    assert SYNTHETIC_SECRET not in repr(result)


def test_failures_publish_only_type_names_and_safe_reports(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, count=1)
    paths[0].write_text(SYNTHETIC_SECRET, encoding="utf-8")

    result = run_trace_files(
        paths,
        _raising_handler,
        store_factory=_fresh_store,
        use_processes=False,
    )

    assert not result.is_complete
    assert result.items[0].error_type == "RuntimeError"
    safe_report = json.dumps(result.to_dict())
    assert SYNTHETIC_SECRET not in safe_report
    assert SYNTHETIC_SECRET not in repr(result)

    with pytest.raises(TraceProcessingError) as raised:
        result.raise_for_errors()
    assert SYNTHETIC_SECRET not in str(raised.value)
    assert raised.value.input_index == 0
    assert raised.value.error_type == "RuntimeError"


def test_dynamic_exception_type_cannot_leak_sensitive_content(tmp_path: Path) -> None:
    sensitive_error = type(SYNTHETIC_SECRET.replace("-", "_"), (RuntimeError,), {})

    def raising_handler(path: Path, store: dict[str, int]) -> None:
        del path, store
        raise sensitive_error()

    result = run_trace_files(
        _write_inputs(tmp_path, count=1),
        raising_handler,
        use_processes=False,
    )

    assert result.items[0].error_type == parallel.UNKNOWN_ERROR_TYPE
    assert SYNTHETIC_SECRET.replace("-", "_") not in json.dumps(result.to_dict())
    assert SYNTHETIC_SECRET.replace("-", "_") not in repr(result.items[0])
