"""Offline tests for actor-backed Ray Data batch de-identification."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations.ray_map_batches import (
    RayDeidentifyBatch,
    map_batches_deidentify,
)


class _FakeBatchPipeline:
    def __init__(self, stats: Any | None = None) -> None:
        self._stats = stats

    def process_texts(self, texts, ids=None):
        if self._stats is not None:
            import ray

            ray.get(self._stats.record_batch.remote(len(texts)))
        items = [
            SimpleNamespace(
                id=ids[index] if ids else f"item_{index}",
                success=True,
                result=SimpleNamespace(
                    deidentified_text=text.replace("Jane Roe", "[PERSON]")
                    .replace("John Doe", "[PERSON]")
                    .replace("555-0100", "[PHONE]")
                    .replace("555-0199", "[PHONE]")
                ),
            )
            for index, text in enumerate(texts)
        ]
        return SimpleNamespace(items=items)


@pytest.mark.parametrize("batch_format", ["pandas", "pyarrow"])
def test_callable_preserves_batch_shape_and_non_target_columns(batch_format):
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c"],
            "note": ["Jane Roe called 555-0100", None, "No identifiers"],
            "score": [1, 2, 3],
        }
    )
    batch = frame
    if batch_format == "pyarrow":
        pa = pytest.importorskip("pyarrow", exc_type=ImportError)
        batch = pa.Table.from_pandas(frame, preserve_index=False)

    actor = RayDeidentifyBatch(
        column="note",
        policy_profile="hipaa_safe_harbor",
        pipeline_factory=lambda **_: _FakeBatchPipeline(),
    )
    output = actor(batch)
    output_frame = output if batch_format == "pandas" else output.to_pandas()

    assert len(output_frame) == len(frame)
    assert output_frame["record_id"].tolist() == frame["record_id"].tolist()
    assert output_frame["score"].tolist() == frame["score"].tolist()
    assert output_frame["note"].tolist()[0] == "[PERSON] called [PHONE]"
    assert pd.isna(output_frame["note"].tolist()[1])
    assert output_frame["note"].tolist()[2] == "No identifiers"


def test_ray_dataset_uses_one_loaded_pipeline_per_actor():
    ray = pytest.importorskip("ray", exc_type=ImportError)
    pytest.importorskip("ray.data", exc_type=ImportError)

    class PipelineStats:
        def __init__(self) -> None:
            self.loads: list[tuple[str, str | None]] = []
            self.batch_sizes: list[int] = []

        def record_load(self, model_name: str, policy_profile: str | None) -> None:
            self.loads.append((model_name, policy_profile))

        def record_batch(self, batch_size: int) -> None:
            self.batch_sizes.append(batch_size)

        def snapshot(self) -> dict[str, Any]:
            return {
                "loads": list(self.loads),
                "batch_sizes": list(self.batch_sizes),
            }

    class FakeBatchPipeline:
        def __init__(self, stats: Any) -> None:
            self._stats = stats

        def process_texts(self, texts, ids=None):
            ray.get(self._stats.record_batch.remote(len(texts)))
            items = [
                SimpleNamespace(
                    id=ids[index] if ids else f"item_{index}",
                    success=True,
                    result=SimpleNamespace(
                        deidentified_text=text.replace("Jane Roe", "[PERSON]")
                        .replace("John Doe", "[PERSON]")
                        .replace("555-0100", "[PHONE]")
                        .replace("555-0199", "[PHONE]")
                    ),
                )
                for index, text in enumerate(texts)
            ]
            return SimpleNamespace(items=items)

    class FakePipelineFactory:
        def __init__(self, stats: Any) -> None:
            self._stats = stats

        def __call__(self, **kwargs):
            ray.get(
                self._stats.record_load.remote(
                    kwargs["model_name"],
                    kwargs["policy_profile"],
                )
            )
            return FakeBatchPipeline(self._stats)

    ray.shutdown()
    try:
        ray.init(local_mode=True, num_cpus=2, include_dashboard=False)
    except RuntimeError as exc:
        if "local_mode` is no longer supported" not in str(exc):
            raise
        ray.init(address="local", num_cpus=2, include_dashboard=False)
    try:
        stats_type = ray.remote(num_cpus=0)(PipelineStats)
        stats = stats_type.remote()
        records = [
            {
                "record_id": "a",
                "note": "Jane Roe called 555-0100",
                "score": 1,
            },
            {"record_id": "b", "note": "No identifiers", "score": 2},
            {
                "record_id": "c",
                "note": "John Doe called 555-0199",
                "score": 3,
            },
            {"record_id": "d", "note": "Follow-up", "score": 4},
            {"record_id": "e", "note": "Discharge summary", "score": 5},
        ]
        dataset = ray.data.from_items(records, override_num_blocks=3)

        redacted = map_batches_deidentify(
            dataset,
            column="note",
            policy_profile="strict_no_leak",
            model_name="fixture-pii-model",
            batch_size=2,
            batch_format="pandas",
            concurrency=1,
            pipeline_factory=FakePipelineFactory(stats),
        ).materialize()
        output = redacted.to_pandas()
        snapshot = ray.get(stats.snapshot.remote())

        assert len(output) == len(records)
        assert output["record_id"].tolist() == [row["record_id"] for row in records]
        assert output["score"].tolist() == [row["score"] for row in records]
        assert "Jane Roe" not in "\n".join(output["note"])
        assert "John Doe" not in "\n".join(output["note"])
        assert snapshot["loads"] == [("fixture-pii-model", "strict_no_leak")]
        assert sum(snapshot["batch_sizes"]) == len(records)
        assert max(snapshot["batch_sizes"]) <= 2
    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    ("concurrency", "message"),
    [
        (0, "at least 1"),
        ((2, 1), "1 <= min <= max"),
        ((1, 3, 4), "between min and max"),
    ],
)
def test_invalid_actor_pool_concurrency_is_rejected(concurrency, message):
    class Strategy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    from openmed.integrations.ray_map_batches import _actor_pool_strategy

    with pytest.raises(ValueError, match=message):
        _actor_pool_strategy(Strategy, concurrency)
