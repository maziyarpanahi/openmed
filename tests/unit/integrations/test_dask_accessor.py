from __future__ import annotations

from types import SimpleNamespace

import pytest


def _fake_batch_result(texts):
    items = [
        SimpleNamespace(
            success=True,
            result=SimpleNamespace(
                deidentified_text=text.replace("Jane Roe", "[PERSON]")
                .replace("John Doe", "[PERSON]")
                .replace("555-0100", "[PHONE]")
                .replace("555-0199", "[PHONE]")
            ),
        )
        for text in texts
    ]
    return SimpleNamespace(items=items)


def test_dataframe_accessor_redacts_once_per_partition_and_preserves_metadata():
    dd = pytest.importorskip("dask.dataframe", exc_type=ImportError)
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    import openmed.integrations.dask_accessor  # noqa: F401

    calls = []

    def fake_process_batch(texts, **kwargs):
        calls.append((list(texts), kwargs))
        assert kwargs["operation"] == "deidentify"
        assert kwargs["method"] == "mask"
        assert kwargs["policy"] == "hipaa_safe_harbor"
        return _fake_batch_result(texts)

    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "note": [
                "Patient Jane Roe called 555-0100.",
                "No identifiers here.",
                "John Doe left voicemail at 555-0199.",
                "Follow-up contains no seeded PHI.",
            ],
            "age": [42, 73, 55, 61],
        },
        index=pd.Index([10, 11, 20, 21], name="row_id"),
    )
    dask_frame = dd.from_pandas(frame, npartitions=2)

    redacted_frame = dask_frame.deid.deidentify(
        target_columns="note",
        policy="hipaa_safe_harbor",
        process_batch_fn=fake_process_batch,
    )
    computed = redacted_frame.compute(scheduler="synchronous")

    assert redacted_frame.divisions == dask_frame.divisions
    assert list(computed.columns) == list(frame.columns)
    assert computed.index.tolist() == frame.index.tolist()
    assert computed["record_id"].tolist() == frame["record_id"].tolist()
    assert computed["age"].tolist() == frame["age"].tolist()
    assert "Jane Roe" not in "\n".join(computed["note"])
    assert "John Doe" not in "\n".join(computed["note"])
    assert "555-0100" not in "\n".join(computed["note"])
    assert "555-0199" not in "\n".join(computed["note"])
    assert len(calls) == dask_frame.npartitions
    assert sum(len(texts) for texts, _ in calls) == len(frame)


def test_series_accessor_redacts_series_partitions():
    dd = pytest.importorskip("dask.dataframe", exc_type=ImportError)
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    import openmed.integrations.dask_accessor  # noqa: F401

    calls = []

    def fake_process_batch(texts, **kwargs):
        calls.append(list(texts))
        assert kwargs["policy"] == "hipaa_safe_harbor"
        return _fake_batch_result(texts)

    series = pd.Series(
        [
            "Patient Jane Roe called 555-0100.",
            "John Doe left voicemail at 555-0199.",
            "No identifiers.",
        ],
        index=pd.Index([3, 4, 5], name="row_id"),
        name="note",
    )
    dask_series = dd.from_pandas(series, npartitions=2)

    redacted_series = dask_series.deid.deidentify(
        policy="hipaa_safe_harbor",
        process_batch_fn=fake_process_batch,
    )
    computed = redacted_series.compute(scheduler="synchronous")

    assert redacted_series.divisions == dask_series.divisions
    assert computed.index.tolist() == series.index.tolist()
    assert computed.name == "note"
    assert "Jane Roe" not in "\n".join(computed)
    assert "John Doe" not in "\n".join(computed)
    assert len(calls) == dask_series.npartitions


def test_map_partitions_helper_requires_dataframe_target_columns():
    dd = pytest.importorskip("dask.dataframe", exc_type=ImportError)
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    from openmed.integrations.dask_accessor import map_partitions_deidentify

    dask_frame = dd.from_pandas(pd.DataFrame({"note": ["Patient Jane Roe"]}), 1)

    with pytest.raises(ValueError, match="target_columns is required"):
        map_partitions_deidentify(dask_frame)
