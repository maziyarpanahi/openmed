"""Version guard of the async job metadata store document."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from openmed.service import jobs  # noqa: E402

_RECORD = {
    "created_at": "2026-01-01T00:00:00Z",
    "expires_at": "2099-01-01T00:00:00Z",
    "id": "job-0001",
    "status": "done",
    "updated_at": "2026-01-01T00:00:00Z",
}


def test_previous_release_layout_is_read_and_rewritten_explicitly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": {"job-0001": _RECORD}}), encoding="utf-8")

    store = jobs.LocalJobStore(path)
    assert store.get("job-0001") is not None
    store.update("job-0001", status="done")

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == jobs.JOB_STORE_SCHEMA_VERSION == 1
    assert set(persisted) == {"jobs", "schema_version"}
    assert jobs.LocalJobStore(path).get("job-0001") is not None


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ({"jobs": {}, "schema_version": 2}, "job_store_schema_unsupported"),
        ({"jobs": {}, "schema_version": 0}, "job_store_schema_unsupported"),
        ({"jobs": {}, "schema_version": True}, "job_store_schema_unsupported"),
        ({"jobs": {}, "schema_version": "1"}, "job_store_schema_unsupported"),
        ({"jobs": {}, "principal_index": {}}, "job_store_unknown_fields"),
    ],
)
def test_newer_or_unknown_documents_are_refused_before_any_write(
    tmp_path: Path, document: dict[str, object], code: str
) -> None:
    path = tmp_path / "jobs.json"
    raw = json.dumps(document)
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(jobs.JobStoreSchemaError) as caught:
        jobs.LocalJobStore(path)

    assert caught.value.code == code
    assert isinstance(caught.value, ValueError)
    assert str(tmp_path) not in str(caught.value)
    assert path.read_text(encoding="utf-8") == raw
