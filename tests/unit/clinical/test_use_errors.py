import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

import pytest

from openmed.clinical.use_errors import UseErrorCode, UseErrorRecorder
from openmed.core.clinician_mode import (
    ClinicianModeError,
    clinician_mode,
    require_clinician_mode,
)
from openmed.core.disclaimer import OpenMedDisclaimer
from openmed.multimodal.notices import (
    NOTICE_CATALOG,
    DraftReviewResult,
    MultimodalNoticeError,
    NoticeKind,
)


def test_count_only_after_explicit_capture_and_restore():
    recorder = UseErrorRecorder()
    with pytest.raises(ClinicianModeError):
        require_clinician_mode("ecg")
    assert recorder.snapshot() == {"schema_version": 1, "counts": []}
    with recorder.capture():
        captured = copy_context()
        for _ in range(2):
            with pytest.raises(ClinicianModeError):
                require_clinician_mode(
                    "openmed.multimodal.ecg.interval_candidates.extract_interval_candidates"
                )
        with pytest.raises(ClinicianModeError), clinician_mode(purpose="review"):
            pytest.fail()
        with pytest.raises(ValueError, match="acknowledgement_required"):
            OpenMedDisclaimer.for_capability("decisions").require_acknowledgement(
                None, capability="decisions"
            )
        with pytest.raises(
            MultimodalNoticeError, match="explicit reviewer confirmation required"
        ):
            DraftReviewResult(
                "a" * 64, NOTICE_CATALOG[NoticeKind.DRAFT]
            ).require_reviewer_confirmation(reviewer_confirmed=False)
    before = recorder.snapshot()
    with pytest.raises(ClinicianModeError):
        captured.run(require_clinician_mode, "ecg")
    assert recorder.snapshot() == before
    assert before == {
        "schema_version": 1,
        "counts": [
            {
                "code": "acknowledgement_skipped",
                "capability": "clinician_mode",
                "tier": "restricted",
                "count": 1,
            },
            {
                "code": "acknowledgement_skipped",
                "capability": "decisions",
                "tier": "restricted",
                "count": 1,
            },
            {
                "code": "acknowledgement_skipped",
                "capability": "notice",
                "tier": "assistive_review",
                "count": 1,
            },
            {
                "code": "restricted_capability_refused",
                "capability": "ecg",
                "tier": "restricted",
                "count": 2,
            },
        ],
    }


def test_unknown_dimensions_and_free_text_are_rejected():
    recorder = UseErrorRecorder()
    for args in [
        ("private-marker", "decisions", "restricted"),
        ("export_before_review", "private-marker", "restricted"),
        ("export_before_review", "decisions", "private-marker"),
    ]:
        with pytest.raises(ValueError) as raised:
            recorder.record(*args)
        assert "private-marker" not in str(raised.value)
    with pytest.raises(TypeError):
        recorder.record(
            "export_before_review", "decisions", "restricted", note="private-marker"
        )
    assert recorder.snapshot()["counts"] == []


def test_concurrent_events_and_detached_snapshots(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    recorder = UseErrorRecorder()

    def record(_):
        recorder.record(UseErrorCode.EXPORT_BEFORE_REVIEW, "decisions", "restricted")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(record, range(75)))
    snapshot = recorder.snapshot()
    assert snapshot["counts"][0]["count"] == 75
    snapshot["counts"].clear()
    assert recorder.snapshot()["counts"][0]["count"] == 75
    assert "private-marker" not in json.dumps(recorder.snapshot())
    assert not list(tmp_path.iterdir())
    assert not caplog.records


def test_swift_shared_code_vector():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    vector = json.loads(
        (root / "tests/fixtures/parity/use_error_codes_v1.json").read_bytes()
    )
    assert sorted(code.value for code in UseErrorCode) == vector
