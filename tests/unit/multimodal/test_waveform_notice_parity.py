"""Keep non-diagnostic waveform notices aligned with the shared Swift fixture."""

import json
from pathlib import Path

import pytest

from openmed.multimodal.aecg import AECG_NOTICE
from openmed.multimodal.edf import EDF_NOTICE
from openmed.multimodal.wfdb import WFDB_NOTICE

FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests/fixtures/parity/waveform_notices_v1.json"
)


@pytest.mark.parametrize(
    ("kind", "notice"),
    [("edf", EDF_NOTICE), ("wfdb", WFDB_NOTICE), ("aecg", AECG_NOTICE)],
)
def test_waveform_notice_matches_shared_fixture(kind: str, notice: str) -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert set(fixture) == {"version", "edf", "wfdb", "aecg"}
    assert fixture["version"] == 1
    assert notice == fixture[kind]
