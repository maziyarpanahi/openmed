"""Synthetic local-file WFDB roundtrip, without external assets or services."""

import pytest

from openmed.multimodal.wfdb import read_wfdb_record
from tests.fixtures.multimodal.wfdb import (
    FORMAT_CASES,
    SYNTHETIC_ANNOTATIONS,
    synthetic_header,
)


@pytest.mark.integration
@pytest.mark.parametrize("fmt,payload,first,second", FORMAT_CASES)
def test_caller_owned_local_file_window(tmp_path, fmt, payload, first, second):
    # Header names intentionally do not match the supplied files: no resolution.
    header_path = tmp_path / "synthetic.hea"
    signal_path = tmp_path / "supplied.dat"
    annotation_path = tmp_path / "supplied.atr"
    header_path.write_bytes(synthetic_header(fmt, first, second))
    signal_path.write_bytes(payload)
    annotation_path.write_bytes(SYNTHETIC_ANNOTATIONS)
    with (
        header_path.open("rb") as header,
        signal_path.open("rb") as signal,
        annotation_path.open("rb") as annotations,
    ):
        result = read_wfdb_record(
            header, [signal], annotations=annotations, start_sample=1, sample_count=2
        )
        assert result.signals[0].samples == first[1:]
        assert result.signals[1].samples == second[1:]
        assert result.annotations.sample_positions == (2,)
        assert header.tell() == signal.tell() == annotations.tell() == 0
        assert all(s.checksum_verified for s in result.signals)
        result.require_reviewer_confirmation(confirmed=True)
