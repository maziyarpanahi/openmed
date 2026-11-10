"""Synthetic local stream-to-review handoff without external providers."""

import io

import pytest

from openmed.multimodal.edf import EdfError, read_edf
from tests.fixtures.multimodal.edf import synthetic_edf


@pytest.mark.integration
def test_local_discontinuous_stream_review_handoff():
    stream = io.BytesIO(synthetic_edf(kind="EDF+D", onsets=("+0", "+5")))
    result = read_edf(stream, start_seconds=5, end_seconds=6)
    assert result.records[0].record_index == 1
    assert result.records[0].signals[0].physical_samples == (1, 0, -0.5, -1)
    assert result.gaps[0].start_seconds == 1
    assert result.gaps[0].end_seconds == 5
    assert result.report()["annotation_count"] == 2
    with pytest.raises(EdfError, match="edf_review_required"):
        result.reviewed(confirmed=False)
    assert result.reviewed(confirmed=True).reviewer_confirmed
    assert stream.tell() == 0
