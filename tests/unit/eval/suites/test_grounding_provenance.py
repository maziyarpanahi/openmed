"""Tests for the grounding provenance round-trip evaluation suite.

All vocabulary content is synthetic and algorithmically generated; no real
patient data or licensed terminology is used.
"""

from __future__ import annotations

import openmed.eval.suites as suites
from openmed.eval.suites.grounding_provenance import (
    GROUNDING_PROVENANCE,
    build_synthetic_gold_set,
    grounding_provenance_metadata,
    run_grounding_provenance_roundtrip,
)


def test_roundtrip_rederives_version_and_method_for_every_code():
    report = run_grounding_provenance_roundtrip()

    assert report["passed"] is True
    assert report["mismatches"] == []
    assert report["emitted_code_count"] == report["case_count"]
    assert report["reproduced"] == report["emitted_code_count"]
    assert report["phi_safe"] is True
    assert report["leaked_raw_text"] == []


def test_roundtrip_is_deterministic():
    first = run_grounding_provenance_roundtrip()
    second = run_grounding_provenance_roundtrip()

    assert first == second
    assert first["repro_hash"] == second["repro_hash"]


def test_gold_set_offsets_are_non_overlapping_and_ordered():
    cases = build_synthetic_gold_set()

    assert cases
    previous_end = -1
    for case in cases:
        assert case.span.start > previous_end
        assert case.span.end > case.span.start
        assert case.method in {
            "sparse",
            "dense",
            "rerank",
            "composite",
            "post-coordinated",
        }
        previous_end = case.span.end


def test_suite_is_not_registered_in_default_suites():
    # The provenance gate must not enroll in the model-benchmark registry, whose
    # membership is pinned by an exact-tuple assertion in the eval metrics tests.
    assert GROUNDING_PROVENANCE not in suites.DEFAULT_SUITES
    assert grounding_provenance_metadata()["registered_in_default_suites"] is False
