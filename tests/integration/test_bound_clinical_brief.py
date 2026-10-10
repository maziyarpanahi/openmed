"""Offline integration of structured generation with the existing safety gates."""

import importlib
import json
import socket
from dataclasses import replace

import pytest

from openmed.clinical.brief import BriefRefusal, build_clinical_brief
from tests.unit.clinical.test_brief import fixture_context
from tests.unit.clinical.test_brief_bindings import SyntheticBoundGenerator

pytestmark = pytest.mark.integration


def test_bound_brief_integrates_offline_generation_and_review(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    result, context = fixture_context()
    backend = SyntheticBoundGenerator()
    brief = build_clinical_brief(result, context=context, model=backend)
    assert brief.refusal_reason is None
    assert len(brief.citations) == len(brief.verdicts) == 3
    assert brief.to_dict()["status"] == "needs_review"
    assert brief.metrics["leakage"]["passed"]
    assert brief.metrics["citation_support"]["deterministic_span_checks"]["passed"]
    assert brief.summary not in json.dumps(brief.to_dict())


@pytest.mark.parametrize(
    "module,name",
    [
        ("nli_assertion_pairs", "build_nli_pair"),
        ("nli_temporal_pairs", "build_temporal_nli_pair"),
        ("nli_experiencer_pairs", "build_experiencer_nli_pair"),
        ("summary_citations", "compute_summary_citation_metrics"),
        ("citation_boundaries", "validate_citation_boundaries"),
        ("citation_minimality", "check_citation_minimality"),
    ],
)
def test_bound_claims_cannot_skip_required_guards(monkeypatch, module, name):
    def fail(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_PRIVATE_ERROR")

    monkeypatch.setattr(
        importlib.import_module("openmed.clinical." + module), name, fail
    )
    value, context = fixture_context()
    brief = build_clinical_brief(
        value, context=context, model=SyntheticBoundGenerator()
    )
    assert brief.refusal_reason is BriefRefusal.STAGE_FAILED
    assert brief.summary == ""
    assert brief.citations == ()
    assert "SYNTHETIC_PRIVATE_ERROR" not in json.dumps(brief.to_dict())


def test_bound_generation_cannot_supply_uncalibrated_confidence():
    value, context = fixture_context()
    brief = build_clinical_brief(
        value,
        model=SyntheticBoundGenerator(),
        context=replace(context, nli_predict=lambda *_: {"confidence": 1.0}),
    )
    assert brief.refusal_reason is BriefRefusal.NLI_UNAVAILABLE
    assert brief.summary == ""
