"""Connected offline synthetic reference, source lineage and terminal fault checks."""

import json
import socket
from dataclasses import replace

import pytest

pytest.importorskip("PIL")

from openmed.multimodal.reference_workflow import run_reference_workflow
from tests.fixtures.multimodal.reference_workflow import (
    CONTROL,
    PRIVATE,
    TRANSCRIPT,
    reference_fixture,
)

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def run(fault=None):
    inputs, controls = reference_fixture(fault, observe_buffers=True)
    report = run_reference_workflow(inputs)
    serialized = json.dumps(report)
    for value in (PRIVATE, CONTROL, TRANSCRIPT):
        assert value not in serialized
    assert report["complete"] is report["clinical_write_allowed"] is False
    assert report["all_data_erased"] is False
    assert report["session"]["owned_bytes"] == 0
    for view in controls.borrowed_audio:
        assert bytes(view) == bytes(len(view))
    return report, controls


def test_connected_reference_reaches_unapproved_review_and_cleans_owned_buffers():
    report, controls = run()
    assert report["reached_review"], report
    assert (controls.vlm_calls, controls.asr_calls, controls.screen_calls) == (1, 1, 1)
    assert report["status"] == "review_required"
    assert report["review"]["clinical_approval"] is False
    assert report["review"]["note_created"] is False
    assert (
        report["audio_retention"]["passed"] is False
    )  # Declared external fixture copies.
    assert report["session"]["deletion_verified"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "malformed",
        "wrong_geometry",
        "truncated_audio",
        "missing_mask",
        "memory",
        "thermal",
        "post_thermal",
        "provider_exception",
        "asr_exception",
        "unresolved_phi",
        "expiry",
        "audio_memory",
        "asr_memory",
        "screen_thermal",
        "slow_asr",
    ],
)
def test_failures_never_reach_review_and_destroy_owned_copies(fault):
    report, controls = run(fault)
    assert report["status"] == "held"
    assert not report["reached_review"] and report["review"] is None
    if fault in {"memory", "thermal"}:
        assert controls.vlm_calls == controls.asr_calls == 0
    if fault == "post_thermal":
        assert controls.vlm_calls == 1 and controls.asr_calls == 0


@pytest.mark.parametrize(
    "stage",
    [
        "preflight",
        "decode",
        "redaction",
        "visual_inference",
        "audio_inference",
        "transcript_screen",
        "evidence_fusion",
        "review",
    ],
)
def test_cancellation_at_each_boundary_is_terminal(stage):
    report, _ = run("cancel:" + stage)
    assert report["reason"] == "cancelled"
    assert not report["reached_review"] and report["review"] is None
    assert report["stages"][-1] == stage


def test_changed_audio_is_not_replayed_as_success():
    inputs, _ = reference_fixture()
    changed = bytearray(inputs.audio)
    changed[-1] ^= 1
    report = run_reference_workflow(replace(inputs, audio=bytes(changed)))
    assert not report["reached_review"]
    assert report["reason"] == "component_failure"


def test_review_links_actual_redaction_and_both_modality_sources():
    import hashlib

    inputs, _ = reference_fixture()
    report = run_reference_workflow(inputs)
    review = report["review"]
    graph = review["visual_graph"]
    sources = {node["node_id"]: node for node in graph["nodes"]}
    claim = graph["claims"][0]
    assert claim["state"] == "cited"
    assert len(claim["source_node_ids"]) == 2
    for source_id in claim["source_node_ids"]:
        assert (
            sources[source_id]["source_digest"]
            == hashlib.sha256(inputs.image).hexdigest()
        )
        assert sources[source_id]["region"] == (4, 24, 60, 40)
    assert (
        report["pixel_redaction"]["source_digest"]
        == hashlib.sha256(inputs.image).hexdigest()
    )
    assert report["pixel_redaction"]["records"][0]["detected_critical_regions"] == [
        [4, 4, 20, 12]
    ]
    assert report["pixel_redaction"]["records"][0]["residual_regions"] == []
    assert len(review["clock_alignment"]) == 2
    assert all(
        item["state"] == "aligned-for-review" for item in review["clock_alignment"]
    )
    assert (
        review["clock_alignment"][0]["source_reference_digest"]
        != review["clock_alignment"][1]["source_reference_digest"]
    )
    assert (
        review["clock_alignment"][0]["target_reference_digest"]
        == review["clock_alignment"][1]["target_reference_digest"]
    )
    token = review["transcript_lineage"]["tokens"][0]
    assert (token["start_ms"], token["end_ms"]) == (0, 1000)
    mapping = review["transcript"]["mappings"][0]
    assert mapping["redacted"] and mapping["source_end"] == len(PRIVATE)
    assert review["transcript"]["revision"] == review["transcript_lineage"]["revision"]
    role = review["speaker_roles"]["assessments"][0]
    assert role["state"] == "unknown" and not role["hypotheses"]
    assert (
        role["primary"]["reference"]["segment_ref"]
        == review["transcript"]["segment_ref"]
    )


def test_visual_disagreement_propagates_to_fused_claim_without_selecting_value():
    report, _ = run("conflict")
    assert report["reached_review"]
    comparison = report["review"]["visual_routing"]["comparisons"][0]
    assert comparison["selected_candidate_id"] is None
    assert comparison["state"] == "review_required"
    assert report["review"]["visual_graph"]["claims"][0]["state"] == "review_required"


def test_wrong_image_dimensions_refuse_before_pixel_decoding(monkeypatch):
    import openmed.multimodal.reference_workflow as workflow

    def unexpected(*args, **kwargs):
        pytest.fail("inventory must not decode outside declared geometry")

    monkeypatch.setattr(workflow, "inventory_pixel_frames", unexpected)
    report = run_reference_workflow(reference_fixture("wrong_geometry")[0])
    assert report["reason"] == "image_geometry_refused"


def test_transcript_timing_is_not_manufactured_when_provider_has_no_tokens():
    inputs, _ = reference_fixture()
    provider = inputs.transcribe
    inputs = replace(
        inputs, transcribe=lambda view: replace(provider(view), token_times=())
    )
    report = run_reference_workflow(inputs)
    assert not report["reached_review"]
    assert report["reason"] == "speech_token_lineage_refused"


def test_one_cleanup_exception_does_not_skip_other_owners(monkeypatch):
    from openmed.multimodal.streaming.privacy.withholding_buffer import (
        WithholdingBuffer,
    )

    def failed(self):
        raise ValueError(PRIVATE)

    monkeypatch.setattr(WithholdingBuffer, "close", failed)
    report, _ = run()
    assert report["reason"] == "cleanup_failed"
    assert report["cleanup_failed"] and report["review"] is None
    assert not report["reached_review"]
    assert any(
        item["action"] == "owned_buffer_overwritten"
        for item in report["audio_retention"]["receipts"]
    )


def test_failed_overwrite_invalidates_review_and_preserves_failure_receipt(monkeypatch):
    from openmed.multimodal.streaming.privacy import audio_retention

    monkeypatch.setattr(audio_retention, "_overwrite", lambda buffer: False)
    inputs, controls = reference_fixture(observe_buffers=True)
    report = run_reference_workflow(inputs)
    assert report["reason"] == "cleanup_failed"
    assert not report["reached_review"] and report["review"] is None
    assert not report["audio_retention"]["passed"]
    assert any(bytes(view).strip(b"\0") for view in controls.borrowed_audio)
    assert report["session"]["owned_bytes"] == 0


def test_synthetic_controls_are_not_used_as_native_model_qualification():
    from openmed.multimodal.vision.providers.base import RuntimeBackend

    inputs, controls = reference_fixture()
    inputs.vlm.metadata = replace(inputs.vlm.metadata, backend=RuntimeBackend.MLX)
    report = run_reference_workflow(inputs)
    assert report["reason"] == "unsupported_reference_input"
    assert controls.vlm_calls == controls.asr_calls == 0


def test_report_is_reproducible_and_no_protected_content_is_logged(caplog, capsys):
    first, _ = run()
    second, _ = run()
    assert first == second
    assert capsys.readouterr().out == ""
    for sentinel in (PRIVATE, CONTROL, TRANSCRIPT):
        assert sentinel not in caplog.text
