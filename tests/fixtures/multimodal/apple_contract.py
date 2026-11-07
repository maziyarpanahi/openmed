"""Versioned synthetic parity vectors built from actual Python source contracts."""

import copy
import hashlib
import json
import struct
from dataclasses import replace

from openmed.multimodal.apple_contract import (
    MultimodalPacket,
    project_asset,
    project_clock,
    project_provider,
    project_transcript,
    project_visual,
)
from openmed.multimodal.asset_manifest import AssetManifest
from openmed.multimodal.audio_windows import plan_audio_windows
from openmed.multimodal.evidence.fused_graph import (
    EvidenceNode,
    NodeKind,
    PageGeometry,
    SpatialUnit,
)
from openmed.multimodal.provenance.clock_alignment import (
    AlignmentPolicy,
    AudioPlanEvidence,
    ClockAnchor,
    ClockBinding,
    ClockEpoch,
    ClockReview,
    ImageEvidence,
    align_evidence,
)
from openmed.multimodal.provider_result import (
    ProviderAbstentionCode,
    ProviderResultEnvelope,
    ProviderResultOutcome,
)
from openmed.multimodal.streaming.transcript.revision_ledger import (
    RevisionLedger,
    TranscriptToken,
)


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def rewrite(wire, changes):
    body = copy.deepcopy(json.loads(wire))
    changes(body)
    body.pop("evidence_digest", None)
    body["evidence_digest"] = hashlib.sha256(canonical(body).encode()).hexdigest()
    return canonical(body)


def clock_result(*, missing_timing=False):
    ns, wall = 10**9, 1_700_000_000_000_000_000

    def epoch(index, origin):
        return ClockEpoch(
            clock_ref=f"{index:032x}",
            epoch_ref=f"{index + 100:032x}",
            wall_domain_digest="a" * 64,
            calibration_digest="b" * 64,
            anchors=(
                ClockAnchor(0, origin, 0),
                ClockAnchor(10 * ns, origin + 10 * ns, 0),
            ),
            maximum_rate_error_ppb=1000000,
            nonlinearity_ns=0,
            breaks=(),
        )

    def binding(epoch, sha):
        return ClockBinding(
            source_digest=sha,
            clock_ref=epoch.clock_ref,
            epoch_ref=epoch.epoch_ref,
            evidence_digest="c" * 64,
            local_start_ns=0,
            local_end_ns=10 * ns,
            monotonic_start_ns=0,
            timing_uncertainty_ns=0,
        )

    first, second = epoch(1, wall), epoch(2, wall - ns)
    plan = plan_audio_windows(10000, window_ms=1000)
    source = AudioPlanEvidence(
        plan, plan.windows[2].window_id, binding(first, "d" * 64)
    )
    node = EvidenceNode(
        node_id="1" * 32,
        kind=NodeKind.PIXELS,
        source_digest="e" * 64,
        geometry=PageGeometry(page=0, width=100, height=100),
        region=(10, 10, 30, 30),
    )
    target = ImageEvidence(
        node,
        None if missing_timing else 2 * ns,
        None if missing_timing else 5 * ns,
        "f" * 64,
        binding(second, "e" * 64),
    )
    policy = AlignmentPolicy(
        version="synthetic-v1",
        evidence_digest="0" * 64,
        maximum_uncertainty_ns=0,
        maximum_anchor_gap_ns=10 * ns,
    )
    return align_evidence(source, target, first, second, policy)


def fixture():
    cases = []

    def add(name, packet, expected):
        cases.append(
            {
                "id": name,
                "wire": packet.wire.decode(),
                "digest": packet.evidence_digest,
                "expected": expected,
            }
        )

    image = project_asset(
        AssetManifest(
            "synthetic-image", "image/png", "a" * 64, 128, width=100, height=50
        )
    )
    add("asset_image", image, {"kind": "asset", "width": 100, "height": 50})
    add(
        "asset_audio",
        project_asset(
            AssetManifest(
                "synthetic-audio", "audio/wav", "b" * 64, 256, duration_seconds=0.1
            )
        ),
        {"kind": "asset", "duration": 0.1},
    )
    add(
        "asset_large_integer",
        project_asset(
            AssetManifest("synthetic-size", "application/dicom", "c" * 64, 2**63 - 1)
        ),
        {"kind": "asset", "bytes": "9223372036854775807"},
    )
    visual = None
    for angle, expected in [
        (0, [10, 5, 30, 20]),
        (90, [30, 10, 45, 30]),
        (180, [70, 30, 90, 45]),
        (270, [5, 70, 20, 90]),
    ]:
        node = EvidenceNode(
            node_id="1" * 32,
            kind=NodeKind.OCR,
            source_digest="a" * 64,
            geometry=PageGeometry(page=0, width=100, height=50),
            region=(10, 5, 30, 20),
            span=(1, 3),
        )
        visual = project_visual(node, rotation=angle)
        add(
            f"visual_{angle}",
            visual,
            {
                "kind": "visual",
                "box": expected,
                "scalar_span": [1, 3],
                "synthetic_text": "A😀éB",
            },
        )
    node = EvidenceNode(
        node_id="2" * 32,
        kind=NodeKind.PIXELS,
        source_digest="a" * 64,
        geometry=PageGeometry(
            page=0, width=100.25, height=50.5, unit=SpatialUnit.POINT
        ),
        region=(0.1, 0.2, 30.25, 20.5),
    )
    add(
        "visual_fractional",
        project_visual(node, rotation=90),
        {"kind": "visual", "box": [30.0, 0.1, 50.3, 30.25]},
    )
    page = EvidenceNode(
        node_id="3" * 32,
        kind=NodeKind.PAGE,
        source_digest="a" * 64,
        geometry=PageGeometry(page=0, width=100, height=50),
    )
    add("visual_page", project_visual(page), {"kind": "visual", "box": None})
    ledger = RevisionLedger(
        "1" * 32,
        "2" * 32,
        4,
        (TranscriptToken(token_ref="3" * 32, start=1, end=3, start_ms=3, end_ms=7),),
    )
    ref = ledger.capture_span("4" * 32, 1, 3)
    initial = project_transcript(ledger, ref, sample_rate_hz=44100)
    add(
        "transcript_partial",
        initial,
        {
            "kind": "transcript",
            "current_span": [1, 3],
            "time_ms": [3, 7],
            "sample_bounds": [132, 309],
            "finalized": False,
        },
    )
    ledger.insert(0, 2, (), expected_revision=0)
    add(
        "transcript_shifted",
        project_transcript(ledger, ref, sample_rate_hz=44100),
        {
            "kind": "transcript",
            "current_span": [3, 5],
            "time_ms": [3, 7],
            "sample_bounds": [132, 309],
            "finalized": False,
        },
    )
    ledger.finalize(expected_revision=1)
    add(
        "transcript_final",
        project_transcript(ledger, ref, sample_rate_hz=44100),
        {
            "kind": "transcript",
            "current_span": [3, 5],
            "sample_bounds": [132, 309],
            "finalized": True,
        },
    )
    invalid = RevisionLedger(
        "1" * 32,
        "5" * 32,
        4,
        (TranscriptToken(token_ref="6" * 32, start=1, end=3, start_ms=3, end_ms=7),),
    )
    badref = invalid.capture_span("7" * 32, 1, 3)
    invalid.delete(1, 3, expected_revision=0)
    add(
        "transcript_invalidated",
        project_transcript(invalid, badref, sample_rate_hz=16000),
        {
            "kind": "transcript",
            "invalidation": "cited_text_changed",
            "current_span": None,
            "sample_bounds": None,
            "finalized": False,
        },
    )
    for outcome in ProviderResultOutcome:
        reasons = (
            list(ProviderAbstentionCode)
            if outcome is ProviderResultOutcome.ABSTENTION
            else [None]
        )
        for reason in reasons:
            value = ProviderResultEnvelope(
                provider_id="synthetic-adapter",
                model_id="synthetic-model",
                input_digest=image.evidence_digest,
                outcome=outcome,
                duration_ms=0.1,
                output_digest="d" * 64
                if outcome is ProviderResultOutcome.SUCCESS
                else None,
                abstention_code=reason,
                count_metadata={
                    "input_items": 1,
                    "output_items": int(outcome is ProviderResultOutcome.SUCCESS),
                },
            )
            add(
                f"provider_{outcome.value}_{reason.value if reason else 'none'}",
                project_provider(value),
                {
                    "kind": "provider",
                    "outcome": outcome.value,
                    "reason": reason.value if reason else None,
                },
            )
    aligned = clock_result()
    add(
        "clock_aligned",
        project_clock(aligned),
        {
            "kind": "clock",
            "overlap": [3000000000, 4000000000],
            "state": "aligned-for-review",
        },
    )
    held = clock_result(missing_timing=True)
    add(
        "clock_hold",
        project_clock(held),
        {"kind": "clock", "overlap": None, "state": "review-required"},
    )
    large = replace(
        aligned,
        target_start_ns=replace(
            aligned.target_start_ns,
            lower=4_000_000_000_000_000_001,
            upper=4_000_000_000_000_000_001,
        ),
        target_end_ns=replace(
            aligned.target_end_ns,
            lower=4_000_000_000_000_000_101,
            upper=4_000_000_000_000_000_101,
        ),
        guaranteed_overlap_ns=(4_000_000_000_000_000_001, 4_000_000_000_000_000_101),
    )
    add(
        "clock_large_integer",
        project_clock(large),
        {
            "kind": "clock",
            "overlap_strings": ["4000000000000000001", "4000000000000000101"],
            "state": "aligned-for-review",
        },
    )
    negatives = []

    def reject(name, original, change, reason):
        wire = rewrite(original.wire, change)
        negatives.append({"id": name, "wire": wire, "reason": reason})

    invalid_reason = "invalid_multimodal_contract"
    provenance = "incomplete_multimodal_provenance"
    reject(
        "unknown_schema",
        image,
        lambda d: d.update(schema_version="future"),
        "unsupported_multimodal_schema",
    )
    reject(
        "fractional_source_schema",
        image,
        lambda d: d["payload"].update(source_schema=1.5),
        "unsupported_multimodal_schema",
    )
    reject(
        "unknown_capability",
        image,
        lambda d: d.update(kind="remote"),
        "unsupported_multimodal_capability",
    )
    reject(
        "missing_digest", image, lambda d: d["payload"].update(sha256=None), provenance
    )
    reject(
        "extra_field",
        image,
        lambda d: d["payload"].update(private_path="synthetic"),
        invalid_reason,
    )
    reject(
        "boolean_integer",
        image,
        lambda d: d["payload"].update(byte_size=True),
        invalid_reason,
    )
    reject(
        "floating_integer",
        image,
        lambda d: d["payload"].update(byte_size=1.0),
        invalid_reason,
    )
    reject(
        "overflow_integer",
        image,
        lambda d: d["payload"].update(byte_size=2**63),
        invalid_reason,
    )
    reject(
        "newline_reference",
        image,
        lambda d: d["payload"].update(asset_ref="1" * 31 + "\n"),
        provenance,
    )
    reject(
        "nonfinite_float",
        image,
        lambda d: d["payload"].update(duration_f64="7ff0000000000000"),
        invalid_reason,
    )
    reject(
        "negative_zero",
        image,
        lambda d: d["payload"].update(duration_f64="8000000000000000"),
        invalid_reason,
    )
    reject(
        "unsupported_rotation",
        visual,
        lambda d: d["payload"].update(rotation=45),
        "invalid_multimodal_transform",
    )
    reject(
        "mismatched_display",
        visual,
        lambda d: d["payload"].update(display_box_f64=None),
        "invalid_multimodal_transform",
    )
    reject(
        "nonfinite_coordinate",
        visual,
        lambda d: d["payload"].update(width_f64="7ff8000000000000"),
        invalid_reason,
    )
    reject(
        "wrong_sample_grid",
        initial,
        lambda d: d["payload"].update(sample_bounds=[132, 308]),
        invalid_reason,
    )
    reject(
        "invalidated_coords_visible",
        project_transcript(invalid, badref, sample_rate_hz=16000),
        lambda d: d["payload"].update(current_span=[1, 3]),
        invalid_reason,
    )
    reject(
        "clock_hold_leaks_time",
        project_clock(held),
        lambda d: d["payload"].update(uncertainty_ns=0),
        invalid_reason,
    )
    reject(
        "visual_precision_collapse",
        visual,
        lambda d: d["payload"].update(
            box_f64=[struct.pack(">d", x).hex() for x in (0.0, 5.0, 1e-300, 20.0)],
            display_box_f64=[
                struct.pack(">d", x).hex() for x in (5.0, 100.0, 20.0, 100.0)
            ],
        ),
        "invalid_multimodal_transform",
    )
    for case_id, wrong_version in (
        (
            "provider_abstention_consent_withdrawn",
            "openmed.multimodal.provider_result.v1",
        ),
        ("provider_abstention_resource_limit", "openmed.multimodal.provider_result.v2"),
    ):
        original = next(row for row in cases if row["id"] == case_id)
        reject(
            case_id + "_wrong_version",
            MultimodalPacket(original["wire"].encode()),
            lambda data: data["payload"].update(schema_version=wrong_version),
            "unsupported_multimodal_schema",
        )
    negatives.append(
        {
            "id": "duplicate_key",
            "wire": image.wire.decode().replace("{", '{"kind":"asset",', 1),
            "reason": invalid_reason,
        }
    )
    negatives.append(
        {
            "id": "nonfinite_json",
            "wire": image.wire.decode().replace(
                '"source_schema":1', '"source_schema":NaN'
            ),
            "reason": invalid_reason,
        }
    )
    negatives.append(
        {
            "id": "noncanonical",
            "wire": image.wire.decode() + "\n",
            "reason": invalid_reason,
        }
    )
    for row in cases:
        assert MultimodalPacket(row["wire"].encode()).evidence_digest == row["digest"]
    return {
        "version": 1,
        "synthetic_only": True,
        "cases": cases,
        "negative_cases": negatives,
        "provider_outcomes": [x.value for x in ProviderResultOutcome],
        "abstention_codes": [x.value for x in ProviderAbstentionCode],
        "clock_reasons": [x.value for x in ClockReview],
    }
