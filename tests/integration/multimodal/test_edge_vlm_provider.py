"""Opt-in real-device qualification; no download or model execution in default CI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import time
from io import BytesIO
from pathlib import Path

import pytest

from openmed.multimodal.generated_text_guard import (
    AssetIdentifierDigest,
    GeneratedTextGuard,
    GuardPolicy,
    PHILabel,
    PHISpan,
    identifier_digest,
)
from openmed.multimodal.vision.providers.base import (
    ImageInput,
    Prompt,
    VLMRequest,
    run_local_vlm,
)
from openmed.multimodal.vision.providers.local_assets import LocalVLMAssets
from openmed.multimodal.vision.providers.smolvlm import SmolVLMProvider
from openmed.multimodal.vision.safety.quality_gate import QualityPolicy

PROFILE_PATH = Path(__file__).parents[2] / "fixtures/multimodal/edge_vlm_profile.json"
pytestmark = [pytest.mark.integration, pytest.mark.slow]


def _policy():
    return QualityPolicy(
        min_short_side=128,
        min_long_side=256,
        min_luminance=0.01,
        max_luminance=0.999,
        min_contrast=0.02,
        min_sharpness=0.00001,
        max_dark_fraction=0.99,
        max_bright_fraction=0.999,
    )


def _image(text=None, color=None, blank=False):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (768, 384), "white")
    draw = ImageDraw.Draw(image)
    if color:
        draw.rectangle((192, 64, 576, 320), fill=color, outline="black", width=4)
    elif not blank:
        draw.text((32, 150), text, fill="black", font=ImageFont.load_default(size=64))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    image.close()
    return ImageInput(
        data=encoded.getvalue(), media_type="image/png", width=768, height=384
    )


def _request(image, prompt):
    return VLMRequest(
        image=image, prompt=Prompt(text=prompt, max_output_tokens=64, seed=0)
    )


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def _fixture_detector(text):
    # Deliberately limited to synthetic fixture names and patterned IDs/dates.
    pattern = (
        r"ALICE\s+EXAMPLE|BOB\s+SAMPLE|CAROL\s+FIXTURE|MRN\s*\d+|\d{4}-\d{2}-\d{2}"
    )
    return [
        PHISpan(m.start(), m.end(), PHILabel.IDENTIFIER)
        for m in re.finditer(pattern, text, re.I)
    ]


def test_real_local_smolvlm_qualification(monkeypatch):
    if os.environ.get("OPENMED_EDGE_VLM_PROFILE") != "smolvlm2-v1":
        pytest.skip("explicit local SmolVLM2 profile is not enabled")
    model_dir = os.environ.get("OPENMED_EDGE_VLM_MODEL_DIR")
    assert model_dir, "the optional profile requires a materialized local model"
    report_path = os.environ.get("OPENMED_EDGE_VLM_REPORT")
    assert report_path, "the optional profile requires an explicit report destination"
    profile_bytes = PROFILE_PATH.read_bytes()
    profile = json.loads(profile_bytes)
    assets = LocalVLMAssets(
        Path(model_dir),
        profile["revision"],
        (Path(model_dir) / "README.md").read_bytes(),
    )
    assert dict(assets.file_digests)["model.safetensors"] == profile["weights_sha256"]
    # Missing optional runtime on an explicitly enabled profile is a failure.
    import resource

    import mlx.core as mx

    network_attempts = []

    def reject_network(*args, **kwargs):
        network_attempts.append(1)
        raise RuntimeError("network is disabled for qualification")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "create_connection", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    guard = GeneratedTextGuard(
        _fixture_detector,
        detector_version="synthetic-fixture-v1",
        policy=GuardPolicy(action="withhold"),
    )
    provider = SmolVLMProvider(assets, quality_policy=_policy(), guard=guard)
    report = {
        "schema_version": profile["schema_version"],
        "profile_digest": hashlib.sha256(profile_bytes).hexdigest(),
        "declared_device_tier": profile["device_tier"],
        "provenance": json.loads(provider.provenance_json()),
        "cases": [],
        "gates": profile["gates"],
    }
    cases = [
        (
            "document",
            _image(text=value),
            "Read the heading in this image. Reply with only the exact heading.",
            value,
        )
        for value in profile["document_headings"]
    ]
    cases += [
        (
            "color",
            _image(color=value),
            "What color is the filled rectangle? Reply with only one color word.",
            value,
        )
        for value in profile["square_colors"]
    ]
    durations = []
    correct = {"document": 0, "color": 0}
    parity = 0
    mx.reset_peak_memory()
    cold_seconds = None
    for case_number, (kind, image, prompt, expected) in enumerate(cases):
        outputs = []
        request = _request(image, prompt)
        for repeat in range(profile["repeats"]):
            started = time.perf_counter()
            result = run_local_vlm(provider, request)
            elapsed = time.perf_counter() - started
            if cold_seconds is None:
                cold_seconds = elapsed
            else:
                durations.append(elapsed)
            outputs.append(result.text if result.abstention is None else "")
            report["cases"].append(
                {
                    "case": case_number,
                    "kind": kind,
                    "repeat": repeat,
                    "result": json.loads(result.to_json()),
                    "correct": _normalize(outputs[-1]) == _normalize(expected),
                }
            )
        correct[kind] += int(
            all(_normalize(x) == _normalize(expected) for x in outputs)
        )
        parity += int(len(set(outputs)) == 1)
    leaked = 0
    for index, value in enumerate(profile["synthetic_identifiers"]):
        image = _image(text=value)
        identifier = AssetIdentifierDigest(
            image.digest,
            PHILabel.IDENTIFIER,
            identifier_digest(value),
            len("".join(value.lower().split())),
        )
        result = provider.infer(
            _request(image, "Read all text in this image exactly."),
            identifiers=(identifier,),
        )
        leaked += int(_normalize(value) in _normalize(result.text))
        report["cases"].append(
            {"case": index, "kind": "privacy", "result": json.loads(result.to_json())}
        )
    invalid = [
        _request(_image(blank=True), "Read the heading."),
        _request(
            ImageInput(data=b"invalid", media_type="image/png", width=768, height=384),
            "Read.",
        ),
        _request(
            ImageInput(
                data=cases[0][1].data, media_type="image/png", width=769, height=384
            ),
            "Read.",
        ),
        VLMRequest(
            image=cases[0][1], prompt=Prompt(text="Read.", max_output_tokens=513)
        ),
    ]
    abstained = sum(
        provider.infer(request).abstention is not None for request in invalid
    )
    unavailable = SmolVLMProvider(
        assets, quality_policy=_policy(), guard=GeneratedTextGuard(None)
    )
    # Reuse loaded runtime only in the test to isolate the missing-detector boundary.
    unavailable._runtime = provider._runtime
    abstained += (
        unavailable.infer(_request(cases[0][1], "Read the heading.")).abstention
        is not None
    )
    metrics = {
        "document_accuracy": correct["document"] / len(profile["document_headings"]),
        "color_accuracy": correct["color"] / len(profile["square_colors"]),
        "overall_accuracy": sum(correct.values()) / len(cases),
        "critical_identifier_leaks": leaked,
        "invalid_input_abstention_fraction": abstained / (len(invalid) + 1),
        "repeat_output_parity_fraction": parity / len(cases),
        "warm_p95_latency_seconds": sorted(durations)[
            __import__("math").ceil(0.95 * len(durations)) - 1
        ],
        "cold_first_request_seconds": cold_seconds,
        "peak_mlx_memory_gib": mx.get_peak_memory() / 1024**3,
        "peak_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024**3,
        "network_attempts": len(network_attempts),
    }
    passed = {}
    for gate, threshold in profile["gates"].items():
        metric, direction = gate.rsplit("_", 1)
        passed[gate] = (
            metrics[metric] >= threshold
            if direction == "min"
            else metrics[metric] <= threshold
        )
    passed["no_network_attempts"] = not network_attempts
    report.update(
        metrics=metrics,
        gate_results=passed,
        passed=all(passed.values()),
        provenance=json.loads(provider.provenance_json()),
    )
    Path(report_path).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    assert all(passed.values()), {k: v for k, v in passed.items() if not v}
