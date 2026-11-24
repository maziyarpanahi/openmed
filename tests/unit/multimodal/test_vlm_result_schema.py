"""Validate the published edge-VLM result schema against runtime payloads."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.vision.providers.base import (
    Confidence,
    ImageInput,
    ImageRegion,
    LocalProviderMetadata,
    RuntimeBackend,
    RuntimeMetadata,
    VisualEvidence,
    VLMRequest,
    VLMResult,
)
from openmed.multimodal.vision.providers.base import Prompt as VLMPrompt

SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "openmed"
    / "core"
    / "schemas"
    / "json"
    / "vlm_result.schema.json"
)
PRIVATE = "synthetic draft text"


@pytest.fixture
def validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _runtime(*, output_tokens: int) -> RuntimeMetadata:
    return RuntimeMetadata(
        provider=LocalProviderMetadata(
            backend=RuntimeBackend.SYNTHETIC,
            model_digest="a" * 64,
            preprocessing_digest="b" * 64,
        ),
        duration_ms=12.5,
        output_tokens=output_tokens,
    )


def _draft_payload() -> dict:
    image = ImageInput(
        data=b"synthetic image bytes", media_type="image/png", width=8, height=8
    )
    region = ImageRegion(image_digest=image.digest, x0=0, y0=0, x1=8, y1=8)
    request = VLMRequest(
        image=image,
        prompt=VLMPrompt(text="synthetic prompt"),
        regions=(region,),
    )
    result = VLMResult(
        request_digest=request.digest,
        runtime=_runtime(output_tokens=3),
        text=PRIVATE,
        evidence=(
            VisualEvidence(
                region=region,
                start=0,
                end=len(PRIVATE),
                confidence=Confidence(score=0.75),
            ),
        ),
        confidence=Confidence(score=0.75),
    )
    return json.loads(result.to_json())


def _abstention_payload() -> dict:
    result = VLMResult(
        request_digest="d" * 64,
        runtime=_runtime(output_tokens=0),
        abstention=AbstentionRecord(
            stage=AbstentionStage.INFERENCE,
            reason=AbstentionReason.PHI_UNCERTAINTY,
        ),
    )
    return json.loads(result.to_json())


def test_runtime_draft_and_abstention_payloads_validate(validator):
    validator.validate(_draft_payload())
    validator.validate(_abstention_payload())


@pytest.mark.parametrize("field", ["extra", "text"])
def test_closed_payload_rejects_protected_or_unknown_fields(validator, field):
    payload = _draft_payload()
    payload[field] = PRIVATE
    assert not validator.is_valid(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_digest", "A" * 64),
        ("model_digest", "not-a-digest"),
        ("preprocessing_digest", "c" * 63),
    ],
)
def test_draft_rejects_non_hex_or_wrong_length_digests(validator, field, value):
    payload = _draft_payload()
    payload[field] = value
    assert not validator.is_valid(payload)


def test_schema_requires_review_and_notice(validator):
    for field in ("notice", "review_required"):
        payload = _draft_payload()
        del payload[field]
        assert not validator.is_valid(payload)

    payload = _draft_payload()
    payload["review_required"] = False
    assert not validator.is_valid(payload)


def test_abstention_has_no_output_or_evidence(validator):
    payload = _abstention_payload()
    payload["output_digest"] = "c" * 64
    assert not validator.is_valid(payload)

    payload = _abstention_payload()
    payload["evidence"] = _draft_payload()["evidence"]
    assert not validator.is_valid(payload)
