"""Synthetic offline contract, provenance and diagnostic-leakage controls."""

import json
import socket
import traceback
from dataclasses import FrozenInstanceError, replace

import pytest

from openmed.multimodal.abstention import (
    AbstentionReason,
    AbstentionRecord,
    AbstentionStage,
)
from openmed.multimodal.vision.providers.base import (
    NON_DIAGNOSTIC_NOTICE,
    Confidence,
    EdgeVLMProvider,
    ImageInput,
    ImageRegion,
    LocalProviderMetadata,
    Prompt,
    RuntimeBackend,
    RuntimeMetadata,
    VisualEvidence,
    VLMContractError,
    VLMRequest,
    VLMResult,
    run_local_vlm,
)

PRIVATE = "SYNTHETIC-PRIVATE-名-معرف-123"


@pytest.fixture
def vlm_request():
    # Opaque synthetic bytes: contract validation does not decode image formats.
    image = ImageInput(data=PRIVATE.encode(), media_type="image/png", width=8, height=8)
    region = ImageRegion(image_digest=image.digest, x0=0, y0=0, x1=8, y1=8)
    return VLMRequest(image=image, prompt=Prompt(text=PRIVATE), regions=(region,))


class SyntheticProvider:
    """Deterministic fixture, not a model or qualified image decoder."""

    def __init__(self, backend=RuntimeBackend.SYNTHETIC):
        self.metadata = LocalProviderMetadata(
            backend=backend, model_digest="a" * 64, preprocessing_digest="b" * 64
        )
        self.calls = 0

    def infer(self, vlm_request):
        self.calls += 1
        return VLMResult(
            request_digest=vlm_request.digest,
            runtime=RuntimeMetadata(
                provider=self.metadata, duration_ms=0, output_tokens=1
            ),
            text=PRIVATE,
            evidence=(
                VisualEvidence(
                    region=vlm_request.regions[0], start=0, end=len(PRIVATE)
                ),
            ),
        )


@pytest.mark.parametrize("backend", list(RuntimeBackend))
def test_offline_deterministic_protocol_and_review(vlm_request, backend, monkeypatch):
    def deny_network(*args, **kwargs):
        pytest.fail("network access attempted")

    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    provider = SyntheticProvider(backend)
    assert isinstance(provider, EdgeVLMProvider)
    first = run_local_vlm(provider, vlm_request)
    second = run_local_vlm(provider, vlm_request)
    assert first == second
    assert first.to_json() == second.to_json()
    for confirmation in (False, None, 1, "yes"):
        with pytest.raises(VLMContractError):
            first.reviewed_text(reviewer_confirmed=confirmation)
    assert first.reviewed_text(reviewer_confirmed=True) == PRIVATE
    payload = json.loads(first.to_json())
    assert payload["notice"] == NON_DIAGNOSTIC_NOTICE
    assert payload["review_required"] is True
    assert payload["model_digest"] == provider.metadata.model_digest
    assert payload["preprocessing_digest"] == provider.metadata.preprocessing_digest
    assert payload["evidence"][0]["end"] == len(PRIVATE)


def test_protected_values_never_enter_repr_reports_or_logs(vlm_request, caplog, capsys):
    result = run_local_vlm(SyntheticProvider(), vlm_request)
    artifacts = repr(vlm_request) + repr(result) + result.to_json()
    assert PRIVATE not in artifacts
    assert PRIVATE.encode().hex() not in artifacts
    assert caplog.text == ""
    assert capsys.readouterr().out == ""
    with pytest.raises(FrozenInstanceError):
        result.text = "changed"


@pytest.mark.parametrize(
    "field,value",
    [
        ("local_only", False),
        ("local_only", 1),
        ("requires_network", True),
        ("requires_network", 0),
        ("deterministic", False),
        ("backend", "cloud"),
        ("model_digest", ""),
        ("preprocessing_digest", ""),
        ("model_digest", "A" * 64),
        ("preprocessing_digest", PRIVATE),
    ],
)
def test_reject_provider_declarations_without_echo(field, value):
    metadata = SyntheticProvider().metadata
    with pytest.raises(VLMContractError) as error:
        replace(metadata, **{field: value})
    assert PRIVATE not in str(error.value)


def test_admission_happens_before_inference(vlm_request):
    provider = SyntheticProvider()
    object.__setattr__(provider.metadata, "requires_network", True)
    with pytest.raises(VLMContractError):
        run_local_vlm(provider, vlm_request)
    assert provider.calls == 0


@pytest.mark.parametrize("phase", ["metadata", "inference"])
def test_provider_exceptions_have_no_sensitive_context(vlm_request, phase):
    class BrokenProvider:
        @property
        def metadata(self):
            if phase == "metadata":
                raise RuntimeError(PRIVATE)
            return SyntheticProvider().metadata

        def infer(self, vlm_request):
            raise RuntimeError(PRIVATE)

    with pytest.raises(VLMContractError) as error:
        run_local_vlm(BrokenProvider(), vlm_request)
    assert error.value.__context__ is None
    assert error.value.__cause__ is None
    assert PRIVATE not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("width", True),
        ("height", 0),
        ("width", 65537),
        ("data", b""),
        ("data", bytearray(b"x")),
        ("data", b"x" * (32 * 1024 * 1024 + 1)),
        ("media_type", PRIVATE),
    ],
)
def test_image_limits_and_safe_errors(vlm_request, field, value):
    with pytest.raises(VLMContractError) as error:
        replace(vlm_request.image, **{field: value})
    assert PRIVATE not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("text", ""),
        ("text", "x" * 65537),
        ("text", "\ud800"),
        ("max_output_tokens", 0),
        ("max_output_tokens", True),
        ("max_output_bytes", 65537),
        ("seed", -1),
    ],
)
def test_prompt_limits(vlm_request, field, value):
    with pytest.raises(VLMContractError):
        replace(vlm_request.prompt, **{field: value})


@pytest.mark.parametrize(
    "score", [float("nan"), float("inf"), -0.1, 1.1, True, PRIVATE]
)
def test_confidence_rejects_invalid_scores(score):
    with pytest.raises(VLMContractError):
        Confidence(score=score)


def test_regions_are_half_open_source_bound_and_immutable(vlm_request):
    region = vlm_request.regions[0]
    for change in ({"x0": 8}, {"y0": -1}, {"x1": True}):
        with pytest.raises(VLMContractError):
            replace(region, **change)
    for change in ({"x1": 9}, {"image_digest": "c" * 64}):
        with pytest.raises(VLMContractError):
            replace(vlm_request, regions=(replace(region, **change),))
    with pytest.raises(VLMContractError):
        replace(vlm_request, regions=list(vlm_request.regions))
    with pytest.raises(VLMContractError):
        replace(vlm_request, regions=vlm_request.regions * 257)


def test_request_identity_binds_all_generation_inputs(vlm_request):
    variants = [
        replace(
            vlm_request, image=replace(vlm_request.image, data=b"other"), regions=()
        ),
        replace(vlm_request, image=replace(vlm_request.image, width=9)),
        replace(vlm_request, image=replace(vlm_request.image, media_type="image/jpeg")),
        replace(vlm_request, regions=()),
    ]
    for change in (
        {"text": "other"},
        {"seed": 1},
        {"max_output_tokens": 1},
        {"max_output_bytes": 128},
    ):
        variants.append(
            replace(vlm_request, prompt=replace(vlm_request.prompt, **change))
        )
    assert len({vlm_request.digest, *(v.digest for v in variants)}) == len(variants) + 1


def test_abstention_uses_shared_reason_codes_and_always_has_digests(vlm_request):
    provider = SyntheticProvider()
    result = VLMResult(
        request_digest=vlm_request.digest,
        runtime=RuntimeMetadata(
            provider=provider.metadata, duration_ms=0, output_tokens=0
        ),
        abstention=AbstentionRecord(
            stage=AbstentionStage.INFERENCE, reason=AbstentionReason.PHI_UNCERTAINTY
        ),
    )
    provider.infer = lambda vlm_request: result
    assert run_local_vlm(provider, vlm_request) == result
    payload = json.loads(result.to_json())
    assert payload["abstention"]["reason"] == "phi_uncertainty"
    assert payload["model_digest"] and payload["preprocessing_digest"]
    with pytest.raises(VLMContractError):
        result.reviewed_text(reviewer_confirmed=True)
    for change in (
        {"text": PRIVATE},
        {"confidence": Confidence(score=0.5)},
        {"evidence": (VisualEvidence(region=vlm_request.regions[0], start=0, end=1),)},
    ):
        with pytest.raises(VLMContractError):
            replace(result, **change)


@pytest.mark.parametrize(
    "change",
    [
        {"request_digest": "c" * 64},
        {"text": "x" * 4097},
        {"runtime_tokens": 513},
        {"model_digest": "c" * 64},
        {"preprocessing_digest": "c" * 64},
        {"region": "foreign"},
        {"region": "out_of_bounds"},
        {"region": "unselected"},
    ],
)
def test_result_validation_rejects_wrong_provenance_and_bounds(vlm_request, change):
    provider = SyntheticProvider()
    result = provider.infer(vlm_request)
    if "runtime_tokens" in change:
        result = replace(
            result,
            runtime=replace(result.runtime, output_tokens=change["runtime_tokens"]),
        )
    elif "model_digest" in change or "preprocessing_digest" in change:
        metadata = replace(provider.metadata, **change)
        result = replace(result, runtime=replace(result.runtime, provider=metadata))
    elif "region" in change:
        region = vlm_request.regions[0]
        changes = {
            "foreign": {"image_digest": "c" * 64},
            "out_of_bounds": {"x1": 9},
            "unselected": {"x1": 7},
        }
        result = replace(
            result,
            evidence=(
                replace(
                    result.evidence[0],
                    region=replace(region, **changes[change["region"]]),
                ),
            ),
        )
    else:
        result = replace(result, **change)
    provider.infer = lambda vlm_request: result
    with pytest.raises(VLMContractError):
        run_local_vlm(provider, vlm_request)


def test_output_span_integrity_and_invalid_nested_contracts(vlm_request):
    result = SyntheticProvider().infer(vlm_request)
    with pytest.raises(VLMContractError):
        replace(result, evidence=(replace(result.evidence[0], end=len(PRIVATE) + 1),))
    with pytest.raises(VLMContractError):
        replace(result, evidence=result.evidence * 257)
    with pytest.raises(VLMContractError):
        replace(result, evidence=[result.evidence[0]])
    for value in (float("nan"), -1, True):
        with pytest.raises(VLMContractError):
            replace(result.runtime, duration_ms=value)
    with pytest.raises(VLMContractError):
        replace(result, runtime=None)
    provider = SyntheticProvider()
    provider.infer = lambda vlm_request: {"text": PRIVATE}
    with pytest.raises(VLMContractError):
        run_local_vlm(provider, vlm_request)
