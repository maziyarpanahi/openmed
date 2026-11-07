"""Offline controls for asset identity, admission, and protected output release."""

import json
import sys
from dataclasses import replace
from io import BytesIO
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from openmed.multimodal.abstention import AbstentionReason
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
    VLMContractError,
    VLMRequest,
    run_local_vlm,
)
from openmed.multimodal.vision.providers.local_assets import LocalVLMAssets
from openmed.multimodal.vision.providers.smolvlm import SmolVLMProvider, _versions
from openmed.multimodal.vision.safety.quality_gate import QualityPolicy


@pytest.fixture
def assets(tmp_path):
    files = {
        "config.json": {"model_type": "smolvlm"},
        "preprocessor_config.json": {},
        "processor_config.json": {},
        "tokenizer.json": {},
        "tokenizer_config.json": {},
        "chat_template.json": {},
        "special_tokens_map.json": {},
    }
    for name, value in files.items():
        (tmp_path / name).write_text(json.dumps(value))
    (tmp_path / "model.safetensors").write_bytes(
        b"synthetic weights; runtime is replaced"
    )
    return LocalVLMAssets(tmp_path, "a" * 40, b"synthetic license snapshot")


@pytest.fixture
def vlm_request():
    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).rectangle((32, 32, 96, 96), fill="blue")
    data = BytesIO()
    image.save(data, format="PNG")
    image.close()
    return VLMRequest(
        image=ImageInput(
            data=data.getvalue(), media_type="image/png", width=128, height=128
        ),
        prompt=Prompt(text="synthetic prompt", max_output_tokens=16),
    )


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(text="blue", output_tokens=1, calls=0, loads=0, error=None)

    class Runtime:
        versions = _versions()

        def __init__(self, assets):
            assets.verify_current()
            state.loads += 1

        def generate(self, image, prompt, max_tokens, seed):
            state.calls += 1
            assert image.mode == "RGB" and not image.info
            if state.error:
                raise state.error
            return SimpleNamespace(text=state.text, output_tokens=state.output_tokens)

    module = ModuleType("openmed.mlx.models.smolvlm")
    module.SmolVLMRuntime = Runtime
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return state


def _provider(assets, detector=lambda text: ()):
    policy = QualityPolicy(
        min_short_side=64,
        min_long_side=64,
        min_luminance=0,
        max_luminance=1,
        min_contrast=0.01,
        min_sharpness=0,
        max_dark_fraction=1,
        max_bright_fraction=1,
    )
    return SmolVLMProvider(
        assets,
        quality_policy=policy,
        guard=GeneratedTextGuard(
            detector,
            detector_version="synthetic",
            policy=GuardPolicy(action="withhold"),
        ),
    )


def test_success_is_guarded_and_metadata_contains_no_payload(
    assets, vlm_request, runtime
):
    provider = _provider(assets)
    result = run_local_vlm(provider, vlm_request)
    assert result.reviewed_text(reviewer_confirmed=True) == "blue"
    assert runtime.calls == runtime.loads == 1
    for payload in (repr(result), result.to_json(), provider.provenance_json()):
        assert "synthetic prompt" not in payload and "blue" not in payload
        assert (
            str(assets.directory) not in payload and "synthetic license" not in payload
        )
    metadata = json.loads(result.to_json())
    assert metadata["model_digest"] == assets.model_digest
    assert metadata["preprocessing_digest"] == provider.metadata.preprocessing_digest
    assert metadata["review_required"]
    with pytest.raises(VLMContractError):
        result.reviewed_text(reviewer_confirmed=False)


def test_rejects_assets_replaced_after_manifest(assets, vlm_request, runtime):
    (assets.directory / "tokenizer.json").write_text('{"changed":true}')
    result = _provider(assets).infer(vlm_request)
    assert result.abstention.reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert runtime.calls == runtime.loads == 0


def test_model_tokenizer_license_and_runtime_changes_invalidate_identity(
    assets, monkeypatch
):
    original = _provider(assets).metadata
    changed_license = LocalVLMAssets(
        assets.directory, assets.revision, b"changed terms"
    )
    assert changed_license.model_digest != assets.model_digest
    (assets.directory / "tokenizer.json").write_text('{"changed":true}')
    changed = LocalVLMAssets(assets.directory, assets.revision, assets.license_metadata)
    assert changed.model_digest != assets.model_digest
    assert changed.preprocessing_digest != assets.preprocessing_digest
    monkeypatch.setattr(
        "openmed.multimodal.vision.providers.smolvlm._versions",
        lambda: {"mlx": "changed"},
    )
    assert (
        _provider(assets).metadata.preprocessing_digest != original.preprocessing_digest
    )


def test_rejects_unsafe_asset_layouts(assets, subtests):
    for name, contents in (
        ("custom.py", "pass"),
        ("generation_config.json", "{}"),
        ("unexpected.safetensors", "other weights"),
    ):
        with subtests.test(name=name):
            path = assets.directory / name
            path.write_text(contents)
            with pytest.raises(VLMContractError):
                LocalVLMAssets(
                    assets.directory, assets.revision, assets.license_metadata
                )
            path.unlink()
    path = assets.directory / "config.json"
    for content in (
        {"model_type": "other"},
        {"model_type": "smolvlm", "auto_map": {}},
        {"model_type": "smolvlm", "quantization": {"bits": 4}},
    ):
        with subtests.test(content=content):
            path.write_text(json.dumps(content))
            with pytest.raises(VLMContractError):
                LocalVLMAssets(
                    assets.directory, assets.revision, assets.license_metadata
                )


def test_invalid_and_large_inputs_never_load_runtime(
    assets, vlm_request, runtime, subtests
):
    provider = _provider(assets)
    invalid = [
        replace(vlm_request, image=replace(vlm_request.image, width=129)),
        replace(vlm_request, image=replace(vlm_request.image, data=b"bad image")),
        replace(vlm_request, image=replace(vlm_request.image, width=1537)),
        replace(vlm_request, prompt=Prompt(text="x" * 4097)),
        replace(vlm_request, prompt=Prompt(text="read", max_output_tokens=513)),
    ]
    for case in invalid:
        with subtests.test(digest=case.digest):
            assert provider.infer(case).abstention is not None
    assert runtime.loads == runtime.calls == 0


def test_oversized_or_invalid_generations_are_withheld(
    assets, vlm_request, runtime, subtests
):
    provider = _provider(assets)
    for text, count in (
        ("", 1),
        ("x" * 4097, 1),
        ("blue", 0),
        ("blue", 17),
        ("blue", True),
    ):
        with subtests.test(count=count, length=len(text)):
            runtime.text, runtime.output_tokens = text, count
            result = provider.infer(vlm_request)
            assert (
                result.text == ""
                and result.abstention.reason is AbstentionReason.RESOURCE_LIMIT
            )


def test_detector_absent_failure_and_findings_fail_closed(
    assets, vlm_request, runtime, subtests
):
    def broken(text):
        raise RuntimeError("synthetic sensitive exception")

    for detector in (None, broken, lambda text: [PHISpan(0, 4, PHILabel.NAME)]):
        with subtests.test(detector=detector is None):
            result = _provider(assets, detector).infer(vlm_request)
            assert result.abstention.reason is AbstentionReason.PHI_UNCERTAINTY
            assert not result.text and "sensitive" not in result.to_json()


def test_same_asset_commitment_blocks_leak_without_raw_identifier_retention(
    assets, vlm_request, runtime
):
    runtime.text = "ALICE EXAMPLE"
    identifier = AssetIdentifierDigest(
        vlm_request.image.digest, PHILabel.NAME, identifier_digest(runtime.text), 12
    )
    result = _provider(assets).infer(vlm_request, identifiers=(identifier,))
    assert result.abstention.reason is AbstentionReason.PHI_UNCERTAINTY
    assert "ALICE" not in result.to_json()


def test_cross_asset_evidence_rejects_before_model(assets, vlm_request, runtime):
    identifier = AssetIdentifierDigest(
        "b" * 64, PHILabel.NAME, identifier_digest("ALICE"), 5
    )
    result = _provider(assets).infer(vlm_request, identifiers=(identifier,))
    assert result.abstention.reason is AbstentionReason.PHI_UNCERTAINTY
    assert runtime.calls == runtime.loads == 0


def test_runtime_failure_never_exposes_exception(assets, vlm_request, runtime):
    runtime.error = RuntimeError("ALICE EXAMPLE synthetic private path")
    result = _provider(assets).infer(vlm_request)
    assert result.abstention.reason is AbstentionReason.PROVIDER_UNAVAILABLE
    assert "ALICE" not in result.to_json()


def test_capabilities_describe_supported_image_path(assets):
    from openmed.multimodal.capability_matrix import ModalityProfile, capability_matrix

    provider = _provider(assets)
    result = capability_matrix(
        (provider.capabilities,),
        (ModalityProfile("image-v1", "image", ("single_image",)),),
    )[0]
    assert result.outcome == "exact"
