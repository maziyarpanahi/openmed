"""Tests for the Privacy Filter Core ML and ONNX export adapter.

The first group is pure Python (configuration, eligibility, decoder). The
second group builds a tiny native Privacy Filter in-test and needs PyTorch
plus a Transformers release that ships ``openai_privacy_filter``.
"""

from __future__ import annotations

import importlib.util
import json
import math
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.eval.export_qualification import (
    ExportQualificationError,
    ExportReason,
    label_map_identity,
)
from openmed.torch import export_adapters
from openmed.torch import privacy_filter_export as pf
from openmed.torch.export_adapters import (
    export_adapter_for_family,
    registered_export_families,
    require_registered_adapter,
    resolve_export_adapter,
)

OPF_LABELS = {
    index: label
    for index, label in enumerate(
        ("O",)
        + tuple(
            f"{tag}-{name}"
            for name in pf.OPENAI_PRIVACY_FILTER_SPAN_LABELS
            for tag in "BIES"
        )
    )
}


def _config(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model_type": "openai_privacy_filter",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 16,
        "head_dim": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_hidden_layers": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "sliding_window": 4,
        "max_position_embeddings": 4096,
        "rope_parameters": {"rope_type": "yarn", "factor": 32.0},
        "id2label": {str(key): value for key, value in OPF_LABELS.items()},
    }
    values.update(overrides)
    return values


@pytest.fixture
def transformers_version(monkeypatch: pytest.MonkeyPatch):
    def install(version: str | None, *, native: bool = True) -> None:
        def fake_version(name: str) -> str:
            if version is None:
                raise metadata.PackageNotFoundError(name)
            return version

        monkeypatch.setattr(
            pf,
            "metadata",
            SimpleNamespace(
                version=fake_version,
                PackageNotFoundError=metadata.PackageNotFoundError,
            ),
        )
        monkeypatch.setattr(
            pf,
            "importlib_util",
            SimpleNamespace(find_spec=lambda name: object() if native else None),
        )

    return install


# ---------------------------------------------------------------------------
# Versioned registry and eligible families
# ---------------------------------------------------------------------------


def test_registry_resolves_only_the_native_spelling() -> None:
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    for model_type in (
        "openai_privacy_filter",
        "openai-privacy-filter",
        "OpenAI_Privacy_Filter",
    ):
        assert resolve_export_adapter(model_type) is adapter
    # Original OPF and MLX family spellings are not export sources.
    for model_type in ("privacy_filter", "privacy-filter", "privacy-filter-nemotron"):
        assert resolve_export_adapter(model_type) is None
    assert {"modernbert", "openai-privacy-filter"} <= set(registered_export_families())
    assert export_adapter_for_family("openai-privacy-filter") is adapter
    description = adapter.describe()
    assert description["version"] == pf.PRIVACY_FILTER_ADAPTER_VERSION
    assert description["support"]["coreml"]["quantizations"] == ["none", "int8"]
    assert description["support"]["onnx"]["providers"] == ["CPUExecutionProvider"]


def test_registered_adapter_check_rejects_unknown_families_and_versions() -> None:
    assert (
        require_registered_adapter("openai-privacy-filter", 1)
        is pf.PRIVACY_FILTER_EXPORT_ADAPTER
    )
    for family, version, reason in (
        ("privacy-filter", 1, ExportReason.UNSUPPORTED_FAMILY),
        (None, 1, ExportReason.UNSUPPORTED_FAMILY),
        ("openai-privacy-filter", 2, ExportReason.ADAPTER_VERSION_MISMATCH),
        ("openai-privacy-filter", True, ExportReason.ADAPTER_VERSION_MISMATCH),
        ("modernbert", "1", ExportReason.ADAPTER_VERSION_MISMATCH),
    ):
        with pytest.raises(ExportQualificationError) as error:
            require_registered_adapter(family, version)
        assert error.value.reason is reason


def test_declared_checkpoints_cover_the_shipped_label_spaces() -> None:
    declared = {item.repo_id: item for item in pf.PRIVACY_FILTER_CHECKPOINTS}
    assert set(declared) == {
        "openai/privacy-filter",
        "OpenMed/privacy-filter-nemotron",
        "OpenMed/privacy-filter-nemotron-v2",
        "OpenMed/privacy-filter-multilingual",
        "OpenMed/privacy-filter-multilingual-v2",
    }
    for item in declared.values():
        assert item.num_labels == 1 + 4 * ((item.num_labels - 1) // 4)
        # No release evidence is committed yet, so nothing claims Android.
        assert not item.qualified_for("onnx", ("onnx-android-fp32",))
        assert item.to_dict()["revision"] is None
    assert (
        pf.privacy_filter_checkpoint("openmed/PRIVACY-filter-nemotron")
        is (declared["OpenMed/privacy-filter-nemotron"])
    )
    assert pf.privacy_filter_checkpoint("attacker/privacy-filter") is None


def test_qualified_for_requires_revision_evidence_and_every_variant() -> None:
    checkpoint = pf.PrivacyFilterCheckpoint(
        "OpenMed/privacy-filter-nemotron",
        "nemotron_fine_v1",
        221,
        revision="c0ffee",
        qualified_variants={"onnx": ("onnx-android-fp32", "onnx-android-fp16")},
        evidence_sha256={"onnx": "ab" * 32},
    )
    assert checkpoint.qualified_for("onnx", ("onnx-android-fp32",))
    assert not checkpoint.qualified_for("onnx", ("onnx-android-fp32", "onnx-int8"))
    assert not checkpoint.qualified_for("coreml", ("coreml-fp16",))
    assert not checkpoint.qualified_for("onnx", ())


# ---------------------------------------------------------------------------
# Configuration normalization and fail-closed shapes
# ---------------------------------------------------------------------------


def test_architecture_normalizes_native_configuration() -> None:
    architecture = pf.privacy_filter_architecture(_config())
    assert architecture.num_experts == 8
    assert architecture.experts_per_token == 2
    assert architecture.sliding_window == 4
    assert architecture.attention_bias is True
    assert architecture.rope_type == "yarn"
    assert architecture.num_labels == 33
    assert len(architecture.sha256) == 64
    assert pf.privacy_filter_architecture(SimpleNamespace(**_config())).sha256 == (
        architecture.sha256
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"auto_map": {"AutoModel": "x.Y"}}, ExportReason.UNSUPPORTED_REMOTE_CODE),
        ({"custom_pipelines": {"x": {}}}, ExportReason.UNSUPPORTED_REMOTE_CODE),
        ({"model_type": "privacy_filter"}, ExportReason.UNSUPPORTED_FAMILY),
        ({"model_type": "bert"}, ExportReason.UNSUPPORTED_FAMILY),
        ({"num_key_value_heads": 3}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"head_dim": 7}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"num_experts_per_tok": 9}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"sliding_window": -1}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"sliding_window": None}, ExportReason.UNSUPPORTED_CONFIGURATION),
        (
            {"rope_parameters": {"rope_type": "longrope"}},
            ExportReason.UNSUPPORTED_CONFIGURATION,
        ),
        (
            {"rope_parameters": {"rope_type": "yarn", "partial_rotary_factor": 0.5}},
            ExportReason.UNSUPPORTED_CONFIGURATION,
        ),
        ({"hidden_size": 0}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"num_local_experts": None}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"rms_norm_eps": float("nan")}, ExportReason.UNSUPPORTED_CONFIGURATION),
        ({"num_labels": 5}, ExportReason.LABEL_MAP_MISMATCH),
        ({"id2label": {}}, ExportReason.MISSING_LABEL_MAP),
        ({"id2label": {"0": "O", "2": "B-x"}}, ExportReason.MISSING_LABEL_MAP),
        (
            {"id2label": {"0": "O", "1": "B-x", "2": "I-x", "3": "S-x", "4": "E-x"}},
            ExportReason.LABEL_MAP_MISMATCH,
        ),
        ({"id2label": {"0": "B-x"}}, ExportReason.LABEL_MAP_MISMATCH),
    ],
)
def test_unsupported_configurations_fail_with_value_free_reasons(
    overrides: dict[str, Any], reason: ExportReason
) -> None:
    with pytest.raises(ExportQualificationError) as error:
        pf.privacy_filter_architecture(_config(**overrides))
    assert error.value.reason is reason
    assert str(error.value).startswith(f"{reason.value}: ")


def test_label_spaces_follow_opf_metadata_and_declarations() -> None:
    assert (
        pf.privacy_filter_label_space(_config()) == pf.OPENAI_PRIVACY_FILTER_LABEL_SPACE
    )
    names = ["O"] + [f"{tag}-email" for tag in "BIES"]
    custom = {str(index): name for index, name in enumerate(names)}
    with pytest.raises(ExportQualificationError) as error:
        pf.privacy_filter_label_space(_config(id2label=custom))
    assert error.value.reason is ExportReason.UNSUPPORTED_FAMILY

    declared = _config(
        id2label=custom,
        opf_metadata={
            "category_version": "nemotron_fine_v1",
            "ner_class_names": names,
            "inference_contract_version": 1,
        },
    )
    assert pf.privacy_filter_label_space(declared) == "nemotron_fine_v1"
    for metadata_block, reason in (
        ({"category_version": "unknown_v9"}, ExportReason.UNSUPPORTED_FAMILY),
        (
            {"category_version": "nemotron_fine_v1", "ner_class_names": names[::-1]},
            ExportReason.LABEL_MAP_MISMATCH,
        ),
        ({"inference_contract_version": 2}, ExportReason.UNSUPPORTED_CONFIGURATION),
    ):
        with pytest.raises(ExportQualificationError) as error:
            pf.privacy_filter_label_space(
                _config(id2label=custom, opf_metadata=metadata_block)
            )
        assert error.value.reason is reason


def test_source_check_requires_declared_checkpoint_and_pin(tmp_path: Path) -> None:
    record = pf.check_privacy_filter_source(
        _config(), model_id="openai/privacy-filter", expected_revision="c0ffee"
    )
    assert record == {
        "family": "openai-privacy-filter",
        "adapter_version": 1,
        "source_kind": "hub",
        "declared_checkpoint": "openai/privacy-filter",
        "declared_revision": None,
        "label_space": "openai-privacy-filter-v1",
        "num_labels": 33,
        "architecture_sha256": pf.privacy_filter_architecture(_config()).sha256,
        "revision_pinned": True,
    }
    local = pf.check_privacy_filter_source(
        _config(), model_id=str(tmp_path), expected_revision="sha256:abc"
    )
    assert local["source_kind"] == "local_directory"
    assert local["declared_checkpoint"] is None
    assert str(tmp_path) not in json.dumps(local)

    cases = (
        ("attacker/privacy-filter-copy", "c0ffee", ExportReason.UNSUPPORTED_FAMILY),
        ("openai/privacy-filter", None, ExportReason.SOURCE_REVISION_UNAVAILABLE),
        (
            "OpenMed/privacy-filter-nemotron",
            "c0ffee",
            ExportReason.LABEL_MAP_MISMATCH,
        ),
    )
    for model_id, revision, reason in cases:
        with pytest.raises(ExportQualificationError) as error:
            pf.check_privacy_filter_source(
                _config(), model_id=model_id, expected_revision=revision
            )
        assert error.value.reason is reason


def test_source_check_enforces_a_reviewed_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pinned = pf.PrivacyFilterCheckpoint(
        "openai/privacy-filter",
        pf.OPENAI_PRIVACY_FILTER_LABEL_SPACE,
        33,
        revision="reviewed123",
    )
    monkeypatch.setattr(pf, "PRIVACY_FILTER_CHECKPOINTS", (pinned,))
    assert pf.check_privacy_filter_source(
        _config(), model_id="openai/privacy-filter", expected_revision="reviewed123"
    )["declared_revision"] == ("reviewed123")
    with pytest.raises(ExportQualificationError) as error:
        pf.check_privacy_filter_source(
            _config(), model_id="openai/privacy-filter", expected_revision="other"
        )
    assert error.value.reason is ExportReason.SOURCE_REVISION_MISMATCH


@pytest.mark.parametrize(
    ("version", "native", "accepted"),
    [
        ("5.22.1", True, True),
        ("5.6.0", True, True),
        ("5.5.0", True, False),
        ("4.57.1", True, False),
        ("5.22.1", False, False),
        (None, True, False),
        ("dev", True, False),
    ],
)
def test_environment_requires_native_transformers_support(
    transformers_version, version: str | None, native: bool, accepted: bool
) -> None:
    transformers_version(version, native=native)
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    if accepted:
        adapter.validate_request(
            runtime="onnx", config=_config(), max_sequence_length=512
        )
        return
    with pytest.raises(ExportQualificationError) as error:
        adapter.validate_request(
            runtime="onnx", config=_config(), max_sequence_length=512
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_CONVERTER_ENVIRONMENT


def test_validate_request_rejects_undeclared_runtime_combinations(
    transformers_version,
) -> None:
    transformers_version("5.22.1")
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    for overrides, reason in (
        ({"runtime": "tflite"}, ExportReason.UNSUPPORTED_RUNTIME),
        ({"runtime": "coreml", "quantizations": ("int4",)}, "unsupported_quantization"),
        ({"runtime": "coreml", "shape_mode": "dynamic"}, "unsupported_sequence_shape"),
        ({"runtime": "onnx", "precisions": ("bfloat16",)}, "unsupported_precision"),
        (
            {"runtime": "onnx", "providers": ("CUDAExecutionProvider",)},
            "unsupported_provider",
        ),
        (
            {"runtime": "onnx", "max_sequence_length": 513},
            "unsupported_sequence_length",
        ),
    ):
        kwargs = {"config": _config(), "max_sequence_length": 512, **overrides}
        with pytest.raises(ExportQualificationError) as error:
            adapter.validate_request(**kwargs)
        assert error.value.reason_code == ExportReason(reason).value
    assert adapter.max_sequence_length(_config(max_position_embeddings=128)) == 128


def test_onnx_operator_set_fails_closed() -> None:
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    adapter.check_onnx_operators(["ai.onnx:MatMul", "ai.onnx:TopK"])
    for extra in ("ai.onnx:NonZero", "ai.onnx:Loop", "com.microsoft:MoE"):
        with pytest.raises(ExportQualificationError) as error:
            adapter.check_onnx_operators(["ai.onnx:MatMul", extra])
        assert error.value.reason is ExportReason.UNSUPPORTED_OPERATOR


# ---------------------------------------------------------------------------
# Decoder and threshold profile
# ---------------------------------------------------------------------------


def test_decoder_profile_is_versioned_digested_and_value_free() -> None:
    biases = {"transition_bias_background_stay": 0.25, "unknown": 9.0}
    profile = pf.privacy_filter_decoder_profile(
        _config(), viterbi_biases=biases, biases_source="viterbi_calibration.json"
    )
    assert profile["decoder"] == "bioes-viterbi"
    assert profile["label_space"] == "openai-privacy-filter-v1"
    assert profile["label_map_sha256"] == label_map_identity(OPF_LABELS)["sha256"]
    assert profile["viterbi"]["biases"]["transition_bias_background_stay"] == 0.25
    assert "unknown" not in profile["viterbi"]["biases"]
    assert len(profile["viterbi"]["biases"]) == 6
    assert profile["threshold_profile"]["confidence_threshold"] == 0.5
    assert profile["span_label_aliases"]["private_person"] == "PERSON"
    assert pf.verify_decoder_profile(profile) == profile

    tampered = dict(profile, threshold_profile={"confidence_threshold": 0.1})
    with pytest.raises(ExportQualificationError) as error:
        pf.verify_decoder_profile(tampered)
    assert error.value.reason is ExportReason.REPORT_INVALID
    for kwargs in (
        {"confidence_threshold": 1.5},
        {"viterbi_biases": {"transition_bias_background_stay": math.inf}},
    ):
        with pytest.raises(ExportQualificationError) as error:
            pf.privacy_filter_decoder_profile(_config(), **kwargs)
        assert error.value.reason is ExportReason.UNSUPPORTED_CONFIGURATION


def test_decoder_profile_round_trips_and_rejects_stale_versions(
    tmp_path: Path,
) -> None:
    profile = pf.privacy_filter_decoder_profile(_config())
    path = pf.write_decoder_profile(profile, tmp_path / pf.DECODER_PROFILE_FILENAME)
    assert pf.load_decoder_profile(path) == profile

    body = {key: value for key, value in profile.items() if key != "sha256"}
    body["adapter_version"] = 7
    body["sha256"] = pf._canonical_digest(body)
    with pytest.raises(ExportQualificationError) as error:
        pf.verify_decoder_profile(body)
    assert error.value.reason is ExportReason.ADAPTER_VERSION_MISMATCH
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ExportQualificationError) as error:
        pf.load_decoder_profile(tmp_path / "broken.json")
    assert error.value.reason is ExportReason.REPORT_INVALID


def test_viterbi_biases_come_from_local_calibration_without_network(
    tmp_path: Path,
) -> None:
    biases, source = pf.privacy_filter_viterbi_biases(str(tmp_path), _config())
    assert source == "default_zero"
    assert set(biases.values()) == {0.0}
    (tmp_path / pf.VITERBI_CALIBRATION_FILENAME).write_text(
        json.dumps(
            {
                "operating_points": {
                    "default": {"biases": {"transition_bias_end_to_start": -0.5}}
                }
            }
        ),
        encoding="utf-8",
    )
    biases, source = pf.privacy_filter_viterbi_biases(str(tmp_path), _config())
    assert source == "viterbi_calibration.json"
    assert biases["transition_bias_end_to_start"] == -0.5
    configured = _config(_mlx_viterbi_biases={"transition_bias_inside_to_end": 0.3})
    biases, source = pf.privacy_filter_viterbi_biases(str(tmp_path), configured)
    assert (source, biases["transition_bias_inside_to_end"]) == ("config", 0.3)
    (tmp_path / pf.VITERBI_CALIBRATION_FILENAME).write_text("[]", encoding="utf-8")
    with pytest.raises(ExportQualificationError) as error:
        pf.privacy_filter_viterbi_biases(str(tmp_path), _config())
    assert error.value.reason is ExportReason.UNSUPPORTED_CONFIGURATION


def _label_id(label: str) -> int:
    return next(index for index, value in OPF_LABELS.items() if value == label)


def _rows(tags: list[str]) -> list[list[float]]:
    rows = []
    for tag in tags:
        row = [0.0] * len(OPF_LABELS)
        row[_label_id(tag)] = 9.0
        rows.append(row)
    return rows


def test_canonical_decoder_uses_viterbi_thresholds_and_refinement() -> None:
    text = "Ana Lima ana@example.org"
    offsets = [(0, 3), (3, 8), (8, 9), (9, 12), (12, 24)]
    tags = [
        "B-private_person",
        "E-private_person",
        "O",
        "B-private_email",
        "E-private_email",
    ]
    spans = pf.decode_privacy_filter_logits(
        _rows(tags), offsets, text, id2label=OPF_LABELS, biases={}
    )
    assert [(span["label"], text[span["start"] : span["end"]]) for span in spans] == [
        ("private_person", "Ana Lima"),
        ("private_email", "ana@example.org"),
    ]
    assert all(span["score"] > 0.99 for span in spans)

    # Viterbi repairs an invalid I-without-B start into a valid span.
    repaired = _rows(["I-private_person", "E-private_person", "O", "O", "O"])
    repaired[0][_label_id("B-private_person")] = 8.5
    spans = pf.decode_privacy_filter_logits(
        repaired, offsets, text, id2label=OPF_LABELS, biases={}
    )
    assert [text[s["start"] : s["end"]] for s in spans] == ["Ana Lima"]

    flat = [[0.0] * len(OPF_LABELS) for _ in offsets]
    for row in flat:
        row[_label_id("S-private_person")] = 1.0
    assert (
        pf.decode_privacy_filter_logits(
            flat,
            offsets,
            text,
            id2label=OPF_LABELS,
            biases={},
            confidence_threshold=0.5,
        )
        == []
    )
    assert (
        pf.decode_privacy_filter_logits([], [], text, id2label=OPF_LABELS, biases={})
        == []
    )


def test_qualification_span_decoder_applies_profile_and_aliases() -> None:
    profile = pf.privacy_filter_decoder_profile(_config())
    decode = pf.privacy_filter_span_decoder(profile)
    text = "Ana Lima"
    rows = _rows(["B-private_person", "E-private_person", "O"])
    spans = decode(
        rows,
        offsets=[(0, 3), (3, 8), (0, 0)],
        attention=[1, 1, 0],
        id2label=OPF_LABELS,
        text=text,
    )
    assert spans == [{"label": "PERSON", "start": 0, "end": 8}]


@pytest.mark.parametrize(
    ("text", "offsets", "surface"),
    [
        (
            "Pt ÉLODIE VARNESSE ok",
            [(0, 2), (3, 10), (10, 18), (19, 21)],
            "ÉLODIE VARNESSE",
        ),
        ("患者欧阳璇珂今日", [(0, 2), (2, 4), (4, 6), (6, 8)], "欧阳璇珂"),
        ("رقم ٧٧٣١٠٠٤٥ هنا", [(0, 3), (4, 8), (8, 12), (13, 16)], "٧٧٣١٠٠٤٥"),
        ("खाता ४४१७२२९० है", [(0, 4), (5, 9), (9, 13), (14, 16)], "४४१७२२९०"),
    ],
)
def test_canonical_decoder_keeps_multilingual_identifiers_whole(
    text: str, offsets: list[tuple[int, int]], surface: str
) -> None:
    label = (
        "private_person"
        if not any(ch.isdigit() for ch in surface)
        else "account_number"
    )
    tags = ["O", f"B-{label}", f"E-{label}", "O"]
    spans = pf.decode_privacy_filter_logits(
        _rows(tags), offsets, text, id2label=OPF_LABELS, biases={}
    )
    assert [(span["label"], text[span["start"] : span["end"]]) for span in spans] == [
        (label, surface)
    ]


def test_suite_is_synthetic_multilingual_and_direct_identifier_heavy() -> None:
    from openmed.eval.export_qualification import (
        direct_identifier_labels,
        load_qualification_fixtures,
    )

    fixtures, identity = load_qualification_fixtures(
        pf.PRIVACY_FILTER_QUALIFICATION_SUITE
    )
    assert identity["synthetic"] is True
    assert len(identity["languages"]) >= 10
    assert {"ar", "hi", "ja", "ko", "zh"} <= set(identity["languages"])
    assert {"PERSON", "EMAIL", "PHONE", "ACCOUNT_NUMBER"} <= set(
        direct_identifier_labels(fixtures)
    )
    rows = [
        json.loads(line)
        for line in pf.PRIVACY_FILTER_QUALIFICATION_SUITE.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for row in rows:
        assert row["metadata"]["synthetic"] is True
        for span in row["gold_spans"]:
            assert row["text"][span["start"] : span["end"]] == span["text"]
    aliases = set(pf.OPENAI_PRIVACY_FILTER_LABEL_ALIASES.values())
    assert set(identity["labels"]) <= aliases


# ---------------------------------------------------------------------------
# Real tracing against the native reference
# ---------------------------------------------------------------------------


def _native_ready() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        return importlib.util.find_spec(pf.PRIVACY_FILTER_NATIVE_MODULE) is not None
    except (ImportError, ValueError):
        return False


requires_native = pytest.mark.skipif(
    not _native_ready(),
    reason="Privacy Filter tracing needs torch and transformers>=5.6",
)


def _stub(tmp_path: Path):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from tests.fixtures.privacy_filter_export_stub import build_tiny_privacy_filter

    checkpoint = build_tiny_privacy_filter(tmp_path / "checkpoint")
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    reference = adapter.reference_model(torch, str(checkpoint))
    model = AutoModelForTokenClassification.from_pretrained(checkpoint).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return torch, checkpoint, reference, model, tokenizer


@requires_native
def test_reference_loader_is_native_eager_float32(tmp_path: Path) -> None:
    torch, _, reference, _, _ = _stub(tmp_path)
    assert type(reference).__module__.startswith(pf.PRIVACY_FILTER_NATIVE_MODULE)
    assert reference.config._attn_implementation == "eager"
    assert next(reference.parameters()).dtype == torch.float32
    with pytest.raises(ExportQualificationError) as error:
        pf.require_native_privacy_filter(object())
    assert error.value.reason is ExportReason.UNSUPPORTED_REMOTE_CODE


@requires_native
def test_weight_mapping_fuses_qkv_flattens_experts_and_keeps_bias(
    tmp_path: Path,
) -> None:
    torch, _, _, model, _ = _stub(tmp_path)
    state = model.state_dict()
    architecture = pf.privacy_filter_architecture(model.config)
    mapped = pf.map_privacy_filter_weights(torch, state, architecture)
    q = state["model.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(mapped["layers.0.qkv.weight"][: q.shape[0]], q)
    gate_up = state["model.layers.1.mlp.experts.gate_up_proj"]
    flat = mapped["layers.1.gate_up_weight"].reshape(
        architecture.hidden_size, architecture.num_experts, -1
    )
    assert torch.equal(flat.permute(1, 0, 2), gate_up)
    assert "classifier.bias" in mapped
    assert all(tensor.dtype == torch.float32 for tensor in mapped.values())

    without_bias = {key: value for key, value in state.items() if key != "score.bias"}
    assert "classifier.bias" not in pf.map_privacy_filter_weights(
        torch, without_bias, architecture
    )
    for broken in (
        {k: v for k, v in state.items() if k != "model.layers.1.self_attn.sinks"},
        {**state, "model.extra.weight": torch.zeros(1)},
        {**state, "model.norm.weight": torch.zeros(3)},
    ):
        with pytest.raises(ExportQualificationError) as error:
            pf.map_privacy_filter_weights(torch, broken, architecture)
        assert error.value.reason is ExportReason.UNSUPPORTED_CONFIGURATION


@requires_native
def test_export_module_reproduces_reference_with_padding(tmp_path: Path) -> None:
    from tests.fixtures.privacy_filter_export_stub import rule_fixture_rows

    torch, _, reference, model, tokenizer = _stub(tmp_path)
    adapter = pf.PRIVACY_FILTER_EXPORT_ADAPTER
    adapter.prepare_model(model)
    wrapper = adapter.build_wrapper(torch, model)
    texts = [row["text"] for row in rule_fixture_rows()]
    batch = tokenizer(texts, padding="max_length", max_length=64, return_tensors="pt")
    mask = batch["attention_mask"].bool()
    with torch.no_grad():
        expected = reference(**batch).logits
        actual = wrapper(batch["input_ids"], batch["attention_mask"])
        assert (expected - actual).abs()[mask].max().item() < 1e-4

        # Each architectural piece matters: breaking it shows up as drift.
        wrapper.sliding_window = 10_000
        broken_window = wrapper(batch["input_ids"], batch["attention_mask"])
        wrapper.sliding_window = model.config.sliding_window
        assert (expected - broken_window).abs()[mask].max().item() > 1e-2

        layer = wrapper.layers[0]
        sinks = layer.sinks.detach().clone()
        layer.sinks.zero_()
        broken_sinks = wrapper(batch["input_ids"], batch["attention_mask"])
        layer.sinks.copy_(sinks)
        assert (expected - broken_sinks).abs()[mask].max().item() > 1e-3

        router = layer.router.weight.detach().clone()
        layer.router.weight.copy_(router.flip(0))
        broken_routing = wrapper(batch["input_ids"], batch["attention_mask"])
        layer.router.weight.copy_(router)
        assert (expected - broken_routing).abs()[mask].max().item() > 1e-3

        scaling = wrapper.attention_scaling
        wrapper.attention_scaling = 1.0
        broken_rope = wrapper(batch["input_ids"], batch["attention_mask"])
        wrapper.attention_scaling = scaling
        assert (expected - broken_rope).abs()[mask].max().item() > 1e-3


@requires_native
def test_binding_decodes_the_stub_suite_like_the_gold_spans(tmp_path: Path) -> None:
    from tests.fixtures.privacy_filter_export_stub import (
        LABELS,
        VITERBI_BIASES,
        rule_fixture_rows,
    )

    torch, checkpoint, reference, model, tokenizer = _stub(tmp_path)
    binding = pf.PRIVACY_FILTER_EXPORT_ADAPTER.binding(model.config, str(checkpoint))
    assert binding.gate_languages is True
    assert binding.fixtures_path == pf.PRIVACY_FILTER_QUALIFICATION_SUITE
    viterbi = binding.decoder_profile["viterbi"]
    assert viterbi["source"] == "viterbi_calibration.json"
    assert viterbi["biases"] == VITERBI_BIASES
    for row in rule_fixture_rows():
        encoded = tokenizer(row["text"], return_offsets_mapping=True)
        ids = torch.tensor([encoded["input_ids"]])
        with torch.no_grad():
            logits = reference(
                input_ids=ids, attention_mask=torch.ones_like(ids)
            ).logits
        spans = binding.span_decoder(
            logits[0].tolist(),
            offsets=encoded["offset_mapping"],
            attention=[1] * ids.shape[1],
            id2label=LABELS,
            text=row["text"],
        )
        assert [(s["label"], row["text"][s["start"] : s["end"]]) for s in spans] == [
            (span["label"], span["text"]) for span in row["gold_spans"]
        ], row["language"]


def test_adapter_registration_survives_either_import_order() -> None:
    assert export_adapters.resolve_export_adapter("openai-privacy-filter") is (
        pf.PRIVACY_FILTER_EXPORT_ADAPTER
    )
