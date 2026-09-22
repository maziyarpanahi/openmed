from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema import Draft202012Validator

from openmed.clinical.model_packs import (
    MODEL_FAMILY_CAPABILITIES,
    CalibrationMetadata,
    ClinicalTaskRequest,
    ClinicalTaskRouter,
    FallbackPolicy,
    LocalArtifactBinding,
    ModelPackEntry,
    ModelPackError,
    ModelPackManifest,
    QuantizationMetadata,
    digest_local_artifact,
    load_model_pack,
    load_model_pack_json,
    load_model_pack_schema,
    load_model_route_schema,
)
from openmed.structured.store import StoreState

MODEL_BYTES = b"bounded-encoder-v1"
MODEL_DIGEST = "sha256:690e1aa7419a956ff38e5887861bb90fc2f724b3ce9c349f1a4beda65d8c8558"
RULE_DIGEST = "sha256:3ae5d5b4b2aadbf4480c136eec6b054d42c97279ba43c498f4baf4245c8f726f"
CALIBRATION_DIGEST = (
    "sha256:e152337e4e85aa3e81482f0ce329aec7bfad531413fe53fef84f1f0d4165caee"
)
HOLDOUT_DIGEST = (
    "sha256:d179b28cc8927b726d766bc0a22ed40ffc879d40f62a100e7d97dabb24253669"
)
QUANT_EVAL_DIGEST = (
    "sha256:0a3d0097d0963800c05fa0b91ce4aeebb5b4d8c7d3d9275b7631770ef70c9940"
)
GOLDEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "clinical"
    / "model_pack_golden.json"
)


def _calibration() -> CalibrationMetadata:
    return CalibrationMetadata(
        method="temperature",
        threshold=0.7,
        calibration_dataset_digest=CALIBRATION_DIGEST,
        holdout_dataset_digest=HOLDOUT_DIGEST,
        seed=17,
    )


def _entry(
    *,
    alias: str = "classifier.deberta",
    task: str = "classification",
    family: str = "deberta_v2",
    artifact_id: str = "local/clinical-classifier",
    artifact_kind: str = "model",
    artifact_digest: str = MODEL_DIGEST,
    runtime: str = "torch",
    model_kind: str = "bounded",
    fallback: str | None = None,
    license_id: str = "apache-2.0",
    quantization: QuantizationMetadata | None = None,
    priority: int = 10,
    experimental: bool = False,
) -> ModelPackEntry:
    return ModelPackEntry(
        alias=alias,
        task=task,
        family=family,
        artifact_id=artifact_id,
        artifact_kind=artifact_kind,
        revision="local-v1" if artifact_kind == "model" else "builtin-v1",
        artifact_digest=artifact_digest,
        license=license_id,
        runtime=runtime,
        model_kind=model_kind,
        output_schema="clinical_component",
        output_schema_version="1.0.0",
        languages=("en",),
        domains=("clinical",),
        calibration=_calibration()
        if artifact_kind == "model"
        else CalibrationMetadata(method="none"),
        quantization=quantization or QuantizationMetadata(mode="fp32"),
        fallback=FallbackPolicy(
            mode="alias" if fallback is not None else "none", alias=fallback
        ),
        priority=priority,
        experimental=experimental,
    )


def _rules(*, alias: str = "rules.classification", task: str = "classification"):
    return _entry(
        alias=alias,
        task=task,
        family="deterministic_rules",
        artifact_id=f"builtin/{alias}",
        artifact_kind="builtin",
        artifact_digest=RULE_DIGEST,
        runtime="builtin",
        model_kind="deterministic",
        priority=900,
    )


def _pack(*entries: ModelPackEntry) -> ModelPackManifest:
    return ModelPackManifest(pack_id="clinical.local.v1", entries=entries)


def _request(**changes: object) -> ClinicalTaskRequest:
    values: dict[str, object] = {
        "task": "classification",
        "output_schema": "clinical_component",
        "output_schema_version": "1.0.0",
        "language": "en",
        "domain": "clinical",
    }
    values.update(changes)
    return ClinicalTaskRequest(**values)  # type: ignore[arg-type]


def _file_binding(path: Path, *, alias: str = "classifier.deberta"):
    path.write_bytes(MODEL_BYTES)
    return LocalArtifactBinding(alias=alias, runtime="torch", path=path)


def _builtin_binding(*, alias: str = "rules.classification"):
    return LocalArtifactBinding(
        alias=alias,
        runtime="builtin",
        builtin_id=f"builtin/{alias}",
        builtin_digest=RULE_DIGEST,
    )


def test_same_manifest_resolves_same_pinned_artifact_offline(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(_file_binding(model),),
        available_runtimes=("torch",),
    )

    first = router.route(_request())
    second = router.route(_request())

    assert first.ok and second.ok
    assert first.value == second.value
    assert first.value is not None
    assert first.value.artifact_digest == MODEL_DIGEST
    assert first.value.local_reference == str(model.resolve())


def test_missing_optional_runtime_uses_explicit_deterministic_fallback() -> None:
    primary = _entry(runtime="mlx", fallback="rules.classification")
    fallback = _rules()
    router = ClinicalTaskRouter(
        _pack(primary, fallback),
        bindings=(_builtin_binding(),),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.ok
    assert result.value is not None
    assert result.value.alias == "rules.classification"
    assert result.value.fallback_from == "classifier.deberta"
    assert result.value.local_reference == "builtin://builtin/rules.classification"


@pytest.mark.parametrize(
    "changes",
    [
        {"output_schema": "different_schema"},
        {"output_schema_version": "2.0.0"},
        {"languages": ("fr",)},
        {"domains": ("legal",)},
    ],
)
def test_fallback_must_satisfy_request_contract(changes: dict[str, object]) -> None:
    primary = _entry(runtime="mlx", fallback="rules.classification")
    fallback = replace(_rules(), **changes)
    router = ClinicalTaskRouter(
        _pack(primary, fallback),
        bindings=(_builtin_binding(),),
        available_runtimes=("torch",),
    )
    result = router.route(_request())
    assert result.state is StoreState.CONFLICT
    assert result.code == "fallback_contract_mismatch"
    assert result.value is None


def test_missing_runtime_without_fallback_is_typed_unsupported() -> None:
    router = ClinicalTaskRouter(
        _pack(_entry(runtime="mlx")),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "runtime_unavailable"


def test_license_gate_fails_before_runtime_or_artifact_resolution() -> None:
    router = ClinicalTaskRouter(
        _pack(_entry(license_id="research-only", runtime="missing")),
        bindings=(),
        available_runtimes=(),
    )

    result = router.route(_request())

    assert result.state is StoreState.DENIED
    assert result.code == "license_denied"


def test_artifact_integrity_mismatch_fails_closed_without_fallback(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"changed")
    primary = _entry(fallback="rules.classification")
    router = ClinicalTaskRouter(
        _pack(primary, _rules()),
        bindings=(
            LocalArtifactBinding(
                alias=primary.alias,
                runtime="torch",
                path=model,
            ),
            _builtin_binding(),
        ),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.state is StoreState.CONFLICT
    assert result.code == "artifact_digest_mismatch"


def test_quantized_delta_gate_fails_before_artifact_resolution() -> None:
    quantization = QuantizationMetadata(
        mode="int8",
        metric="macro_f1",
        observed_delta=0.08,
        maximum_delta=0.02,
        evaluation_digest=QUANT_EVAL_DIGEST,
    )
    router = ClinicalTaskRouter(
        _pack(_entry(quantization=quantization)),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.state is StoreState.DENIED
    assert result.code == "quantization_delta_exceeded"


def test_quantized_entry_with_seeded_calibration_and_holdout_routes(
    tmp_path: Path,
) -> None:
    quantization = QuantizationMetadata(
        mode="int8",
        metric="macro_f1",
        observed_delta=0.01,
        maximum_delta=0.02,
        evaluation_digest=QUANT_EVAL_DIGEST,
    )
    model = tmp_path / "model.bin"
    router = ClinicalTaskRouter(
        _pack(_entry(quantization=quantization)),
        bindings=(_file_binding(model),),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.ok


def test_generative_entry_is_never_implicit(tmp_path: Path) -> None:
    entry = _entry(model_kind="generative", experimental=True)
    model = tmp_path / "model.bin"
    router = ClinicalTaskRouter(
        _pack(entry),
        bindings=(_file_binding(model),),
        available_runtimes=("torch",),
    )

    denied = router.route(_request())
    allowed = router.route(
        _request(
            requested_alias=entry.alias,
            allow_experimental_generative=True,
        )
    )

    assert denied.state is StoreState.DENIED
    assert denied.code == "generative_profile_required"
    assert allowed.ok


def test_bounded_entry_wins_even_when_generative_has_higher_priority(
    tmp_path: Path,
) -> None:
    generative = _entry(
        alias="generator.experimental",
        model_kind="generative",
        experimental=True,
        priority=0,
    )
    bounded = _entry(priority=100)
    model = tmp_path / "model.bin"
    generator = tmp_path / "generator.bin"
    router = ClinicalTaskRouter(
        _pack(generative, bounded),
        bindings=(
            _file_binding(model),
            _file_binding(generator, alias="generator.experimental"),
        ),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.ok
    assert result.value is not None
    assert result.value.alias == "classifier.deberta"


@pytest.mark.parametrize(
    ("task", "family"),
    [
        ("span_extraction", "gliner2_5"),
        ("classification", "deberta_v2"),
        ("pair_scoring", "modernbert"),
        ("relation_extraction", "deberta_v2"),
        ("assertion", "modernbert"),
        ("temporality", "modernbert"),
        ("token_classification", "openmed_token_classifier"),
    ],
)
def test_bounded_specialist_task_shapes_are_registered(
    tmp_path: Path, task: str, family: str
) -> None:
    alias = f"specialist.{task}"
    model = tmp_path / f"{task}.bin"
    entry = _entry(alias=alias, task=task, family=family)
    router = ClinicalTaskRouter(
        _pack(entry),
        bindings=(_file_binding(model, alias=alias),),
        available_runtimes=("torch",),
    )

    result = router.route(_request(task=task, requested_alias=alias))

    assert result.ok
    assert result.value is not None
    assert result.value.task == task


def test_default_family_capabilities_cover_bounded_specialists() -> None:
    assert MODEL_FAMILY_CAPABILITIES["gliner2_5"].zero_shot_labels is True
    assert MODEL_FAMILY_CAPABILITIES["gliner2_5"].tasks == ("span_extraction",)
    assert "classification" in MODEL_FAMILY_CAPABILITIES["deberta_v2"].tasks
    assert "pair_scoring" in MODEL_FAMILY_CAPABILITIES["modernbert"].tasks
    assert MODEL_FAMILY_CAPABILITIES["openmed_token_classifier"].tasks == (
        "token_classification",
    )
    assert set(MODEL_FAMILY_CAPABILITIES["deterministic_rules"].tasks) == {
        "assertion",
        "classification",
        "pair_scoring",
        "relation_extraction",
        "span_extraction",
        "temporality",
        "token_classification",
    }


def test_language_and_domain_mismatch_is_typed_unsupported() -> None:
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request(language="fr", domain="oncology"))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "task_not_supported"


def test_same_major_output_schema_is_compatible(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(_file_binding(model),),
        available_runtimes=("torch",),
    )

    result = router.route(_request(output_schema_version="1.9.0"))

    assert result.ok


def test_different_major_output_schema_is_unsupported() -> None:
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request(output_schema_version="2.0.0"))

    assert result.state is StoreState.UNSUPPORTED


def test_unknown_requested_alias_does_not_fall_through() -> None:
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request(requested_alias="unknown.local"))

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "alias_not_supported"


def test_remote_identifier_is_metadata_and_never_an_implicit_binding() -> None:
    entry = replace(_entry(), artifact_id="models.example/model")
    router = ClinicalTaskRouter(
        _pack(entry),
        bindings=(),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.state is StoreState.UNKNOWN
    assert result.code == "alias_unbound"


def test_binding_runtime_must_match_manifest(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(MODEL_BYTES)
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(
            LocalArtifactBinding(alias="classifier.deberta", runtime="mlx", path=model),
        ),
        available_runtimes=("torch", "mlx"),
    )

    result = router.route(_request())

    assert result.state is StoreState.CONFLICT
    assert result.code == "runtime_binding_mismatch"


def test_manifest_json_round_trip_and_schemas(tmp_path: Path) -> None:
    manifest = _pack(_entry(), _rules())
    path = tmp_path / "model-pack.json"
    path.write_text(manifest.to_json(), encoding="utf-8")

    loaded = load_model_pack(path)
    pack_schema = load_model_pack_schema()
    route_schema = load_model_route_schema()

    assert loaded == manifest
    Draft202012Validator.check_schema(pack_schema)
    Draft202012Validator(pack_schema).validate(manifest.to_dict())
    Draft202012Validator.check_schema(route_schema)


def test_static_golden_model_pack_matches_public_contract() -> None:
    manifest = load_model_pack(GOLDEN_PATH)

    Draft202012Validator(load_model_pack_schema()).validate(manifest.to_dict())
    assert manifest.pack_id == "clinical.local.v1"
    assert manifest.entries[0].fallback.alias == "rules.classification"


def test_route_serialization_excludes_local_path(tmp_path: Path) -> None:
    model = tmp_path / "private-cache" / "model.bin"
    model.parent.mkdir()
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(_file_binding(model),),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.value is not None
    serialized = result.value.to_dict()
    Draft202012Validator(load_model_route_schema()).validate(serialized)
    assert "local_reference" not in serialized
    assert str(tmp_path) not in json.dumps(serialized)


def test_duplicate_json_keys_and_nonfinite_values_are_rejected() -> None:
    with pytest.raises(ModelPackError, match="duplicate"):
        load_model_pack_json('{"pack_id":"a","pack_id":"b"}')
    with pytest.raises(ModelPackError, match="non-finite"):
        load_model_pack_json('{"value":NaN}')


def test_manifest_rejects_missing_alias_and_fallback_cycles() -> None:
    with pytest.raises(ModelPackError, match="not present"):
        _pack(_entry(fallback="missing.alias"))
    first = _entry(alias="first.entry", fallback="second.entry")
    second = _entry(alias="second.entry", fallback="first.entry")
    with pytest.raises(ModelPackError, match="cycle"):
        _pack(first, second)


def test_unpinned_revision_and_unseeded_calibration_are_rejected() -> None:
    with pytest.raises(ModelPackError, match="immutable"):
        replace(_entry(), revision="main")
    with pytest.raises(ModelPackError, match="seed"):
        replace(_calibration(), seed=None)


def test_digest_local_artifact_is_stable_and_path_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.bin").write_bytes(b"same")
    (first / "b.bin").write_bytes(b"other")
    (second / "a.bin").write_bytes(b"same")
    (second / "c.bin").write_bytes(b"other")

    assert digest_local_artifact(first) == digest_local_artifact(first)
    assert digest_local_artifact(first) != digest_local_artifact(second)


def test_symlink_artifact_is_denied(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(MODEL_BYTES)
    link = tmp_path / "link.bin"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")
    router = ClinicalTaskRouter(
        _pack(_entry()),
        bindings=(
            LocalArtifactBinding(
                alias="classifier.deberta", runtime="torch", path=link
            ),
        ),
        available_runtimes=("torch",),
    )

    result = router.route(_request())

    assert result.state is StoreState.DENIED
    assert result.code == "artifact_path_unsafe"


@given(priority=st.integers(min_value=0, max_value=10_000))
def test_manifest_digest_and_resolution_are_deterministic_for_priority(
    priority: int,
) -> None:
    entry = replace(_rules(), priority=priority)
    manifest = _pack(entry)
    router = ClinicalTaskRouter(
        manifest,
        bindings=(_builtin_binding(),),
        available_runtimes=(),
    )

    first = router.route(_request())
    second = router.route(_request())

    assert manifest.digest == load_model_pack_json(manifest.to_json()).digest
    assert first.value == second.value


def test_manifest_digest_changes_with_pinned_artifact() -> None:
    first = _pack(_entry())
    changed = _pack(
        replace(
            _entry(),
            artifact_digest="sha256:" + "1" * 64,
        )
    )

    assert first.digest != changed.digest
