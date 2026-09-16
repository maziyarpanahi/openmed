"""Chinese clinical NER foundation, leakage-gate, and local-asset tests."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import pytest

from openmed.core.labels import (
    BODY_SITE,
    CANONICAL_LABELS,
    CONDITION,
    DEVICE,
    JOB_DEPARTMENT,
    LAB_TEST,
    MEDICATION,
    MICROORGANISM,
    OTHER,
    PROCEDURE,
    normalize_label,
)
from openmed.eval.datasets.cmeee import load_cmeee, map_cmeee_label
from openmed.eval.datasets.multilingual_ner import MultilingualNerCorpusRequired
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan
from openmed.eval.suites.chinese_clinical_ner import (
    CHINESE_CLINICAL_NER,
    CHINESE_CLINICAL_NER_MODEL_DIR_ENV,
    MULTILINGUAL_FALLBACK_MODEL,
    ChineseClinicalNerAssetUnavailable,
    ChineseClinicalNerContractError,
    ChineseClinicalNerLeakageError,
    ChineseLocalModelAsset,
    chinese_clinical_ner_metadata,
    chinese_local_model_asset_skip_reason,
    load_chinese_clinical_ner_fixtures,
    resolve_chinese_local_model_asset,
    run_chinese_clinical_ner_conformance,
    run_chinese_clinical_ner_suite,
    run_synthetic_chinese_clinical_ner_smoke,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

CMEEE_EXPECTED = {
    "bod": BODY_SITE,
    "dep": JOB_DEPARTMENT,
    "dis": CONDITION,
    "dru": MEDICATION,
    "equ": DEVICE,
    "ite": LAB_TEST,
    "mic": MICROORGANISM,
    "pro": PROCEDURE,
    "sym": CONDITION,
}


def test_cmeee_labels_normalize_through_chinese_core_mapping() -> None:
    for source_label, expected in CMEEE_EXPECTED.items():
        assert normalize_label(source_label, lang="zh") == expected
        mapping = map_cmeee_label(source_label)
        assert mapping.canonical_label == expected
        assert mapping.mapped is True
        assert mapping.canonical_label in CANONICAL_LABELS

    core_clinical = {"bod", "dis", "dru", "equ", "ite", "mic", "pro", "sym"}
    assert all(CMEEE_EXPECTED[label] != OTHER for label in core_clinical)


@pytest.mark.parametrize(
    ("source_label", "expected"),
    [
        ("body_site", BODY_SITE),
        ("Body Site", BODY_SITE),
        ("lab_test", LAB_TEST),
        ("Lab-Test", LAB_TEST),
    ],
)
def test_cmeee_long_labels_normalize_for_chinese_locales(
    source_label: str,
    expected: str,
) -> None:
    assert normalize_label(source_label, lang="zh_CN") == expected


def test_cmeee_real_loader_requires_an_explicit_external_path() -> None:
    with pytest.raises(MultilingualNerCorpusRequired, match="explicit local"):
        load_cmeee()


def test_bundled_chinese_fixture_covers_all_categories_and_synthetic_phi() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    assert len(fixtures) == 2
    assert all(fixture.language == "zh" for fixture in fixtures)
    assert all(fixture.metadata["synthetic"] is True for fixture in fixtures)
    assert all(fixture.metadata["contains_real_phi"] is False for fixture in fixtures)
    clinical = fixtures[0]
    assert {span.metadata["source_label"] for span in clinical.gold_spans} == set(
        CMEEE_EXPECTED
    )
    assert all(
        clinical.text[span.start : span.end] == span.text
        for span in clinical.gold_spans
    )
    assert len(fixtures[1].metadata["phi_spans"]) == 3


def test_synthetic_suite_reports_per_label_metrics_and_zero_leakage() -> None:
    report = run_synthetic_chinese_clinical_ner_smoke()

    assert report.suite == CHINESE_CLINICAL_NER
    assert report.metrics["gate"]["passed"] is True
    assert report.metrics["phi_token_leakage"] == {
        "findings": [],
        "leaked_tokens": 0,
        "rate": 0.0,
        "total_tokens": 3,
    }
    assert report.metrics["per_label"][JOB_DEPARTMENT]["recall"] == 1.0
    assert report.metrics["per_label"][MICROORGANISM]["precision"] == 1.0
    assert "user-supplied local inputs" in report.metadata["data_boundary"]
    assert "model-card evidence" in report.metadata["model_notice"]


def test_model_evidence_tracks_the_routed_chinese_default() -> None:
    from openmed.core.language_pack_catalog import (
        DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
        DEFAULT_PII_MODELS,
    )

    evidence = chinese_clinical_ner_metadata()["model_evidence"]
    placeholder_routed = "zh" in DEFAULT_MODEL_PLACEHOLDER_LANGUAGES

    # Track the catalog in both directions: if zh were ever demoted back to a
    # placeholder the suite must report the fallback, not the catalog entry.
    if placeholder_routed:
        assert evidence["routed_default_model"] == MULTILINGUAL_FALLBACK_MODEL
    else:
        assert evidence["routed_default_model"] == DEFAULT_PII_MODELS["zh"]
    assert evidence["weights_bundled"] is False
    assert evidence["dedicated_zh_model"] is (
        not placeholder_routed
        and DEFAULT_PII_MODELS["zh"] != MULTILINGUAL_FALLBACK_MODEL
    )
    # The suite must not re-assert a fallback that master no longer routes to.
    assert (
        "multilingual routing placeholder"
        not in (chinese_clinical_ner_metadata()["model_notice"])
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, float("nan")])
def test_suite_rejects_invalid_per_label_recall_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="min_per_label_recall"):
        run_chinese_clinical_ner_suite(
            load_chinese_clinical_ner_fixtures(),
            model_name="synthetic-oracle",
            runner=_identity_runner,
            redactor=lambda fixture, predicted: "",
            min_per_label_recall=threshold,
        )


def test_suite_raises_without_exposing_surviving_identifier_text() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    with pytest.raises(ChineseClinicalNerLeakageError) as caught:
        run_chinese_clinical_ner_suite(
            fixtures,
            model_name="synthetic-leaky",
            runner=_identity_runner,
            redactor=lambda fixture, predicted: fixture.text,
        )

    report = caught.value.report
    leakage = report.metrics["phi_token_leakage"]
    assert leakage["rate"] == 1.0
    assert leakage["leaked_tokens"] == 3
    assert all(
        set(finding) == {"end", "fixture_id", "label", "start", "token_hash"}
        for finding in leakage["findings"]
    )
    serialized = report.to_json()
    assert "王芳" not in serialized
    assert "CN123456" not in serialized
    assert "13800138000" not in serialized


def test_conformance_stub_adapter_matches_gold_offsets_and_canonical_labels() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    report = run_chinese_clinical_ner_conformance(
        _OracleLocalModelAdapter(fixtures),
        fixtures=fixtures,
        min_per_label_recall=1.0,
        clock=_StepClock(),
    )

    assert report.suite == CHINESE_CLINICAL_NER
    assert report.model_name == "synthetic-local-checkpoint"
    assert report.metrics["gate"]["passed"] is True
    assert report.metrics["gate"]["failures"] == []
    assert report.metrics["exact_span_f1"]["precision"] == 1.0
    assert report.metrics["exact_span_f1"]["recall"] == 1.0
    assert set(report.metrics["per_label"]) <= set(CANONICAL_LABELS)
    assert all(
        metrics["recall"] == 1.0 for metrics in report.metrics["per_label"].values()
    )


def test_conformance_report_carries_per_label_latency_and_hashed_leakage() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    with pytest.raises(ChineseClinicalNerLeakageError) as caught:
        run_chinese_clinical_ner_conformance(
            _LeakyLocalModelAdapter(fixtures),
            fixtures=fixtures,
            asset=_asset_stub(),
            clock=_StepClock(),
        )

    report = caught.value.report
    # Two fixtures, one deterministic 1.0 ms tick per prediction.
    assert report.metrics["latency"] == {
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
        "count": 2,
    }
    assert report.metrics["per_label"][MICROORGANISM]["recall"] == 1.0
    leakage = report.metrics["phi_token_leakage"]
    assert leakage["leaked_tokens"] == 3
    assert all(
        set(finding) == {"end", "fixture_id", "label", "start", "token_hash"}
        for finding in leakage["findings"]
    )
    asset_evidence = report.metadata["local_model_asset"]
    assert asset_evidence["fingerprint"].startswith("sha256:")
    assert asset_evidence["weight_file_suffixes"] == [".safetensors"]
    assert "not the measured quality" in report.metadata["conformance"]

    serialized = report.to_json()
    assert "王芳" not in serialized
    assert "CN123456" not in serialized
    assert "13800138000" not in serialized
    # No component of the local path survives: not the directory, not the
    # parent, and not the username embedded in it.
    assert "/home/alice/zh-clinical-ner-stub" not in serialized
    assert "zh-clinical-ner-stub" not in serialized
    assert "alice" not in serialized


@pytest.mark.parametrize(
    ("mutation", "expected_problem"),
    [
        ("out_of_bounds", "span_out_of_bounds"),
        ("inverted", "span_not_positive_length"),
        ("non_canonical_label", "label_not_canonical"),
        ("stale_text", "span_text_offset_mismatch"),
    ],
)
def test_conformance_enforces_the_span_and_label_contract(
    mutation: str,
    expected_problem: str,
) -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    with pytest.raises(ChineseClinicalNerContractError) as caught:
        run_chinese_clinical_ner_conformance(
            _ContractBreakingAdapter(fixtures, mutation),
            fixtures=fixtures,
            fail_on_leakage=False,
            clock=_StepClock(),
        )

    findings = caught.value.findings
    assert findings
    assert any(expected_problem in finding["problems"] for finding in findings)
    # Contract evidence is surface-free.
    assert all(
        set(finding) == {"fixture_id", "start", "end", "label", "problems"}
        for finding in findings
    )
    assert "王芳" not in json.dumps(findings, ensure_ascii=False)


def test_conformance_allows_nested_and_overlapping_spans() -> None:
    """Nested entities are normal in CMeEE data and must not break the contract."""

    fixtures = load_chinese_clinical_ner_fixtures()

    report = run_chinese_clinical_ner_conformance(
        _NestedSpanAdapter(fixtures),
        fixtures=fixtures,
        fail_on_leakage=False,
        clock=_StepClock(),
    )

    assert report.metrics["latency"]["count"] == len(fixtures)


def test_conformance_defaults_to_the_bundled_synthetic_fixture() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()

    report = run_chinese_clinical_ner_conformance(
        _OracleLocalModelAdapter(fixtures),
        clock=_StepClock(),
    )

    assert report.fixture_count == len(fixtures)
    assert report.metadata["local_model_asset"] is None
    assert report.metrics["latency"]["count"] == len(fixtures)


def test_conformance_requires_an_adapter_model_name() -> None:
    fixtures = load_chinese_clinical_ner_fixtures()
    adapter = _OracleLocalModelAdapter(fixtures)
    adapter.model_name = "  "

    with pytest.raises(ValueError, match="model_name"):
        run_chinese_clinical_ner_conformance(adapter, fixtures=fixtures)


def test_missing_local_asset_skips_with_env_var_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CHINESE_CLINICAL_NER_MODEL_DIR_ENV, raising=False)

    reason = chinese_local_model_asset_skip_reason()

    assert reason is not None
    assert CHINESE_CLINICAL_NER_MODEL_DIR_ENV in reason
    assert "never downloads or bundles" in reason
    with pytest.raises(
        ChineseClinicalNerAssetUnavailable,
        match=CHINESE_CLINICAL_NER_MODEL_DIR_ENV,
    ):
        resolve_chinese_local_model_asset()


def test_local_asset_resolution_accepts_a_provisioned_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # The operator-controlled parts of the path are a username and a filename
    # that must never reach the report.
    parent = tmp_path / "home" / "alice"
    checkpoint = parent / "zh-clinical-ner"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "alice-macbook-run1.safetensors").write_bytes(b"")
    monkeypatch.setenv(CHINESE_CLINICAL_NER_MODEL_DIR_ENV, str(checkpoint))

    assert chinese_local_model_asset_skip_reason() is None
    asset = resolve_chinese_local_model_asset()

    assert asset.path == checkpoint.resolve()
    assert asset.tokenizer_files == ("tokenizer.json",)
    assert asset.weight_file_suffixes == (".safetensors",)
    assert asset.weight_file_count == 1

    evidence = json.dumps(asset.to_dict(), ensure_ascii=False)
    assert CHINESE_CLINICAL_NER_MODEL_DIR_ENV in evidence
    # Neither the path, the username, nor the operator-chosen weight filename
    # may appear anywhere in the serialized evidence.
    assert str(checkpoint) not in evidence
    assert "alice" not in evidence
    assert "zh-clinical-ner" not in evidence
    assert "alice-macbook-run1" not in evidence


def test_unreadable_local_asset_skips_instead_of_raising_oserror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "locked"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"")
    monkeypatch.setenv(CHINESE_CLINICAL_NER_MODEL_DIR_ENV, str(checkpoint))

    def _deny(self: Path) -> Any:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "iterdir", _deny)

    reason = chinese_local_model_asset_skip_reason()

    assert reason is not None
    assert CHINESE_CLINICAL_NER_MODEL_DIR_ENV in reason
    assert "permissions" in reason


@pytest.mark.parametrize(
    "weight_name",
    ["tf_model.h5", "flax_model.msgpack", "model.pt", "weights.pth", "model.ckpt"],
)
def test_non_pytorch_checkpoint_formats_are_accepted(
    tmp_path: Path,
    weight_name: str,
) -> None:
    checkpoint = tmp_path / weight_name.replace(".", "-")
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "tokenizer.json").write_text("{}", encoding="utf-8")
    (checkpoint / weight_name).write_bytes(b"")

    asset = resolve_chinese_local_model_asset(checkpoint)

    assert asset.weight_file_suffixes == (Path(weight_name).suffix,)


@pytest.mark.parametrize(
    ("present", "expected"),
    [
        ((), "config.json"),
        (("config.json",), "tokenizer artifact"),
        (("config.json", "tokenizer.json"), "model weights"),
    ],
)
def test_incomplete_local_asset_names_the_env_var_and_the_gap(
    tmp_path: Path,
    present: tuple[str, ...],
    expected: str,
) -> None:
    checkpoint = tmp_path / "partial"
    checkpoint.mkdir()
    for name in present:
        (checkpoint / name).write_text("{}", encoding="utf-8")

    with pytest.raises(ChineseClinicalNerAssetUnavailable) as caught:
        resolve_chinese_local_model_asset(checkpoint)

    message = str(caught.value)
    assert CHINESE_CLINICAL_NER_MODEL_DIR_ENV in message
    assert expected in message


def test_absent_local_asset_directory_names_the_env_var(tmp_path: Path) -> None:
    with pytest.raises(ChineseClinicalNerAssetUnavailable) as caught:
        resolve_chinese_local_model_asset(tmp_path / "does-not-exist")

    assert CHINESE_CLINICAL_NER_MODEL_DIR_ENV in str(caught.value)


def test_repo_ships_no_zh_model_weights() -> None:
    """Fail when model weights sit inside the repository at all.

    This deliberately exceeds "no weights are committed". It scans tracked
    files *and* untracked files that are not git-ignored, because the accident
    this guard exists to prevent begins as an untracked file: a checkpoint
    parked in the working tree is one ``git add -A`` away from being
    committed, and by the time it is committed the guard has already failed at
    its job. A red test at the moment the checkpoint lands is the intended
    outcome, not noise.

    If you are working against a local checkout, keep the checkpoint outside
    the repository and point ``OPENMED_ZH_CLINICAL_NER_MODEL_DIR`` at it, or
    git-ignore the directory you store it in.
    """

    listings = []
    for args in (
        ["git", "ls-files", "-z"],
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
    ):
        completed = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, check=False, text=True
        )
        if completed.returncode != 0:
            pytest.skip("git is unavailable; the repository guard cannot run")
        listings.append(completed.stdout)

    candidates = [name for name in "".join(listings).split("\0") if name]
    assert candidates, "git listed no repository files; the guard would be vacuous"

    weights = sorted(
        name
        for name in candidates
        if _looks_like_model_weights(name)
        # A symlink is the documented "keep it outside the repository"
        # workflow, so it is a pointer, not a checkpoint living in the tree.
        and not (REPO_ROOT / name).is_symlink()
    )
    assert weights == [], f"model weights must stay outside the repository: {weights}"


@pytest.mark.integration
def test_local_checkpoint_meets_the_span_and_label_contract() -> None:
    """Opt-in check against a real, user-provisioned Chinese NER checkpoint.

    Asserts what a Chinese clinical *entity* checkpoint can actually satisfy:
    the span and canonical-label contract, plus a populated scorecard. It
    deliberately does not assert ``gate.passed``. The gate includes a
    zero-tolerance PHI-leakage check that judges de-identification behaviour,
    and a CMeEE-trained model emits ``bod``/``dis``/``sym`` and no PHI types,
    so it will legitimately leave the synthetic identifiers standing. That is
    reported as evidence via ``fail_on_leakage=False``, not scored as a
    failure of the checkpoint.

    Quality numbers are only meaningful for the corpus the caller supplies;
    nothing here is measured in CI.
    """

    reason = chinese_local_model_asset_skip_reason()
    if reason is not None:
        pytest.skip(reason)

    pytest.importorskip("transformers")
    asset = resolve_chinese_local_model_asset()

    # Raises ChineseClinicalNerContractError if any predicted span is out of
    # bounds, inverted, misaligned, or carries a non-canonical label.
    report = run_chinese_clinical_ner_conformance(
        _TransformersLocalAdapter(asset),
        asset=asset,
        fail_on_leakage=False,
    )

    assert report.metrics["latency"]["count"] > 0
    assert report.metrics["per_label"]
    assert set(report.metrics["per_label"]) <= set(CANONICAL_LABELS)
    assert report.metadata["local_model_asset"]["path_env"] == (
        CHINESE_CLINICAL_NER_MODEL_DIR_ENV
    )
    leakage = report.metrics["phi_token_leakage"]
    assert all(
        set(finding) == {"end", "fixture_id", "label", "start", "token_hash"}
        for finding in leakage["findings"]
    )


def _looks_like_model_weights(name: str) -> bool:
    """Return True for filenames that are model weight artifacts.

    Name-scoped on purpose: a bare suffix match would fail this Chinese
    clinical NER test for any unrelated ``testdata.npz`` elsewhere in the
    tree. Recognizes the Hugging Face artifact conventions plus the
    unambiguous single-purpose checkpoint suffixes.
    """

    base = Path(name).name
    if base in {"tf_model.h5", "flax_model.msgpack"}:
        return True
    if Path(base).suffix in {".safetensors", ".gguf", ".ckpt", ".pt", ".pth"}:
        return True
    return bool(
        base.startswith(("pytorch_model", "model", "weights"))
        and Path(base).suffix in {".bin", ".onnx", ".tflite", ".h5", ".msgpack"}
    )


class _StepClock:
    """Monotonic stub advancing one millisecond per read."""

    def __init__(self, step_ms: float = 1.0) -> None:
        self._step_seconds = step_ms / 1000.0
        self._reads = 0

    def __call__(self) -> float:
        value = self._reads * self._step_seconds
        self._reads += 1
        return value


class _OracleLocalModelAdapter:
    """Deterministic stand-in for weights provisioned outside the repository."""

    def __init__(self, fixtures: Sequence[BenchmarkFixture]) -> None:
        self.model_name = "synthetic-local-checkpoint"
        self._by_text = {fixture.text: fixture for fixture in fixtures}

    def predict_spans(self, text: str, *, language: str) -> Sequence[Any]:
        fixture = self._by_text[text]
        assert fixture.language == language
        return tuple(fixture.gold_spans)

    def redact(self, text: str, *, spans: Sequence[Any]) -> str:
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            text = f"{text[: span.start]}[{span.label}]{text[span.end :]}"
        return text


class _LeakyLocalModelAdapter(_OracleLocalModelAdapter):
    """Adapter that predicts correctly but forgets to redact."""

    def redact(self, text: str, *, spans: Sequence[Any]) -> str:
        _ = spans
        return text


class _ContractBreakingAdapter(_OracleLocalModelAdapter):
    """Adapter that emits one malformed span, to prove enforcement is real."""

    def __init__(self, fixtures: Sequence[BenchmarkFixture], mutation: str) -> None:
        super().__init__(fixtures)
        self.model_name = f"synthetic-broken-{mutation}"
        self._mutation = mutation

    def predict_spans(self, text: str, *, language: str) -> Sequence[Any]:
        spans = list(super().predict_spans(text, language=language))
        first = spans[0]
        if self._mutation == "out_of_bounds":
            spans[0] = replace(first, start=0, end=len(text) + 10_000)
        elif self._mutation == "inverted":
            spans[0] = replace(first, start=first.end, end=first.start)
        elif self._mutation == "non_canonical_label":
            spans[0] = replace(first, label="NOT_A_CANONICAL_LABEL")
        elif self._mutation == "stale_text":
            shifted = min(first.start + 1, len(text) - 1)
            spans[0] = replace(first, start=shifted, end=shifted + len(first.text))
        return spans


class _NestedSpanAdapter(_OracleLocalModelAdapter):
    """Adapter emitting a nested span pair, which the contract must allow."""

    def predict_spans(self, text: str, *, language: str) -> Sequence[Any]:
        spans = list(super().predict_spans(text, language=language))
        outer = spans[0]
        if outer.end - outer.start >= 2:
            spans.append(
                replace(
                    outer,
                    end=outer.end - 1,
                    text=text[outer.start : outer.end - 1],
                )
            )
        return spans


class _TransformersLocalAdapter:
    """Thin opt-in adapter over a locally provisioned token-classification model."""

    def __init__(self, asset: Any) -> None:
        from transformers import pipeline  # noqa: PLC0415

        # A fixed label, never the directory basename: that basename is
        # operator-controlled and can carry a username.
        self.model_name = "local-zh-clinical-ner"
        self._pipeline = pipeline(
            "token-classification",
            model=str(asset.path),
            tokenizer=str(asset.path),
            aggregation_strategy="simple",
        )

    def predict_spans(self, text: str, *, language: str) -> Sequence[Any]:
        return [
            EvalSpan(
                start=int(item["start"]),
                end=int(item["end"]),
                label=normalize_label(str(item["entity_group"]), lang="zh"),
                text=text[int(item["start"]) : int(item["end"])],
                language=language,
            )
            for item in self._pipeline(text)
        ]

    def redact(self, text: str, *, spans: Sequence[Any]) -> str:
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            text = f"{text[: span.start]}[{span.label}]{text[span.end :]}"
        return text


def _asset_stub() -> ChineseLocalModelAsset:
    return ChineseLocalModelAsset(
        path=Path("/home/alice/zh-clinical-ner-stub"),
        config_file="config.json",
        tokenizer_files=("tokenizer.json",),
        weight_file_suffixes=(".safetensors",),
        weight_file_count=1,
        fingerprint="sha256:" + "0" * 64,
    )


def _identity_runner(
    fixture: BenchmarkFixture,
    model_name: str,
    device: str,
) -> tuple[Any, ...]:
    _ = (model_name, device)
    return tuple(fixture.gold_spans)
