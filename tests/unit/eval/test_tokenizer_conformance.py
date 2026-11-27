"""Portable tokenizer conformance vectors, verifier and Python runners."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from openmed.core.registry_compatibility import build_rollback_compatibility_report
from openmed.eval.tokenizer_conformance import (
    CONFORMANCE_FIELDS,
    DEFAULT_FIXTURE_DIR,
    MANIFEST_FILENAME,
    SUPPORT_FILENAME,
    SYNTHETIC_CASES,
    TOKENIZER_CONTRACT_VERSION,
    TOKENIZER_DIRNAME,
    VECTORS_FILENAME,
    RuntimeEncoding,
    TokenizerConformanceError,
    build_fixture_files,
    check_fixture_dir,
    load_fixture_dir,
    require_tokenizer_conformance,
    tokenizer_contract_reference,
    validate_runtime_support,
    verify_tokenizer_conformance,
    write_fixture_dir,
)
from openmed.eval.tokenizer_reference import (
    ReferenceTokenizer,
    UnsupportedTokenizerFeature,
)
from openmed.onnx.inference import _windows_from_encoding

ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = ROOT / DEFAULT_FIXTURE_DIR
TOKENIZER_DIR = FIXTURE_DIR / TOKENIZER_DIRNAME
SCRIPT = ROOT / "scripts" / "eval" / "generate_tokenizer_conformance.py"
VECTORS, SUPPORT, MANIFEST = load_fixture_dir(FIXTURE_DIR)
CASES: list[dict[str, Any]] = VECTORS["cases"]
CASE_IDS = [case["id"] for case in CASES]
RUNTIMES = (
    "python-tokenizers",
    "swift-transformers",
    "djl-tokenizers",
    "tokenizers-js",
)
_PHI_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),
    re.compile(r"\b(?:John|Jane)\s+Doe\b", re.IGNORECASE),
)


def _tokenizer_json() -> dict[str, Any]:
    return json.loads((TOKENIZER_DIR / "tokenizer.json").read_text(encoding="utf-8"))


def _reference_runner(
    payload: Mapping[str, Any],
) -> Callable[[Mapping[str, Any]], RuntimeEncoding]:
    reference = ReferenceTokenizer(payload)

    def run(case: Mapping[str, Any]) -> RuntimeEncoding:
        text = case["text"]
        encoding = reference.encode(text)
        truncated = ranges = window_ids = None
        if "window" in case:
            windows = reference.windows(
                text,
                max_length=case["window"]["max_length"],
                stride=case["window"]["stride"],
            )
            truncated = windows[0].input_ids
            ranges = [(item.content_start, item.content_end) for item in windows]
            window_ids = [item.input_ids for item in windows]
        return RuntimeEncoding(
            normalized=reference.normalize(text),
            tokens=encoding.tokens,
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            token_type_ids=encoding.token_type_ids,
            special_tokens_mask=encoding.special_tokens_mask,
            offsets=encoding.offsets,
            truncated_input_ids=truncated,
            window_ranges=ranges,
            window_input_ids=window_ids,
        )

    return run


def _all_supported(divergences: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "schema": SUPPORT["schema"],
        "version": SUPPORT["version"],
        "contract_version": TOKENIZER_CONTRACT_VERSION,
        "runtimes": {
            "python-reference": {
                "library": "openmed.eval.tokenizer_reference",
                "adapter": "reference",
                "fields": {
                    name: {"status": "supported"} for name in CONFORMANCE_FIELDS
                },
                "divergences": divergences or [],
            }
        },
    }


def _diagnostic_text(report_payload: Mapping[str, Any]) -> str:
    return json.dumps(report_payload, ensure_ascii=False)


def test_committed_fixtures_match_a_fresh_generation() -> None:
    assert check_fixture_dir(FIXTURE_DIR) == ()


def test_generator_check_cli_reports_roles_and_codes_only(tmp_path: Path) -> None:
    passing = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert passing.returncode == 0, passing.stdout + passing.stderr

    stale = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_DIR, stale)
    vectors_path = stale / VECTORS_FILENAME
    vectors_path.write_text(
        vectors_path.read_text(encoding="utf-8").replace(
            '"input_ids":[2,', '"input_ids":[3,', 1
        ),
        encoding="utf-8",
    )
    failing = subprocess.run(
        [sys.executable, str(SCRIPT), "--check", "--fixture-dir", str(stale)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failing.returncode == 1
    assert "stale:vectors" in failing.stdout
    assert "stale:manifest" not in failing.stdout
    assert str(tmp_path) not in failing.stdout + failing.stderr


def test_regeneration_is_byte_identical_and_support_is_never_rewritten(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "fixtures"
    shutil.copytree(FIXTURE_DIR, copied)
    support_before = (copied / SUPPORT_FILENAME).read_bytes()

    assert write_fixture_dir(copied) == ()
    assert (copied / SUPPORT_FILENAME).read_bytes() == support_before
    assert build_fixture_files(support_before) == build_fixture_files(support_before)


def test_manifest_binds_every_fixture_file_and_contract_reference() -> None:
    listed = {entry["path"]: entry for entry in MANIFEST["files"]}
    expected = {
        f"{TOKENIZER_DIRNAME}/config.json",
        f"{TOKENIZER_DIRNAME}/tokenizer.json",
        f"{TOKENIZER_DIRNAME}/tokenizer_config.json",
        VECTORS_FILENAME,
        SUPPORT_FILENAME,
    }
    assert set(listed) == expected
    for path, entry in listed.items():
        data = (FIXTURE_DIR / path).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data)
    assert (
        VECTORS["tokenizer"]["sha256"]
        == listed[f"{TOKENIZER_DIRNAME}/tokenizer.json"]["sha256"]
    )
    assert MANIFEST_FILENAME not in listed

    reference = tokenizer_contract_reference(MANIFEST)
    assert reference == {
        "version": TOKENIZER_CONTRACT_VERSION,
        "digest": MANIFEST["contract_digest"],
    }
    tampered = copy.deepcopy(MANIFEST)
    tampered["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        tokenizer_contract_reference(tampered)
    with pytest.raises(ValueError):
        tokenizer_contract_reference({**MANIFEST, "contract_version": "v0"})


def test_bundle_tokenizer_contract_feeds_rollback_compatibility() -> None:
    def checkpoint(model_id: str, version: str, tokenizer_contract: object) -> dict:
        return {
            "model_id": model_id,
            "family": "PII",
            "version": version,
            "semver_constraint": ">=1.0.0,<3.0.0",
            "lineage": (
                [{"relation": "supersedes", "from": "synthetic-v1", "to": model_id}]
                if model_id == "synthetic-v2"
                else ()
            ),
            "policy_fingerprint": "synthetic-policy-v1",
            "tokenizer_contract": tokenizer_contract,
            "evidence_schema_versions": ("openmed.evidence.v1",),
        }

    contract = tokenizer_contract_reference(MANIFEST)
    same = build_rollback_compatibility_report(
        checkpoint("synthetic-v2", "2.0.0", contract),
        checkpoint("synthetic-v1", "1.0.0", dict(contract)),
    )
    assert same.check("tokenizer_contract").passed

    drifted = build_rollback_compatibility_report(
        checkpoint("synthetic-v2", "2.0.0", contract),
        checkpoint(
            "synthetic-v1", "1.0.0", {**contract, "digest": "sha256:" + "1" * 64}
        ),
    )
    assert "TOKENIZER_CONTRACT_MISMATCH" in drifted.blocked_reasons
    assert contract["digest"] not in drifted.to_json()


def test_vectors_cover_the_declared_reference_semantics() -> None:
    assert VECTORS["contract_version"] == TOKENIZER_CONTRACT_VERSION
    assert VECTORS["offset_unit"] == "unicode_scalar"
    assert VECTORS["fields"] == list(CONFORMANCE_FIELDS)
    assert len(CASE_IDS) == len(set(CASE_IDS)) == len(SYNTHETIC_CASES)
    features = set(VECTORS["features"])
    for required in (
        "combining.precomposed",
        "combining.decomposed",
        "combining.leading_mark",
        "rtl.arabic",
        "rtl.hebrew",
        "rtl.bidi_control",
        "cjk.han",
        "cjk.supplementary",
        "indic.devanagari",
        "indic.bengali",
        "indic.tamil",
        "emoji.single",
        "emoji.zwj_sequence",
        "emoji.skin_tone",
        "emoji.flag",
        "empty",
        "whitespace_only",
        "special_tokens.in_text",
        "unknown.word",
        "truncation.overflow",
        "window.exact_fit",
        "window.overflow",
        "window.stride",
        "window.stride_zero",
        "window.subword_boundary",
    ):
        assert required in features
    by_id = {case["id"]: case for case in CASES}
    assert by_id["empty-01"]["expected"]["input_ids"] == [2, 3]
    assert by_id["whitespace-only-01"]["expected"]["offsets"] == [[0, 0], [0, 0]]
    assert by_id["special-token-in-text-01"]["expected"]["input_ids"][2] == 3


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "window" in case],
    ids=lambda case: case["id"],
)
def test_window_vectors_follow_stride_semantics(case: dict[str, Any]) -> None:
    expected = case["expected"]
    capacity = (
        case["window"]["max_length"] - VECTORS["tokenizer"]["template_special_tokens"]
    )
    stride = case["window"]["stride"]
    ranges = expected["window_ranges"]
    content = len(expected["input_ids"]) - 2

    assert ranges[0][0] == 0 and ranges[-1][1] == content
    for (start, stop), ids in zip(ranges, expected["window_input_ids"]):
        assert stop - start <= capacity
        assert ids == [2, *expected["input_ids"][1 + start : 1 + stop], 3]
    for previous, current in zip(ranges, ranges[1:]):
        assert previous[1] - current[0] == stride
    assert expected["truncated_input_ids"] == expected["window_input_ids"][0]
    assert ("truncation.overflow" in case["features"]) == (content > capacity)
    if "window.exact_fit" in case["features"]:
        assert ranges == [[0, capacity]]
    if "window.overflow" in case["features"]:
        assert len(ranges) == 2 and content == capacity + 1
    if "window.subword_boundary" in case["features"]:
        assert any(
            expected["tokens"][1 + start].startswith("##") for start, _ in ranges[1:]
        )


def test_all_fixture_text_is_synthetic() -> None:
    assert VECTORS["synthetic_only"] is True
    assert MANIFEST["synthetic_only"] is True
    for case in CASES:
        for pattern in _PHI_PATTERNS:
            assert pattern.search(case["text"]) is None, case["id"]


def test_reference_runner_conforms_and_report_is_value_free() -> None:
    report = require_tokenizer_conformance(
        VECTORS,
        _all_supported(),
        "python-reference",
        _reference_runner(_tokenizer_json()),
    )

    assert report.passed
    assert report.unsupported_fields == ()
    assert report.checked == sum(
        len([name for name in CONFORMANCE_FIELDS if name in case["expected"]])
        for case in CASES
    )
    payload = report.to_dict()
    assert payload["status"] == "pass"
    assert payload["contract_version"] == TOKENIZER_CONTRACT_VERSION
    assert re.fullmatch(r"[0-9a-f]{64}", payload["vectors_sha256"])


@pytest.mark.parametrize(
    ("mutate", "case_id", "field", "index"),
    [
        pytest.param(
            lambda payload: payload["post_processor"].update(
                single=[
                    item
                    for item in payload["post_processor"]["single"]
                    if item.get("SpecialToken", {}).get("id") != "[CLS]"
                ]
            ),
            "ascii-basic-01",
            "input_ids",
            0,
            id="special-token-insertion",
        ),
        pytest.param(
            lambda payload: payload["normalizer"].update(
                lowercase=False, strip_accents=False
            ),
            "ascii-basic-01",
            "normalized",
            0,
            id="lowercase-normalization",
        ),
        pytest.param(
            lambda payload: payload["normalizer"].update(handle_chinese_chars=False),
            "cjk-han-01",
            "normalized",
            0,
            id="cjk-padding",
        ),
    ],
)
def test_tokenizer_changes_fail_before_model_execution(
    mutate: Callable[[dict[str, Any]], None],
    case_id: str,
    field: str,
    index: int,
) -> None:
    payload = _tokenizer_json()
    mutate(payload)
    model_calls: list[str] = []

    with pytest.raises(TokenizerConformanceError) as raised:
        require_tokenizer_conformance(
            VECTORS, _all_supported(), "python-reference", _reference_runner(payload)
        )
        model_calls.append("executed")

    assert model_calls == []
    mismatches = raised.value.report.mismatches
    assert any(
        item.case_id == case_id and item.field == field and item.index == index
        for item in mismatches
    )
    diagnostics = str(raised.value) + _diagnostic_text(raised.value.report.to_dict())
    for case in CASES:
        if len(case["text"].strip()) > 3:
            assert case["text"] not in diagnostics
    assert "synth" not in diagnostics


def test_declarations_are_explicit_and_stale_ones_fail() -> None:
    runner = _reference_runner(_tokenizer_json())
    stale = verify_tokenizer_conformance(
        VECTORS,
        _all_supported(
            [{"feature": "ascii.basic", "fields": ["tokens"], "reason": "synthetic"}]
        ),
        "python-reference",
        runner,
    )
    assert [item.to_dict() for item in stale.mismatches] == [
        {
            "case_id": "*",
            "field": "tokens",
            "index": None,
            "code": "TOKENIZER_DECLARED_DIVERGENCE_STALE",
            "feature": "ascii.basic",
        }
    ]

    support = _all_supported()
    support["runtimes"]["python-reference"]["fields"]["offsets"] = {
        "status": "unsupported",
        "reason": "synthetic_reason",
    }
    report = verify_tokenizer_conformance(VECTORS, support, "python-reference", runner)
    assert report.passed
    assert report.unsupported_fields == (("offsets", "synthetic_reason"),)


def test_missing_fields_runtime_errors_and_undeclared_runtimes_fail_closed() -> None:
    reference = _reference_runner(_tokenizer_json())

    def without_offsets(case: Mapping[str, Any]) -> RuntimeEncoding:
        produced = reference(case)
        return RuntimeEncoding(**{**produced.__dict__, "offsets": None})

    def broken(case: Mapping[str, Any]) -> RuntimeEncoding:
        raise RuntimeError(case["text"])

    support = _all_supported(
        [{"feature": "ascii.basic", "fields": ["offsets"], "reason": "synthetic"}]
    )
    missing = verify_tokenizer_conformance(
        VECTORS, support, "python-reference", without_offsets
    )
    assert {
        (item.case_id, item.code)
        for item in missing.mismatches
        if item.field == "offsets"
    } >= {("ascii-basic-01", "TOKENIZER_FIELD_MISSING")}

    errored = verify_tokenizer_conformance(
        VECTORS, _all_supported(), "python-reference", broken
    )
    assert {item.code for item in errored.mismatches} == {"TOKENIZER_RUNTIME_ERROR"}
    assert "Synthetic patient" not in _diagnostic_text(errored.to_dict())

    undeclared = verify_tokenizer_conformance(
        VECTORS, SUPPORT, "unknown-runtime", broken
    )
    assert undeclared.mismatches[0].code == "TOKENIZER_RUNTIME_UNDECLARED"
    wrong = verify_tokenizer_conformance(
        {**VECTORS, "contract_version": "v0"}, SUPPORT, "python-tokenizers", broken
    )
    assert wrong.mismatches[0].code == "TOKENIZER_CONTRACT_VERSION_MISMATCH"


def test_runtime_support_declarations_are_complete_and_controlled() -> None:
    assert validate_runtime_support(SUPPORT, VECTORS) == ()
    assert set(SUPPORT["runtimes"]) == set(RUNTIMES)
    for entry in SUPPORT["runtimes"].values():
        assert set(entry["fields"]) == set(CONFORMANCE_FIELDS)

    broken = copy.deepcopy(SUPPORT)
    broken["runtimes"]["tokenizers-js"]["divergences"].append(
        {"feature": "unknown.feature", "fields": ["tokens"], "reason": "synthetic"}
    )
    broken["runtimes"]["djl-tokenizers"]["fields"].pop("offsets")
    broken["runtimes"]["swift-transformers"]["fields"]["tokens"] = {
        "status": "unsupported",
        "reason": "Free text with spaces",
    }
    problems = validate_runtime_support(broken, VECTORS)
    assert "support_divergence:tokenizers-js" in problems
    assert "support_fields:djl-tokenizers" in problems
    assert "support_field_status:swift-transformers" in problems
    assert "support_divergence_unsupported_field:swift-transformers" in problems


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda payload: payload["model"].update(type="BPE"), "model"),
        (
            lambda payload: payload.update(pre_tokenizer={"type": "Metaspace"}),
            "pre_tokenizer",
        ),
        (
            lambda payload: payload.update(truncation={"max_length": 8}),
            "embedded_truncation",
        ),
        (
            lambda payload: payload.update(padding={"strategy": "BatchLongest"}),
            "embedded_padding",
        ),
        (
            lambda payload: payload["added_tokens"][0].update(normalized=True),
            "added_token_flags",
        ),
        (
            lambda payload: payload["post_processor"].update(type="RobertaProcessing"),
            "post_processor",
        ),
    ],
)
def test_unsupported_tokenizer_components_fail_closed(
    mutate: Callable[[dict[str, Any]], None], code: str
) -> None:
    payload = _tokenizer_json()
    mutate(payload)

    with pytest.raises(UnsupportedTokenizerFeature) as raised:
        ReferenceTokenizer(payload)

    assert raised.value.code == code
    assert str(raised.value) == f"unsupported tokenizer feature: {code}"


def test_reference_rejects_inputs_it_cannot_align_and_bad_windows() -> None:
    reference = ReferenceTokenizer(_tokenizer_json())
    reordered = "e" + chr(0x301) + chr(0x323)

    with pytest.raises(UnsupportedTokenizerFeature) as raised:
        reference.encode(reordered)
    assert raised.value.code == "nfd_reordering"
    for max_length, stride in ((2, 0), (6, 4), (6, -1)):
        with pytest.raises(UnsupportedTokenizerFeature):
            reference.windows("note dose", max_length=max_length, stride=stride)


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if "window" in case],
    ids=lambda case: case["id"],
)
def test_python_onnx_window_adapter_matches_window_vectors(
    case: dict[str, Any],
) -> None:
    expected = case["expected"]
    content = [
        index
        for index, special in enumerate(expected["special_tokens_mask"])
        if not special
    ]

    windows = _windows_from_encoding(
        expected["input_ids"],
        [tuple(item) for item in expected["offsets"]],
        expected["attention_mask"],
        expected["token_type_ids"],
        content,
        0,
        case["window"]["max_length"],
        case["window"]["stride"],
    )

    ranges = []
    for window in windows:
        indices = [index for index in window.token_indices if index is not None]
        ranges.append([indices[0], indices[-1] + 1])
    assert ranges == expected["window_ranges"]
    assert [window.input_ids for window in windows] == expected["window_input_ids"]


def test_python_tokenizers_runtime_matches_vectors_before_inference() -> None:
    tokenizers = pytest.importorskip(
        "tokenizers",
        reason=(
            "TOKENIZER_CONFORMANCE runtime=python-tokenizers status=unsupported "
            "reason=tokenizers_not_installed"
        ),
    )
    from openmed.onnx.inference import _TokenizersTokenizerAdapter

    raw = tokenizers.Tokenizer.from_file(str(TOKENIZER_DIR / "tokenizer.json"))
    adapter = _TokenizersTokenizerAdapter(raw)

    def run(case: Mapping[str, Any]) -> RuntimeEncoding:
        text = case["text"]
        inputs = adapter(
            text, return_offsets_mapping=True, return_tensors="np", truncation=False
        )
        encoding = raw.encode(text)
        truncated = ranges = window_ids = None
        if "window" in case:
            truncated = adapter(
                text,
                return_offsets_mapping=True,
                return_tensors="np",
                truncation=True,
                max_length=case["window"]["max_length"],
            )["input_ids"][0]
            windows = adapter.encode_windows(
                text,
                document=0,
                max_length=case["window"]["max_length"],
                stride=case["window"]["stride"],
            )
            ranges = []
            for window in windows:
                indices = [index for index in window.token_indices if index is not None]
                ranges.append([indices[0], indices[-1] + 1])
            window_ids = [window.input_ids for window in windows]
        return RuntimeEncoding(
            normalized=raw.normalizer.normalize_str(text),
            tokens=encoding.tokens,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            token_type_ids=inputs["token_type_ids"][0],
            special_tokens_mask=encoding.special_tokens_mask,
            offsets=inputs["offset_mapping"][0],
            truncated_input_ids=truncated,
            window_ranges=ranges,
            window_input_ids=window_ids,
        )

    report = require_tokenizer_conformance(VECTORS, SUPPORT, "python-tokenizers", run)

    assert report.unsupported_fields == ()
    assert report.declared_divergences == ()
    assert report.checked > 0
