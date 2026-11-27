"""Tests for the portable task pack manifest, generator and resolver."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.core import task_packs
from openmed.core.labels import DIRECT_IDENTIFIER, policy_label_for
from openmed.core.model_registry import load_manifest_rows
from openmed.core.task_packs import (
    NOTICE_MEDICATION_EXTRACTION_ONLY,
    PERMISSIVE_LICENSES,
    RUNTIME_FORMATS,
    STATUS_NO_CANDIDATE,
    STATUS_RESOLVED,
    TaskPackDeclaration,
    TaskPackError,
    TaskPackSelection,
    available_packs,
    build_task_packs,
    load_task_pack_schema,
    load_task_packs,
    render_task_packs,
    resolve,
    validate_task_packs,
)
from scripts.manifest import build_task_packs as generator

ROOT = Path(__file__).resolve().parents[3]
COMMITTED = ROOT / "openmed" / "core" / "data" / "task_packs.json"
PACKS = ("clinical_entity", "medication", "pii")


def _row(repo_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "repo_id": repo_id,
        "family": "NER",
        "task": "token-classification",
        "languages": ["en"],
        "formats": ["onnx", "mlx-fp"],
        "canonical_labels": ["CHEM"],
        "param_count": 50_000_000,
        "license": "apache-2.0",
    }
    row.update(overrides)
    return row


def _declaration(**selection: object) -> TaskPackDeclaration:
    rules: dict[str, object] = {
        "family": "NER",
        "family_token": "PharmaDetect",
        "manifest_task": "token-classification",
        "languages": ("en",),
        "max_param_count": 100_000_000,
        "max_candidates": 2,
    }
    rules.update(selection)
    return TaskPackDeclaration(
        pack_id="medication",
        version="1.0.0",
        task="medication_mention_extraction",
        canonical_labels=("CHEMICAL",),
        selection=TaskPackSelection(**rules),  # type: ignore[arg-type]
        notice_key=NOTICE_MEDICATION_EXTRACTION_ONLY,
    )


def _resolved(rows: list[dict[str, object]], runtime: str = "android") -> list[str]:
    document = build_task_packs(rows, [_declaration()])
    entry = document["packs"][0]["runtimes"][runtime]
    return [candidate["repo_id"] for candidate in entry["candidates"]]


def test_schema_is_valid_and_accepts_the_committed_manifest() -> None:
    schema = load_task_pack_schema()
    Draft202012Validator.check_schema(schema)
    document = json.loads(COMMITTED.read_text(encoding="utf-8"))

    assert list(Draft202012Validator(schema).iter_errors(document)) == []
    assert document["schema"] == "openmed.task_packs"
    assert document["schema_version"] == task_packs.TASK_PACKS_SCHEMA_VERSION


def test_committed_manifest_matches_a_fresh_build_from_models_jsonl() -> None:
    expected = render_task_packs(build_task_packs(load_manifest_rows()))

    assert COMMITTED.read_text(encoding="utf-8") == expected
    assert generator.main(["--check"]) == 0


def test_check_mode_fails_on_a_stale_manifest_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = tmp_path / "task_packs.json"
    stale.write_text("{}\n", encoding="utf-8")

    assert generator.main(["--check", "--output", str(stale)]) == 1
    assert stale.read_text(encoding="utf-8") == "{}\n"
    assert "stale" in capsys.readouterr().err

    assert generator.main(["--output", str(stale)]) == 0
    assert generator.main(["--check", "--output", str(stale)]) == 0


@pytest.mark.parametrize("pack_id", PACKS)
@pytest.mark.parametrize("runtime", ["apple", "android"])
def test_each_pack_resolves_or_records_no_candidate(pack_id: str, runtime: str) -> None:
    resolution = resolve(pack_id, runtime)

    assert resolution.format == RUNTIME_FORMATS[runtime]
    if resolution.status == STATUS_NO_CANDIDATE:
        assert resolution.candidates == ()
    else:
        assert resolution.status == STATUS_RESOLVED
        assert resolution.candidates
    # The current models.jsonl serves every pack on both mobile runtimes.
    assert resolution.status == STATUS_RESOLVED


def test_every_candidate_is_permissive_and_matches_its_manifest_row() -> None:
    rows = {row["repo_id"]: row for row in load_manifest_rows()}
    document = load_task_packs()

    for pack in document["packs"]:
        for runtime, entry in pack["runtimes"].items():
            for candidate in entry["candidates"]:
                row = rows[candidate["repo_id"]]
                assert candidate["license"] in PERMISSIVE_LICENSES
                assert row["license"] == candidate["license"]
                assert row["param_count"] == candidate["param_count"]
                assert RUNTIME_FORMATS[runtime] in row["formats"]


def test_candidates_are_ordered_best_first() -> None:
    for pack_id in available_packs():
        for runtime in RUNTIME_FORMATS:
            candidates = resolve(pack_id, runtime).candidates
            keys = [(item.param_count, item.repo_id) for item in candidates]
            assert keys == sorted(keys)
            assert [item.rank for item in candidates] == list(
                range(1, len(candidates) + 1)
            )


def test_medication_pack_declares_its_extraction_only_notice() -> None:
    resolution = resolve("medication", "apple")
    notices = load_task_packs()["notices"]

    assert resolution.notice_key == NOTICE_MEDICATION_EXTRACTION_ONLY
    text = notices[resolution.notice_key]
    assert "mentions and their offsets only" in text
    assert "dosing advice" in text


def test_every_pack_notice_key_resolves_to_text() -> None:
    document = load_task_packs()

    for pack in document["packs"]:
        assert document["notices"][pack["notice_key"]]


def test_pii_pack_labels_are_direct_identifiers() -> None:
    labels = resolve("pii", "android").canonical_labels

    assert labels
    assert all(policy_label_for(label) == DIRECT_IDENTIFIER for label in labels)


def test_resolve_reads_metadata_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("task pack resolution must not open a socket")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)

    for pack_id in PACKS:
        for runtime in RUNTIME_FORMATS:
            resolve(pack_id, runtime, manifest=load_task_packs(COMMITTED))


@pytest.mark.parametrize(
    ("pack_id", "runtime", "reason"),
    [
        ("pii", "windows", "unknown_runtime"),
        ("pii", "", "unknown_runtime"),
        ("dosing", "apple", "unknown_pack"),
        ("../pii", "apple", "unknown_pack"),
    ],
)
def test_unknown_pack_or_runtime_fails_with_a_value_free_reason(
    pack_id: str, runtime: str, reason: str
) -> None:
    with pytest.raises(TaskPackError) as raised:
        resolve(pack_id, runtime)

    assert raised.value.reason == reason
    assert str(raised.value) == reason


def test_selection_orders_by_size_then_repo_id_and_caps_candidates() -> None:
    rows = [
        _row("OpenMed/OpenMed-NER-PharmaDetect-B-60M", param_count=60_000_000),
        _row("OpenMed/OpenMed-NER-PharmaDetect-A-60M", param_count=60_000_000),
        _row("OpenMed/OpenMed-NER-PharmaDetect-C-20M", param_count=20_000_000),
    ]

    assert _resolved(rows) == [
        "OpenMed/OpenMed-NER-PharmaDetect-C-20M",
        "OpenMed/OpenMed-NER-PharmaDetect-A-60M",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"license": "other"},
        {"license": None},
        {"license": "gpl-3.0"},
        {"param_count": 200_000_000},
        {"param_count": None},
        {"formats": ["pytorch"]},
        {"languages": ["pt"]},
        {"canonical_labels": ["DISEASE"]},
        {"family": "ZeroShot"},
        {"task": "text-generation"},
        {"repo_id": "OpenMed/OpenMed-NER-ChemicalDetect-Small-20M"},
    ],
)
def test_selection_rules_exclude_non_matching_rows(overrides: dict) -> None:
    good = _row("OpenMed/OpenMed-NER-PharmaDetect-Ok-90M", param_count=90_000_000)
    bad = _row("OpenMed/OpenMed-NER-PharmaDetect-Bad-10M", param_count=10_000_000)
    bad.update(overrides)

    assert _resolved([bad, good]) == ["OpenMed/OpenMed-NER-PharmaDetect-Ok-90M"]


def test_label_coverage_uses_normalized_canonical_labels() -> None:
    rows = [_row("OpenMed/OpenMed-NER-PharmaDetect-Alias-10M")]
    rows[0]["canonical_labels"] = ["SIMPLE_CHEMICAL"]

    assert _resolved(rows) == ["OpenMed/OpenMed-NER-PharmaDetect-Alias-10M"]


def test_runtime_without_a_matching_row_records_no_candidate() -> None:
    rows = [_row("OpenMed/OpenMed-NER-PharmaDetect-Onnx-10M", formats=["onnx"])]
    document = build_task_packs(rows, [_declaration()])
    runtimes = document["packs"][0]["runtimes"]

    assert runtimes["apple"] == {
        "candidates": [],
        "format": "mlx-fp",
        "status": STATUS_NO_CANDIDATE,
    }
    assert runtimes["android"]["status"] == STATUS_RESOLVED
    resolution = resolve("medication", "apple", manifest=document)
    assert resolution.status == STATUS_NO_CANDIDATE
    assert resolution.candidates == ()


def test_build_ignores_non_object_rows() -> None:
    rows = ["not-a-row", 7, _row("OpenMed/OpenMed-NER-PharmaDetect-Ok-10M")]

    assert _resolved(rows) == ["OpenMed/OpenMed-NER-PharmaDetect-Ok-10M"]  # type: ignore[arg-type]


def _tampered(mutate) -> dict:
    document = copy.deepcopy(load_task_packs())
    mutate(document)
    return document


def _first_candidate(document: dict) -> dict:
    return document["packs"][0]["runtimes"]["android"]["candidates"][0]


def _swap_first_two_candidates(document: dict) -> None:
    candidates = document["packs"][0]["runtimes"]["android"]["candidates"]
    first, second = candidates[0], candidates[1]
    first["repo_id"], second["repo_id"] = second["repo_id"], first["repo_id"]
    first["param_count"], second["param_count"] = (
        second["param_count"],
        first["param_count"],
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda d: _first_candidate(d).update(license="other"), "schema_invalid"),
        (lambda d: d["packs"][0].update(unexpected=True), "schema_invalid"),
        (lambda d: d.update(schema_version="2.0.0"), "schema_invalid"),
        (
            lambda d: d["packs"][0]["runtimes"]["android"].update(
                status="no_candidate"
            ),
            "schema_invalid",
        ),
        (
            lambda d: d["packs"][0]["runtimes"]["android"].update(format="mlx-fp"),
            "schema_invalid",
        ),
        (lambda d: _first_candidate(d).update(rank=2), "rank_not_sequential"),
        (
            lambda d: d["packs"][0].update(notice_key="openmed.notice.unknown.missing"),
            "unknown_notice_key",
        ),
        (
            lambda d: d["packs"][1].update(pack_id=d["packs"][0]["pack_id"]),
            "duplicate_pack",
        ),
        (
            lambda d: d["packs"][0]["selection"].update(max_param_count=1),
            "size_cap_exceeded",
        ),
        (
            lambda d: d["packs"][0]["runtimes"]["android"]["candidates"].reverse(),
            "rank_not_sequential",
        ),
        (
            lambda d: d["packs"][0].update(canonical_labels=["NOT_A_LABEL"]),
            "label_not_canonical",
        ),
        (lambda d: _swap_first_two_candidates(d), "candidate_order_invalid"),
    ],
)
def test_validation_fails_closed_on_tampered_manifests(mutate, reason: str) -> None:
    with pytest.raises(TaskPackError) as raised:
        validate_task_packs(_tampered(mutate))

    assert raised.value.reason == reason


def test_load_task_packs_reports_an_unreadable_file_without_its_path(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "private-dir" / "task_packs.json"
    broken.parent.mkdir()
    broken.write_text("{not json", encoding="utf-8")

    for path in (broken, tmp_path / "missing.json"):
        with pytest.raises(TaskPackError) as raised:
            load_task_packs(path)
        assert raised.value.reason == "manifest_unreadable"
        assert "private-dir" not in str(raised.value)
