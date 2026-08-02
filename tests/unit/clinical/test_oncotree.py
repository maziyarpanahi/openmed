"""Unit tests for the offline OncoTree tumor-type mapper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import (
    ONCOTREE_ADVISORY,
    load_oncotree,
    map_tumor_type,
)

ROOT = Path(__file__).resolve().parents[3]
STUB = ROOT / "tests" / "fixtures" / "clinical" / "oncotree_stub.json"
FIXTURE = ROOT / "tests" / "fixtures" / "clinical" / "oncotree_map.jsonl"
VERSION = "synthetic-oncotree-1"


def _load_gold() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@pytest.fixture(scope="module")
def release():
    return load_oncotree(STUB, version=VERSION)


def test_mapping_fixture_is_present_and_synthetic():
    rows = _load_gold()
    assert len(rows) >= 12
    assert all(row["metadata"]["synthetic"] is True for row in rows)


def test_load_stub_stamps_version(release):
    assert release.version == VERSION
    assert release.node_count >= 10


def test_absent_release_raises_clear_error(tmp_path: Path):
    missing = tmp_path / "missing_oncotree.json"
    with pytest.raises(FileNotFoundError, match="OncoTree release"):
        load_oncotree(missing, version=VERSION)


def test_absent_env_path_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENMED_ONCOTREE_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="OPENMED_ONCOTREE_PATH"):
        load_oncotree(version=VERSION)


def test_env_path_loads_release(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_ONCOTREE_PATH", str(STUB))
    release = load_oncotree(version=VERSION)
    assert release.version == VERSION
    assert release.node_count >= 10
    mapped = map_tumor_type("Vestra Pigment Tumor", release)
    assert mapped["code"] == "SYN_PIG"
    assert mapped["oncotree_version"] == VERSION


def test_version_always_required(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENMED_ONCOTREE_VERSION", raising=False)
    with pytest.raises(ValueError, match="version is required"):
        load_oncotree(STUB)


def test_advisory_constant_and_attachment(release):
    assert isinstance(ONCOTREE_ADVISORY, str)
    assert "user-supplied" in ONCOTREE_ADVISORY.casefold()
    assert "ConceptNormalizer" not in ONCOTREE_ADVISORY
    mapped = map_tumor_type("Vestra Pigment Tumor", release)
    assert mapped["advisory"] == ONCOTREE_ADVISORY


def test_gold_mentions_map_with_provenance(release):
    rows = _load_gold()
    checked = 0
    for row in rows:
        result = map_tumor_type(row["mention"], release)
        gold = row["gold"]
        assert result["code"] == gold["code"], row["mention"]
        assert result["name"] == gold["name"], row["mention"]
        assert result["main_type"] == gold["main_type"], row["mention"]
        assert result["tissue"] == gold["tissue"], row["mention"]
        assert result["match_confidence"] == gold["match_confidence"], row["mention"]
        assert result["reason"] == gold["reason"], row["mention"]
        assert result["oncotree_version"] == VERSION
        assert result["advisory"] == ONCOTREE_ADVISORY
        checked += 1
    assert checked >= 12


def test_empty_mention_unmapped(release):
    result = map_tumor_type("   ", release)
    assert result["code"] is None
    assert result["reason"] == "empty_mention"
    assert result["match_confidence"] == 0.0
    assert result["oncotree_version"] == VERSION


def test_ambiguous_collision_unmapped(release):
    result = map_tumor_type("Shared Collision Tumor", release)
    assert result["code"] is None
    assert result["reason"] == "ambiguous"
    assert result["oncotree_version"] == VERSION


def test_paraphrase_stays_unmapped_without_fuzzy_fallback(release):
    result = map_tumor_type("Adenoid Tumor of the helion", release)
    assert result["code"] is None
    assert result["reason"] == "no_match"
    assert result["match_confidence"] == 0.0


def test_unrelated_mention_unmapped(release):
    result = map_tumor_type("not a real tumor type xyz", release)
    assert result["code"] is None
    assert result["reason"] == "no_match"
    assert result["match_confidence"] == 0.0


def test_exact_code_match_confidence(release):
    result = map_tumor_type("SYN_BCL", release)
    assert result["code"] == "SYN_BCL"
    assert result["match_confidence"] == 1.0
    assert result["oncotree_version"] == VERSION


def test_history_and_revocation_codes_map_to_current_node(release):
    history = map_tumor_type("SYN_OLD_MAR", release)
    assert history["code"] == "SYN_MAR"
    assert history["name"] == "Marrowoid Blast Tumor"
    assert history["match_confidence"] == 1.0

    revoked = map_tumor_type("SYN_OLD_PIG", release)
    assert revoked["code"] == "SYN_PIG"
    assert revoked["name"] == "Vestra Pigment Tumor"
    assert revoked["match_confidence"] == 1.0


def test_history_alias_colliding_with_live_code_prefers_live(tmp_path: Path):
    """Former-code aliases must not shadow still-live codes in the release."""

    path = tmp_path / "live_shadow.json"
    path.write_text(
        json.dumps(
            [
                {
                    "code": "LIVE_A",
                    "name": "Alpha Tumor",
                    "mainType": "Alpha",
                    "tissue": "Tissue",
                    "history": ["LIVE_B"],
                    "revocations": [],
                },
                {
                    "code": "LIVE_B",
                    "name": "Beta Tumor",
                    "mainType": "Beta",
                    "tissue": "Tissue",
                    "history": [],
                    "revocations": ["LIVE_A"],
                },
            ]
        ),
        encoding="utf-8",
    )
    release = load_oncotree(path, version="shadow-v1")
    mapped_a = map_tumor_type("LIVE_A", release)
    mapped_b = map_tumor_type("LIVE_B", release)
    assert mapped_a["code"] == "LIVE_A"
    assert mapped_a["name"] == "Alpha Tumor"
    assert mapped_a["reason"] is None
    assert mapped_b["code"] == "LIVE_B"
    assert mapped_b["name"] == "Beta Tumor"
    assert mapped_b["reason"] is None


def test_caller_supplied_synonyms_are_exact_and_normalized(tmp_path: Path):
    path = tmp_path / "synonyms.json"
    path.write_text(
        json.dumps(
            [
                {
                    "code": "SYN_LUNG",
                    "name": "Synthetic Pulmonary Adenocarcinoma",
                    "mainType": "Synthetic Adenocarcinoma",
                    "tissue": "Synthetic Lung",
                    "synonyms": ["Synthetic Lung Adenocarcinoma"],
                }
            ]
        ),
        encoding="utf-8",
    )
    release = load_oncotree(path, version="synonyms-v1")

    exact = map_tumor_type("Synthetic Lung Adenocarcinoma", release)
    normalized = map_tumor_type("synthetic-lung-adenocarcinoma", release)

    assert exact["code"] == "SYN_LUNG"
    assert exact["match_confidence"] == 1.0
    assert normalized["code"] == "SYN_LUNG"
    assert normalized["match_confidence"] == 0.95


def test_live_code_wins_over_caller_supplied_synonym(tmp_path: Path):
    path = tmp_path / "code_synonym_collision.json"
    path.write_text(
        json.dumps(
            [
                {"code": "LIVE_A", "name": "Alpha Tumor"},
                {
                    "code": "LIVE_B",
                    "name": "Beta Tumor",
                    "synonyms": ["LIVE_A"],
                },
            ]
        ),
        encoding="utf-8",
    )
    release = load_oncotree(path, version="collision-v1")

    mapped = map_tumor_type("LIVE_A", release)

    assert mapped["code"] == "LIVE_A"
    assert mapped["name"] == "Alpha Tumor"


def test_duplicate_live_codes_rejected(tmp_path: Path):
    path = tmp_path / "duplicate_codes.json"
    path.write_text(
        json.dumps(
            [
                {
                    "code": "DUP_CODE",
                    "name": "First Tumor",
                    "mainType": "First",
                    "tissue": "Tissue",
                },
                {
                    "code": "DUP_CODE",
                    "name": "Second Tumor",
                    "mainType": "Second",
                    "tissue": "Tissue",
                },
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate OncoTree code"):
        load_oncotree(path, version="dup-v1")


@pytest.mark.parametrize(
    "payload,match",
    [
        (
            [
                {
                    "code": "X1",
                    "name": "Example Tumor",
                    "history": "not-a-list",
                }
            ],
            r"'history' must be a list",
        ),
        (
            [
                {
                    "code": "X1",
                    "name": "Example Tumor",
                    "revocations": {"OLD": True},
                }
            ],
            r"'revocations' must be a list",
        ),
        (
            [
                {
                    "code": "X1",
                    "name": "Example Tumor",
                    "history": ["OK", 42],
                }
            ],
            r"'history' entries must be strings",
        ),
        (
            [
                {
                    "code": "X1",
                    "name": "Example Tumor",
                    "revocations": [None],
                }
            ],
            r"'revocations' entries must be strings",
        ),
        (
            [{"code": "X1", "name": "Example Tumor", "synonyms": "alias"}],
            r"'synonyms' must be a list",
        ),
        (
            [{"code": "X1", "name": "Example Tumor", "synonyms": [42]}],
            r"'synonyms' entries must be strings",
        ),
    ],
)
def test_malformed_string_list_fields_rejected(
    tmp_path: Path, payload: list, match: str
):
    path = tmp_path / "malformed_list.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_oncotree(path, version="malformed-v1")


def test_release_must_be_a_list(tmp_path: Path):
    path = tmp_path / "wrapped.json"
    path.write_text(
        json.dumps({"nodes": [{"code": "X1", "name": "Example Tumor"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="flat JSON list.*nested tree dumps"):
        load_oncotree(path, version="user-v1")


def test_nested_tree_dump_unsupported(tmp_path: Path):
    path = tmp_path / "nested_tree.json"
    path.write_text(
        json.dumps(
            {
                "code": "ROOT",
                "name": "Root",
                "children": {
                    "X1": {
                        "code": "X1",
                        "name": "Example Tumor",
                        "children": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="nested tree dumps"):
        load_oncotree(path, version="user-v1")


def test_bare_list_loads_with_caller_version(tmp_path: Path):
    path = tmp_path / "bare.json"
    path.write_text(
        json.dumps(
            [
                {
                    "code": "X1",
                    "name": "Example Tumor",
                    "mainType": "Example",
                    "tissue": "Other",
                }
            ]
        ),
        encoding="utf-8",
    )
    release = load_oncotree(path, version="user-v1")
    assert release.version == "user-v1"
    mapped = map_tumor_type("Example Tumor", release)
    assert mapped["code"] == "X1"
    assert mapped["oncotree_version"] == "user-v1"


def test_version_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENMED_ONCOTREE_VERSION", "env-v2")
    release = load_oncotree(STUB)
    assert release.version == "env-v2"
