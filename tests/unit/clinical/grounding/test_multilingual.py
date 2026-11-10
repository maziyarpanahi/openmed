"""Focused synthetic tests for pure-offline multilingual grounding."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from eval.suites.multilingual_grounding import run_multilingual_grounding_eval
from openmed.clinical.grounding import (
    CrosswalkLicenseError,
    MultilingualGrounder,
    UserKeyVocabularyLoader,
    ground_multilingual,
    load_crosswalk,
    load_default_crosswalks,
)
from openmed.clinical.grounding.embeddings import AliasEncoder
from openmed.clinical.grounding.vocab import RestrictedVocabularyError
from openmed.interop.bridges.icd10cn import (
    ICD10CNBridge,
    load_icd10cn_crosswalk,
)

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "openmed/clinical/grounding/data"
RESTRICTED_RESOURCE_MARKERS = (
    "cpt",
    "mrconso",
    "mrrel",
    "mrsty",
    "sct2",
    "snomed",
    "umls",
)


class _SyntheticConceptEncoder:
    """Deterministic stand-in for a local cross-lingual model."""

    encoder_id = "synthetic-cross-lingual-v1"
    dimension = 3

    def encode(self, texts):
        vectors = []
        for text in texts:
            if text in {"अर्थ-समान परीक्षण", "Fever"}:
                vectors.append((1.0, 0.0, 0.0))
            else:
                vectors.append((0.0, 1.0, 0.0))
        return tuple(vectors)


def _write_crosswalk(
    path: Path,
    *,
    redistributable: bool = True,
) -> Path:
    payload = {
        "schema_version": 1,
        "name": "synthetic-user-crosswalk",
        "version": "1.0",
        "license": "CC0-1.0",
        "redistributable": redistributable,
        "entries": [
            {
                "source_system": "SYNTHETIC-HI",
                "source_code": "SYN-HI-1",
                "locale": "hi-IN",
                "aliases": ["स्थानीय ताप संकेत"],
                "target_system": "HPO",
                "target_code": "HP:0001945",
                "target_display": "Fever",
            }
        ],
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_ground_multilingual_links_chinese_and_indic_to_international_codes() -> None:
    chinese = ground_multilingual("2型糖尿病", "zh-CN")
    hindi = ground_multilingual("बुखार", "hi_IN")

    assert (chinese.system, chinese.code) == ("ICD10", "E11.9")
    assert chinese.surface == "2型糖尿病"
    assert chinese.source_language == "zh"
    assert chinese.score == 1.0
    assert chinese.provenance["source_locale"] == "zh-CN"
    assert chinese.provenance["mapping_resource_version"].startswith(
        "openmed-icd10cn-icd10-starter@2026.08.0+sha256:"
    )
    assert chinese.to_dict()["cross_lingual_match_score"] == 1.0

    assert (hindi.system, hindi.code) == ("HPO", "HP:0001945")
    assert hindi.source_language == "hi"
    assert hindi.provenance["offline"] is True


def test_all_shipped_icd10cn_crosswalk_entries_map_exactly_both_directions() -> None:
    resource = load_icd10cn_crosswalk()
    bridge = ICD10CNBridge(resource)

    for entry in resource.entries:
        forward = bridge.to_icd10(entry.source_code)
        assert forward is not None
        assert forward.source_code == entry.source_code
        assert forward.target_code == entry.target_code
        assert forward.target_display == entry.target_display
        assert forward.resource_version == resource.resource_version

        reverse = bridge.from_icd10(entry.target_code)
        assert entry.source_code in {mapping.source_code for mapping in reverse}

    assert bridge.to_icd10("E11.9000") is None


def test_no_encoder_degrades_to_local_alias_and_string_matching(tmp_path: Path) -> None:
    grounder = MultilingualGrounder(encoder_path=str(tmp_path / "missing-weights"))

    exact = grounder.ground("发烧", "zh")
    fuzzy = grounder.ground("मांसपेशियों मे कमजोरी", "hi-IN")

    assert grounder.encoder_enabled is False
    assert exact.code == "HP:0001945"
    assert exact.candidates[0].match_kind == "exact-crosswalk"
    assert fuzzy.code == "HP:0001324"
    assert fuzzy.candidates[0].match_kind == "string-crosswalk"


def test_local_crosslingual_encoder_adds_semantic_dense_candidate(
    tmp_path: Path,
) -> None:
    resource = load_crosswalk(_write_crosswalk(tmp_path / "crosswalk.json"))
    encoder: AliasEncoder = _SyntheticConceptEncoder()

    result = ground_multilingual(
        "अर्थ-समान परीक्षण",
        "hi-IN",
        resources=[resource],
        encoder=encoder,
    )

    assert result.code == "HP:0001945"
    assert result.candidates[0].source == "cross-lingual-dense"
    assert result.candidates[0].match_kind == "dense-cross-lingual"
    assert result.score == 1.0
    assert result.provenance["encoder_id"] == encoder.encoder_id


def test_crosswalk_loader_rejects_nonredistributable_resource(tmp_path: Path) -> None:
    path = _write_crosswalk(
        tmp_path / "restricted.json",
        redistributable=False,
    )

    with pytest.raises(CrosswalkLicenseError, match="not declared redistributable"):
        load_crosswalk(path)


def test_multilingual_grounding_is_socket_blocked_and_pure_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*_args, **_kwargs):
        raise AssertionError("network egress attempted")

    monkeypatch.setattr(socket.socket, "connect", fail_socket)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_socket)
    monkeypatch.setattr(socket, "create_connection", fail_socket)

    result = ground_multilingual("తలనొప్పి", "te-IN")

    assert result.code == "HP:0002315"
    assert result.provenance["offline"] is True


def test_umls_aliases_activate_only_through_user_key_loader(tmp_path: Path) -> None:
    aliases = tmp_path / "caller_aliases.csv"
    aliases.write_text(
        "code,preferred_term,synonyms\nC-SYN-725,कृत्रिम संकेत,कृत्रिम खोज|बुखार\n",
        encoding="utf-8",
    )

    without_loader = MultilingualGrounder(resources=[]).ground("कृत्रिम संकेत", "hi-IN")
    assert without_loader.candidates == ()

    with pytest.raises(RestrictedVocabularyError, match="license key"):
        UserKeyVocabularyLoader("umls", aliases, license_key="")

    loader = UserKeyVocabularyLoader(
        "umls",
        aliases,
        license_key="caller-accepted-license",
    )
    with_loader = ground_multilingual(
        "कृत्रिम संकेत",
        "hi-IN",
        resources=[],
        restricted_loaders={"umls": loader},
    )

    assert with_loader.cui == "C-SYN-725"
    assert with_loader.system == "UMLS"
    assert with_loader.candidates[0].source == "user-key-local"
    assert "license_key" not in vars(loader)

    free_and_gated = ground_multilingual(
        "बुखार",
        "hi-IN",
        restricted_loaders={"umls": loader},
    )
    assert free_and_gated.system == "HPO"
    assert free_and_gated.cui == "C-SYN-725"


def test_synthetic_per_language_acc_at_5_meets_floor() -> None:
    report = run_multilingual_grounding_eval()

    assert set(report.per_language_acc_at_5) == {"bn", "hi", "ta", "te", "zh"}
    assert all(score >= 0.80 for score in report.per_language_acc_at_5.values())
    assert report.overall_acc_at_5 >= 0.80
    assert report.synthetic_provenance is True
    assert report.passed is True


def test_packaged_crosswalks_are_permissive_and_contain_no_restricted_data() -> None:
    resources = load_default_crosswalks()

    assert resources
    assert all(resource.redistributable for resource in resources)
    assert {resource.license_id for resource in resources} == {"CC0-1.0"}
    for path in DATA_DIR.glob("*.json"):
        folded_name = path.name.casefold()
        folded_content = path.read_text(encoding="utf-8").casefold()
        assert not any(marker in folded_name for marker in RESTRICTED_RESOURCE_MARKERS)
        assert not any(
            marker in folded_content for marker in RESTRICTED_RESOURCE_MARKERS
        )
