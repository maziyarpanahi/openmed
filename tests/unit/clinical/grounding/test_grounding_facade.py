"""Focused canonical grounding facade and restricted-vocabulary tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    GroundedSpan,
    RestrictedVocabularyError,
    UserKeyVocabularyLoader,
    VocabLoader,
    VocabSource,
    ground,
)

ROOT = Path(__file__).resolve().parents[4]
FREE_FIXTURE = ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"


def _free_loader(tmp_path: Path) -> VocabLoader:
    checksum = hashlib.sha256(FREE_FIXTURE.read_bytes()).hexdigest()
    return VocabLoader(
        cache_dir=tmp_path / "cache",
        registry={
            "icd10cm": VocabSource(
                system="icd10cm",
                path=FREE_FIXTURE,
                sha256=checksum,
            )
        },
    )


def _default_system_loader(tmp_path: Path) -> VocabLoader:
    checksum = hashlib.sha256(FREE_FIXTURE.read_bytes()).hexdigest()
    return VocabLoader(
        cache_dir=tmp_path / "default-cache",
        registry={
            system: VocabSource(
                system=system,
                path=FREE_FIXTURE,
                sha256=checksum,
            )
            for system in ("rxnorm", "icd10cm", "loinc", "hpo")
        },
    )


def test_ground_returns_canonical_grounded_span_contract(tmp_path: Path) -> None:
    first = ground(
        [
            {
                "text": "type 2 diabetes",
                "start": 4,
                "end": 19,
                "label": "condition",
                "assertion": {
                    "temporality": "recent",
                    "certainty": "certain",
                    "negation": "affirmed",
                    "experiencer": "patient",
                },
            }
        ],
        systems=["icd10cm"],
        loader=_free_loader(tmp_path),
    )
    second = ground(
        [GroundedSpan(text="type 2 diabetes", start=4, end=19)],
        systems=["icd10cm"],
        loader=_free_loader(tmp_path),
    )

    assert len(first) == 1
    assert isinstance(first[0], GroundedSpan)
    assert first[0].codes == {"icd10cm": "E11.9"}
    assert first[0].cui is None
    assert first[0].score == 1.0
    assert first[0].canonical_label == "CONDITION"
    assert first[0].assertion is not None
    assert first[0].to_dict()["codes"] == {"icd10cm": "E11.9"}
    assert second[0].codes == first[0].codes


def test_ground_accepts_the_canonical_default_system_set(tmp_path: Path) -> None:
    result = ground(
        [{"text": "renal disorder zeta001", "language": "en"}],
        loader=_default_system_loader(tmp_path),
    )

    assert len(result) == 1
    assert isinstance(result[0], GroundedSpan)
    assert "rxnorm" in result[0].codes


def test_restricted_system_requires_explicit_user_key_loader() -> None:
    with pytest.raises(RestrictedVocabularyError, match="explicit matching"):
        ground(["synthetic finding"], systems=["umls"])


def test_user_key_loader_activates_only_local_normalized_aliases(
    tmp_path: Path,
) -> None:
    aliases = tmp_path / "local_aliases.csv"
    aliases.write_text(
        "code,preferred_term,synonyms\nC-SYN-1,synthetic finding,synthetic sign\n",
        encoding="utf-8",
    )
    with pytest.raises(RestrictedVocabularyError, match="license key"):
        UserKeyVocabularyLoader("umls", aliases, license_key="")

    loader = UserKeyVocabularyLoader(
        "umls",
        aliases,
        license_key="caller-accepted-license",
    )
    result = ground(
        ["synthetic sign"],
        systems=["umls"],
        restricted_loaders={"umls": loader},
    )

    assert result[0].cui == "C-SYN-1"
    assert result[0].codes == {"umls": "C-SYN-1"}
    assert result[0].candidates[0].vocab_version.startswith("sha256:")
    assert "license_key" not in vars(loader)


def test_cpt_remains_out_of_process() -> None:
    with pytest.raises(RestrictedVocabularyError, match="out of process"):
        ground(["synthetic procedure"], systems=["cpt"])
