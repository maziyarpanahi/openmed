"""Focused canonical grounding facade and restricted-vocabulary tests."""

from __future__ import annotations

import hashlib
import socket
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


# Each restricted URI paired with the short name it must behave like.
RESTRICTED_URI_PAIRS = [
    ("snomed", "http://snomed.info/sct"),
    ("snomed", "http://snomed.info/sct/"),
    ("umls", "http://terminology.hl7.org/CodeSystem/umls"),
    ("cpt", "urn:openmed:restricted:cpt"),
    ("cpt", "http://www.ama-assn.org/go/cpt"),
]

# One synthetic alias per free system, matched by FREE_FIXTURE.
FREE_SYSTEM_CASES = [
    (
        "rxnorm",
        "http://www.nlm.nih.gov/research/umls/rxnorm",
        "renal disorder zeta001",
        "RXNORM-0001",
    ),
    ("icd10cm", "http://hl7.org/fhir/sid/icd-10-cm", "type 2 diabetes", "E11.9"),
    ("loinc", "http://loinc.org", "cardiac disorder zeta002", "LOINC-0002"),
    (
        "hpo",
        "http://human-phenotype-ontology.org",
        "pulmonary disorder zeta003",
        "HPO-0003",
    ),
    ("mesh", "https://www.nlm.nih.gov/mesh", "gastric disorder zeta004", "MESH-0004"),
]


def _all_free_system_loader(tmp_path: Path) -> VocabLoader:
    checksum = hashlib.sha256(FREE_FIXTURE.read_bytes()).hexdigest()
    return VocabLoader(
        cache_dir=tmp_path / "free-cache",
        local_only=True,
        registry={
            system: VocabSource(system=system, path=FREE_FIXTURE, sha256=checksum)
            for system, *_ in FREE_SYSTEM_CASES
        },
    )


@pytest.mark.parametrize(("short_name", "uri"), RESTRICTED_URI_PAIRS)
def test_restricted_system_uri_matches_short_name_without_network(
    short_name: str,
    uri: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted: list[str] = []

    def fail_network(*_: object, **__: object) -> None:
        attempted.append(uri)
        raise AssertionError("grounding attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    with pytest.raises(RestrictedVocabularyError) as by_name:
        ground(["synthetic finding"], systems=[short_name])
    with pytest.raises(RestrictedVocabularyError) as by_uri:
        ground(["synthetic finding"], systems=[uri])

    assert type(by_uri.value) is type(by_name.value)
    assert str(by_uri.value) == str(by_name.value)
    assert "synthetic finding" not in str(by_uri.value)
    assert attempted == []


@pytest.mark.parametrize(("short_name", "uri", "surface", "code"), FREE_SYSTEM_CASES)
def test_free_system_uri_returns_the_short_name_coding(
    short_name: str,
    uri: str,
    surface: str,
    code: str,
    tmp_path: Path,
) -> None:
    loader = _all_free_system_loader(tmp_path)
    text = f"Synthetic note: {surface}."
    by_name = ground(text, systems=[short_name], snapshot=loader)

    concept = by_name.concepts[0]
    assert (concept.system, concept.code, concept.system_uri) == (
        short_name,
        code,
        uri,
    )
    assert concept.display
    for requested in (uri, f"{uri}/"):
        by_uri = ground(text, systems=[requested], snapshot=loader)
        assert by_uri.to_dict() == by_name.to_dict()


@pytest.mark.parametrize("system", ["cvx", "http://hl7.org/fhir/sid/cvx"])
def test_cvx_short_name_and_uri_stay_unsupported(system: str, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported grounding system"):
        ground(
            "synthetic vaccine",
            systems=[system],
            snapshot=_all_free_system_loader(tmp_path),
        )
