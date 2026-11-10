"""Tests for the sparse alias candidate generator.

All vocabulary content is synthetic and algorithmically generated; no real
patient data or licensed terminology is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from openmed.clinical.grounding import (
    Candidate,
    SparseCandidateGenerator,
    generate_candidates,
    get_linker,
)
from openmed.clinical.grounding.vocab import (
    VocabLoader,
    VocabLoaderError,
    VocabSource,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = _REPO_ROOT / "openmed/eval/golden/fixtures/grounding_vocab_synthetic.jsonl"
_SYSTEMS = ("icd10cm", "rxnorm", "loinc", "hpo", "mesh")


def _fixture_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in _FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _loader(tmp_path: Path, path: Path = _FIXTURE) -> VocabLoader:
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    registry = {
        system: VocabSource(system=system, path=path, sha256=sha256)
        for system in _SYSTEMS
    }
    return VocabLoader(cache_dir=tmp_path / "cache", registry=registry)


def test_generate_returns_e11_family_top_result(tmp_path):
    candidates = generate_candidates(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=_loader(tmp_path),
    )

    assert candidates
    top = candidates[0]
    assert isinstance(top, Candidate)
    assert top.system == "ICD10CM"
    assert top.code == "E11.9"
    assert top.concept_id == "E11.9"
    assert top.code.startswith("E11")
    assert top.match_kind == "exact"
    assert top.score == 1.0


def test_every_candidate_carries_provenance(tmp_path):
    candidates = generate_candidates(
        "type 2 diabetes",
        systems=["icd10cm"],
        loader=_loader(tmp_path),
    )

    assert candidates
    for candidate in candidates:
        assert candidate.source == "sparse"
        assert candidate.matched_alias
        assert candidate.match_kind in {"exact", "fuzzy"}
        assert candidate.vocab_version
        assert candidate.vocab_version.startswith("sha256:")


def test_fuzzy_match_recovers_misspelled_mention(tmp_path):
    candidates = generate_candidates(
        "type 2 diabetis mellitis",
        systems=["icd10cm"],
        loader=_loader(tmp_path),
    )

    codes = {candidate.code for candidate in candidates}
    assert "E11.9" in codes
    recovered = next(candidate for candidate in candidates if candidate.code == "E11.9")
    assert recovered.match_kind == "fuzzy"
    assert 0.0 < recovered.score < 1.0


def test_results_are_byte_identical_across_runs(tmp_path):
    first = generate_candidates(
        "high blood pressure", systems=list(_SYSTEMS), loader=_loader(tmp_path)
    )
    second = generate_candidates(
        "high blood pressure", systems=list(_SYSTEMS), loader=_loader(tmp_path)
    )

    assert [repr(candidate) for candidate in first] == [
        repr(candidate) for candidate in second
    ]


def test_tie_break_prefers_earlier_system_then_concept_id(tmp_path):
    shared = tmp_path / "shared.jsonl"
    shared.write_text(
        json.dumps(
            {
                "concept_id": "RX-1",
                "system": "rxnorm",
                "canonical_term": "Shared synthetic agent",
                "aliases": ["shared synthetic term"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "concept_id": "ICD-1",
                "system": "icd10cm",
                "canonical_term": "Shared synthetic disorder",
                "aliases": ["shared synthetic term"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loader = _loader(tmp_path, path=shared)

    icd_first = generate_candidates(
        "shared synthetic term", systems=["icd10cm", "rxnorm"], loader=loader
    )
    rx_first = generate_candidates(
        "shared synthetic term", systems=["rxnorm", "icd10cm"], loader=loader
    )

    assert [candidate.score for candidate in icd_first] == [1.0, 1.0]
    assert icd_first[0].system == "ICD10CM"
    assert rx_first[0].system == "RXNORM"


def test_unknown_system_raises_clear_error(tmp_path):
    with pytest.raises(VocabLoaderError, match="Unsupported free vocabulary system"):
        generate_candidates(
            "type 2 diabetes",
            systems=["not_a_system"],
            loader=_loader(tmp_path),
        )


def test_grounding_subpackage_ships_no_vocabulary_content():
    # The grounding subpackage itself ships no vocabulary tables — vocab data is
    # user-supplied at runtime. (The only bundled grounding data is a synthetic
    # eval fixture under openmed/eval/golden/fixtures/, following the existing
    # grounding_crosslingual.jsonl precedent; no licensed/real terminology ships.)
    import openmed.clinical.grounding as grounding

    package_dir = Path(grounding.__file__).resolve().parent
    vocab_suffixes = {".tsv", ".rrf", ".obo", ".csv", ".jsonl", ".txt", ".xml"}
    bundled = [
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in vocab_suffixes
    ]
    assert bundled == []


def test_generator_is_resolvable_via_registry(tmp_path):
    factory = get_linker("sparse")
    assert factory is SparseCandidateGenerator

    generator = factory(_loader(tmp_path))
    candidates = generator.generate("type 2 diabetes", ["icd10cm"], k=3)
    assert candidates[0].code == "E11.9"
    assert len(candidates) <= 3


def test_top5_recall_meets_threshold_on_synthetic_gold(tmp_path):
    loader = _loader(tmp_path)
    generator = SparseCandidateGenerator(loader)

    gold: list[tuple[str, str]] = []
    for row in _fixture_rows():
        for alias in row["aliases"]:
            gold.append((alias, row["concept_id"]))

    mentions: list[tuple[str, str]] = []
    index = 0
    while len(mentions) < 200:
        alias, code = gold[index % len(gold)]
        # Perturb roughly one in five mentions with a deterministic single-char
        # substitution to exercise the fuzzy stage.
        if index % 5 == 0 and len(alias) > 8:
            midpoint = len(alias) // 2
            alias = alias[:midpoint] + "x" + alias[midpoint + 1 :]
        mentions.append((alias, code))
        index += 1

    hits = 0
    for mention, code in mentions:
        candidates = generator.generate(mention, list(_SYSTEMS), k=5)
        if code in {candidate.code for candidate in candidates}:
            hits += 1

    recall = hits / len(mentions)
    assert recall >= 0.85
