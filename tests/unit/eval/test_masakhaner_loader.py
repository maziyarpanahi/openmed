"""Tests for the offline-only MasakhaNER evaluation suite."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import pytest

from openmed.core.labels import DATE, HIPAA_NAME, LOCATION, ORGANIZATION, PERSON
from openmed.eval import build_dataset_card
from openmed.eval.datasets import (
    MASAKHANER,
    MASAKHANER_1_LICENSE_ID,
    MASAKHANER_2_LICENSE_ID,
    MASAKHANER_2_LICENSE_NOTICE,
    MASAKHANER_LANGUAGES,
    MASAKHANER_ORG_HANDLING,
    MasakhaNerLicenseRequired,
    load_masakhaner_corpus,
    map_masakhaner_label,
    masakhaner_suite_metadata,
    records_from_masakhaner_conll,
    run_masakhaner_benchmark,
)
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.suites import load_suite_fixtures, suite_metadata


def test_registry_covers_all_masakhaner_2_languages_and_labels() -> None:
    assert len(MASAKHANER_LANGUAGES) == 20
    assert {"swa", "hau", "yor", "ibo", "zul", "xho"}.issubset(MASAKHANER_LANGUAGES)
    assert map_masakhaner_label("PER") == PERSON
    assert map_masakhaner_label("LOC") == LOCATION
    assert map_masakhaner_label("DATE") == DATE
    assert map_masakhaner_label("ORG") == ORGANIZATION
    metadata = masakhaner_suite_metadata(languages=("swa",))
    assert metadata["label_mapping"]["PER"] == HIPAA_NAME
    assert metadata["canonical_label_mapping"]["PER"] == PERSON
    assert "does not imply" in MASAKHANER_ORG_HANDLING


def test_loader_requires_explicit_cc_by_nc_license_acceptance(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")

    with pytest.raises(MasakhaNerLicenseRequired) as exc_info:
        load_masakhaner_corpus("swa", path)

    message = str(exc_info.value)
    assert "CC-BY-NC-4.0" in message
    assert "accept_license=True" in message
    assert "commercial-use legal guidance" in message
    assert "card metadata tags it AFL-3.0" in message


def test_masakhaner_1_license_error_names_per_card_license(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")

    with pytest.raises(MasakhaNerLicenseRequired, match="per-card license"):
        load_masakhaner_corpus("hau", path, version="1.0")


def test_masakhaner_1_dataset_card_uses_per_card_license(tmp_path: Path) -> None:
    card = build_dataset_card(
        MASAKHANER,
        language="hau",
        path=_write_conll(tmp_path / "test.txt"),
        version="1.0",
        accept_license=True,
    )

    assert card.license_id == MASAKHANER_1_LICENSE_ID
    assert card.source_url.endswith("/masakhane/masakhaner")


def test_conll_parser_round_trips_exact_reconstructed_offsets() -> None:
    records = records_from_masakhaner_conll(
        _synthetic_conll(),
        language="swa",
    )

    assert len(records) == 2
    first = records[0]
    assert first.text == "12 Mei 2025 Amina Diallo yuko Bamako ."
    assert [span.source_label for span in first.spans] == ["DATE", "PER", "LOC"]
    assert [(span.start, span.end) for span in first.spans] == [
        (0, 11),
        (12, 24),
        (30, 36),
    ]
    assert [first.text[span.start : span.end] for span in first.spans] == [
        "12 Mei 2025",
        "Amina Diallo",
        "Bamako",
    ]

    fixture = first.to_benchmark_fixture()
    assert [fixture.text[span.start : span.end] for span in fixture.gold_spans] == [
        span.text for span in fixture.gold_spans
    ]
    assert fixture.gold_spans[0].label == DATE
    assert records[1].spans[0].canonical_label == ORGANIZATION


def test_loader_records_license_source_and_content_hash(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")
    corpus = load_masakhaner_corpus("swa", path, accept_license=True)
    fixture = corpus.to_benchmark_fixtures()[0]
    expected_hash = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"

    assert corpus.provenance.license_id == MASAKHANER_2_LICENSE_ID
    assert corpus.provenance.source.endswith("/masakhane/masakhaner2")
    assert corpus.provenance.content_hash == expected_hash
    assert corpus.provenance.source_hash == expected_hash
    assert fixture.metadata["license_id"] == MASAKHANER_2_LICENSE_ID
    assert fixture.metadata["content_hash"] == expected_hash
    assert fixture.metadata["provenance"]["source_hash"] == expected_hash
    assert _synthetic_conll() not in str(fixture.metadata)


def test_dataset_card_records_source_hash_without_corpus_text(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")

    card = build_dataset_card(
        MASAKHANER,
        language="swa",
        path=path,
        accept_license=True,
    )
    rendered = card.to_json() + card.to_markdown()

    assert card.record_count == 2
    assert card.license_id == MASAKHANER_2_LICENSE_ID
    assert card.languages == ("swa",)
    assert card.labels == (DATE, LOCATION, ORGANIZATION, PERSON)
    assert card.content_hash.startswith("sha256:")
    assert card.source_hash == card.content_hash
    assert "Amina Diallo" not in rendered


def test_suite_loads_language_subset_from_prepopulated_cache(tmp_path: Path) -> None:
    _write_conll(tmp_path / "swa" / "test.txt")
    _write_conll(tmp_path / "hau" / "test.txt")

    fixtures = load_suite_fixtures(
        MASAKHANER,
        cache_dir=tmp_path,
        languages=("swa", "hau"),
        accept_license=True,
    )
    metadata = suite_metadata(MASAKHANER, languages=("swa", "hau"))

    assert len(fixtures) == 4
    assert {fixture.language for fixture in fixtures} == {"swa", "hau"}
    assert {fixture.metadata["source_kind"] for fixture in fixtures} == {
        "pre-populated-hf-cache"
    }
    assert metadata["license_id"] == MASAKHANER_2_LICENSE_ID
    assert metadata["redistribution"].startswith("user-supplied only")


def test_loader_resolves_official_repository_layout(tmp_path: Path) -> None:
    expected_path = _write_conll(
        tmp_path / "MasakhaNER2.0" / "data" / "swa" / "test.txt"
    )

    corpus = load_masakhaner_corpus(
        "swa",
        cache_dir=tmp_path,
        accept_license=True,
    )

    assert corpus.fixture_count == 2
    assert Path(corpus.source_path) == expected_path.resolve()


def test_single_file_requires_exactly_one_language(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")

    with pytest.raises(ValueError, match="requires exactly one language"):
        load_suite_fixtures(
            MASAKHANER,
            path=path,
            languages=("swa", "hau"),
            accept_license=True,
        )


def test_string_language_and_case_normalized_path_mapping(tmp_path: Path) -> None:
    path = _write_conll(tmp_path / "test.txt")

    fixtures = load_suite_fixtures(
        MASAKHANER,
        paths={"SWA": path},
        languages="SWA",
        accept_license=True,
    )

    assert len(fixtures) == 2
    assert {fixture.language for fixture in fixtures} == {"swa"}


def test_benchmark_emits_per_language_precision_recall_f1_offline(
    tmp_path: Path,
) -> None:
    swa = load_masakhaner_corpus(
        "swa",
        _write_conll(tmp_path / "swa.txt"),
        accept_license=True,
    ).to_benchmark_fixtures()
    hau = load_masakhaner_corpus(
        "hau",
        _write_conll(tmp_path / "hau.txt"),
        accept_license=True,
    ).to_benchmark_fixtures()
    checkpoint = tmp_path / "afroxlmr-local"
    checkpoint.mkdir()
    received_models: list[str] = []

    def identity_runner(
        fixture: BenchmarkFixture,
        model_name: str,
        device: str,
    ) -> tuple[Any, ...]:
        del device
        received_models.append(model_name)
        return fixture.gold_spans

    report = run_masakhaner_benchmark(
        [*swa, *hau],
        model_name="configured-detector",
        runner=identity_runner,
        languages=("swa", "hau"),
        checkpoint_path=checkpoint,
    )

    assert report.suite == MASAKHANER
    assert report.model_name == str(checkpoint.resolve())
    assert report.metadata["inference_only"] is True
    assert report.metadata["training_performed"] is False
    assert set(received_models) == {str(checkpoint.resolve())}
    assert {row["language"] for row in report.metrics["per_language"]} == {
        "hau",
        "swa",
    }
    for row in report.metrics["per_language"]:
        assert row["precision"] == pytest.approx(1.0)
        assert row["recall"] == pytest.approx(1.0)
        assert row["f1"] == pytest.approx(1.0)
        assert row["exact_span_f1"]["f1"] == pytest.approx(1.0)


def test_invalid_bio_continuation_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid BIO continuation"):
        records_from_masakhaner_conll(
            "Amina I-PER\n",
            language="swa",
        )


def test_no_masakhaner_dataset_artifacts_are_tracked() -> None:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "*masakhaner*",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    assert tracked
    assert all(Path(item).suffix == ".py" for item in tracked)
    assert all("fixtures" not in Path(item).parts for item in tracked)


def test_suite_metadata_is_offline_and_row_free() -> None:
    metadata = masakhaner_suite_metadata(languages=("zul", "xho"))

    assert metadata["languages"] == ["zul", "xho"]
    assert metadata["sources"]["zul"]["display_name"] == "isiZulu"
    assert metadata["sources"]["xho"]["display_name"] == "isiXhosa"
    assert metadata["license_notice"] == MASAKHANER_2_LICENSE_NOTICE
    assert "tokens" not in str(metadata)


def _write_conll(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_synthetic_conll(), encoding="utf-8")
    return path


def _synthetic_conll() -> str:
    return "\n".join(
        (
            "12 B-DATE",
            "Mei I-DATE",
            "2025 I-DATE",
            "Amina B-PER",
            "Diallo I-PER",
            "yuko O",
            "Bamako B-LOC",
            ". O",
            "",
            "Shirika B-ORG",
            "Afya I-ORG",
            "lilifunguliwa O",
            ". O",
            "",
        )
    )
