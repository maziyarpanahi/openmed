"""Provider-version capture and drift checks for the openmed-synth corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from faker.providers.address import fr_FR as faker_fr_address

from openmed.cli.main import main
from openmed.core.anonymizer.providers import clinical_ids
from openmed.eval.suites import OPENMED_SYNTH, openmed_synth_suite_metadata
from openmed.eval.synth_corpus import (
    CORPUS_ID,
    CORPUS_VERSION,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_SEED,
    check_provider_drift,
    corpus_content_hash,
    corpus_locales,
    corpus_provider_manifest,
    generate_corpus,
    load_published_provider_receipt,
)
from openmed.training.synthetic import provider_versions
from openmed.training.synthetic.provider_versions import (
    PROVIDER_VERSION_DRIFT,
    faker_version,
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
_FAKER_LOCK_RE = re.compile(r'name = "faker"\nversion = "([^"]+)"')


def _locked_faker_version() -> str:
    match = _FAKER_LOCK_RE.search(Path("uv.lock").read_text(encoding="utf-8"))
    assert match is not None
    return match.group(1)


def _generated_values(rows) -> set[str]:
    return {span["text"] for row in rows for span in row["gold_spans"]}


def _assert_value_free(payload, rows) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    for value in _generated_values(rows):
        assert value not in serialized


def _patch_fr_city_table(monkeypatch) -> None:
    # Simulate a Faker data release that changes one locale's city table.
    monkeypatch.setattr(
        faker_fr_address.Provider,
        "city_formats",
        ("{{last_name}}-Synthetique",),
    )


def _patch_mrn_provider(monkeypatch) -> None:
    # Simulate an edited repository-authored clinical-ID provider.
    monkeypatch.setattr(
        clinical_ids.MedicalRecordNumberProvider,
        "medical_record_number",
        lambda self: f"MRN-{self.numerify('########')}",
    )


def test_receipt_matches_dataset_card_and_corpus_constants() -> None:
    receipt = load_published_provider_receipt()
    card = Path("docs/datasets/openmed-synth-phi.md").read_text(encoding="utf-8")
    rows = generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)

    assert receipt["corpus"] == {
        "content_hash": corpus_content_hash(rows),
        "dataset": CORPUS_ID,
        "record_count": DEFAULT_CORPUS_SIZE,
        "seed": DEFAULT_SEED,
        "version": CORPUS_VERSION,
    }
    assert sorted(receipt["locales"]) == list(corpus_locales())
    assert f"| Faker version | {receipt['providers']['faker']} |" in card
    assert f"| Provider probe digest | {receipt['probe']['digest']} |" in card
    assert receipt["providers"]["faker"] == _locked_faker_version()
    _assert_value_free(receipt, rows)


def test_locked_environment_reports_no_drift() -> None:
    if faker_version() != _locked_faker_version():
        pytest.skip("requires the Faker version pinned in uv.lock")

    report = check_provider_drift()

    assert report.status == "match"
    assert report.reasons == ()
    assert report.content_hash_match is True
    assert corpus_provider_manifest() == load_published_provider_receipt()


def test_suite_metadata_records_provider_versions_without_values() -> None:
    metadata = openmed_synth_suite_metadata()
    rows = generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)
    providers = metadata["surrogate_providers"]

    assert providers["providers"] == {"faker": faker_version()}
    assert _DIGEST_RE.fullmatch(providers["probe"]["digest"])
    assert providers["corpus"]["content_hash"] == metadata["content_hash"]
    for locale, entry in providers["locales"].items():
        assert entry["backend_locale"] == locale
        assert _DIGEST_RE.fullmatch(entry["output"])
        assert set(entry["providers"]) == {"faker", "openmed.clinical_ids"}
        assert set(entry["tables"]) == {"openmed.synth_corpus"}
    assert metadata["provider_drift"]["status"] in {"match", "drift"}
    _assert_value_free(providers, rows)
    _assert_value_free(metadata["provider_drift"], rows)


def test_faker_version_change_produces_drift_reason(monkeypatch) -> None:
    monkeypatch.setattr(provider_versions, "faker_version", lambda: "0.0.0+drift")

    report = check_provider_drift()

    assert report.reasons == (PROVIDER_VERSION_DRIFT,)
    assert "faker:version" in report.changed
    version = next(f for f in report.findings if f.aspect == "version")
    assert version.current == "0.0.0+drift"
    assert version.recorded == load_published_provider_receipt()["providers"]["faker"]


def test_faker_locale_table_change_names_provider_and_locale(monkeypatch) -> None:
    receipt = load_published_provider_receipt()
    _patch_fr_city_table(monkeypatch)
    rows = generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)

    report = check_provider_drift(receipt=receipt)

    assert report.reasons == (PROVIDER_VERSION_DRIFT,)
    assert report.content_hash_match is False
    digest_changes = {
        label for label in report.changed if not label.endswith(":version")
    }
    assert digest_changes == {
        "faker[fr_FR]:probe_digest",
        "corpus[fr_FR]:output_digest",
        "corpus:content_hash",
    }
    _assert_value_free(report.to_dict(), rows)


def test_repository_provider_change_names_clinical_ids(monkeypatch) -> None:
    receipt = load_published_provider_receipt()
    _patch_mrn_provider(monkeypatch)

    report = check_provider_drift(receipt=receipt)

    probe_changes = {
        finding.locale
        for finding in report.findings
        if finding.component == "openmed.clinical_ids"
        and finding.aspect == "probe_digest"
    }
    assert probe_changes == set(corpus_locales())
    assert not any(
        finding.component == "faker" and finding.aspect == "probe_digest"
        for finding in report.findings
    )
    assert "corpus:content_hash" in report.changed


def test_non_default_seed_compares_providers_but_not_outputs() -> None:
    receipt = load_published_provider_receipt()
    current = corpus_provider_manifest(seed=DEFAULT_SEED + 1, size=3)
    expected = provider_versions.compare_provider_manifests(receipt, current)

    report = check_provider_drift(seed=DEFAULT_SEED + 1, size=3, receipt=receipt)

    assert report == expected
    assert report.corpus_comparable is False
    assert report.content_hash_match is None
    assert not any(f.component == "corpus" for f in report.findings)


def test_cli_default_mode_warns_and_keeps_running_on_drift(monkeypatch, capsys) -> None:
    monkeypatch.setattr(provider_versions, "faker_version", lambda: "0.0.0+drift")

    result = main(["benchmark", "pii", "--suite", OPENMED_SYNTH])

    assert result == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["fixture_count"] == DEFAULT_CORPUS_SIZE
    assert payload["metadata"]["provider_drift"]["reasons"] == [PROVIDER_VERSION_DRIFT]
    assert f"Warning: {PROVIDER_VERSION_DRIFT}" in captured.err
    assert "faker:version" in captured.err


def test_cli_strict_mode_exits_non_zero_on_drift(monkeypatch, capsys) -> None:
    receipt = load_published_provider_receipt()
    rows = generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)
    _patch_fr_city_table(monkeypatch)

    result = main(["benchmark", "pii", "--suite", OPENMED_SYNTH, "--strict", "--json"])

    assert result == 1
    envelope = json.loads(capsys.readouterr().out)
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == PROVIDER_VERSION_DRIFT
    assert "faker[fr_FR]:probe_digest" in envelope["error"]["message"]
    _assert_value_free(envelope, rows)
    assert receipt == load_published_provider_receipt()


def test_cli_strict_mode_passes_in_locked_environment(capsys) -> None:
    if faker_version() != _locked_faker_version():
        pytest.skip("requires the Faker version pinned in uv.lock")

    result = main(["benchmark", "pii", "--suite", OPENMED_SYNTH, "--strict"])

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["metadata"]["provider_drift"]["status"] == "match"
