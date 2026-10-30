"""Tests for surrogate-provider version capture and drift comparison."""

from __future__ import annotations

import copy
import json
import re

import pytest
from faker import Faker
from faker.providers import BaseProvider

from openmed.core.anonymizer.providers import clinical_ids
from openmed.training.synthetic import provider_versions
from openmed.training.synthetic.locale_phi import LocalePhiGenerator
from openmed.training.synthetic.provider_versions import (
    CLINICAL_IDS_COMPONENT,
    EXTERNAL_COMPONENT,
    FAKER_COMPONENT,
    PROVIDER_PROBE_SEED,
    PROVIDER_VERSION_DRIFT,
    SURROGATE_PROVIDER_SCHEMA_VERSION,
    build_provider_manifest,
    clear_provider_probe_cache,
    compare_provider_manifests,
    faker_version,
    probe_locale_providers,
    provider_component,
    surrogate_provider_stamp,
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    clear_provider_probe_cache()
    yield
    clear_provider_probe_cache()


class _SyntheticOverrideProvider(BaseProvider):
    """Caller-registered provider that takes over the MRN probe method."""

    def medical_record_number(self) -> str:
        return f"MRN-{self.numerify('########')}"


def _probe_values(locale: str) -> list[str]:
    faker = Faker(locale)
    clinical_ids.register_clinical_providers(faker)
    values = []
    for method in ("name", "city", "street_address", "email", "phone_number"):
        faker.seed_instance(PROVIDER_PROBE_SEED)
        values.append(str(getattr(faker, method)()))
    return values


def test_provider_component_maps_modules_to_stable_names() -> None:
    assert provider_component("faker.providers.person.en_US") == FAKER_COMPONENT
    assert (
        provider_component("openmed.core.anonymizer.providers.clinical_ids")
        == CLINICAL_IDS_COMPONENT
    )
    assert provider_component("openmed.eval.synth_corpus") == "openmed"
    assert provider_component("site_plugins.providers") == EXTERNAL_COMPONENT


def test_probe_is_deterministic_component_scoped_and_value_free() -> None:
    first = probe_locale_providers("en_US")
    second = probe_locale_providers("en_US")

    assert first == second
    assert first.backend_locale == "en_US"
    assert set(first.component_digests) == {FAKER_COMPONENT, CLINICAL_IDS_COMPONENT}
    assert all(_DIGEST_RE.fullmatch(d) for d in first.component_digests.values())
    serialized = json.dumps(first.to_dict(), ensure_ascii=False)
    for value in _probe_values("en_US"):
        assert value not in serialized


def test_probe_records_backend_fallback_for_unavailable_faker_locale() -> None:
    probe = probe_locale_providers("xx_XX")

    assert probe.locale == "xx_XX"
    assert probe.backend_locale == "en_US"


def test_probe_rejects_non_positive_draws() -> None:
    with pytest.raises(ValueError, match="draws"):
        probe_locale_providers("en_US", draws=0)


def test_manifest_records_versions_probe_tables_and_outputs() -> None:
    manifest = build_provider_manifest(
        ["fr_FR", "en_US", "fr_FR"],
        table_digests={"fr_FR": {"repo.table": "sha256:" + "a" * 64}},
        output_digests={"en_US": "sha256:" + "b" * 64},
        corpus={"dataset": "synthetic-test", "content_hash": "sha256:" + "c" * 64},
    )

    assert manifest["schema_version"] == SURROGATE_PROVIDER_SCHEMA_VERSION
    assert manifest["providers"] == {FAKER_COMPONENT: faker_version()}
    assert manifest["probe"]["seed"] == PROVIDER_PROBE_SEED
    assert _DIGEST_RE.fullmatch(manifest["probe"]["digest"])
    assert list(manifest["locales"]) == ["en_US", "fr_FR"]
    assert manifest["locales"]["fr_FR"]["tables"] == {
        "repo.table": "sha256:" + "a" * 64
    }
    assert manifest["locales"]["en_US"]["output"] == "sha256:" + "b" * 64
    assert manifest["corpus"]["dataset"] == "synthetic-test"


def test_identical_manifests_report_no_drift() -> None:
    manifest = build_provider_manifest(["en_US"])

    report = compare_provider_manifests(manifest, copy.deepcopy(manifest))

    assert report.status == "match"
    assert report.reasons == ()
    assert report.to_dict()["findings"] == []


def test_version_change_names_provider_component() -> None:
    recorded = build_provider_manifest(["en_US"])
    current = copy.deepcopy(recorded)
    current["providers"][FAKER_COMPONENT] = "999.0.0"

    report = compare_provider_manifests(recorded, current)

    assert report.reasons == (PROVIDER_VERSION_DRIFT,)
    assert report.changed == ("faker:version",)
    assert report.findings[0].recorded == recorded["providers"][FAKER_COMPONENT]
    assert report.findings[0].current == "999.0.0"


def test_digest_changes_name_component_table_and_locale() -> None:
    recorded = build_provider_manifest(
        ["de_DE", "en_US"],
        table_digests={
            "de_DE": {"repo.table": "sha256:" + "1" * 64},
            "en_US": {"repo.table": "sha256:" + "2" * 64},
        },
    )
    current = copy.deepcopy(recorded)
    current["locales"]["de_DE"]["providers"][FAKER_COMPONENT] = "sha256:" + "f" * 64
    current["locales"]["en_US"]["tables"]["repo.table"] = "sha256:" + "e" * 64

    report = compare_provider_manifests(recorded, current)

    assert report.changed == (
        "faker[de_DE]:probe_digest",
        "repo.table[en_US]:table_digest",
    )
    assert report.corpus_comparable is False
    assert report.content_hash_match is None


def test_outputs_compare_only_for_the_same_corpus_identity() -> None:
    corpus = {
        "content_hash": "sha256:" + "0" * 64,
        "dataset": "synthetic-test",
        "record_count": 2,
        "seed": 1,
        "version": "1.0.0",
    }
    recorded = build_provider_manifest(
        ["en_US"], output_digests={"en_US": "sha256:" + "3" * 64}, corpus=corpus
    )
    changed_output = copy.deepcopy(recorded)
    changed_output["locales"]["en_US"]["output"] = "sha256:" + "4" * 64
    changed_output["corpus"]["content_hash"] = "sha256:" + "5" * 64

    same_corpus = compare_provider_manifests(recorded, changed_output)
    other_seed = copy.deepcopy(changed_output)
    other_seed["corpus"]["seed"] = 2
    different_corpus = compare_provider_manifests(recorded, other_seed)

    assert same_corpus.changed == ("corpus[en_US]:output_digest", "corpus:content_hash")
    assert same_corpus.content_hash_match is False
    assert different_corpus.status == "match"
    assert different_corpus.corpus_comparable is False


def test_unrecorded_locale_is_reported_but_unused_locale_is_not() -> None:
    recorded = build_provider_manifest(["en_US", "fr_FR"])
    current = build_provider_manifest(["en_US", "de_DE"])

    report = compare_provider_manifests(recorded, current)

    assert report.changed == ("locale[de_DE]:unrecorded_locale",)


def test_registered_provider_takeover_names_external_component(monkeypatch) -> None:
    recorded = build_provider_manifest(["en_US"])
    monkeypatch.setattr(clinical_ids, "_extra_providers", [])
    clinical_ids.register_extra_clinical_provider(_SyntheticOverrideProvider)

    report = compare_provider_manifests(recorded, build_provider_manifest(["en_US"]))

    assert report.reasons == (PROVIDER_VERSION_DRIFT,)
    assert set(report.changed) == {
        "external[en_US]:probe_digest",
        "openmed.clinical_ids[en_US]:probe_digest",
    }
    external = next(f for f in report.findings if f.component == EXTERNAL_COMPONENT)
    assert external.recorded is None
    assert _DIGEST_RE.fullmatch(external.current or "")


def test_stamp_is_compact_value_free_and_keyed_by_provider_version(
    monkeypatch,
) -> None:
    stamp = surrogate_provider_stamp("fr_FR", "en_US", "fr_FR")

    assert stamp == {
        "locales": {"en_US": "en_US", "fr_FR": "fr_FR"},
        "probe_digest": build_provider_manifest(["en_US", "fr_FR"])["probe"]["digest"],
        "providers": {FAKER_COMPONENT: faker_version()},
        "schema_version": SURROGATE_PROVIDER_SCHEMA_VERSION,
    }
    serialized = json.dumps(stamp, ensure_ascii=False)
    for value in _probe_values("fr_FR"):
        assert value not in serialized

    monkeypatch.setattr(provider_versions, "faker_version", lambda: "0.0.0+test")
    patched = surrogate_provider_stamp("en_US", "fr_FR")
    assert patched["providers"] == {FAKER_COMPONENT: "0.0.0+test"}
    assert patched["probe_digest"] == stamp["probe_digest"]


def test_stamp_requires_a_locale() -> None:
    with pytest.raises(ValueError, match="locale"):
        surrogate_provider_stamp()


def test_locale_phi_metadata_records_provider_stamp_without_values() -> None:
    example = LocalePhiGenerator(seed=41).generate("fr")
    stamp = example.metadata["surrogate_providers"]

    assert stamp == surrogate_provider_stamp(example.locale)
    assert stamp["providers"][FAKER_COMPONENT] == faker_version()
    assert _DIGEST_RE.fullmatch(stamp["probe_digest"])
    serialized = json.dumps(stamp, ensure_ascii=False)
    for span in example.gold_spans:
        assert span.text not in serialized
    assert example.to_training_item()["metadata"]["surrogate_providers"] == stamp


def test_locale_phi_stamp_tracks_a_patched_provider_version(monkeypatch) -> None:
    baseline = LocalePhiGenerator(seed=41).generate("en")
    monkeypatch.setattr(provider_versions, "faker_version", lambda: "0.0.0+test")
    drifted = LocalePhiGenerator(seed=41).generate("en")

    assert drifted.text == baseline.text
    assert drifted.metadata["surrogate_providers"]["providers"] == {
        FAKER_COMPONENT: "0.0.0+test"
    }


def test_burned_in_metadata_records_name_and_mrn_provider_locales() -> None:
    pytest.importorskip("PIL")
    from openmed.training.synthetic.burned_in import BurnedInGenerator

    example = BurnedInGenerator(seed=1790).generate("de")
    stamp = example.metadata["surrogate_providers"]

    assert stamp == surrogate_provider_stamp("de_DE", "en_US")
    serialized = json.dumps(stamp, ensure_ascii=False)
    for span in example.gold_spans:
        assert span.text not in serialized
