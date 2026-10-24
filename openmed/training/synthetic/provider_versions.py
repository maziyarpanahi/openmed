"""Surrogate-provider version capture and drift comparison for seeded data.

A fixed seed only reproduces a synthetic corpus while its surrogate providers
stay the same. A new Faker release, a changed Faker locale table, or an edited
repository-authored provider can silently change the values that seed draws.
This module records which providers stand behind a seeded generator without
storing any generated value:

* installed provider versions, currently the Faker distribution version;
* the Faker backend locale used for each requested locale;
* one digest per locale and provider component over a fixed probe set, where
  each probe method is drawn from a freshly reseeded provider stack; and
* optional caller-supplied digests of repository-authored tables and outputs.

:func:`compare_provider_manifests` compares a recorded manifest with a
regenerated one and names the provider component, locale, and aspect that
changed. Manifests, stamps, and drift findings hold only version strings,
locale codes, method names, and ``sha256:`` digests.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata as importlib_metadata
from typing import Any, Final, Mapping, Sequence

from faker import Faker

from openmed.core.anonymizer.locales import resolve_faker_backend_locale
from openmed.core.anonymizer.providers.clinical_ids import (
    register_clinical_providers,
)

SURROGATE_PROVIDER_SCHEMA_VERSION: Final = "openmed.surrogate-providers.v1"
PROVIDER_VERSION_DRIFT: Final = "provider_version_drift"
PROVIDER_PROBE_SEED: Final = 7919
PROVIDER_PROBE_DRAWS: Final = 3

FAKER_COMPONENT: Final = "faker"
CLINICAL_IDS_COMPONENT: Final = "openmed.clinical_ids"
REPOSITORY_COMPONENT: Final = "openmed"
EXTERNAL_COMPONENT: Final = "external"

# Fixed probe set: the Faker and clinical-ID methods that seeded OpenMed
# corpora draw from. Changing it changes every probe digest, so it is versioned
# by SURROGATE_PROVIDER_SCHEMA_VERSION.
_PROBE_CALLS: Final[tuple[tuple[str, tuple[int, ...]], ...]] = (
    ("city", ()),
    ("email", ()),
    ("medical_record_number", ()),
    ("name", ()),
    ("phone_number", ()),
    ("random_int", (0, 999_999)),
    ("street_address", ()),
)
PROVIDER_PROBE_METHODS: Final[tuple[str, ...]] = tuple(
    method for method, _args in _PROBE_CALLS
)
_FALLBACK_LOCALE: Final = "en_US"
_CLINICAL_PROVIDER_MODULE_PREFIX: Final = "openmed.core.anonymizer.providers"


def faker_version() -> str:
    """Return the installed Faker distribution version.

    Returns:
        The version string reported by the installed ``faker`` distribution,
        or ``"unknown"`` when no version can be determined.
    """

    try:
        return importlib_metadata.version("faker")
    except importlib_metadata.PackageNotFoundError:
        import faker

        return str(getattr(faker, "VERSION", "unknown"))


def installed_provider_versions() -> dict[str, str]:
    """Return the versions of third-party surrogate providers.

    Returns:
        A mapping from provider component name to installed version.
    """

    return {FAKER_COMPONENT: faker_version()}


def provider_component(module_name: str) -> str:
    """Map a provider class module to a stable provider component name.

    Args:
        module_name: ``__module__`` of the provider class that served a method.

    Returns:
        ``"faker"`` for Faker's own providers, ``"openmed.clinical_ids"`` for
        the repository's anonymizer providers, ``"openmed"`` for other
        repository modules, and ``"external"`` for caller-registered providers.
    """

    if module_name == "faker" or module_name.startswith("faker."):
        return FAKER_COMPONENT
    if module_name.startswith(_CLINICAL_PROVIDER_MODULE_PREFIX):
        return CLINICAL_IDS_COMPONENT
    if module_name == "openmed" or module_name.startswith("openmed."):
        return REPOSITORY_COMPONENT
    return EXTERNAL_COMPONENT


def digest_payload(payload: Any) -> str:
    """Return a ``sha256:`` digest of a JSON-compatible payload.

    Args:
        payload: JSON-compatible value hashed in canonical sorted-key form.

    Returns:
        A ``sha256:<hex>`` digest string.
    """

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class LocaleProviderProbe:
    """Value-free probe digests for one requested locale.

    Attributes:
        locale: Locale requested by the generator.
        backend_locale: Faker locale that actually served the probe.
        component_digests: One digest per provider component that served at
            least one probe method.
    """

    locale: str
    backend_locale: str
    component_digests: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible manifest entry for this locale."""

        return {
            "backend_locale": self.backend_locale,
            "providers": dict(sorted(self.component_digests.items())),
        }


def _probe_faker(locale: str) -> tuple[Faker, str]:
    backend_locale = resolve_faker_backend_locale(locale)
    try:
        faker = Faker(backend_locale)
    except (AttributeError, KeyError):
        backend_locale = _FALLBACK_LOCALE
        faker = Faker(backend_locale)
    register_clinical_providers(faker)
    return faker, backend_locale


def _serving_component(bound_method: Any) -> str:
    owner = getattr(bound_method, "__self__", None)
    module_name = type(owner).__module__ if owner is not None else ""
    return provider_component(module_name)


def probe_locale_providers(
    locale: str,
    *,
    seed: int = PROVIDER_PROBE_SEED,
    draws: int = PROVIDER_PROBE_DRAWS,
) -> LocaleProviderProbe:
    """Fingerprint the surrogate-provider stack behind one locale.

    The stack mirrors the anonymizer and corpus generators: a Faker instance
    for the resolved backend locale plus every registered OpenMed clinical
    provider. Each probe method is drawn ``draws`` times after reseeding the
    instance, so one component's change cannot shift another's digest.

    Args:
        locale: Requested Faker or conceptual locale.
        seed: Probe seed applied before each probe method.
        draws: Number of draws per probe method.

    Returns:
        Per-component digests. Generated probe values are hashed and dropped.

    Raises:
        ValueError: If ``draws`` is not a positive integer.
    """

    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        faker, backend_locale = _probe_faker(locale)
        material: dict[str, dict[str, list[str] | None]] = {}
        for method, args in _PROBE_CALLS:
            bound = getattr(faker, method, None)
            if bound is None:
                material.setdefault(FAKER_COMPONENT, {})[method] = None
                continue
            faker.seed_instance(seed)
            outputs = [str(bound(*args)) for _ in range(draws)]
            material.setdefault(_serving_component(bound), {})[method] = outputs
    digests = {
        component: digest_payload(
            {"backend_locale": backend_locale, "methods": methods}
        )
        for component, methods in material.items()
    }
    return LocaleProviderProbe(
        locale=locale,
        backend_locale=backend_locale,
        component_digests=digests,
    )


def probe_definition(
    *, seed: int = PROVIDER_PROBE_SEED, draws: int = PROVIDER_PROBE_DRAWS
) -> dict[str, Any]:
    """Return the value-free description of the fixed probe set."""

    return {
        "draws": draws,
        "methods": list(PROVIDER_PROBE_METHODS),
        "seed": seed,
    }


def _aggregate_probe_digest(entries: Mapping[str, Mapping[str, Any]]) -> str:
    return digest_payload({locale: dict(entry) for locale, entry in entries.items()})


def build_provider_manifest(
    locales: Sequence[str],
    *,
    table_digests: Mapping[str, Mapping[str, str]] | None = None,
    output_digests: Mapping[str, str] | None = None,
    corpus: Mapping[str, Any] | None = None,
    seed: int = PROVIDER_PROBE_SEED,
    draws: int = PROVIDER_PROBE_DRAWS,
) -> dict[str, Any]:
    """Build a value-free surrogate-provider manifest for a seeded corpus.

    Args:
        locales: Locales whose provider stacks the corpus draws from.
        table_digests: Optional per-locale digests of repository-authored
            provider tables, keyed by locale and then table owner.
        output_digests: Optional per-locale digests of the generated output.
        corpus: Optional corpus identity, such as dataset, version, seed,
            record count, and content hash.
        seed: Probe seed.
        draws: Draws per probe method.

    Returns:
        A JSON-compatible manifest containing provider versions, the probe
        definition and digest, and per-locale digests.
    """

    table_digests = table_digests or {}
    output_digests = output_digests or {}
    probes = {
        locale: probe_locale_providers(locale, seed=seed, draws=draws)
        for locale in sorted(set(locales))
    }
    probe_entries = {locale: probe.to_dict() for locale, probe in probes.items()}
    locale_entries: dict[str, dict[str, Any]] = {}
    for locale, entry in probe_entries.items():
        locale_entry = dict(entry)
        if locale in table_digests:
            locale_entry["tables"] = dict(sorted(table_digests[locale].items()))
        if locale in output_digests:
            locale_entry["output"] = output_digests[locale]
        locale_entries[locale] = locale_entry

    manifest: dict[str, Any] = {
        "schema_version": SURROGATE_PROVIDER_SCHEMA_VERSION,
        "providers": installed_provider_versions(),
        "probe": {
            **probe_definition(seed=seed, draws=draws),
            "digest": _aggregate_probe_digest(probe_entries),
        },
        "locales": locale_entries,
    }
    if corpus is not None:
        manifest["corpus"] = dict(corpus)
    return manifest


@lru_cache(maxsize=256)
def _cached_locale_probe(locale: str, provider_version: str) -> LocaleProviderProbe:
    # ``provider_version`` is part of the cache key so an upgraded or patched
    # Faker version is never served a stale probe.
    del provider_version
    return probe_locale_providers(locale)


def clear_provider_probe_cache() -> None:
    """Drop cached probe digests after swapping providers at runtime."""

    _cached_locale_probe.cache_clear()


def surrogate_provider_stamp(*locales: str) -> dict[str, Any]:
    """Return a compact provider stamp for synthetic example metadata.

    Probe digests are cached per process and Faker version. Call
    :func:`clear_provider_probe_cache` after registering or patching
    providers at runtime.

    Args:
        locales: Locales whose provider stacks produced the example.

    Returns:
        Schema version, provider versions, locales, and the aggregate probe
        digest. The stamp contains no generated values.

    Raises:
        ValueError: If no locale is given.
    """

    if not locales:
        raise ValueError("at least one locale is required")
    versions = installed_provider_versions()
    ordered = sorted(set(locales))
    entries = {
        locale: _cached_locale_probe(locale, versions[FAKER_COMPONENT]).to_dict()
        for locale in ordered
    }
    return {
        "schema_version": SURROGATE_PROVIDER_SCHEMA_VERSION,
        "providers": versions,
        "locales": {
            locale: entry["backend_locale"] for locale, entry in entries.items()
        },
        "probe_digest": _aggregate_probe_digest(entries),
    }


@dataclass(frozen=True)
class ProviderDriftFinding:
    """One value-free difference between a recorded and a current manifest.

    Attributes:
        component: Provider component, table owner, ``"corpus"``, ``"probe"``,
            ``"locale"``, or ``"manifest"``.
        aspect: What differs, for example ``"version"``, ``"probe_digest"``,
            ``"table_digest"``, ``"output_digest"``, or ``"content_hash"``.
        locale: Affected locale, when the difference is locale-specific.
        recorded: Recorded version, locale code, or digest.
        current: Current version, locale code, or digest.
    """

    component: str
    aspect: str
    locale: str | None = None
    recorded: str | None = None
    current: str | None = None

    @property
    def label(self) -> str:
        """Return a stable ``component[locale]:aspect`` label."""

        scope = f"{self.component}[{self.locale}]" if self.locale else self.component
        return f"{scope}:{self.aspect}"

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible finding."""

        return {
            "aspect": self.aspect,
            "component": self.component,
            "current": self.current,
            "locale": self.locale,
            "recorded": self.recorded,
        }


@dataclass(frozen=True)
class ProviderDriftReport:
    """Comparison of a recorded provider manifest with a regenerated one.

    Attributes:
        findings: Value-free differences, in deterministic order.
        corpus_comparable: Whether both manifests describe the same corpus
            identity, so output digests and content hashes were compared.
        content_hash_match: Content-hash comparison result, or ``None`` when
            the corpus is not comparable.
    """

    findings: tuple[ProviderDriftFinding, ...] = ()
    corpus_comparable: bool = False
    content_hash_match: bool | None = None

    @property
    def drifted(self) -> bool:
        """Return whether any provider, table, or output digest differs."""

        return bool(self.findings)

    @property
    def status(self) -> str:
        """Return ``"drift"`` or ``"match"``."""

        return "drift" if self.drifted else "match"

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return the stable reason codes for this report."""

        return (PROVIDER_VERSION_DRIFT,) if self.drifted else ()

    @property
    def changed(self) -> tuple[str, ...]:
        """Return the ``component[locale]:aspect`` label of each finding."""

        return tuple(finding.label for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible report."""

        return {
            "changed": list(self.changed),
            "content_hash_match": self.content_hash_match,
            "corpus_comparable": self.corpus_comparable,
            "findings": [finding.to_dict() for finding in self.findings],
            "reasons": list(self.reasons),
            "status": self.status,
        }


_CORPUS_IDENTITY_KEYS: Final = ("dataset", "record_count", "seed", "version")
_PROBE_DEFINITION_KEYS: Final = ("draws", "methods", "seed")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _compare_digest_maps(
    findings: list[ProviderDriftFinding],
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    aspect: str,
    locale: str,
) -> None:
    for component in sorted(set(recorded) | set(current)):
        if recorded.get(component) != current.get(component):
            findings.append(
                ProviderDriftFinding(
                    component=component,
                    aspect=aspect,
                    locale=locale,
                    recorded=_optional_text(recorded.get(component)),
                    current=_optional_text(current.get(component)),
                )
            )


def compare_provider_manifests(
    recorded: Mapping[str, Any],
    current: Mapping[str, Any],
) -> ProviderDriftReport:
    """Compare a recorded provider manifest with a regenerated one.

    Provider versions, the probe definition, and per-locale provider and table
    digests are always compared. Per-locale output digests and the content
    hash are compared only when both manifests describe the same corpus
    identity (dataset, version, seed, and record count). A locale recorded
    only in ``recorded`` is ignored, because the current run did not use it.

    Args:
        recorded: Manifest captured when the corpus was published.
        current: Manifest built from the current environment.

    Returns:
        A drift report whose findings name what changed without echoing any
        generated value.
    """

    findings: list[ProviderDriftFinding] = []
    if recorded.get("schema_version") != current.get("schema_version"):
        findings.append(
            ProviderDriftFinding(
                component="manifest",
                aspect="schema_version",
                recorded=_optional_text(recorded.get("schema_version")),
                current=_optional_text(current.get("schema_version")),
            )
        )

    recorded_versions = _mapping(recorded.get("providers"))
    current_versions = _mapping(current.get("providers"))
    for component in sorted(set(recorded_versions) | set(current_versions)):
        if recorded_versions.get(component) != current_versions.get(component):
            findings.append(
                ProviderDriftFinding(
                    component=component,
                    aspect="version",
                    recorded=_optional_text(recorded_versions.get(component)),
                    current=_optional_text(current_versions.get(component)),
                )
            )

    recorded_probe = _mapping(recorded.get("probe"))
    current_probe = _mapping(current.get("probe"))
    if any(
        recorded_probe.get(key) != current_probe.get(key)
        for key in _PROBE_DEFINITION_KEYS
    ):
        findings.append(
            ProviderDriftFinding(
                component="probe",
                aspect="probe_set",
                recorded=_optional_text(recorded_probe.get("digest")),
                current=_optional_text(current_probe.get("digest")),
            )
        )

    recorded_corpus = _mapping(recorded.get("corpus"))
    current_corpus = _mapping(current.get("corpus"))
    corpus_comparable = bool(recorded_corpus) and all(
        recorded_corpus.get(key) == current_corpus.get(key)
        for key in _CORPUS_IDENTITY_KEYS
    )

    recorded_locales = _mapping(recorded.get("locales"))
    current_locales = _mapping(current.get("locales"))
    for locale in sorted(current_locales):
        current_entry = _mapping(current_locales[locale])
        if locale not in recorded_locales:
            findings.append(
                ProviderDriftFinding(
                    component="locale",
                    aspect="unrecorded_locale",
                    locale=locale,
                    current=_optional_text(current_entry.get("backend_locale")),
                )
            )
            continue
        recorded_entry = _mapping(recorded_locales[locale])
        if recorded_entry.get("backend_locale") != current_entry.get("backend_locale"):
            findings.append(
                ProviderDriftFinding(
                    component=FAKER_COMPONENT,
                    aspect="backend_locale",
                    locale=locale,
                    recorded=_optional_text(recorded_entry.get("backend_locale")),
                    current=_optional_text(current_entry.get("backend_locale")),
                )
            )
        _compare_digest_maps(
            findings,
            _mapping(recorded_entry.get("providers")),
            _mapping(current_entry.get("providers")),
            aspect="probe_digest",
            locale=locale,
        )
        _compare_digest_maps(
            findings,
            _mapping(recorded_entry.get("tables")),
            _mapping(current_entry.get("tables")),
            aspect="table_digest",
            locale=locale,
        )
        if corpus_comparable and recorded_entry.get("output") != current_entry.get(
            "output"
        ):
            findings.append(
                ProviderDriftFinding(
                    component="corpus",
                    aspect="output_digest",
                    locale=locale,
                    recorded=_optional_text(recorded_entry.get("output")),
                    current=_optional_text(current_entry.get("output")),
                )
            )

    content_hash_match: bool | None = None
    if corpus_comparable:
        content_hash_match = recorded_corpus.get("content_hash") == current_corpus.get(
            "content_hash"
        )
        if not content_hash_match:
            findings.append(
                ProviderDriftFinding(
                    component="corpus",
                    aspect="content_hash",
                    recorded=_optional_text(recorded_corpus.get("content_hash")),
                    current=_optional_text(current_corpus.get("content_hash")),
                )
            )

    return ProviderDriftReport(
        findings=tuple(findings),
        corpus_comparable=corpus_comparable,
        content_hash_match=content_hash_match,
    )


__all__ = [
    "CLINICAL_IDS_COMPONENT",
    "EXTERNAL_COMPONENT",
    "FAKER_COMPONENT",
    "PROVIDER_PROBE_DRAWS",
    "PROVIDER_PROBE_METHODS",
    "PROVIDER_PROBE_SEED",
    "PROVIDER_VERSION_DRIFT",
    "REPOSITORY_COMPONENT",
    "SURROGATE_PROVIDER_SCHEMA_VERSION",
    "LocaleProviderProbe",
    "ProviderDriftFinding",
    "ProviderDriftReport",
    "build_provider_manifest",
    "clear_provider_probe_cache",
    "compare_provider_manifests",
    "digest_payload",
    "faker_version",
    "installed_provider_versions",
    "probe_definition",
    "probe_locale_providers",
    "provider_component",
    "surrogate_provider_stamp",
]
