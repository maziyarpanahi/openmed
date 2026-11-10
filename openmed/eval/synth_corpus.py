"""Deterministic synthetic clinical-PHI corpus generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from faker import Faker

from openmed.core.anonymizer.providers.clinical_ids import (
    register_clinical_providers,
)
from openmed.core.labels import (
    DATE_OF_BIRTH,
    EMAIL,
    ID_NUM,
    LOCATION,
    PERSON,
    PHONE,
    STREET_ADDRESS,
    ZIPCODE,
)
from openmed.training.synthetic.provider_versions import (
    ProviderDriftReport,
    build_provider_manifest,
    compare_provider_manifests,
    digest_payload,
)

CORPUS_ID = "openmed-synth"
CORPUS_VERSION = "1.0.0"
DEFAULT_SEED = 2352
DEFAULT_CORPUS_SIZE = 14
CORPUS_LICENSE = "Apache-2.0"
PROVIDER_RECEIPT_RESOURCE = "data/openmed_synth_provider_receipt.json"
SYNTH_CORPUS_TABLE_COMPONENT = "openmed.synth_corpus"

_DATE_START = date(1935, 1, 1)
_DATE_END = date(2000, 12, 31)
_PLACEHOLDER_RE = re.compile(r"\{([A-Z_]+)\}")
# Locale-shaped postcode draws: (minimum, maximum, zero-padded width).
_POSTCODE_RULES: Mapping[str, tuple[int, int, int]] = {
    "de_DE": (1_000, 99_999, 5),
    "en_US": (10_000, 99_999, 5),
    "es_ES": (1_000, 52_999, 5),
    "fr_FR": (1_000, 98_999, 5),
    "hi_IN": (110_000, 999_999, 6),
    "pt_PT": (1_000_000, 9_999_999, 7),
    "zh_CN": (100_000, 999_999, 6),
}
# Postcodes rendered with a hyphen after this many digits.
_POSTCODE_SPLITS: Mapping[str, int] = {"pt_PT": 4}


@dataclass(frozen=True)
class _LocaleProfile:
    """A locale and clinical-note shell used by the corpus generator."""

    language: str
    locale: str
    template: str


_LOCALE_PROFILES = (
    _LocaleProfile(
        "en",
        "en_US",
        (
            "Synthetic clinical note: patient {PERSON} was born on "
            "{DATE_OF_BIRTH}; phone {PHONE}; record {ID_NUM}; address "
            "{STREET_ADDRESS}, {ZIPCODE}, {LOCATION}; email {EMAIL}."
        ),
    ),
    _LocaleProfile(
        "fr",
        "fr_FR",
        (
            "Note clinique synthétique : patient {PERSON}, né le "
            "{DATE_OF_BIRTH} ; téléphone {PHONE} ; dossier {ID_NUM} ; "
            "adresse {STREET_ADDRESS}, {ZIPCODE}, {LOCATION} ; courriel "
            "{EMAIL}."
        ),
    ),
    _LocaleProfile(
        "de",
        "de_DE",
        (
            "Synthetische klinische Notiz: Patient {PERSON}, geboren am "
            "{DATE_OF_BIRTH}; Telefon {PHONE}; Akte {ID_NUM}; Adresse "
            "{STREET_ADDRESS}, {ZIPCODE}, {LOCATION}; E-Mail {EMAIL}."
        ),
    ),
    _LocaleProfile(
        "es",
        "es_ES",
        (
            "Nota clínica sintética: paciente {PERSON}, nacido el "
            "{DATE_OF_BIRTH}; teléfono {PHONE}; historia {ID_NUM}; dirección "
            "{STREET_ADDRESS}, {ZIPCODE}, {LOCATION}; correo {EMAIL}."
        ),
    ),
    _LocaleProfile(
        "pt",
        "pt_PT",
        (
            "Nota clínica sintética: paciente {PERSON}, nascido em "
            "{DATE_OF_BIRTH}; telefone {PHONE}; registo {ID_NUM}; morada "
            "{STREET_ADDRESS}, {ZIPCODE}, {LOCATION}; e-mail {EMAIL}."
        ),
    ),
    _LocaleProfile(
        "hi",
        "hi_IN",
        (
            "सिंथेटिक क्लिनिकल नोट: रोगी {PERSON}, जन्म {DATE_OF_BIRTH}; "
            "फोन {PHONE}; रिकॉर्ड {ID_NUM}; पता {STREET_ADDRESS}, {ZIPCODE}, "
            "{LOCATION}; ईमेल {EMAIL}।"
        ),
    ),
    _LocaleProfile(
        "zh",
        "zh_CN",
        (
            "合成临床记录：患者{PERSON}，出生日期{DATE_OF_BIRTH}；电话{PHONE}；"
            "病历号{ID_NUM}；地址{STREET_ADDRESS}，{ZIPCODE}，{LOCATION}；"
            "电子邮箱{EMAIL}。"
        ),
    ),
)


def _single_line(value: Any) -> str:
    """Normalize a Faker value without changing its semantic surface."""

    return " ".join(str(value).split())


def _stable_postcode(faker: Faker, locale: str) -> str:
    """Generate a locale-shaped postcode from Faker's seeded integer stream."""

    minimum, maximum, width = _POSTCODE_RULES[locale]
    digits = f"{faker.random_int(min=minimum, max=maximum):0{width}d}"
    split = _POSTCODE_SPLITS.get(locale)
    if split is None:
        return digits
    return f"{digits[:split]}-{digits[split:]}"


def _build_values(faker: Faker, *, locale: str) -> dict[str, str]:
    """Generate one record's locale-aware synthetic PHI values."""

    birth_date = _DATE_START + timedelta(
        days=faker.random_int(min=0, max=(_DATE_END - _DATE_START).days)
    )
    return {
        PERSON: _single_line(faker.name()),
        DATE_OF_BIRTH: birth_date.isoformat(),
        PHONE: _single_line(faker.phone_number()),
        ID_NUM: _single_line(faker.medical_record_number()),
        STREET_ADDRESS: _single_line(faker.street_address()),
        ZIPCODE: _stable_postcode(faker, locale),
        LOCATION: _single_line(faker.city()),
        EMAIL: _single_line(faker.email()),
    }


def _render_record(
    profile: _LocaleProfile,
    values: Mapping[str, str],
    *,
    record_id: str,
    seed: int,
) -> dict[str, Any]:
    """Render one row and calculate offsets from the composed segments."""

    parts: list[str] = []
    spans: list[dict[str, Any]] = []
    cursor = 0
    template_cursor = 0
    for match in _PLACEHOLDER_RE.finditer(profile.template):
        literal = profile.template[template_cursor : match.start()]
        parts.append(literal)
        cursor += len(literal)

        label = match.group(1)
        value = values[label]
        start = cursor
        parts.append(value)
        cursor += len(value)
        span_metadata: dict[str, Any] = {
            "generator": "faker",
            "synthetic": True,
        }
        if label == ID_NUM:
            span_metadata["identifier_type"] = "medical_record_number"
            span_metadata["provider"] = "openmed.clinical_ids"
        spans.append(
            {
                "start": start,
                "end": cursor,
                "label": label,
                "text": value,
                "metadata": span_metadata,
            }
        )
        template_cursor = match.end()

    trailing = profile.template[template_cursor:]
    parts.append(trailing)
    text = "".join(parts)

    expected_text = text
    for span in reversed(spans):
        expected_text = (
            expected_text[: span["start"]]
            + f"[{span['label']}]"
            + expected_text[span["end"] :]
        )

    return {
        "id": record_id,
        "language": profile.language,
        "text": text,
        "gold_spans": spans,
        "metadata": {
            "category": "multilingual",
            "contains_real_phi": False,
            "dataset": CORPUS_ID,
            "dataset_version": CORPUS_VERSION,
            "expected_output": {
                "method": "mask",
                "text": expected_text,
            },
            "locale": profile.locale,
            "seed": seed,
            "split": "synthetic",
            "synthetic": True,
        },
    }


def generate_corpus(
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_CORPUS_SIZE,
) -> list[dict[str, Any]]:
    """Generate a deterministic list of synthetic golden-fixture rows.

    Args:
        seed: Integer seed used to isolate each record's Faker stream.
        size: Number of records to generate. Locale profiles cycle when a
            caller requests more rows than the default corpus size.

    Returns:
        JSON-ready rows containing source text, canonical gold spans, and the
        expected mask output.

    Raises:
        ValueError: If *size* is not positive or *seed* is not an integer.
    """

    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not -(2**63) <= seed < 2**63
    ):
        raise ValueError("seed must be a signed 64-bit integer")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 10_000:
        raise ValueError("size must be between 1 and 10000")

    rows: list[dict[str, Any]] = []
    for index in range(size):
        profile = _LOCALE_PROFILES[index % len(_LOCALE_PROFILES)]
        faker = Faker(profile.locale)
        register_clinical_providers(faker)
        row_seed = seed + (index * 1_000_003)
        faker.seed_instance(row_seed)
        values = _build_values(faker, locale=profile.locale)
        rows.append(
            _render_record(
                profile,
                values,
                record_id=f"{CORPUS_ID}-{index + 1:04d}",
                seed=seed,
            )
        )
    return rows


def render_corpus(rows: Sequence[Mapping[str, Any]]) -> str:
    """Serialize corpus rows as canonical JSONL."""

    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )


def corpus_content_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the SHA-256 digest of the canonical corpus JSONL."""

    payload = render_corpus(rows).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def label_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count canonical gold spans by label in deterministic order."""

    counts = Counter(
        str(span["label"]) for row in rows for span in row.get("gold_spans", [])
    )
    return dict(sorted(counts.items()))


def corpus_locales() -> tuple[str, ...]:
    """Return the Faker locales the corpus generator draws from, sorted."""

    return tuple(sorted(profile.locale for profile in _LOCALE_PROFILES))


def provider_table_digests() -> dict[str, dict[str, str]]:
    """Digest the repository-authored tables behind each corpus locale.

    The digest covers the note template, the date-of-birth range, and the
    locale postcode rule. It never covers generated values.

    Returns:
        Per-locale mapping from table owner to ``sha256:`` digest.
    """

    digests: dict[str, dict[str, str]] = {}
    for profile in _LOCALE_PROFILES:
        table = {
            "date_of_birth_range": [_DATE_START.isoformat(), _DATE_END.isoformat()],
            "language": profile.language,
            "postcode_rule": list(_POSTCODE_RULES[profile.locale]),
            "postcode_split": _POSTCODE_SPLITS.get(profile.locale),
            "template": profile.template,
        }
        digests[profile.locale] = {SYNTH_CORPUS_TABLE_COMPONENT: digest_payload(table)}
    return dict(sorted(digests.items()))


def locale_output_digests(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Return one canonical-JSONL ``sha256:`` digest per generated locale.

    Args:
        rows: Corpus rows from :func:`generate_corpus`.

    Returns:
        Mapping from Faker locale to the digest of that locale's rows.
    """

    by_locale: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        locale = str(row.get("metadata", {}).get("locale", ""))
        by_locale.setdefault(locale, []).append(row)
    return {
        locale: corpus_content_hash(locale_rows)
        for locale, locale_rows in sorted(by_locale.items())
    }


def corpus_provider_manifest(
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_CORPUS_SIZE,
    rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return the value-free surrogate-provider manifest for a corpus.

    Args:
        seed: Corpus seed.
        size: Corpus record count.
        rows: Optional rows already generated for ``seed`` and ``size``.

    Returns:
        Faker version, per-locale provider probe digests, repository table
        digests, per-locale output digests, and the corpus content hash.
    """

    corpus_rows = (
        list(rows) if rows is not None else generate_corpus(seed=seed, size=size)
    )
    return build_provider_manifest(
        corpus_locales(),
        table_digests=provider_table_digests(),
        output_digests=locale_output_digests(corpus_rows),
        corpus={
            "content_hash": corpus_content_hash(corpus_rows),
            "dataset": CORPUS_ID,
            "record_count": len(corpus_rows),
            "seed": seed,
            "version": CORPUS_VERSION,
        },
    )


def load_published_provider_receipt() -> dict[str, Any]:
    """Load the provider manifest recorded for the published corpus.

    Returns:
        The manifest captured in the locked environment for the default seed
        and size, matching the dataset card.
    """

    resource = resources.files("openmed.eval").joinpath(PROVIDER_RECEIPT_RESOURCE)
    return json.loads(resource.read_text(encoding="utf-8"))


def check_provider_drift(
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_CORPUS_SIZE,
    rows: Sequence[Mapping[str, Any]] | None = None,
    receipt: Mapping[str, Any] | None = None,
) -> ProviderDriftReport:
    """Regenerate the corpus and compare it with the published receipt.

    Provider versions and probe digests are always compared. Output digests
    and the content hash are compared only for the published seed and size.

    Args:
        seed: Corpus seed.
        size: Corpus record count.
        rows: Optional rows already generated for ``seed`` and ``size``.
        receipt: Optional recorded manifest; defaults to the published one.

    Returns:
        A drift report naming each changed provider, locale, or table.
    """

    recorded = receipt if receipt is not None else load_published_provider_receipt()
    current = corpus_provider_manifest(seed=seed, size=size, rows=rows)
    return compare_provider_manifests(recorded, current)


def write_corpus(
    path: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_CORPUS_SIZE,
) -> Path:
    """Generate and write canonical corpus JSONL to *path*."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_corpus(generate_corpus(seed=seed, size=size)),
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Run the corpus generator from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--size",
        "--corpus-size",
        dest="size",
        type=int,
        default=DEFAULT_CORPUS_SIZE,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--hash-only",
        action="store_true",
        help="print only the content hash instead of JSONL rows",
    )
    parser.add_argument(
        "--provider-manifest",
        action="store_true",
        help=(
            "print the value-free surrogate-provider manifest (Faker version, "
            "probe, table, and output digests) instead of JSONL rows"
        ),
    )
    args = parser.parse_args(argv)
    rows = generate_corpus(seed=args.seed, size=args.size)
    if args.provider_manifest:
        manifest = corpus_provider_manifest(seed=args.seed, size=args.size, rows=rows)
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.hash_only:
        print(corpus_content_hash(rows))
    elif args.output is not None:
        path = write_corpus(args.output, seed=args.seed, size=args.size)
        print(path)
    else:
        print(render_corpus(rows), end="")
    return 0


__all__ = [
    "CORPUS_ID",
    "CORPUS_LICENSE",
    "CORPUS_VERSION",
    "DEFAULT_CORPUS_SIZE",
    "DEFAULT_SEED",
    "PROVIDER_RECEIPT_RESOURCE",
    "SYNTH_CORPUS_TABLE_COMPONENT",
    "check_provider_drift",
    "corpus_content_hash",
    "corpus_locales",
    "corpus_provider_manifest",
    "generate_corpus",
    "label_distribution",
    "load_published_provider_receipt",
    "locale_output_digests",
    "main",
    "provider_table_digests",
    "render_corpus",
    "write_corpus",
]
