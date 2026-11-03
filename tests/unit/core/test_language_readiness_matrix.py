"""Tests for the clinical language-pack readiness matrix (#3100).

The suite drives the matrix through synthetic readiness entries and a private
:class:`LanguagePackRegistry`, never the process-global one, so registrations
stay isolated. Golden payloads pin both the semantic shape and the rendered
bytes, and every locale, pack value, and digest is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import pytest

from openmed.core import LanguagePack, LanguagePackRegistry
from openmed.core.language_readiness_matrix import (
    BLOCKED,
    COMPLETE,
    MATRIX_SCHEMA_VERSION,
    MISSING_EVIDENCE,
    NOT_REPORTED,
    PARTIAL,
    READINESS_CAPABILITIES,
    READINESS_STATES,
    LanguageReadinessEntry,
    LanguageReadinessError,
    build_language_readiness_matrix,
    normalize_readiness_entries,
    readiness_entries_from_packs,
    render_language_readiness_json,
    render_language_readiness_markdown,
)

_VERSION = "2026.09"
_DIGEST_A = "sha256:" + "a" * 64
_DIGEST_B = "sha256:" + "b" * 64
_DIGEST_C = "sha256:" + "c" * 64

_FIXTURE_ENTRIES = [
    LanguageReadinessEntry("en", "detection", COMPLETE, _DIGEST_A),
    LanguageReadinessEntry("en", "surrogates", COMPLETE, _DIGEST_B),
    LanguageReadinessEntry("en", "validation", PARTIAL),
    LanguageReadinessEntry("fr", "detection", COMPLETE, _DIGEST_C),
    LanguageReadinessEntry("fr", "surrogates", PARTIAL),
    LanguageReadinessEntry("fr", "validation", BLOCKED),
    LanguageReadinessEntry("fr", "fixtures", MISSING_EVIDENCE),
]

_EXPECTED_MATRIX = {
    "capabilities": [
        "detection",
        "surrogates",
        "validation",
        "fixtures",
        "release_evidence",
    ],
    "cells": [
        {
            "capability": "detection",
            "evidence_digest": _DIGEST_A,
            "locale": "en",
            "state": COMPLETE,
        },
        {
            "capability": "surrogates",
            "evidence_digest": _DIGEST_B,
            "locale": "en",
            "state": COMPLETE,
        },
        {
            "capability": "validation",
            "evidence_digest": None,
            "locale": "en",
            "state": PARTIAL,
        },
        {
            "capability": "detection",
            "evidence_digest": _DIGEST_C,
            "locale": "fr",
            "state": COMPLETE,
        },
        {
            "capability": "surrogates",
            "evidence_digest": None,
            "locale": "fr",
            "state": PARTIAL,
        },
        {
            "capability": "validation",
            "evidence_digest": None,
            "locale": "fr",
            "state": BLOCKED,
        },
        {
            "capability": "fixtures",
            "evidence_digest": None,
            "locale": "fr",
            "state": MISSING_EVIDENCE,
        },
    ],
    "locales": ["en", "fr"],
    "schema_version": MATRIX_SCHEMA_VERSION,
    "states": ["blocked", "complete", "missing_evidence", "partial"],
    "totals": {"blocked": 1, "complete": 3, "missing_evidence": 1, "partial": 2},
    "version": _VERSION,
}

# Rendering digests pinned as literals; they change only when the matrix payload
# or the Markdown layout changes.
_GOLDEN_JSON_SHA256 = "80c9441d5133c64e0bb253ca3f8a25b9e410e42edafa041c74f029af0909f8a3"
_GOLDEN_MARKDOWN_SHA256 = (
    "984b1458b909066a2b4776de16c15ddaae14f9802a3e56634534e0e6d49bfac9"
)

_GOLDEN_MARKDOWN = f"""\
# Language readiness matrix

Schema: `{MATRIX_SCHEMA_VERSION}`

Version: `{_VERSION}`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `en` | `complete` | `complete` | `partial` | `not_reported` | `not_reported` |
| `fr` | `complete` | `partial` | `blocked` | `missing_evidence` | `not_reported` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 3 |
| `partial` | 2 |
| `blocked` | 1 |
| `missing_evidence` | 1 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `en` | detection | `{_DIGEST_A}` |
| `en` | surrogates | `{_DIGEST_B}` |
| `fr` | detection | `{_DIGEST_C}` |
"""


def _pack(code: str = "en", **overrides: object) -> LanguagePack:
    """Return a coherent synthetic pack unless overridden."""

    values: dict[str, object] = {
        "code": code,
        "scripts": ["Latin"],
        "default_model": "OpenMed/synthetic-pii",
        "segmenter_id": "unicode-sentence",
        "recognizers": ["regex", "model"],
        "surrogate_locale": "en_US",
        "national_id_providers": {"ssn": "en_US"},
        "policy_overrides": {"profile": "strict_no_leak"},
        "recall_floor_overrides": {"PERSON": 0.99},
    }
    values.update(overrides)
    return LanguagePack(**values)  # type: ignore[arg-type]


def _registry(*packs: LanguagePack) -> LanguagePackRegistry:
    registry = LanguagePackRegistry()
    for pack in packs:
        registry.register(pack)
    return registry


def _degraded_registry() -> LanguagePackRegistry:
    """Return one fully coherent pack and one pack with a broken segmenter."""

    return _registry(
        _pack("en"),
        _pack(
            "fr",
            segmenter_id="not-a-real-segmenter",
            surrogate_locale="fr_FR",
            national_id_providers={},
            policy_overrides={},
            recall_floor_overrides={},
        ),
    )


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


def test_closed_vocabularies_are_pinned():
    assert READINESS_CAPABILITIES == (
        "detection",
        "surrogates",
        "validation",
        "fixtures",
        "release_evidence",
    )
    assert READINESS_STATES == ("complete", "partial", "blocked", "missing_evidence")
    assert MATRIX_SCHEMA_VERSION == "openmed.language_readiness_matrix.v1"
    # The Markdown placeholder is a rendering token, not a readiness state.
    assert NOT_REPORTED not in READINESS_STATES


# ---------------------------------------------------------------------------
# Golden payloads
# ---------------------------------------------------------------------------


def test_matrix_projects_the_golden_payload():
    matrix = build_language_readiness_matrix(_FIXTURE_ENTRIES, version=_VERSION)

    assert matrix == _EXPECTED_MATRIX


def test_json_render_matches_the_pinned_bytes():
    rendered = render_language_readiness_json(_FIXTURE_ENTRIES, version=_VERSION)

    assert json.loads(rendered) == _EXPECTED_MATRIX
    assert "\n" not in rendered
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == _GOLDEN_JSON_SHA256


def test_markdown_render_matches_the_golden_text():
    rendered = render_language_readiness_markdown(_FIXTURE_ENTRIES, version=_VERSION)

    assert rendered == _GOLDEN_MARKDOWN
    assert hashlib.sha256(rendered.encode("utf-8")).hexdigest() == (
        _GOLDEN_MARKDOWN_SHA256
    )


def test_reordered_entries_render_identical_bytes():
    reordered = list(reversed(_FIXTURE_ENTRIES))

    assert render_language_readiness_json(reordered, version=_VERSION) == (
        render_language_readiness_json(_FIXTURE_ENTRIES, version=_VERSION)
    )
    assert render_language_readiness_markdown(reordered, version=_VERSION) == (
        _GOLDEN_MARKDOWN
    )


def test_json_payload_carries_only_the_documented_keys():
    payload = json.loads(
        render_language_readiness_json(_FIXTURE_ENTRIES, version=_VERSION)
    )

    assert set(payload) == {
        "capabilities",
        "cells",
        "locales",
        "schema_version",
        "states",
        "totals",
        "version",
    }
    for cell in payload["cells"]:
        assert set(cell) == {"capability", "evidence_digest", "locale", "state"}


def test_markdown_uses_only_closed_vocabulary_tokens():
    tokens = set(re.findall(r"`([^`]+)`", _GOLDEN_MARKDOWN))

    for token in tokens:
        if token.startswith("sha256:"):
            continue
        assert token in READINESS_STATES or token in {
            "en",
            "fr",
            MATRIX_SCHEMA_VERSION,
            _VERSION,
            NOT_REPORTED,
        }


# ---------------------------------------------------------------------------
# Locale aliases
# ---------------------------------------------------------------------------


def test_alias_locale_folds_onto_the_canonical_golden():
    canonical = [
        LanguageReadinessEntry("he", "detection", COMPLETE, _DIGEST_A),
        LanguageReadinessEntry("he", "validation", BLOCKED),
    ]
    aliased = [
        LanguageReadinessEntry.from_tag(
            "iw", "detection", COMPLETE, _DIGEST_A, aliases={"iw": "he"}
        ),
        LanguageReadinessEntry.from_tag(
            "iw", "validation", BLOCKED, aliases={"iw": "he"}
        ),
    ]

    assert [entry.locale for entry in aliased] == ["he", "he"]
    assert normalize_readiness_entries(aliased) == tuple(canonical)
    assert render_language_readiness_json(aliased, version=_VERSION) == (
        render_language_readiness_json(canonical, version=_VERSION)
    )
    assert render_language_readiness_markdown(aliased, version=_VERSION) == (
        render_language_readiness_markdown(canonical, version=_VERSION)
    )


def test_region_alias_folds_onto_the_primary_locale():
    entry = LanguageReadinessEntry.from_tag(
        "en-US", "detection", COMPLETE, _DIGEST_A, aliases={"en-US": "en"}
    )

    assert entry.locale == "en"


def test_underscore_alias_folds_onto_the_canonical_tag():
    entry = LanguageReadinessEntry.from_tag(
        "en_US", "detection", COMPLETE, _DIGEST_A, aliases={"en_US": "en-US"}
    )

    assert entry.locale == "en-US"


def test_underscore_tag_without_an_alias_is_rejected():
    with pytest.raises(LanguageReadinessError, match="locale_tag_invalid"):
        LanguageReadinessEntry.from_tag("en_US", "detection", COMPLETE, _DIGEST_A)


def test_identity_alias_keeps_the_requested_tag():
    entry = LanguageReadinessEntry.from_tag(
        "iw", "detection", COMPLETE, _DIGEST_A, aliases={"iw": "iw"}
    )

    assert entry.locale == "iw"


def test_alias_collision_is_rejected():
    entries = [
        LanguageReadinessEntry.from_tag(
            "iw", "detection", COMPLETE, _DIGEST_A, aliases={"iw": "he"}
        ),
        LanguageReadinessEntry("he", "detection", COMPLETE, _DIGEST_A),
    ]

    with pytest.raises(LanguageReadinessError, match="duplicate readiness entry"):
        normalize_readiness_entries(entries)


@pytest.mark.parametrize(
    ("aliases", "message"),
    [
        (["iw", "he"], "aliases must be a mapping of locale tags"),
        ({1: "he"}, "alias keys and values must be strings"),
        ({"iw": 2}, "alias keys and values must be strings"),
        ({"iw": "en_US"}, "locale_alias_invalid"),
        ({"iw": "he", "he": "iw"}, "locale_alias_chain_unsupported"),
        ({"iw": "he", "IW": "de"}, "locale_alias_duplicate"),
    ],
)
def test_malformed_aliases_are_rejected(aliases, message):
    with pytest.raises(LanguageReadinessError, match=message):
        LanguageReadinessEntry.from_tag(
            "iw", "detection", COMPLETE, _DIGEST_A, aliases=aliases
        )


# ---------------------------------------------------------------------------
# Entry validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"capability": "nope"}, "unknown readiness capability"),
        ({"state": "ready"}, "unknown readiness state"),
    ],
)
def test_unknown_vocabulary_members_are_rejected(overrides, message):
    values: dict[str, object] = {
        "locale": "en",
        "capability": "detection",
        "state": COMPLETE,
        "evidence_digest": _DIGEST_A,
    }
    values.update(overrides)

    with pytest.raises(LanguageReadinessError, match=message):
        LanguageReadinessEntry(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "digest",
    [
        "sha256:",
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
        "a" * 64,
        123,
    ],
)
def test_evidence_digest_shape_is_enforced(digest):
    with pytest.raises(LanguageReadinessError, match="evidence_digest must be"):
        LanguageReadinessEntry("en", "detection", COMPLETE, digest)  # type: ignore[arg-type]


def test_complete_readiness_requires_an_evidence_digest():
    with pytest.raises(
        LanguageReadinessError, match="complete readiness requires an evidence digest"
    ):
        LanguageReadinessEntry("en", "detection", COMPLETE)


def test_locale_must_be_a_non_empty_string():
    for locale in ("  ", None, 7):
        with pytest.raises(
            LanguageReadinessError, match="locale must be a non-empty string"
        ):
            LanguageReadinessEntry(locale, "detection", COMPLETE, _DIGEST_A)  # type: ignore[arg-type]


def test_underscore_locales_are_rejected_by_the_shared_normalizer():
    with pytest.raises(LanguageReadinessError, match="locale_tag_invalid"):
        LanguageReadinessEntry("en_US", "detection", COMPLETE, _DIGEST_A)


def test_blank_version_is_rejected():
    with pytest.raises(
        LanguageReadinessError, match="version must be a non-empty string"
    ):
        build_language_readiness_matrix(_FIXTURE_ENTRIES, version=" ")


def test_entries_must_be_language_readiness_entries():
    with pytest.raises(LanguageReadinessError, match="must be an iterable"):
        normalize_readiness_entries("en")  # type: ignore[arg-type]

    with pytest.raises(
        LanguageReadinessError, match="must contain LanguageReadinessEntry"
    ):
        normalize_readiness_entries(["en"])  # type: ignore[list-item]


def test_duplicate_cells_are_rejected():
    with pytest.raises(LanguageReadinessError, match="duplicate readiness entry"):
        normalize_readiness_entries([_FIXTURE_ENTRIES[0], _FIXTURE_ENTRIES[0]])


# ---------------------------------------------------------------------------
# Bounded rendering
# ---------------------------------------------------------------------------


def test_empty_matrix_is_explicit_and_bounded():
    matrix = build_language_readiness_matrix([], version=_VERSION)

    assert matrix["locales"] == []
    assert matrix["cells"] == []
    assert matrix["totals"] == dict.fromkeys(READINESS_STATES, 0)

    rendered = render_language_readiness_markdown([], version=_VERSION)

    assert "_No readiness entries are registered._" in rendered
    assert NOT_REPORTED not in rendered
    assert json.loads(render_language_readiness_json([], version=_VERSION)) == matrix


def test_unreported_cells_render_a_placeholder_without_entering_json():
    entries = [
        LanguageReadinessEntry("en", "detection", COMPLETE, _DIGEST_A),
        LanguageReadinessEntry("fr", "surrogates", PARTIAL),
    ]

    matrix = build_language_readiness_matrix(entries, version=_VERSION)
    rendered = render_language_readiness_markdown(entries, version=_VERSION)
    fr_row = next(line for line in rendered.splitlines() if line.startswith("| `fr` |"))

    assert len(matrix["cells"]) == 2
    assert fr_row.count(f"`{NOT_REPORTED}`") == len(READINESS_CAPABILITIES) - 1
    assert f"`{PARTIAL}`" in fr_row
    assert NOT_REPORTED not in render_language_readiness_json(entries, version=_VERSION)


# ---------------------------------------------------------------------------
# Derivation from the language-pack registry
# ---------------------------------------------------------------------------


def test_derived_entries_cover_every_capability_from_an_isolated_registry():
    derived = readiness_entries_from_packs(registry=_degraded_registry())
    cells = {(entry.locale, entry.capability): entry for entry in derived}

    assert len(derived) == 2 * len(READINESS_CAPABILITIES)
    assert set(cells) == {
        (locale, capability)
        for locale in ("en", "fr")
        for capability in READINESS_CAPABILITIES
    }

    assert cells[("en", "detection")].state == COMPLETE
    assert cells[("en", "detection")].evidence_digest is not None
    assert cells[("en", "surrogates")].state == COMPLETE
    assert cells[("en", "validation")].state == COMPLETE

    # A broken segmenter degrades detection to partial and cites no evidence.
    assert cells[("fr", "detection")].state == PARTIAL
    assert cells[("fr", "detection")].evidence_digest is None
    assert cells[("fr", "surrogates")].state == COMPLETE
    # The fr pack declares no national-ID provider, so validation has no evidence.
    assert cells[("fr", "validation")].state == MISSING_EVIDENCE

    for locale in ("en", "fr"):
        assert cells[(locale, "fixtures")].state == MISSING_EVIDENCE
        assert cells[(locale, "release_evidence")].state == MISSING_EVIDENCE


def test_derived_entries_are_deterministic_and_json_serializable():
    registry = _degraded_registry()

    first = readiness_entries_from_packs(registry=registry)
    second = readiness_entries_from_packs(registry=registry)

    assert first == second

    payload = json.loads(render_language_readiness_json(first, version="packs"))

    assert payload["locales"] == ["en", "fr"]
    assert all(cell["state"] in READINESS_STATES for cell in payload["cells"])
    assert all(
        cell["evidence_digest"] is not None
        for cell in payload["cells"]
        if cell["state"] == COMPLETE
    )


def test_builtin_catalog_derivation_is_bounded_and_normalized():
    derived = readiness_entries_from_packs()
    locales = sorted({entry.locale for entry in derived})

    assert len(derived) == len(locales) * len(READINESS_CAPABILITIES)
    assert normalize_readiness_entries(derived) == derived
    assert all(entry.state in READINESS_STATES for entry in derived)
    assert all(
        entry.evidence_digest is not None
        for entry in derived
        if entry.state == COMPLETE
    )


# ---------------------------------------------------------------------------
# Golden Markdown fixtures (#3122)
# ---------------------------------------------------------------------------

_GOLDEN_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "language_readiness_expected.md"
)
_SCENARIO_MARKER = re.compile(r"^<!-- scenario: ([a-z_]+) -->\n", re.MULTILINE)
_HEBREW_ALIAS = {"iw": "he"}
_REGION_ALIAS = {"en_GB": "en-GB"}


def _digest(character: str) -> str:
    return "sha256:" + character * 64


# Compact synthetic registries, one per readiness shape. The fixture file holds
# their renders in this order, each under a ``<!-- scenario: name -->`` marker.
_GOLDEN_SCENARIOS: dict[str, tuple[str, list[LanguageReadinessEntry]]] = {
    "empty": ("2026.10", []),
    "complete": (
        "2026.10",
        [
            LanguageReadinessEntry("en-GB", capability, COMPLETE, _digest(character))
            for capability, character in zip(READINESS_CAPABILITIES, "01234")
        ],
    ),
    "partial": (
        "2026.10",
        [
            LanguageReadinessEntry("de", "detection", PARTIAL),
            LanguageReadinessEntry("de", "surrogates", PARTIAL),
            LanguageReadinessEntry("de", "validation", COMPLETE, _digest("5")),
        ],
    ),
    "blocked": (
        "2026.10",
        [
            LanguageReadinessEntry("fa", "detection", COMPLETE, _digest("6")),
            LanguageReadinessEntry("fa", "validation", BLOCKED),
            LanguageReadinessEntry("am", "validation", BLOCKED),
            LanguageReadinessEntry("am", "release_evidence", BLOCKED),
        ],
    ),
    "aliased": (
        "2026.10",
        [
            LanguageReadinessEntry.from_tag(
                "iw", "detection", COMPLETE, _digest("7"), aliases=_HEBREW_ALIAS
            ),
            LanguageReadinessEntry.from_tag(
                "iw", "surrogates", PARTIAL, aliases=_HEBREW_ALIAS
            ),
            LanguageReadinessEntry.from_tag(
                "en_GB", "validation", BLOCKED, aliases=_REGION_ALIAS
            ),
        ],
    ),
    "missing_evidence": (
        "2026.10",
        [
            LanguageReadinessEntry("sw", "fixtures", MISSING_EVIDENCE),
            LanguageReadinessEntry("sw", "release_evidence", MISSING_EVIDENCE),
        ],
    ),
    "ordering_and_escaping": (
        "``2026.10`-rc|1",
        [
            LanguageReadinessEntry("zh-Hant-TW", "detection", COMPLETE, _digest("8")),
            LanguageReadinessEntry("sr-Latn", "surrogates", PARTIAL),
            LanguageReadinessEntry("es-419", "validation", BLOCKED),
            LanguageReadinessEntry("en-GB", "fixtures", MISSING_EVIDENCE),
            LanguageReadinessEntry("en", "release_evidence", COMPLETE, _digest("9")),
            LanguageReadinessEntry("am", "detection", COMPLETE, _digest("a")),
        ],
    ),
}
_PLAIN_CELLS = frozenset(
    {"Locale", "Capability", "Digest", "State", "Count", "---", *READINESS_CAPABILITIES}
)


def _golden_sections() -> dict[str, str]:
    parts = _SCENARIO_MARKER.split(_GOLDEN_FIXTURE.read_text(encoding="utf-8"))
    assert parts[0] == ""
    return dict(zip(parts[1::2], parts[2::2]))


def _table_cells(markdown: str) -> list[str]:
    cells = []
    for line in markdown.splitlines():
        if line.startswith("| "):
            cells.extend(cell.strip() for cell in line.strip("|").split("|"))
    return cells


def _assert_metadata_only_cells(
    markdown: str, entries: list[LanguageReadinessEntry]
) -> None:
    locales = {entry.locale for entry in entries}
    digests = {entry.evidence_digest for entry in entries} - {None}
    allowed = {*READINESS_STATES, NOT_REPORTED, *locales, *digests}
    for cell in _table_cells(markdown):
        if cell in _PLAIN_CELLS or cell.isdigit():
            continue
        assert cell.startswith("`") and cell.endswith("`"), cell
        assert cell.strip("`") in allowed, cell


def test_golden_fixture_holds_every_scenario_once_in_order():
    assert list(_golden_sections()) == list(_GOLDEN_SCENARIOS)


@pytest.mark.parametrize("scenario", list(_GOLDEN_SCENARIOS))
def test_scenario_renders_the_golden_markdown(scenario):
    version, entries = _GOLDEN_SCENARIOS[scenario]

    assert (
        render_language_readiness_markdown(entries, version=version)
        == (_golden_sections()[scenario])
    )


@pytest.mark.parametrize("scenario", list(_GOLDEN_SCENARIOS))
def test_equivalent_orderings_render_byte_identical_markdown(scenario):
    version, entries = _GOLDEN_SCENARIOS[scenario]
    expected = _golden_sections()[scenario].encode("utf-8")
    orderings = [list(reversed(entries))]
    for seed in range(5):
        shuffled = list(entries)
        random.Random(seed).shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        rendered = render_language_readiness_markdown(ordering, version=version)
        assert rendered.encode("utf-8") == expected


def test_aliased_registry_matches_its_canonical_golden():
    canonical = [
        LanguageReadinessEntry("he", "detection", COMPLETE, _digest("7")),
        LanguageReadinessEntry("he", "surrogates", PARTIAL),
        LanguageReadinessEntry("en-GB", "validation", BLOCKED),
    ]

    assert (
        render_language_readiness_markdown(canonical, version="2026.10")
        == (_golden_sections()["aliased"])
    )


@pytest.mark.parametrize("scenario", list(_GOLDEN_SCENARIOS))
def test_golden_cells_carry_only_locale_and_evidence_metadata(scenario):
    _, entries = _GOLDEN_SCENARIOS[scenario]

    _assert_metadata_only_cells(_golden_sections()[scenario], entries)


def test_registry_derived_markdown_carries_no_pack_values():
    derived = list(readiness_entries_from_packs(registry=_degraded_registry()))
    rendered = render_language_readiness_markdown(derived, version="packs")

    _assert_metadata_only_cells(rendered, derived)
    assert "/" not in rendered
    for pack_value in (
        "OpenMed/synthetic-pii",
        "synthetic-pii",
        "unicode-sentence",
        "not-a-real-segmenter",
        "en_US",
        "fr_FR",
        "Latin",
        "strict_no_leak",
        "PERSON",
    ):
        assert pack_value not in rendered


@pytest.mark.parametrize(
    ("version", "line"),
    [
        ("2026.10", "Version: `2026.10`"),
        ("a|b", "Version: `a|b`"),
        ("v`1|x", "Version: ``v`1|x``"),
        ("``rc``", "Version: ``` ``rc`` ```"),
        ("`rc", "Version: `` `rc ``"),
        (" padded", "Version: `  padded `"),
    ],
)
def test_version_renders_as_one_code_span(version, line):
    rendered = render_language_readiness_markdown(
        _GOLDEN_SCENARIOS["partial"][1], version=version
    )

    assert rendered.splitlines()[4] == line


@pytest.mark.parametrize("version", ["2026.10\n# injected", "2026\r10", "a\tb"])
def test_versions_that_could_break_markdown_are_rejected(version):
    message = "version must be a single printable line"

    with pytest.raises(LanguageReadinessError, match=message):
        render_language_readiness_markdown([], version=version)
    with pytest.raises(LanguageReadinessError, match=message):
        render_language_readiness_json([], version=version)
