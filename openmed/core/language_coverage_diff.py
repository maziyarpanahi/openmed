"""Deterministic metadata-only comparisons of language coverage manifests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_SCHEMA = "1.0.0"
_LOCALE = re.compile(r"[a-z]{2,3}(?:-[A-Z][a-z]{3})?(?:-[A-Z]{2}|-[0-9]{3})?")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SURROGATES = {"missing": 0, "approximated": 1, "supported": 2}


def _locale(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid coverage locale")
    parts = value.replace("_", "-").split("-")
    parts[0] = parts[0].lower()
    for i in range(1, len(parts)):
        parts[i] = parts[i].title() if len(parts[i]) == 4 else parts[i].upper()
    canonical = "-".join(parts)
    if not _LOCALE.fullmatch(canonical):
        raise ValueError("invalid coverage locale")
    return canonical


def _names(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > 256:
        raise ValueError("invalid coverage capability list")
    if any(not isinstance(v, str) or not _TOKEN.fullmatch(v) for v in value):
        raise ValueError("invalid coverage capability token")
    if len(set(value)) != len(value):
        raise ValueError("duplicate coverage capability token")
    return tuple(sorted(value))


def _manifest(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or value.get("schema_version") != _SCHEMA:
        raise ValueError("unsupported coverage manifest version")
    packs = value.get("packs")
    if not isinstance(packs, (list, tuple)) or len(packs) > 1000:
        raise ValueError("invalid coverage manifest packs")
    result = {}
    for pack in packs:
        if not isinstance(pack, Mapping):
            raise ValueError("invalid coverage pack")
        code = _locale(pack.get("locale"))
        if code in result:
            raise ValueError("duplicate canonical coverage locale")
        surrogate = pack.get("surrogate_support")
        if not isinstance(surrogate, str) or surrogate not in _SURROGATES:
            raise ValueError("invalid coverage surrogate state")
        digest = pack.get("evidence_digest")
        if digest is not None and (
            not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
        ):
            raise ValueError("invalid coverage evidence digest")
        result[code] = {
            "scripts": _names(pack.get("scripts")),
            "identifier_classes": _names(pack.get("identifier_classes")),
            "surrogate_support": surrogate,
            "evidence_digest": digest,
        }
    return result


def diff_language_coverage(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare versioned coverage declarations without reading clinical text.

    Losses and gains are independent: a pack can improve and regress in the
    same comparison. Evidence changes never imply capability improvement.
    Unknown input fields are excluded from the output.
    """
    old, new = _manifest(before), _manifest(after)
    output: dict[str, Any] = {
        "schema_version": _SCHEMA,
        "added": sorted(new.keys() - old.keys()),
        "removed": sorted(old.keys() - new.keys()),
        "improved": [],
        "regressed": [],
        "changed": [],
    }
    for locale in sorted(old.keys() & new.keys()):
        a, b = old[locale], new[locale]
        gains, losses = [], []
        for field in ("scripts", "identifier_classes"):
            if set(b[field]) - set(a[field]):
                gains.append(field)
            if set(a[field]) - set(b[field]):
                losses.append(field)
        delta = (
            _SURROGATES[b["surrogate_support"]] - _SURROGATES[a["surrogate_support"]]
        )
        if delta > 0:
            gains.append("surrogate_support")
        elif delta < 0:
            losses.append("surrogate_support")
        evidence_changed = a["evidence_digest"] != b["evidence_digest"]
        if a["evidence_digest"] is None and b["evidence_digest"] is not None:
            gains.append("evidence")
        elif a["evidence_digest"] is not None and b["evidence_digest"] is None:
            losses.append("evidence")
        if gains:
            output["improved"].append({"locale": locale, "fields": gains})
        if losses:
            output["regressed"].append({"locale": locale, "fields": losses})
        if a != b:
            output["changed"].append(
                {"locale": locale, "evidence_changed": evidence_changed}
            )
    return output


def render_language_coverage_diff(diff: Mapping[str, Any]) -> str:
    """Render a validated metadata-only diff as byte-stable JSON."""
    expected = {
        "schema_version",
        "added",
        "removed",
        "improved",
        "regressed",
        "changed",
    }
    if (
        not isinstance(diff, Mapping)
        or set(diff) != expected
        or diff.get("schema_version") != _SCHEMA
    ):
        raise ValueError("invalid coverage diff")
    # Rebuild all fields rather than serializing arbitrary caller metadata.
    clean = {"schema_version": _SCHEMA}
    for key in ("added", "removed"):
        values = diff[key]
        if not isinstance(values, (list, tuple)) or len(values) > 1000:
            raise ValueError("invalid coverage diff entries")
        locales = [_locale(v) for v in values]
        if len(set(locales)) != len(locales):
            raise ValueError("duplicate canonical coverage diff locale")
        clean[key] = sorted(locales)
    for key in ("improved", "regressed", "changed"):
        entries = diff[key]
        if not isinstance(entries, (list, tuple)) or len(entries) > 1000:
            raise ValueError("invalid coverage diff entries")
        rows = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("invalid coverage diff entry")
            row = {"locale": _locale(entry.get("locale"))}
            if key == "changed":
                if type(entry.get("evidence_changed")) is not bool:
                    raise ValueError("invalid coverage evidence change")
                row["evidence_changed"] = entry["evidence_changed"]
            else:
                fields = entry.get("fields")
                allowed = {
                    "scripts",
                    "identifier_classes",
                    "surrogate_support",
                    "evidence",
                }
                if not isinstance(fields, (list, tuple)) or any(
                    not isinstance(f, str) or f not in allowed for f in fields
                ):
                    raise ValueError("invalid coverage change category")
                row["fields"] = sorted(set(fields))
            rows.append(row)
        if len({row["locale"] for row in rows}) != len(rows):
            raise ValueError("duplicate canonical coverage diff locale")
        clean[key] = sorted(rows, key=lambda row: row["locale"])
    return json.dumps(clean, sort_keys=True, separators=(",", ":")) + "\n"
