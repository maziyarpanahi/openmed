"""Read user-supplied i2b2 2012 temporal annotations in memory."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from openmed.eval.datasets.dua_stubs import require_credentialed_path

FOLD_VERSION = 1
TLINK_FOLD = {
    "BEFORE": "BEFORE",
    "AFTER": "AFTER",
    "OVERLAP": "OVERLAP",
    "BEFORE_OVERLAP": "BEFORE",
    "ENDED_BY": "BEFORE",
    "BEGUN_BY": "AFTER",
    "SIMULTANEOUS": "OVERLAP",
    "DURING": "OVERLAP",
    "OVERLAP_AFTER": "AFTER",
    "BEGINS": "BEFORE",
    "ENDS": "AFTER",
    "DURING_BY": "OVERLAP",
}
PREDICTION_FOLD = {
    **TLINK_FOLD,
    **dict.fromkeys(("CONTAINS", "BEGINS_ON", "ENDS_ON"), "OVERLAP"),
}
EVENT_TYPES = {
    "PROBLEM",
    "TREATMENT",
    "TEST",
    "OCCURRENCE",
    "EVIDENTIAL",
    "CLINICAL_DEPT",
}


class TemporalCorpusError(ValueError):
    """Fixed-code refusal for malformed temporal annotations."""


@dataclass(frozen=True, repr=False)
class TemporalDocument:
    """Annotations scoped to a single document in the credentialed process."""

    text: str = field(repr=False)
    events: tuple[dict, ...]
    timexes: tuple[dict, ...]
    sectimes: tuple[dict, ...]
    links: tuple[tuple[str, str, str], ...]


def _parse(path: Path) -> TemporalDocument:
    raw = path.read_text(encoding="utf-8-sig")
    if "<!DOCTYPE" in raw or "<!ENTITY" in raw:
        raise TemporalCorpusError("invalid_temporal_xml")
    root = ET.fromstring(raw)
    text = root.findtext("TEXT")
    tags = root.find("TAGS")
    if not text or tags is None:
        raise TemporalCorpusError("missing_temporal_annotations")
    groups = {"EVENT": [], "TIMEX3": [], "SECTIME": []}
    ids = set()
    aliases = {}
    links = []
    for node in tags:
        attrs = dict(node.attrib)
        code = attrs["type"].upper()
        if node.tag == "TLINK":
            if code not in TLINK_FOLD:
                raise TemporalCorpusError("unknown_tlink_type")
            links.append((TLINK_FOLD[code], attrs["fromID"], attrs["toID"]))
            continue
        if node.tag not in groups:
            raise TemporalCorpusError("unknown_temporal_annotation")
        start, end = int(attrs["start"]), int(attrs["end"])
        if not 0 <= start < end <= len(text) or text[start:end] != attrs["text"]:
            raise TemporalCorpusError("temporal_span_mismatch")
        identifier = attrs["id"]
        if not identifier or identifier in ids:
            raise TemporalCorpusError("invalid_temporal_identifier")
        ids.add(identifier)
        span = {"id": identifier, "start": start, "end": end, "type": code}
        if node.tag == "EVENT":
            if code not in EVENT_TYPES or attrs["polarity"] not in (
                "POS",
                "NEG",
                "POSITIVE",
                "NEGATIVE",
            ):
                raise TemporalCorpusError("unknown_event_annotation")
            span.update(polarity=attrs["polarity"], modality=attrs["modality"])
        elif node.tag == "TIMEX3":
            if code not in ("DATE", "TIME", "DURATION", "FREQUENCY", "SET"):
                raise TemporalCorpusError("unknown_timex_type")
            span.update(
                type="SET" if code == "FREQUENCY" else code,
                val=attrs["val"],
                mod=attrs.get("mod"),
            )
        else:
            if code not in ("ADMISSION", "DISCHARGE") or code in aliases:
                raise TemporalCorpusError("invalid_section_time")
            aliases[code] = identifier
            span["val"] = attrs.get("dvalue") or attrs.get("val")
        groups[node.tag].append(span)

    def resolve(identifier):
        if identifier in ids:
            return identifier
        if identifier.upper() in aliases:
            return aliases[identifier.upper()]
        raise TemporalCorpusError("unknown_tlink_endpoint")

    return TemporalDocument(
        text,
        tuple(groups["EVENT"]),
        tuple(groups["TIMEX3"]),
        tuple(groups["SECTIME"]),
        tuple((r, resolve(a), resolve(b)) for r, a, b in links),
    )


def load_i2b2_2012(path: str | Path | None = None) -> tuple[TemporalDocument, ...]:
    """Read temporal XML beneath the caller's authorized corpus directory."""
    root = require_credentialed_path(
        path, dataset="i2b2-2012", authority="i2b2", env_var="OPENMED_I2B2_2012_PATH"
    )
    try:
        paths = sorted(root.rglob("*.xml")) if root.is_dir() else []
        if not paths or any(
            p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root)
            for p in paths
        ):
            raise TemporalCorpusError("invalid_temporal_corpus")
        return tuple(_parse(p) for p in paths)
    except TemporalCorpusError:
        raise
    except (OSError, UnicodeError, ValueError, KeyError, ET.ParseError):
        raise TemporalCorpusError("invalid_temporal_xml") from None
