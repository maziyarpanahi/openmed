"""CoNLL-style BIO column readers and writers for ``OpenMedSpan``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openmed.core.schemas import OpenMedSpan
from openmed.eval.annotation.toolkit import (
    AnnotationIssue,
    AnnotationTask,
    AnnotationValidationError,
    span_from_offsets,
    validate_spans,
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
_TAG_RE = re.compile(r"^(?P<prefix>[BIESUL])-(?P<label>\S+)$")


@dataclass(frozen=True)
class _TokenRow:
    token: str
    tag: str
    start: int
    end: int
    line: int
    sentence_break: bool


@dataclass
class _ActiveSpan:
    label: str
    start: int
    end: int
    line: int


def parse_conll(
    text: str,
    columns: str,
    *,
    doc_id: str,
    hash_secret: str | bytes,
) -> tuple[OpenMedSpan, ...]:
    """Parse CoNLL token columns and align them to exact source text.

    Two-column ``TOKEN TAG`` and traditional multi-column rows are accepted;
    the first column is the token and the last is the NER tag. BIO is the
    canonical output, while BIOES/BILOU tags are accepted on import. Blank
    lines mark sentence boundaries.

    Args:
        text: Exact source text used to recover character offsets.
        columns: CoNLL rows to parse.
        doc_id: Document identifier stored on imported spans.
        hash_secret: Key used to HMAC imported span surfaces.

    Returns:
        Imported spans in deterministic offset order.

    Raises:
        AnnotationValidationError: If rows, alignment, or tag transitions fail.
    """

    if not doc_id:
        raise AnnotationValidationError(
            AnnotationIssue("doc_id must be non-empty"), format_name="CoNLL"
        )
    rows, issues = _align_rows(text, columns)
    if issues:
        raise AnnotationValidationError(issues, format_name="CoNLL")

    spans: list[OpenMedSpan] = []
    active: _ActiveSpan | None = None

    def close_active() -> None:
        nonlocal active
        if active is None:
            return
        try:
            spans.append(
                span_from_offsets(
                    doc_id=doc_id,
                    text=text,
                    start=active.start,
                    end=active.end,
                    label=active.label,
                    hash_secret=hash_secret,
                    entity_type=active.label,
                    metadata={"annotation_format": "conll"},
                )
            )
        except AnnotationValidationError as exc:
            issues.extend(
                AnnotationIssue(issue.message, line=active.line) for issue in exc.issues
            )
        active = None

    for row in rows:
        if row.sentence_break:
            close_active()
        if row.tag == "O":
            close_active()
            continue
        match = _TAG_RE.fullmatch(row.tag)
        if match is None:
            issues.append(
                AnnotationIssue(
                    "tag must be O or PREFIX-LABEL using BIO/BIOES/BILOU",
                    line=row.line,
                )
            )
            close_active()
            continue
        prefix = match.group("prefix")
        label = match.group("label")
        if prefix == "B":
            close_active()
            active = _ActiveSpan(label, row.start, row.end, row.line)
        elif prefix in {"S", "U"}:
            close_active()
            active = _ActiveSpan(label, row.start, row.end, row.line)
            close_active()
        elif prefix == "I":
            if active is None:
                issues.append(
                    AnnotationIssue(
                        f"I-{label} cannot start an entity; use B-{label}",
                        line=row.line,
                    )
                )
            elif active.label != label:
                issues.append(
                    AnnotationIssue(
                        f"I-{label} does not match active B-{active.label}",
                        line=row.line,
                    )
                )
                close_active()
            else:
                active.end = row.end
        else:  # E or L
            if active is None:
                issues.append(
                    AnnotationIssue(
                        f"{prefix}-{label} has no matching entity start",
                        line=row.line,
                    )
                )
            elif active.label != label:
                issues.append(
                    AnnotationIssue(
                        f"{prefix}-{label} does not match active B-{active.label}",
                        line=row.line,
                    )
                )
                close_active()
            else:
                active.end = row.end
                close_active()
    close_active()

    if issues:
        raise AnnotationValidationError(issues, format_name="CoNLL")
    try:
        return validate_spans(
            text,
            spans,
            doc_id=doc_id,
            hash_secret=hash_secret,
            allow_overlap=False,
        )
    except AnnotationValidationError as exc:
        raise AnnotationValidationError(exc.issues, format_name="CoNLL") from exc


def format_conll(text: str, spans: Iterable[OpenMedSpan]) -> str:
    """Serialize non-overlapping spans as two-column CoNLL BIO rows.

    Args:
        text: Exact source text whose offsets the spans reference.
        spans: Canonical, non-overlapping spans from one document.

    Returns:
        UTF-8-ready ``TOKEN<TAB>TAG`` rows using BIO labels.

    Raises:
        AnnotationValidationError: If spans overlap, cross documents, or use
            whitespace-only boundaries that CoNLL cannot represent.
    """

    materialized = tuple(spans)
    doc_id = materialized[0].doc_id if materialized else None
    ordered = validate_spans(
        text,
        materialized,
        doc_id=doc_id,
        allow_overlap=False,
    )
    tokens = _tokens_with_span_boundaries(text, ordered)
    issues: list[AnnotationIssue] = []
    for span in ordered:
        covered = [
            token for token in tokens if token[0] >= span.start and token[1] <= span.end
        ]
        if not covered or covered[0][0] != span.start or covered[-1][1] != span.end:
            issues.append(
                AnnotationIssue(
                    f"span {span.start}:{span.end} must begin and end on a "
                    "non-whitespace token boundary"
                )
            )
    if issues:
        raise AnnotationValidationError(issues, format_name="CoNLL")

    lines: list[str] = []
    for start, end, token in tokens:
        owner = next(
            (span for span in ordered if span.start <= start and end <= span.end),
            None,
        )
        if owner is None:
            tag = "O"
        else:
            prefix = "B" if start == owner.start else "I"
            tag = f"{prefix}-{owner.canonical_label}"
        lines.append(f"{token}\t{tag}")
    return "\n".join(lines) + ("\n" if lines else "")


def read_conll(
    path: str | Path,
    *,
    text: str,
    doc_id: str,
    hash_secret: str | bytes,
    synthetic: bool = False,
) -> AnnotationTask:
    """Read a CoNLL column file aligned against its exact source text.

    Args:
        path: Path to the CoNLL column file.
        text: Exact paired source text used to recover offsets.
        doc_id: Document identifier stored on imported spans.
        hash_secret: Key used to HMAC imported span surfaces.
        synthetic: Whether the caller attests that the source is synthetic.

    Returns:
        The validated source document and imported spans.
    """

    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8", newline="") as handle:
        columns = handle.read()
    spans = parse_conll(
        text,
        columns,
        doc_id=doc_id,
        hash_secret=hash_secret,
    )
    return AnnotationTask(
        doc_id=doc_id,
        text=text,
        spans=spans,
        synthetic=synthetic,
        metadata={"annotation_format": "conll", "synthetic": synthetic},
    )


def write_conll(
    path: str | Path,
    text: str,
    spans: Iterable[OpenMedSpan],
) -> Path:
    """Write a two-column CoNLL BIO file and return its path.

    Args:
        path: Destination for the CoNLL column file.
        text: Exact source text.
        spans: Canonical, non-overlapping spans from one document.

    Returns:
        The resolved output path.
    """

    resolved_path = Path(path)
    contents = format_conll(text, spans)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(contents)
    return resolved_path


def _align_rows(
    text: str,
    columns: str,
) -> tuple[list[_TokenRow], list[AnnotationIssue]]:
    rows: list[_TokenRow] = []
    issues: list[AnnotationIssue] = []
    cursor = 0
    sentence_break = False

    for line_number, line in enumerate(columns.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            sentence_break = True
            continue
        if stripped == "#" or stripped.startswith("# "):
            continue
        fields = stripped.split()
        if fields[0] == "-DOCSTART-":
            sentence_break = True
            continue
        if len(fields) < 2:
            issues.append(
                AnnotationIssue(
                    "expected at least TOKEN and TAG columns", line=line_number
                )
            )
            continue
        token, tag = fields[0], fields[-1]
        start = text.find(token, cursor)
        if start < 0:
            issues.append(
                AnnotationIssue(
                    "token cannot be aligned after the preceding source offset",
                    line=line_number,
                )
            )
            continue
        if text[cursor:start].strip():
            issues.append(
                AnnotationIssue(
                    "source text contains non-whitespace content before this token",
                    line=line_number,
                )
            )
        end = start + len(token)
        rows.append(_TokenRow(token, tag, start, end, line_number, sentence_break))
        cursor = end
        sentence_break = False

    if text[cursor:].strip():
        issues.append(
            AnnotationIssue("source text contains content after the final CoNLL token")
        )
    return rows, issues


def _tokens_with_span_boundaries(
    text: str,
    spans: tuple[OpenMedSpan, ...],
) -> list[tuple[int, int, str]]:
    boundaries = {value for span in spans for value in (span.start, span.end)}
    tokens: list[tuple[int, int, str]] = []
    for match in _TOKEN_RE.finditer(text):
        cuts = [
            match.start(),
            *sorted(
                boundary
                for boundary in boundaries
                if match.start() < boundary < match.end()
            ),
            match.end(),
        ]
        tokens.extend(
            (start, end, text[start:end])
            for start, end in zip(cuts, cuts[1:])
            if start < end
        )
    return tokens


__all__ = ["format_conll", "parse_conll", "read_conll", "write_conll"]
