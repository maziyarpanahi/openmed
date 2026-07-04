"""Shared types for clinical concept grounding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    """A ranked grounding candidate: a coded concept for a clinical span.

    ``system`` is the vocabulary (e.g. ``"RXNORM"``), ``code`` its identifier
    (e.g. an RxCUI), ``display`` a human-readable name, and ``score`` a
    ``0.0``-``1.0`` match confidence (``1.0`` for an exact alias match).
    """

    system: str
    code: str
    display: str
    score: float
