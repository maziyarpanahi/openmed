"""Deterministic, synthetic social-history sections with SDOH gold events.

The templates are repository-authored. Restricted SHAC records are never read,
bundled, or used for training by this module.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from openmed.clinical.sdoh import SDOHFinding

SYNTHETIC_SOCIAL_HISTORY_SOURCE = "openmed.synthetic.social_history"
SOCIAL_HISTORY_CATEGORIES = (
    "alcohol",
    "drug",
    "tobacco",
    "employment",
    "living_status",
)


@dataclass(frozen=True)
class SDOHEvent:
    """Gold determinant event using the runtime finding's SHAC-style axes.

    Args:
        category: One of the five social-history determinant categories.
        value: Normalized determinant value.
        status: Normalized status, including ``none`` for negated use.
        amount: Optional extent or quantity expressed in the synthetic text.
        temporality: Recent or historical qualifier.
        span: Half-open offsets for the evidence clause in the section text.
    """

    category: str
    value: str
    status: str
    amount: str | None
    temporality: str
    span: tuple[int, int]

    def to_finding(self) -> SDOHFinding:
        """Adapt the gold event to the runtime SDOH finding contract."""

        return SDOHFinding(
            category=self.category,
            value=self.value,
            status=self.status,
            extent=self.amount,
            temporality=self.temporality,
            span=self.span,
            score=1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible event with SHAC-style arguments."""

        return {
            "category": self.category,
            "value": self.value,
            "status": self.status,
            "attributes": {
                "extent": self.amount,
                "temporality": self.temporality,
            },
            "span": list(self.span),
        }


@dataclass(frozen=True)
class SyntheticSocialHistory:
    """One explicitly synthetic Social History section and its gold events."""

    text: str
    events: tuple[SDOHEvent, ...]
    synthetic: bool = True
    source: str = SYNTHETIC_SOCIAL_HISTORY_SOURCE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible training record with provenance."""

        return {
            "text": self.text,
            "events": [event.to_dict() for event in self.events],
            "metadata": {
                "synthetic": self.synthetic,
                "source": self.source,
                "restricted_data": False,
            },
        }


@dataclass(frozen=True)
class _Template:
    text: str
    value: str
    status: str
    amount: str | None = None
    temporality: str = "recent"


_TEMPLATES: dict[str, tuple[tuple[_Template, ...], ...]] = {
    "alcohol": (
        (
            _Template(
                "drinks alcohol, 2 drinks/week", "alcohol", "current", "2 drinks/week"
            ),
            _Template("reports 1 drink/week", "alcohol", "current", "1 drink/week"),
        ),
        (
            _Template(
                "former alcohol use, stopped in 2019",
                "alcohol",
                "past",
                temporality="historical",
            ),
            _Template(
                "stopped drinking alcohol in 2020",
                "alcohol",
                "past",
                temporality="historical",
            ),
        ),
        (
            _Template("denies alcohol use", "alcohol", "none"),
            _Template("never drinks alcohol", "alcohol", "none"),
        ),
    ),
    "drug": (
        (
            _Template(
                "reports recreational drug use weekly", "drug", "current", "weekly"
            ),
            _Template(
                "uses recreational drugs occasionally",
                "drug",
                "current",
                "occasionally",
            ),
        ),
        (
            _Template(
                "past drug use, stopped in 2018",
                "drug",
                "past",
                temporality="historical",
            ),
            _Template(
                "former recreational drug use", "drug", "past", temporality="historical"
            ),
        ),
        (
            _Template("denies illicit drug use", "drug", "none"),
            _Template("never used illicit drugs", "drug", "none"),
        ),
    ),
    "tobacco": (
        (
            _Template(
                "current smoker, 5 cigarettes/day",
                "tobacco",
                "current",
                "5 cigarettes/day",
            ),
            _Template(
                "uses tobacco, 10 cigarettes/day",
                "tobacco",
                "current",
                "10 cigarettes/day",
            ),
        ),
        (
            _Template(
                "former smoker, quit in 2019",
                "tobacco",
                "past",
                temporality="historical",
            ),
            _Template(
                "stopped smoking in 2020", "tobacco", "past", temporality="historical"
            ),
        ),
        (
            _Template("denies tobacco use", "tobacco", "none"),
            _Template("never smoked tobacco", "tobacco", "none"),
        ),
    ),
    "employment": (
        (
            _Template("currently employed as a teacher", "teacher", "employed"),
            _Template("working as an engineer", "engineer", "employed"),
        ),
        (
            _Template(
                "previously employed as a driver",
                "driver",
                "former",
                temporality="historical",
            ),
            _Template(
                "former office worker",
                "office_worker",
                "former",
                temporality="historical",
            ),
        ),
        (
            _Template("currently unemployed", "unemployed", "unemployed"),
            _Template("without work", "unemployed", "unemployed"),
        ),
    ),
    "living_status": (
        (
            _Template("has stable housing", "housed", "housed"),
            _Template("lives with family", "lives_with_family", "lives_with_family"),
        ),
        (
            _Template(
                "formerly homeless, now housed",
                "former",
                "former",
                temporality="historical",
            ),
            _Template(
                "past homelessness, now housed",
                "former",
                "former",
                temporality="historical",
            ),
        ),
        (
            _Template("currently homeless", "homeless", "homeless"),
            _Template("living in a shelter", "homeless", "homeless"),
        ),
    ),
}


def generate_social_history_examples(
    count: int,
    *,
    seed: int = 0,
) -> tuple[SyntheticSocialHistory, ...]:
    """Generate balanced synthetic sections for all five determinant categories.

    Each three consecutive sections cover current, historical, and negative or
    absent states for every category. The same seed yields identical text and
    offsets; no external data or network access is used.

    Args:
        count: Number of complete Social History sections to generate.
        seed: Deterministic template and clause-order seed.

    Returns:
        Synthetic sections with offset-aligned SDOHEvent gold annotations.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    rng = random.Random(seed)
    examples: list[SyntheticSocialHistory] = []
    for index in range(count):
        clauses = [
            (
                category,
                rng.choice(_TEMPLATES[category][(index + category_index) % 3]),
            )
            for category_index, category in enumerate(SOCIAL_HISTORY_CATEGORIES)
        ]
        rng.shuffle(clauses)
        text = "Social History:\n"
        events: list[SDOHEvent] = []
        for category, template in clauses:
            start = len(text)
            text += template.text
            events.append(
                SDOHEvent(
                    category=category,
                    value=template.value,
                    status=template.status,
                    amount=template.amount,
                    temporality=template.temporality,
                    span=(start, len(text)),
                )
            )
            text += ".\n"
        examples.append(SyntheticSocialHistory(text=text, events=tuple(events)))
    return tuple(examples)


__all__ = [
    "SDOHEvent",
    "SOCIAL_HISTORY_CATEGORIES",
    "SYNTHETIC_SOCIAL_HISTORY_SOURCE",
    "SyntheticSocialHistory",
    "generate_social_history_examples",
]
