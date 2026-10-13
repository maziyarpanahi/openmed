"""Bounded provenance graphs for caller-supplied transliteration candidates."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SCRIPT = re.compile(r"[A-Z][a-z]{1,31}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _code(value: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError("invalid transliteration metadata code")


@dataclass(frozen=True, slots=True)
class TransliterationNode:
    """An opaque term identity with explicit language and script provenance."""

    digest: str
    language: str
    script: str

    def __post_init__(self) -> None:
        _code(self.digest, _DIGEST)
        _code(self.language, _CODE)
        _code(self.script, _SCRIPT)

    @classmethod
    def from_text(
        cls, text: str, *, language: str, script: str
    ) -> "TransliterationNode":
        """Fingerprint a local term without retaining or normalizing its spelling."""
        if not isinstance(text, str) or not text or len(text) > 4096:
            raise ValueError("invalid transliteration term")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(digest, language, script)


@dataclass(frozen=True, slots=True)
class TransliterationRule:
    """One user-supplied candidate edge; reversibility is an explicit assertion."""

    source: TransliterationNode
    target: TransliterationNode
    rule_source: str
    evidence_digest: str
    confidence: float
    reversible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.source, TransliterationNode) or not isinstance(
            self.target, TransliterationNode
        ):
            raise ValueError("invalid transliteration endpoint")
        _code(self.rule_source, _CODE)
        _code(self.evidence_digest, _DIGEST)
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise ValueError("invalid transliteration confidence")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid transliteration confidence")
        if type(self.reversible) is not bool:
            raise ValueError("invalid transliteration reversibility")
        if self.source == self.target:
            raise ValueError("self-referential transliteration rule")


@dataclass(frozen=True, slots=True)
class TransliterationPath:
    """A distinct candidate route, retaining all rule provenance and lossiness."""

    nodes: tuple[TransliterationNode, ...]
    rules: tuple[TransliterationRule, ...]
    inverse_steps: tuple[bool, ...]

    def __post_init__(self) -> None:
        for name, maximum in (("nodes", 9), ("rules", 8), ("inverse_steps", 8)):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)):
                raise ValueError("invalid transliteration path collection")
            try:
                snapshot = tuple(islice(value, maximum + 1))
            except (TypeError, ValueError):
                raise ValueError("invalid transliteration path collection") from None
            if len(snapshot) > maximum:
                raise ValueError("transliteration path limit exceeded")
            object.__setattr__(self, name, snapshot)
        if (
            not self.rules
            or len(self.nodes) != len(self.rules) + 1
            or len(self.inverse_steps) != len(self.rules)
        ):
            raise ValueError("inconsistent transliteration path lengths")
        if any(not isinstance(n, TransliterationNode) for n in self.nodes) or len(
            set(self.nodes)
        ) != len(self.nodes):
            raise ValueError("invalid transliteration path nodes")
        for index, (rule, inverse) in enumerate(zip(self.rules, self.inverse_steps)):
            if not isinstance(rule, TransliterationRule) or type(inverse) is not bool:
                raise ValueError("invalid transliteration path provenance")
            if inverse and not rule.reversible:
                raise ValueError("lossy transliteration path cannot be inverted")
            source, target = (
                (rule.target, rule.source) if inverse else (rule.source, rule.target)
            )
            if self.nodes[index] != source or self.nodes[index + 1] != target:
                raise ValueError("transliteration path endpoints disagree")

    @property
    def confidence(self) -> float:
        """Return the product of declared hints; this is not calibrated accuracy."""
        return math.prod(rule.confidence for rule in self.rules)

    @property
    def reversible(self) -> bool:
        """Report reversibility only when every edge declares it."""
        return all(rule.reversible for rule in self.rules)

    def to_dict(self) -> dict[str, object]:
        """Export fingerprints and controlled provenance without term spellings."""
        return {
            "nodes": [
                {"digest": n.digest, "language": n.language, "script": n.script}
                for n in self.nodes
            ],
            "rules": [
                {
                    "source": r.rule_source,
                    "evidence_digest": r.evidence_digest,
                    "confidence": r.confidence,
                    "inverse": inverse,
                }
                for r, inverse in zip(self.rules, self.inverse_steps)
            ],
            "confidence": self.confidence,
            "reversible": self.reversible,
        }


class TransliterationGraph:
    """A deterministic bounded graph; no vocabulary or network provider is bundled."""

    def __init__(self, rules: Iterable[TransliterationRule] = ()) -> None:
        self._rules: set[TransliterationRule] = set()
        for index, rule in enumerate(rules):
            if index >= 10000:
                raise ValueError("transliteration graph input limit exceeded")
            self.add_rule(rule)

    def add_rule(self, rule: TransliterationRule) -> None:
        """Register explicit local terminology edges without dropping alternatives."""
        if not isinstance(rule, TransliterationRule):
            raise ValueError("invalid transliteration rule")
        if rule not in self._rules and len(self._rules) >= 10000:
            raise ValueError("transliteration graph rule limit exceeded")
        self._rules.add(rule)

    def candidates(
        self,
        source: TransliterationNode,
        *,
        language: str,
        script: str,
        max_depth: int = 3,
        max_paths: int = 100,
        max_steps: int = 10000,
    ) -> tuple[TransliterationPath, ...]:
        """Enumerate simple candidate paths; a budget overflow rejects, never truncates.

        Lossy rules cannot be traversed backward. Alternate routes remain
        separate even when their target term fingerprints match.
        """
        if not isinstance(source, TransliterationNode):
            raise ValueError("invalid transliteration source")
        _code(language, _CODE)
        _code(script, _SCRIPT)
        for value, upper in ((max_depth, 8), (max_paths, 1000), (max_steps, 100000)):
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError("invalid transliteration traversal budget")
        adjacency: dict[
            TransliterationNode, list[tuple[TransliterationRule, bool]]
        ] = {}
        for rule in sorted(
            self._rules,
            key=lambda r: (
                r.source.digest,
                r.source.language,
                r.source.script,
                r.target.digest,
                r.target.language,
                r.target.script,
                r.rule_source,
                r.evidence_digest,
                r.confidence,
                r.reversible,
            ),
        ):
            adjacency.setdefault(rule.source, []).append((rule, False))
            if rule.reversible:
                adjacency.setdefault(rule.target, []).append((rule, True))
        paths = []
        steps = 0
        stack = [(source, (source,), (), ())]
        while stack:
            node, nodes, rules, inverses = stack.pop()
            if len(rules) >= max_depth:
                continue
            for rule, inverse in reversed(adjacency.get(node, [])):
                steps += 1
                if steps > max_steps:
                    raise ValueError("transliteration traversal budget exceeded")
                target = rule.source if inverse else rule.target
                if target in nodes:
                    continue
                path = TransliterationPath(
                    nodes + (target,), rules + (rule,), inverses + (inverse,)
                )
                if target.language == language and target.script == script:
                    paths.append(path)
                    if len(paths) > max_paths:
                        raise ValueError("transliteration path limit exceeded")
                stack.append((target, path.nodes, path.rules, path.inverse_steps))
        return tuple(
            sorted(
                paths,
                key=lambda p: (
                    -p.confidence,
                    len(p.rules),
                    tuple((n.digest, n.language, n.script) for n in p.nodes),
                    tuple((r.rule_source, r.evidence_digest) for r in p.rules),
                    p.inverse_steps,
                ),
            )
        )
