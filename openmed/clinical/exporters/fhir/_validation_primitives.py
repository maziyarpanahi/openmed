"""Shared, dependency-free traversal primitives for FHIR validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Occurrence:
    value: Any
    expression: str
    key: str | None = None
    repeated: bool = False


@dataclass(frozen=True)
class _OccurrenceGroup:
    expression: str
    occurrences: tuple[_Occurrence, ...]


def _occurrence_groups(
    node: Any,
    segments: Sequence[str],
    root_expression: str,
) -> list[_OccurrenceGroup]:
    """Resolve element paths while retaining per-parent cardinality groups."""

    if not segments:
        occurrence = (
            (_Occurrence(node, root_expression),) if _is_present(node) else tuple()
        )
        return [_OccurrenceGroup(root_expression, occurrence)]

    parents = [_Occurrence(node, root_expression)]
    for segment in segments[:-1]:
        next_parents: list[_Occurrence] = []
        for parent in parents:
            next_parents.extend(_children(parent, segment))
        parents = next_parents
        if not parents:
            return []

    final_segment = segments[-1]
    groups: list[_OccurrenceGroup] = []
    for parent in parents:
        children = tuple(_children(parent, final_segment))
        groups.append(
            _OccurrenceGroup(
                expression=_child_expression(parent.expression, final_segment),
                occurrences=children,
            )
        )
    return groups


def _children(parent: _Occurrence, segment: str) -> list[_Occurrence]:
    if not isinstance(parent.value, Mapping):
        return []
    children: list[_Occurrence] = []
    for key in _matching_keys(parent.value, segment):
        value = parent.value[key]
        expression = f"{parent.expression}.{key}"
        if isinstance(value, list):
            children.extend(
                _Occurrence(
                    item,
                    f"{expression}[{index}]",
                    key=key,
                    repeated=True,
                )
                for index, item in enumerate(value)
                if _is_present(item)
            )
        elif _is_present(value):
            children.append(_Occurrence(value, expression, key=key))
    return children


def _matching_keys(node: Mapping[str, Any], segment: str) -> list[str]:
    if segment.endswith("[x]"):
        prefix = segment[:-3]
        return sorted(
            key
            for key in node
            if isinstance(key, str)
            and key.startswith(prefix)
            and len(key) > len(prefix)
            and key[len(prefix)].isupper()
        )
    return [segment] if segment in node else []


def _child_expression(parent_expression: str, segment: str) -> str:
    if segment.endswith("[x]"):
        segment = segment[:-3]
    return f"{parent_expression}.{segment}"


def _path_expression(root_expression: str, segments: Sequence[str]) -> str:
    expression = root_expression
    for segment in segments:
        expression = _child_expression(expression, segment)
    return expression


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes)):
        return bool(value)
    return True


def _extract_codes(value: Any) -> list[tuple[str, str]]:
    """Extract ``(system, code)`` pairs from code/Coding/CodeableConcept values."""

    if isinstance(value, str):
        return [("", value)]
    if not isinstance(value, Mapping):
        return []
    code = value.get("code")
    if isinstance(code, str):
        system = value.get("system")
        return [((system if isinstance(system, str) else ""), code)]
    coding = value.get("coding")
    if isinstance(coding, list):
        codes: list[tuple[str, str]] = []
        for item in coding:
            codes.extend(_extract_codes(item))
        return codes
    return []
