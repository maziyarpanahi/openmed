"""Offline loader for user-supplied Human Phenotype Ontology releases.

The loader accepts a local HPO OBO file or an OBO-JSON/simple JSON release. It
does not download, cache, or bundle HPO data. HPO releases are openly licensed
under the Creative Commons Attribution 4.0 International license (CC BY 4.0);
callers remain responsible for complying with the terms of the exact release
they provide.

The returned terms implement the shared :class:`VocabularyLoader` contract.
HPO-specific graph operations are available on :class:`HPOVocabularyLoader`,
while lexical matches carry their deterministic root-to-term path in
``ConceptMatch.metadata["ontology_path"]``.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..matcher import ConceptMatch, LexicalConcept, LexicalMatcher, VocabularyTerms
from ..vocab import VocabLoaderError

__all__ = [
    "HPOConcept",
    "HPO_LICENSE_NOTE",
    "HPO_SYSTEM_URI",
    "HPOVocabularyError",
    "HPOVocabularyLoader",
    "HPOLoader",
    "HpoLoader",
    "HpoVocabularyLoader",
]

# Keep the loader aligned with the repository's established HPO/FHIR coding
# system URI; the purl is the release source URL, not the coding-system key.
HPO_SYSTEM_URI = "http://human-phenotype-ontology.org"
HPO_LICENSE_NOTE = (
    "Human Phenotype Ontology releases are openly licensed under the Creative "
    "Commons Attribution 4.0 International license (CC BY 4.0)."
)

_OBO_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:[^\s!]+")
_OBO_QUOTED_RE = re.compile(r'^"((?:\\.|[^"\\])*)"(?:\s+(.+))?$')
_JSON_GRAPH_PREDICATES = {
    "is_a",
    "is-a",
    "subclassof",
    "subclass_of",
    "subclass",
    "http://www.w3.org/2000/01/rdf-schema#subclassof",
}
_JSON_NODE_KEYS = ("nodes", "terms", "concepts", "items")
_JSON_ID_KEYS = ("id", "code", "identifier", "curie", "@id")
_JSON_LABEL_KEYS = ("lbl", "label", "name", "preferred_term", "title")
_JSON_PARENT_KEYS = (
    "parents",
    "parent",
    "is_a",
    "is-a",
    "isa",
    "superClassOf",
    "superclass_of",
)
_JSON_SYNONYM_KEYS = (
    "synonyms",
    "synonym",
    "aliases",
    "alt_labels",
    "exact_synonyms",
    "related_synonyms",
)


class HPOVocabularyError(VocabLoaderError, ValueError):
    """Raised when a supplied HPO release cannot be loaded safely."""


@dataclass(frozen=True)
class HPOConcept:
    """One HPO concept parsed from the caller's local release."""

    id: str
    label: str
    synonyms: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()

    @property
    def code(self) -> str:
        """Return the HPO identifier as a code-compatible alias."""

        return self.id

    @property
    def display(self) -> str:
        """Return the preferred HPO label."""

        return self.label


@dataclass
class _ParsedConcept:
    identifier: str
    label: str
    synonyms: list[str] = field(default_factory=list)
    parents: set[str] = field(default_factory=set)
    obsolete: bool = False


class HPOVocabularyLoader:
    """Load and index a caller-supplied HPO OBO or JSON release.

    Args:
        path: Local path to an HPO OBO or JSON release. The file is read only
            when :meth:`load` or another query method is first called.
        source: Keyword alias for ``path``.
        include_obsolete: Include terms marked obsolete in the supplied
            release. Obsolete terms are skipped by default.

    The loader is deliberately data-free: there is no default release URL,
    download path, or bundled HPO fixture. ``redistributable`` is true because
    HPO is openly licensed, while the actual release remains user-supplied.
    """

    system_uri = HPO_SYSTEM_URI
    redistributable = True
    license_note = HPO_LICENSE_NOTE

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        source: str | Path | None = None,
        include_obsolete: bool = False,
    ) -> None:
        if path is not None and source is not None:
            raise TypeError("provide either path or source, not both")
        if path is None:
            path = source
        if path is None:
            raise TypeError("an HPO release path is required")
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or pathlib.Path")
        if not isinstance(include_obsolete, bool):
            raise TypeError("include_obsolete must be a boolean")

        self.path = Path(path).expanduser()
        self.include_obsolete = include_obsolete
        self._concepts: dict[str, HPOConcept] | None = None
        self._terms: VocabularyTerms | None = None
        self._matcher: LexicalMatcher | None = None
        self._parents: dict[str, tuple[str, ...]] | None = None
        self._children: dict[str, tuple[str, ...]] | None = None

    @property
    def concepts(self) -> Mapping[str, HPOConcept]:
        """Return the loaded HPO concepts keyed by canonical identifier."""

        self._ensure_loaded()
        assert self._concepts is not None
        return MappingProxyType(self._concepts)

    @property
    def concept_count(self) -> int:
        """Return the number of non-obsolete concepts in the release."""

        return len(self.concepts)

    @property
    def term_count(self) -> int:
        """Return the number of indexed primary labels and synonyms."""

        return len(self.load())

    @property
    def matcher(self) -> LexicalMatcher:
        """Return the lazy lexical matcher for the supplied HPO release."""

        self._ensure_loaded()
        assert self._matcher is not None
        return self._matcher

    def load(self) -> VocabularyTerms:
        """Return primary labels and exact/related synonyms for matching."""

        self._ensure_loaded()
        assert self._terms is not None
        return self._terms

    def lookup(
        self, query: str, *, limit: int | None = None
    ) -> tuple[ConceptMatch, ...]:
        """Resolve a finding mention to deterministic HPO concept matches."""

        return self.matcher.lookup(query, limit=limit)

    def match(
        self, query: str, *, limit: int | None = None
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup`."""

        return self.lookup(query, limit=limit)

    def resolve(
        self, query: str, *, limit: int | None = None
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup` for terminology-resolution callers."""

        return self.lookup(query, limit=limit)

    def get_concept(self, concept_id: str) -> HPOConcept:
        """Return one concept, raising ``KeyError`` for an unknown ID."""

        identifier = self._require_concept(concept_id)
        assert self._concepts is not None
        return self._concepts[identifier]

    def ontology_path(self, concept_id: str) -> tuple[str, ...]:
        """Return one stable root-to-term path through the ``is_a`` DAG.

        HPO terms may have more than one parent. The shortest available path is
        selected, with lexicographic ordering as the deterministic tie-breaker.
        Only concepts present in the supplied release are included.
        """

        identifier = self._require_concept(concept_id)
        assert self._parents is not None
        assert self._concepts is not None

        def build(node: str, visiting: frozenset[str]) -> tuple[str, ...]:
            if node in visiting:
                return (node,)
            parents = [
                parent
                for parent in self._parents.get(node, ())
                if parent in self._concepts
            ]
            if not parents:
                return (node,)
            paths = [build(parent, visiting | {node}) + (node,) for parent in parents]
            return min(paths, key=lambda path: (len(path), path))

        return build(identifier, frozenset())

    def path(self, concept_id: str) -> tuple[str, ...]:
        """Alias for :meth:`ontology_path`."""

        return self.ontology_path(concept_id)

    def ancestors(
        self, concept_id: str, *, include_self: bool = False
    ) -> frozenset[str]:
        """Return strict ``is_a`` ancestors of an HPO concept as IDs."""

        identifier = self._require_concept(concept_id)
        assert self._parents is not None
        found: set[str] = set()
        pending = list(self._parents.get(identifier, ()))
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(self._parents.get(current, ()))
        if include_self:
            found.add(identifier)
        return frozenset(found)

    def descendants(
        self, concept_id: str, *, include_self: bool = False
    ) -> frozenset[str]:
        """Return strict ``is_a`` descendants of an HPO concept as IDs."""

        identifier = self._require_concept(concept_id)
        assert self._children is not None
        found: set[str] = set()
        pending = list(self._children.get(identifier, ()))
        while pending:
            current = pending.pop()
            if current in found:
                continue
            found.add(current)
            pending.extend(self._children.get(current, ()))
        if include_self:
            found.add(identifier)
        return frozenset(found)

    def ancestor_ids(
        self, concept_id: str, *, include_self: bool = False
    ) -> tuple[str, ...]:
        """Return :meth:`ancestors` in deterministic identifier order."""

        return tuple(sorted(self.ancestors(concept_id, include_self=include_self)))

    def descendant_ids(
        self, concept_id: str, *, include_self: bool = False
    ) -> tuple[str, ...]:
        """Return :meth:`descendants` in deterministic identifier order."""

        return tuple(sorted(self.descendants(concept_id, include_self=include_self)))

    def ancestor_terms(
        self, concept_id: str, *, include_self: bool = False
    ) -> tuple[HPOConcept, ...]:
        """Return ancestor concepts in deterministic identifier order."""

        ids = self.ancestor_ids(concept_id, include_self=include_self)
        return tuple(self.get_concept(identifier) for identifier in ids)

    def descendant_terms(
        self, concept_id: str, *, include_self: bool = False
    ) -> tuple[HPOConcept, ...]:
        """Return descendant concepts in deterministic identifier order."""

        ids = self.descendant_ids(concept_id, include_self=include_self)
        return tuple(self.get_concept(identifier) for identifier in ids)

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Return whether ``ancestor_id`` is a strict ancestor of a term."""

        ancestor = self._require_concept(ancestor_id)
        return ancestor in self.ancestors(descendant_id)

    def is_descendant(self, descendant_id: str, ancestor_id: str) -> bool:
        """Return whether ``descendant_id`` is a strict descendant of a term."""

        descendant = self._require_concept(descendant_id)
        return descendant in self.descendants(ancestor_id)

    def subsumes(
        self,
        ancestor_id: str,
        descendant_id: str,
        *,
        include_self: bool = True,
    ) -> bool:
        """Return whether one HPO term subsumes another for phenotype roll-up."""

        ancestor = self._require_concept(ancestor_id)
        descendant = self._require_concept(descendant_id)
        if include_self and ancestor == descendant:
            return True
        return ancestor in self.ancestors(descendant)

    def is_subsumed_by(
        self,
        concept_id: str,
        subsumer_id: str,
        *,
        include_self: bool = True,
    ) -> bool:
        """Return whether ``concept_id`` is subsumed by ``subsumer_id``."""

        return self.subsumes(
            subsumer_id,
            concept_id,
            include_self=include_self,
        )

    def roll_up(self, concept_id: str, *, include_self: bool = False) -> frozenset[str]:
        """Return the ancestor IDs useful for phenotype roll-up."""

        return self.ancestors(concept_id, include_self=include_self)

    def subsumers(
        self, concept_id: str, *, include_self: bool = False
    ) -> frozenset[str]:
        """Alias for :meth:`roll_up`."""

        return self.roll_up(concept_id, include_self=include_self)

    def _require_concept(self, concept_id: str) -> str:
        self._ensure_loaded()
        identifier = _canonical_id(concept_id)
        assert self._concepts is not None
        if identifier not in self._concepts:
            raise KeyError(f"Unknown HPO concept {concept_id!r}.")
        return identifier

    def _ensure_loaded(self) -> None:
        if self._concepts is not None:
            return

        parsed = _parse_release(self.path, include_obsolete=self.include_obsolete)
        merged: dict[str, _ParsedConcept] = {}
        for record in parsed:
            if record.obsolete and not self.include_obsolete:
                continue
            existing = merged.get(record.identifier)
            if existing is None:
                merged[record.identifier] = record
                continue
            if not existing.label and record.label:
                existing.label = record.label
            existing.synonyms.extend(record.synonyms)
            existing.parents.update(record.parents)
            existing.obsolete = existing.obsolete and record.obsolete

        if not merged:
            raise HPOVocabularyError(
                f"No usable HPO concepts were found in {self.path}."
            )

        concepts: dict[str, HPOConcept] = {}
        for identifier in sorted(merged):
            record = merged[identifier]
            if not record.label:
                raise HPOVocabularyError(
                    f"HPO concept {identifier!r} is missing a label."
                )
            synonyms = _unique_terms(record.synonyms, exclude=record.label)
            parents = tuple(sorted(record.parents - {identifier}))
            concepts[identifier] = HPOConcept(
                id=identifier,
                label=record.label,
                synonyms=synonyms,
                parents=parents,
            )

        parents = {
            identifier: concept.parents for identifier, concept in concepts.items()
        }
        children_mutable: dict[str, set[str]] = defaultdict(set)
        for child, parent_ids in parents.items():
            for parent in parent_ids:
                children_mutable[parent].add(child)
        children = {
            identifier: tuple(sorted(child_ids))
            for identifier, child_ids in children_mutable.items()
        }

        terms_mutable: dict[str, list[LexicalConcept]] = defaultdict(list)
        for identifier in sorted(concepts):
            concept = concepts[identifier]
            metadata = {
                "ontology_path": self._path_for_concepts(identifier, concepts, parents),
                "parents": concept.parents,
                "synonyms": concept.synonyms,
            }
            lexical = LexicalConcept(
                system_uri=self.system_uri,
                code=concept.id,
                display=concept.label,
                metadata=metadata,
            )
            for term in (concept.label, *concept.synonyms):
                if lexical not in terms_mutable[term]:
                    terms_mutable[term].append(lexical)

        terms: dict[str, LexicalConcept | tuple[LexicalConcept, ...]] = {}
        for term, lexical_concepts in terms_mutable.items():
            terms[term] = (
                lexical_concepts[0]
                if len(lexical_concepts) == 1
                else tuple(lexical_concepts)
            )

        self._concepts = concepts
        self._parents = parents
        self._children = children
        self._terms = terms
        self._matcher = LexicalMatcher(terms, system_uri=self.system_uri)

    @staticmethod
    def _path_for_concepts(
        identifier: str,
        concepts: Mapping[str, HPOConcept],
        parents: Mapping[str, Sequence[str]],
    ) -> tuple[str, ...]:
        """Build a path before the loader's public graph state is installed."""

        def build(node: str, visiting: frozenset[str]) -> tuple[str, ...]:
            if node in visiting:
                return (node,)
            known_parents = [
                parent for parent in parents.get(node, ()) if parent in concepts
            ]
            if not known_parents:
                return (node,)
            paths = [
                build(parent, visiting | {node}) + (node,) for parent in known_parents
            ]
            return min(paths, key=lambda path: (len(path), path))

        return build(identifier, frozenset())


def _parse_release(path: Path, *, include_obsolete: bool) -> list[_ParsedConcept]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise HPOVocabularyError(f"Unable to read HPO release {path}: {exc}") from exc

    if not text.strip():
        raise HPOVocabularyError(f"HPO release {path} is empty.")

    stripped = text.lstrip()
    looks_like_json = stripped[:1] in {"{", "["} and not stripped.startswith("[Term]")
    if path.suffix.casefold() == ".json" or looks_like_json:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise HPOVocabularyError(
                f"Invalid JSON HPO release {path}: {exc.msg}."
            ) from exc
        return _parse_json(payload, include_obsolete=include_obsolete)
    return _parse_obo(text, include_obsolete=include_obsolete)


def _parse_obo(text: str, *, include_obsolete: bool) -> list[_ParsedConcept]:
    records: list[_ParsedConcept] = []
    current: dict[str, Any] | None = None

    def finish() -> None:
        if current is None:
            return
        identifier = _canonical_id(current.get("id"))
        label = _text(current.get("name"))
        if not identifier or not label:
            return
        obsolete = _as_bool(current.get("is_obsolete"))
        if obsolete and not include_obsolete:
            return
        records.append(
            _ParsedConcept(
                identifier=identifier,
                label=label,
                synonyms=list(current.get("synonyms", [])),
                parents=set(current.get("parents", [])),
                obsolete=obsolete,
            )
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        if line == "[Term]":
            finish()
            current = {"synonyms": [], "parents": []}
            continue
        if line.startswith("["):
            finish()
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if key == "id":
            current["id"] = _first_id(value)
        elif key == "name":
            current["name"] = value
        elif key == "is_a":
            parent = _first_id(value)
            if parent:
                current.setdefault("parents", []).append(parent)
        elif key in {"synonym", "exact_synonym", "related_synonym"}:
            synonym, scope = _obo_synonym(value)
            if synonym and (key != "synonym" or scope in {None, "EXACT", "RELATED"}):
                current.setdefault("synonyms", []).append(synonym)
        elif key == "is_obsolete":
            current["is_obsolete"] = value

    finish()
    return records


def _parse_json(payload: Any, *, include_obsolete: bool) -> list[_ParsedConcept]:
    if not isinstance(payload, (Mapping, list)):
        raise HPOVocabularyError("JSON HPO release must contain an object or array.")

    node_containers: list[Any] = []
    edge_containers: list[Any] = []
    if isinstance(payload, list):
        node_containers.append(payload)
    elif "graphs" in payload:
        graphs = payload["graphs"]
        if not isinstance(graphs, Sequence) or isinstance(graphs, (str, bytes)):
            raise HPOVocabularyError("JSON HPO release 'graphs' must be an array.")
        for graph in graphs:
            if not isinstance(graph, Mapping):
                continue
            node_containers.append(_first_present(graph, _JSON_NODE_KEYS, ()))
            edge_containers.append(graph.get("edges", ()))
    else:
        node_container = _first_present(payload, _JSON_NODE_KEYS, None)
        if node_container is None:
            node_container = payload
        node_containers.append(node_container)
        edge_containers.append(payload.get("edges", ()))

    records: list[_ParsedConcept] = []
    for container in node_containers:
        for raw, fallback_id in _iter_json_nodes(container):
            record = _json_record(raw, fallback_id=fallback_id)
            if record is None:
                continue
            if record.obsolete and not include_obsolete:
                continue
            records.append(record)

    merged = _merge_parsed_records(records)
    for container in edge_containers:
        for child, parent in _iter_json_edges(container):
            record = merged.get(child)
            if record is not None:
                record.parents.add(parent)
    return list(merged.values())


def _iter_json_nodes(container: Any) -> Iterable[tuple[Mapping[str, Any], str | None]]:
    if isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
        for item in container:
            if isinstance(item, Mapping):
                yield item, None
        return
    if not isinstance(container, Mapping):
        return
    if _looks_like_json_node(container):
        yield container, None
        return
    for key, value in container.items():
        if isinstance(value, Mapping):
            yield value, str(key)


def _iter_json_edges(container: Any) -> Iterable[tuple[str, str]]:
    if not isinstance(container, Sequence) or isinstance(container, (str, bytes)):
        return
    for raw_edge in container:
        if not isinstance(raw_edge, Mapping):
            continue
        predicate = _text(
            _first_present(raw_edge, ("pred", "predicate", "relation", "type"), "")
        )
        if not _is_is_a_predicate(predicate):
            continue
        child = _canonical_id(
            _first_present(raw_edge, ("sub", "subject", "child", "source"), None)
        )
        parent = _canonical_id(
            _first_present(raw_edge, ("obj", "object", "parent", "target"), None)
        )
        if child and parent and child != parent:
            yield child, parent


def _json_record(
    raw: Mapping[str, Any], *, fallback_id: str | None
) -> _ParsedConcept | None:
    meta = raw.get("meta")
    metadata = meta if isinstance(meta, Mapping) else {}
    identifier = _canonical_id(_first_present(raw, _JSON_ID_KEYS, fallback_id))
    if not identifier:
        return None
    label = _text(_first_present(raw, _JSON_LABEL_KEYS, None))
    if not label:
        return None

    synonyms: list[str] = []
    for key in _JSON_SYNONYM_KEYS:
        if key in raw:
            synonyms.extend(_synonym_texts(raw[key], key=key))
    if "synonyms" in metadata:
        synonyms.extend(_synonym_texts(metadata["synonyms"], key="synonyms"))

    parents: set[str] = set()
    for key in _JSON_PARENT_KEYS:
        if key in raw:
            parents.update(_parent_ids(raw[key]))
    if "parents" in metadata:
        parents.update(_parent_ids(metadata["parents"]))

    obsolete = _as_bool(
        _first_present(
            raw,
            ("is_obsolete", "obsolete", "deprecated"),
            _first_present(metadata, ("deprecated", "is_obsolete"), False),
        )
    )
    return _ParsedConcept(
        identifier=identifier,
        label=label,
        synonyms=synonyms,
        parents=parents,
        obsolete=obsolete,
    )


def _merge_parsed_records(
    records: Iterable[_ParsedConcept],
) -> dict[str, _ParsedConcept]:
    merged: dict[str, _ParsedConcept] = {}
    for record in records:
        existing = merged.get(record.identifier)
        if existing is None:
            merged[record.identifier] = record
            continue
        if not existing.label and record.label:
            existing.label = record.label
        existing.synonyms.extend(record.synonyms)
        existing.parents.update(record.parents)
        existing.obsolete = existing.obsolete and record.obsolete
    return merged


def _obo_synonym(value: str) -> tuple[str | None, str | None]:
    match = _OBO_QUOTED_RE.match(value)
    if match is None:
        return None, None
    text = _unescape_obo(match.group(1))
    remainder = (match.group(2) or "").strip()
    scope = remainder.split(None, 1)[0].upper() if remainder else None
    return text, scope


def _unescape_obo(value: str) -> str:
    return (
        value.replace(r"\n", "\n")
        .replace(r"\t", "\t")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def _synonym_texts(value: Any, *, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        predicate = _text(
            _first_present(value, ("pred", "predicate", "type", "kind"), "")
        )
        if predicate and not _is_accepted_synonym_predicate(predicate):
            return []
        text = _text(
            _first_present(value, ("val", "value", "label", "text", "name"), None)
        )
        if text:
            return [text]
        values: list[str] = []
        for item in value.values():
            values.extend(_synonym_texts(item, key=key))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = []
        for item in value:
            values.extend(_synonym_texts(item, key=key))
        return values
    return []


def _parent_ids(value: Any) -> set[str]:
    if isinstance(value, str):
        identifier = _canonical_id(value)
        return {identifier} if identifier else set()
    if isinstance(value, Mapping):
        direct = _first_present(
            value, _JSON_ID_KEYS + ("obj", "object", "target"), None
        )
        if direct is not None:
            identifier = _canonical_id(direct)
            return {identifier} if identifier else set()
        result: set[str] = set()
        for item in value.values():
            result.update(_parent_ids(item))
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = set()
        for item in value:
            result.update(_parent_ids(item))
        return result
    return set()


def _unique_terms(
    values: Iterable[str], *, exclude: str | None = None
) -> tuple[str, ...]:
    excluded = _term_key(exclude) if exclude else ""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _text(value)
        key = _term_key(text)
        if not text or not key or key == excluded or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return tuple(result)


def _term_key(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.casefold().split())


def _canonical_id(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    text = _first_id(text)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    if "#" in text:
        text = text.rsplit("#", 1)[-1]
    if "_" in text and ":" not in text:
        prefix, suffix = text.split("_", 1)
        if prefix and suffix:
            text = f"{prefix}:{suffix}"
    return text


def _first_id(value: Any) -> str:
    text = _text(value)
    if not text:
        return ""
    match = _OBO_ID_RE.match(text)
    return match.group(0) if match else text.split(None, 1)[0]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _first_present(
    mapping: Mapping[str, Any], keys: Sequence[str], default: Any
) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _looks_like_json_node(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in _JSON_ID_KEYS) and any(
        key in value for key in _JSON_LABEL_KEYS
    )


def _is_is_a_predicate(value: str) -> bool:
    normalized = value.strip().casefold().rstrip("/")
    return normalized in _JSON_GRAPH_PREDICATES or normalized.endswith("/is_a")


def _is_accepted_synonym_predicate(value: str) -> bool:
    normalized = value.strip().casefold().rstrip("/")
    return normalized.endswith(
        ("hasexactsynonym", "hasrelatedsynonym")
    ) or normalized in {
        "exact",
        "related",
        "exact_synonym",
        "related_synonym",
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).casefold() in {"1", "true", "yes"}


# Compatibility aliases follow the existing ``HpoLinker`` naming while the
# all-caps class remains the descriptive public name for this HPO-specific API.
HpoVocabularyLoader = HPOVocabularyLoader
HpoLoader = HPOVocabularyLoader
HPOLoader = HPOVocabularyLoader
