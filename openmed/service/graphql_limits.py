"""Document, variable, and execution budgets for the OpenMed GraphQL endpoint.

The checks in this module run before any resolver executes. Request-level
budgets (document bytes and variables) run before parsing, lexical budgets
(tokens and bracket nesting) run before the parser builds an AST, and
structural budgets (operations, fragments, depth, aliases, nodes, and selected
fields) run before graphql-core's own validation rules. Every traversal is
iterative and fragment expansion is cycle-safe, so adversarial documents cannot
exhaust the stack or amplify work.

Rejections carry only a closed-vocabulary limit category, the configured
ceiling, and an aggregate count. They never echo a document, a variable value,
an operation name, or a field name.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Final, Optional, Union

from graphql import GraphQLError
from graphql.error import GraphQLSyntaxError
from graphql.language import (
    DocumentNode,
    FieldNode,
    FragmentDefinitionNode,
    FragmentSpreadNode,
    InlineFragmentNode,
    Lexer,
    Node,
    OperationDefinitionNode,
    SelectionSetNode,
    Source,
    TokenKind,
)
from strawberry.extensions import SchemaExtension

GRAPHQL_LIMITS_SCHEMA_VERSION: Final = "1.0.0"
GRAPHQL_LIMITS_COMPATIBILITY: Final = "same_major"

GRAPHQL_LIMIT_EXCEEDED_CODE: Final = "OPENMED_GRAPHQL_LIMIT_EXCEEDED"
GRAPHQL_TIMEOUT_CODE: Final = "OPENMED_GRAPHQL_TIMEOUT"
GRAPHQL_BATCH_UNSUPPORTED_CODE: Final = "OPENMED_GRAPHQL_BATCH_UNSUPPORTED"

GRAPHQL_LIMIT_EXCEEDED_MESSAGE: Final = (
    "The GraphQL request exceeds a configured limit."
)
GRAPHQL_TIMEOUT_MESSAGE: Final = "The GraphQL request exceeded its execution deadline."
GRAPHQL_BATCH_UNSUPPORTED_MESSAGE: Final = "Batched GraphQL requests are not supported."

# Expanded fragment counts can grow exponentially; aggregates saturate here.
SATURATED_COUNT: Final = 1_000_000_000

Ceiling = Union[int, float]


class GraphQLLimitCategory(str, Enum):
    """Closed vocabulary of GraphQL budget categories.

    The value is used as the ``limit`` extension on rejected operations and as
    the ``reason`` label on the aggregate rejection metric.
    """

    DOCUMENT_BYTES = "document_bytes"
    TOKENS = "tokens"
    NESTING = "nesting"
    NODES = "nodes"
    OPERATIONS = "operations"
    FRAGMENTS = "fragments"
    FRAGMENT_CYCLES = "fragment_cycles"
    VARIABLES = "variables"
    VARIABLE_BYTES = "variable_bytes"
    VARIABLE_DEPTH = "variable_depth"
    DEPTH = "depth"
    ALIASES = "aliases"
    SELECTED_FIELDS = "selected_fields"
    DEADLINE = "deadline"
    BATCH = "batch"


_CEILING_FIELDS: Final[Mapping[GraphQLLimitCategory, str]] = MappingProxyType(
    {
        GraphQLLimitCategory.DOCUMENT_BYTES: "max_document_bytes",
        GraphQLLimitCategory.TOKENS: "max_tokens",
        GraphQLLimitCategory.NESTING: "max_nesting",
        GraphQLLimitCategory.NODES: "max_nodes",
        GraphQLLimitCategory.OPERATIONS: "max_operations",
        GraphQLLimitCategory.FRAGMENTS: "max_fragments",
        GraphQLLimitCategory.VARIABLES: "max_variables",
        GraphQLLimitCategory.VARIABLE_BYTES: "max_variable_bytes",
        GraphQLLimitCategory.VARIABLE_DEPTH: "max_variable_depth",
        GraphQLLimitCategory.DEPTH: "max_depth",
        GraphQLLimitCategory.ALIASES: "max_aliases",
        GraphQLLimitCategory.SELECTED_FIELDS: "max_selected_fields",
        GraphQLLimitCategory.DEADLINE: "deadline_seconds",
    }
)

# Inclusive safe ranges. Nesting and depth stay well below the interpreter
# recursion limit that graphql-core's recursive-descent parser relies on.
_INTEGER_RANGES: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "max_document_bytes": (1, 16 * 1024 * 1024),
        "max_tokens": (1, 1_000_000),
        "max_nesting": (1, 128),
        "max_nodes": (1, 1_000_000),
        "max_operations": (1, 1_000),
        "max_fragments": (0, 10_000),
        "max_variables": (0, 10_000),
        "max_variable_bytes": (0, 256 * 1024 * 1024),
        "max_variable_depth": (1, 128),
        "max_depth": (1, 128),
        "max_aliases": (0, 1_000_000),
        "max_selected_fields": (1, 1_000_000),
    }
)
_MAX_DEADLINE_SECONDS: Final = 86_400.0

GRAPHQL_LIMIT_ENV_VARS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "max_document_bytes": "OPENMED_SERVICE_GRAPHQL_MAX_DOCUMENT_BYTES",
        "max_tokens": "OPENMED_SERVICE_GRAPHQL_MAX_TOKENS",
        "max_nesting": "OPENMED_SERVICE_GRAPHQL_MAX_NESTING",
        "max_nodes": "OPENMED_SERVICE_GRAPHQL_MAX_NODES",
        "max_operations": "OPENMED_SERVICE_GRAPHQL_MAX_OPERATIONS",
        "max_fragments": "OPENMED_SERVICE_GRAPHQL_MAX_FRAGMENTS",
        "max_variables": "OPENMED_SERVICE_GRAPHQL_MAX_VARIABLES",
        "max_variable_bytes": "OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_BYTES",
        "max_variable_depth": "OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_DEPTH",
        "max_depth": "OPENMED_SERVICE_GRAPHQL_MAX_DEPTH",
        "max_aliases": "OPENMED_SERVICE_GRAPHQL_MAX_ALIASES",
        "max_selected_fields": "OPENMED_SERVICE_GRAPHQL_MAX_SELECTED_FIELDS",
        "deadline_seconds": "OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS",
    }
)


def _validate_ceiling(name: str, value: object) -> None:
    if name == "deadline_seconds":
        valid = (
            isinstance(value, (int, float))
            and type(value) is not bool
            and math.isfinite(value)
            and 0 < value <= _MAX_DEADLINE_SECONDS
        )
    else:
        minimum, maximum = _INTEGER_RANGES[name]
        valid = type(value) is int and minimum <= value <= maximum
    if not valid:
        raise ValueError(f"GraphQL limit {name} is outside the safe range")


@dataclass(frozen=True, slots=True)
class GraphQLLimits:
    """Versioned resource contract for one GraphQL HTTP request.

    Attributes:
        max_document_bytes: UTF-8 bytes allowed in the query document.
        max_tokens: Lexical tokens allowed in the document, comments excluded.
        max_nesting: Deepest ``{``, ``[``, or ``(`` nesting in the document.
        max_nodes: Syntax-tree nodes allowed in the parsed document.
        max_operations: Operation definitions allowed in one document.
        max_fragments: Fragment definitions allowed in one document.
        max_variables: Variables supplied with, or declared by, the request.
        max_variable_bytes: Compact UTF-8 JSON size of the supplied variables.
        max_variable_depth: Container nesting of the supplied variables.
        max_depth: Field nesting after fragment expansion.
        max_aliases: Aliased fields after fragment expansion.
        max_selected_fields: Field selections after fragment expansion; this is
            the document's complexity budget.
        deadline_seconds: Wall-clock budget from body parsing through resolver
            execution, cancellation, and response cleanup.
        schema_version: Contract version for these limits.
        compatibility_policy: Compatibility rule for the contract version.
    """

    max_document_bytes: int = 128 * 1024
    max_tokens: int = 2_000
    max_nesting: int = 32
    max_nodes: int = 5_000
    max_operations: int = 8
    max_fragments: int = 64
    max_variables: int = 32
    max_variable_bytes: int = 8 * 1024 * 1024
    max_variable_depth: int = 16
    max_depth: int = 20
    max_aliases: int = 16
    max_selected_fields: int = 1_000
    deadline_seconds: float = 300.0
    schema_version: str = GRAPHQL_LIMITS_SCHEMA_VERSION
    compatibility_policy: str = GRAPHQL_LIMITS_COMPATIBILITY

    def __post_init__(self) -> None:
        for name in GRAPHQL_LIMIT_ENV_VARS:
            _validate_ceiling(name, getattr(self, name))
        if self.schema_version != GRAPHQL_LIMITS_SCHEMA_VERSION:
            raise ValueError("unsupported GraphQL limits schema version")
        if self.compatibility_policy != GRAPHQL_LIMITS_COMPATIBILITY:
            raise ValueError("unsupported GraphQL limits compatibility policy")
        object.__setattr__(self, "deadline_seconds", float(self.deadline_seconds))

    def ceiling(self, category: GraphQLLimitCategory) -> Optional[Ceiling]:
        """Return the configured ceiling for one category.

        Args:
            category: Budget category to look up.

        Returns:
            The ceiling, ``0`` for fragment cycles (never allowed), or ``None``
            for batches, which have no numeric ceiling.
        """
        category = GraphQLLimitCategory(category)
        if category is GraphQLLimitCategory.FRAGMENT_CYCLES:
            return 0
        name = _CEILING_FIELDS.get(category)
        return None if name is None else getattr(self, name)

    def to_dict(self) -> dict[str, object]:
        """Return the configured ceilings and contract version."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


DEFAULT_GRAPHQL_LIMITS: Final = GraphQLLimits()


def graphql_limits_from_env(
    environ: Optional[Mapping[str, str]] = None,
) -> GraphQLLimits:
    """Load GraphQL limits, applying any ``OPENMED_SERVICE_GRAPHQL_*`` overrides.

    Args:
        environ: Mapping to read instead of ``os.environ``.

    Returns:
        Validated limits. Unset or empty variables keep the safe default.

    Raises:
        ValueError: An override is not a number or is outside its safe range.
            The message names the variable but never echoes its value.
    """
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for name, env_var in GRAPHQL_LIMIT_ENV_VARS.items():
        raw = source.get(env_var)
        if raw is None or not raw.strip():
            continue
        try:
            value: Ceiling = (
                float(raw) if name == "deadline_seconds" else int(raw.strip())
            )
            _validate_ceiling(name, value)
        except ValueError:
            kind = "a number" if name == "deadline_seconds" else "an integer"
            raise ValueError(
                f"{env_var} must be {kind} within the supported GraphQL range"
            ) from None
        overrides[name] = value
    return GraphQLLimits(**overrides)


@dataclass(frozen=True, slots=True)
class GraphQLLimitViolation:
    """Value-free description of one rejected GraphQL budget.

    Attributes:
        category: Budget that rejected the request.
        maximum: Configured ceiling, when the category has one.
        observed: Aggregate count that crossed the ceiling. Scans that stop
            early report the running count at the point of crossing.
    """

    category: GraphQLLimitCategory
    maximum: Optional[Ceiling] = None
    observed: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", GraphQLLimitCategory(self.category))
        if self.observed is not None and (
            type(self.observed) is not int or self.observed < 0
        ):
            raise ValueError("GraphQL limit observation must be a non-negative count")

    @property
    def code(self) -> str:
        """Return the stable GraphQL ``extensions.code`` for this violation."""
        if self.category is GraphQLLimitCategory.DEADLINE:
            return GRAPHQL_TIMEOUT_CODE
        if self.category is GraphQLLimitCategory.BATCH:
            return GRAPHQL_BATCH_UNSUPPORTED_CODE
        return GRAPHQL_LIMIT_EXCEEDED_CODE

    @property
    def message(self) -> str:
        """Return the fixed, value-free client message for this violation."""
        if self.category is GraphQLLimitCategory.DEADLINE:
            return GRAPHQL_TIMEOUT_MESSAGE
        if self.category is GraphQLLimitCategory.BATCH:
            return GRAPHQL_BATCH_UNSUPPORTED_MESSAGE
        return GRAPHQL_LIMIT_EXCEEDED_MESSAGE

    def to_extensions(self) -> dict[str, object]:
        """Return the GraphQL error extensions: code, category, and counts."""
        extensions: dict[str, object] = {
            "code": self.code,
            "limit": self.category.value,
        }
        if self.maximum is not None:
            extensions["maximum"] = self.maximum
        if self.observed is not None:
            extensions["observed"] = self.observed
        return extensions

    def to_graphql_error(self) -> GraphQLError:
        """Return a GraphQL error without locations, paths, or source text."""
        return GraphQLError(self.message, extensions=self.to_extensions())

    def to_response_error(self) -> dict[str, object]:
        """Return the serialized GraphQL error entry for an HTTP response."""
        return {"message": self.message, "extensions": self.to_extensions()}


class GraphQLLimitError(Exception):
    """Raised by transport hooks when a request is rejected before execution."""

    def __init__(self, violation: GraphQLLimitViolation) -> None:
        self.violation = violation
        super().__init__(violation.code)


def _violation(
    limits: GraphQLLimits,
    category: GraphQLLimitCategory,
    observed: Optional[int],
) -> GraphQLLimitViolation:
    return GraphQLLimitViolation(
        category=category,
        maximum=limits.ceiling(category),
        observed=observed,
    )


def deadline_violation(limits: GraphQLLimits) -> GraphQLLimitViolation:
    """Return the violation reported when the execution deadline expires."""
    return _violation(limits, GraphQLLimitCategory.DEADLINE, None)


def batch_violation() -> GraphQLLimitViolation:
    """Return the violation reported for a batched operation array."""
    return GraphQLLimitViolation(category=GraphQLLimitCategory.BATCH)


def variable_depth_violation(limits: GraphQLLimits) -> GraphQLLimitViolation:
    """Return the violation for a request body nested beyond the JSON decoder."""
    return _violation(
        limits,
        GraphQLLimitCategory.VARIABLE_DEPTH,
        limits.max_variable_depth + 1,
    )


# Request-level budgets -------------------------------------------------------


def check_request_budget(
    query: Optional[str],
    variables: Optional[Mapping[str, Any]],
    limits: GraphQLLimits = DEFAULT_GRAPHQL_LIMITS,
) -> Optional[GraphQLLimitViolation]:
    """Check document bytes and supplied variables before parsing.

    Args:
        query: GraphQL document text, if any.
        variables: Decoded variables object, if any.
        limits: Budgets to enforce.

    Returns:
        The first violated budget, or ``None`` when the request fits.
    """
    if query is not None:
        document_bytes = len(query.encode("utf-8", errors="surrogatepass"))
        if document_bytes > limits.max_document_bytes:
            return _violation(
                limits, GraphQLLimitCategory.DOCUMENT_BYTES, document_bytes
            )
    if variables is None:
        return None
    if isinstance(variables, Mapping) and len(variables) > limits.max_variables:
        return _violation(limits, GraphQLLimitCategory.VARIABLES, len(variables))
    return _check_variable_values(variables, limits)


def compact_json_size(value: Any) -> tuple[int, int]:
    """Measure a decoded JSON value without recursion.

    Args:
        value: Decoded JSON value.

    Returns:
        ``(size, depth)`` where ``size`` equals the UTF-8 length of
        ``json.dumps(value, ensure_ascii=False, separators=(",", ":"))`` and
        ``depth`` is the container nesting (a scalar has depth 0).
    """
    size, depth, _ = _walk_json(value)
    return size, depth


def _check_variable_values(
    variables: Any,
    limits: GraphQLLimits,
) -> Optional[GraphQLLimitViolation]:
    size, depth, category = _walk_json(
        variables,
        max_depth=limits.max_variable_depth,
        max_bytes=limits.max_variable_bytes,
    )
    if category is GraphQLLimitCategory.VARIABLE_DEPTH:
        return _violation(limits, category, depth)
    if category is GraphQLLimitCategory.VARIABLE_BYTES:
        return _violation(limits, category, size)
    return None


def _walk_json(
    value: Any,
    *,
    max_depth: Optional[int] = None,
    max_bytes: Optional[int] = None,
) -> tuple[int, int, Optional[GraphQLLimitCategory]]:
    """Return compact size, depth, and the first ceiling crossed, if any."""
    size = 0
    deepest = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, Mapping):
            deepest = max(deepest, depth)
            size += 2 + max(len(item) - 1, 0)
            for key, child in item.items():
                size += _scalar_size(str(key)) + 1
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            deepest = max(deepest, depth)
            size += 2 + max(len(item) - 1, 0)
            stack.extend((child, depth + 1) for child in item)
        else:
            size += _scalar_size(item)
        if max_depth is not None and deepest > max_depth:
            return size, deepest, GraphQLLimitCategory.VARIABLE_DEPTH
        if max_bytes is not None and size > max_bytes:
            return size, deepest, GraphQLLimitCategory.VARIABLE_BYTES
    return size, deepest, None


def _scalar_size(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return len(encoded.encode("utf-8", errors="surrogatepass"))


# Lexical budgets -------------------------------------------------------------

_OPENING_TOKENS: Final = frozenset(
    {TokenKind.BRACE_L, TokenKind.BRACKET_L, TokenKind.PAREN_L}
)
_CLOSING_TOKENS: Final = frozenset(
    {TokenKind.BRACE_R, TokenKind.BRACKET_R, TokenKind.PAREN_R}
)


def check_document_text(
    query: str,
    limits: GraphQLLimits = DEFAULT_GRAPHQL_LIMITS,
) -> Optional[GraphQLLimitViolation]:
    """Bound tokens and bracket nesting before the recursive parser runs.

    The scan stops at the first token above either ceiling. Lexical errors are
    left to the parser, which reports them as ordinary syntax errors.

    Args:
        query: GraphQL document text.
        limits: Budgets to enforce.

    Returns:
        The first violated budget, or ``None`` when the text fits.
    """
    lexer = Lexer(Source(query))
    tokens = 0
    nesting = 0
    try:
        while True:
            token = lexer.advance()
            if token.kind is TokenKind.EOF:
                return None
            tokens += 1
            if tokens > limits.max_tokens:
                return _violation(limits, GraphQLLimitCategory.TOKENS, tokens)
            if token.kind in _OPENING_TOKENS:
                nesting += 1
                if nesting > limits.max_nesting:
                    return _violation(limits, GraphQLLimitCategory.NESTING, nesting)
            elif token.kind in _CLOSING_TOKENS:
                nesting = max(nesting - 1, 0)
    except GraphQLSyntaxError:
        return None


# Structural budgets ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphQLDocumentProfile:
    """Aggregate, value-free shape of one parsed GraphQL document.

    Expanded counts follow every fragment spread with multiplicity and saturate
    at :data:`SATURATED_COUNT`. Fragments that no operation uses add to
    ``nodes`` and ``fragments`` but not to the expanded counts.

    Attributes:
        nodes: Syntax-tree nodes in the document.
        operations: Operation definitions.
        fragments: Fragment definitions.
        fragment_cycles: Fragment spreads that close a cycle.
        variables: Variable definitions across all operations.
        depth: Deepest field nesting of any operation after expansion.
        aliases: Aliased fields across all operations after expansion.
        selected_fields: Field selections across all operations after expansion.
    """

    nodes: int
    operations: int
    fragments: int
    fragment_cycles: int
    variables: int
    depth: int
    aliases: int
    selected_fields: int


@dataclass
class _SelectionSummary:
    fields: int = 0
    aliases: int = 0
    depth: int = 0
    spreads: tuple[tuple[str, int], ...] = ()


def profile_document(document: DocumentNode) -> GraphQLDocumentProfile:
    """Measure a parsed document with iterative, cycle-safe traversals.

    Args:
        document: Parsed GraphQL document.

    Returns:
        Aggregate counts for every structural budget.
    """
    operations: list[_SelectionSummary] = []
    variables = 0
    fragment_count = 0
    fragments: dict[str, _SelectionSummary] = {}
    for definition in document.definitions:
        if isinstance(definition, OperationDefinitionNode):
            operations.append(_summarize(definition.selection_set))
            variables += len(definition.variable_definitions or ())
        elif isinstance(definition, FragmentDefinitionNode):
            fragment_count += 1
            fragments.setdefault(
                definition.name.value, _summarize(definition.selection_set)
            )

    order, cycles = _fragment_post_order(fragments)
    expanded: dict[str, tuple[int, int, int]] = {}
    for name in order:
        expanded[name] = _expand(fragments[name], expanded)

    depth = 0
    aliases = 0
    selected_fields = 0
    for operation in operations:
        op_fields, op_aliases, op_depth = _expand(operation, expanded)
        selected_fields = _saturating_add(selected_fields, op_fields)
        aliases = _saturating_add(aliases, op_aliases)
        depth = max(depth, op_depth)

    return GraphQLDocumentProfile(
        nodes=_count_nodes(document),
        operations=len(operations),
        fragments=fragment_count,
        fragment_cycles=cycles,
        variables=variables,
        depth=depth,
        aliases=aliases,
        selected_fields=selected_fields,
    )


_STRUCTURAL_CHECKS: Final = (
    (GraphQLLimitCategory.NODES, "nodes"),
    (GraphQLLimitCategory.OPERATIONS, "operations"),
    (GraphQLLimitCategory.FRAGMENTS, "fragments"),
    (GraphQLLimitCategory.FRAGMENT_CYCLES, "fragment_cycles"),
    (GraphQLLimitCategory.VARIABLES, "variables"),
    (GraphQLLimitCategory.DEPTH, "depth"),
    (GraphQLLimitCategory.ALIASES, "aliases"),
    (GraphQLLimitCategory.SELECTED_FIELDS, "selected_fields"),
)


def check_document(
    document: DocumentNode,
    limits: GraphQLLimits = DEFAULT_GRAPHQL_LIMITS,
) -> Optional[GraphQLLimitViolation]:
    """Check structural budgets in a fixed, deterministic order.

    Args:
        document: Parsed GraphQL document.
        limits: Budgets to enforce.

    Returns:
        The first violated budget, or ``None`` when the document fits.
    """
    profile = profile_document(document)
    for category, attribute in _STRUCTURAL_CHECKS:
        observed = getattr(profile, attribute)
        ceiling = limits.ceiling(category)
        if ceiling is not None and observed > ceiling:
            return _violation(limits, category, observed)
    return None


def _summarize(selection_set: Optional[SelectionSetNode]) -> _SelectionSummary:
    summary = _SelectionSummary()
    spreads: list[tuple[str, int]] = []
    stack: list[tuple[Optional[SelectionSetNode], int]] = [(selection_set, 0)]
    while stack:
        current, depth = stack.pop()
        if current is None:
            continue
        for selection in current.selections:
            if isinstance(selection, FieldNode):
                summary.fields += 1
                if selection.alias is not None:
                    summary.aliases += 1
                summary.depth = max(summary.depth, depth + 1)
                stack.append((selection.selection_set, depth + 1))
            elif isinstance(selection, InlineFragmentNode):
                stack.append((selection.selection_set, depth))
            elif isinstance(selection, FragmentSpreadNode):
                spreads.append((selection.name.value, depth))
    summary.spreads = tuple(spreads)
    return summary


def _fragment_post_order(
    fragments: Mapping[str, _SelectionSummary],
) -> tuple[list[str], int]:
    """Return fragments children-first plus the number of cycle-closing spreads."""
    white, gray, black = 0, 1, 2
    state = dict.fromkeys(fragments, white)
    order: list[str] = []
    cycles = 0
    for root in fragments:
        if state[root] != white:
            continue
        state[root] = gray
        stack: list[tuple[str, Iterator[tuple[str, int]]]] = [
            (root, iter(fragments[root].spreads))
        ]
        while stack:
            name, children = stack[-1]
            for child, _depth in children:
                child_state = state.get(child)
                if child_state == white:
                    state[child] = gray
                    stack.append((child, iter(fragments[child].spreads)))
                    break
                if child_state == gray:
                    cycles += 1
            else:
                stack.pop()
                state[name] = black
                order.append(name)
    return order, cycles


def _expand(
    summary: _SelectionSummary,
    expanded: Mapping[str, tuple[int, int, int]],
) -> tuple[int, int, int]:
    total_fields = summary.fields
    total_aliases = summary.aliases
    depth = summary.depth
    for name, spread_depth in summary.spreads:
        child = expanded.get(name)
        if child is None:
            # Unknown fragments are reported by standard validation; spreads
            # that close a cycle are rejected before expansion matters.
            continue
        child_fields, child_aliases, child_depth = child
        total_fields = _saturating_add(total_fields, child_fields)
        total_aliases = _saturating_add(total_aliases, child_aliases)
        depth = max(depth, spread_depth + child_depth)
    return total_fields, total_aliases, depth


def _saturating_add(left: int, right: int) -> int:
    return min(left + right, SATURATED_COUNT)


def _count_nodes(document: DocumentNode) -> int:
    count = 0
    stack: list[Node] = [document]
    while stack:
        node = stack.pop()
        count += 1
        for key in node.keys:
            if key == "loc":
                continue
            value = getattr(node, key, None)
            if isinstance(value, Node):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(item for item in value if isinstance(item, Node))
    return count


# Strawberry integration ------------------------------------------------------


def graphql_limits_for_request(request: Any) -> GraphQLLimits:
    """Return the limits attached to the service app, or the safe defaults.

    Args:
        request: Starlette request (or any object exposing ``app.state``).

    Returns:
        The app's ``graphql_limits`` when configured, else the defaults.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    limits = getattr(state, "graphql_limits", None)
    return limits if isinstance(limits, GraphQLLimits) else DEFAULT_GRAPHQL_LIMITS


def record_graphql_rejection(request: Any, violation: GraphQLLimitViolation) -> None:
    """Count one rejection on the service metrics registry, when enabled.

    Args:
        request: Starlette request (or any object exposing ``app.state``).
        violation: Rejected budget; only its category becomes a label.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    recorder = getattr(
        getattr(state, "metrics", None), "record_graphql_rejection", None
    )
    if callable(recorder):
        recorder(reason=violation.category.value)


class GraphQLLimitsExtension(SchemaExtension):
    """Enforce :class:`GraphQLLimits` before parsing, validation, and resolvers.

    Register this extension ahead of other validation extensions so oversized
    documents are rejected before any other rule walks them.
    """

    def _limits(self) -> GraphQLLimits:
        return graphql_limits_for_request(self._service_request())

    def _service_request(self) -> Any:
        return getattr(self.execution_context.context, "request", None)

    def _reject(self, violation: Optional[GraphQLLimitViolation]) -> None:
        if violation is None:
            return
        record_graphql_rejection(self._service_request(), violation)
        raise violation.to_graphql_error()

    def on_operation(self) -> Iterator[None]:
        """Check document bytes and supplied variables before parsing."""
        context = self.execution_context
        limits = self._limits()
        self._reject(check_request_budget(context.query, context.variables, limits))
        yield

    def on_parse(self) -> Iterator[None]:
        """Check tokens and nesting before the recursive parser runs."""
        context = self.execution_context
        if context.graphql_document is None and context.query:
            self._reject(check_document_text(context.query, self._limits()))
        yield

    def on_validate(self) -> Iterator[None]:
        """Check structural budgets before graphql-core validation rules run."""
        document = self.execution_context.graphql_document
        if document is not None:
            self._reject(check_document(document, self._limits()))
        yield


__all__ = [
    "DEFAULT_GRAPHQL_LIMITS",
    "GRAPHQL_BATCH_UNSUPPORTED_CODE",
    "GRAPHQL_LIMITS_COMPATIBILITY",
    "GRAPHQL_LIMITS_SCHEMA_VERSION",
    "GRAPHQL_LIMIT_ENV_VARS",
    "GRAPHQL_LIMIT_EXCEEDED_CODE",
    "GRAPHQL_TIMEOUT_CODE",
    "SATURATED_COUNT",
    "GraphQLDocumentProfile",
    "GraphQLLimitCategory",
    "GraphQLLimitError",
    "GraphQLLimitViolation",
    "GraphQLLimits",
    "GraphQLLimitsExtension",
    "batch_violation",
    "check_document",
    "check_document_text",
    "check_request_budget",
    "compact_json_size",
    "deadline_violation",
    "graphql_limits_for_request",
    "graphql_limits_from_env",
    "profile_document",
    "record_graphql_rejection",
    "variable_depth_violation",
]
