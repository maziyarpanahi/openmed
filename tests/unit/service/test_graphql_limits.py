"""GraphQL document, variable, batch, and execution budget tests.

Every document in this module is synthetic. Sentinels stand in for clinical
text so tests can prove rejected documents and variables are never echoed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from graphql import (
    build_client_schema,
    get_introspection_query,
    parse,
)
from hypothesis import given, settings
from hypothesis import strategies as st

import openmed
from openmed.processing.outputs import PredictionResult
from openmed.service import graphql_schema
from openmed.service import runtime as service_runtime
from openmed.service.app import _SERVICE_WORK_TASKS, create_app
from openmed.service.graphql_limits import (
    DEFAULT_GRAPHQL_LIMITS,
    GRAPHQL_BATCH_UNSUPPORTED_CODE,
    GRAPHQL_LIMIT_ENV_VARS,
    GRAPHQL_LIMIT_EXCEEDED_CODE,
    GRAPHQL_LIMITS_SCHEMA_VERSION,
    GRAPHQL_TIMEOUT_CODE,
    SATURATED_COUNT,
    GraphQLLimitCategory,
    GraphQLLimits,
    GraphQLLimitViolation,
    check_document,
    check_document_text,
    check_request_budget,
    compact_json_size,
    graphql_limits_from_env,
    profile_document,
)
from openmed.service.metrics import (
    GRAPHQL_REJECTION_NAME,
    GRAPHQL_REJECTION_REASONS,
    PrometheusMetricsRegistry,
)

LOOPBACK_BASE_URL = "http://127.0.0.1"
SENTINEL = "SYNTHETIC-PHI-3545"
LIMIT_MESSAGE = "The GraphQL request exceeds a configured limit."
# Bounds the wait for a model worker thread to start on a loaded runner.
_WORKER_START_WAIT_SECONDS = 5
_DEADLINE_SECONDS = 30.0

_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_KEEP_ALIVE",
    "OPENMED_SERVICE_MAX_TEXT_LENGTH",
    "OPENMED_SERVICE_BATCHING_ENABLED",
    "OPENMED_SERVICE_COALESCING_ENABLED",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    "OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY",
    "OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS",
    "OPENMED_SERVICE_THROTTLE_KEY",
    "OPENMED_SERVICE_METRICS_ENABLED",
    "OPENMED_SERVICE_RETRY_MAX_ATTEMPTS",
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_CORS_ORIGINS",
    "OPENMED_SERVICE_TRUSTED_HOSTS",
    *GRAPHQL_LIMIT_ENV_VARS.values(),
)


class FakeLoader:
    """Offline model loader double; it never touches model weights."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: Any) -> object:
        del model_name
        return object()

    def loaded_models(self) -> dict[str, Any]:
        return {}


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep service startup deterministic, offline, and on default limits."""
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in _SERVICE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


@pytest.fixture
def resolver_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every model-backed resolver entry and model invocation."""
    calls: list[str] = []
    original_analyze = graphql_schema._run_analyze
    original_deidentify = graphql_schema._run_deidentify

    async def run_analyze(runtime: Any, payload: Any) -> Any:
        calls.append("analyze")
        return await original_analyze(runtime, payload)

    async def run_deidentify(runtime: Any, payload: Any) -> Any:
        calls.append("deidentify")
        return await original_deidentify(runtime, payload)

    def analyze_text(text: str, **_: Any) -> PredictionResult:
        calls.append("model")
        return _prediction_result(text)

    monkeypatch.setattr(graphql_schema, "_run_analyze", run_analyze)
    monkeypatch.setattr(graphql_schema, "_run_deidentify", run_deidentify)
    monkeypatch.setattr(openmed, "analyze_text", analyze_text)
    return calls


def _prediction_result(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name="disease_detection_superclinical",
        timestamp=datetime(2026, 11, 7).isoformat(),
        processing_time=0.01,
    )


def _nested_fields(depth: int) -> str:
    return "query { " + "f { " * (depth - 1) + "f" + " }" * (depth - 1) + " }"


def _limit_extensions(
    category: GraphQLLimitCategory, maximum: Any, observed: int
) -> dict[str, Any]:
    return {
        "code": GRAPHQL_LIMIT_EXCEEDED_CODE,
        "limit": category.value,
        "maximum": maximum,
        "observed": observed,
    }


# Contract and configuration ---------------------------------------------------


def test_default_limits_are_a_versioned_contract() -> None:
    limits = graphql_limits_from_env({})

    assert limits == DEFAULT_GRAPHQL_LIMITS == GraphQLLimits()
    payload = limits.to_dict()
    assert payload["schema_version"] == GRAPHQL_LIMITS_SCHEMA_VERSION
    assert payload["compatibility_policy"] == "same_major"
    assert set(GRAPHQL_LIMIT_ENV_VARS) <= set(payload)
    for name, env_var in GRAPHQL_LIMIT_ENV_VARS.items():
        assert env_var.startswith("OPENMED_SERVICE_GRAPHQL_")
        assert isinstance(payload[name], (int, float))
    assert limits.ceiling(GraphQLLimitCategory.FRAGMENT_CYCLES) == 0
    assert limits.ceiling(GraphQLLimitCategory.BATCH) is None
    assert limits.ceiling(GraphQLLimitCategory.DEADLINE) == limits.deadline_seconds


def test_env_overrides_apply_and_invalid_values_fail_closed(subtests) -> None:
    limits = graphql_limits_from_env(
        {
            "OPENMED_SERVICE_GRAPHQL_MAX_DEPTH": " 7 ",
            "OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS": "2.5",
            "OPENMED_SERVICE_GRAPHQL_MAX_ALIASES": "",
        }
    )
    assert limits.max_depth == 7
    assert limits.deadline_seconds == 2.5
    assert limits.max_aliases == DEFAULT_GRAPHQL_LIMITS.max_aliases

    for env_var, raw in [
        ("OPENMED_SERVICE_GRAPHQL_MAX_DEPTH", f"{SENTINEL}"),
        ("OPENMED_SERVICE_GRAPHQL_MAX_DEPTH", "0"),
        ("OPENMED_SERVICE_GRAPHQL_MAX_DEPTH", "129"),
        ("OPENMED_SERVICE_GRAPHQL_MAX_NESTING", "1.5"),
        ("OPENMED_SERVICE_GRAPHQL_MAX_TOKENS", "-1"),
        ("OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS", "0"),
        ("OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS", "nan"),
        ("OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS", "inf"),
        ("OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS", f"{SENTINEL}"),
    ]:
        with subtests.test(env_var=env_var, raw=raw):
            with pytest.raises(ValueError) as error:
                graphql_limits_from_env({env_var: raw})
            assert env_var in str(error.value)
            assert raw not in str(error.value)


def test_limits_reject_unsafe_ceilings_and_unknown_versions(subtests) -> None:
    for overrides in [
        {"max_depth": True},
        {"max_nesting": 129},
        {"max_selected_fields": 0},
        {"deadline_seconds": -1.0},
        {"deadline_seconds": float("inf")},
        {"schema_version": "2.0.0"},
        {"compatibility_policy": "any"},
    ]:
        with subtests.test(overrides=overrides):
            with pytest.raises(ValueError):
                GraphQLLimits(**overrides)


def test_service_loads_limits_from_env_and_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_GRAPHQL_MAX_DEPTH", "9")
    assert create_app().state.graphql_limits.max_depth == 9

    monkeypatch.setenv("OPENMED_SERVICE_GRAPHQL_MAX_DEPTH", SENTINEL)
    with pytest.raises(ValueError) as error:
        create_app()
    assert SENTINEL not in str(error.value)


def test_rejection_metric_vocabulary_matches_limit_categories() -> None:
    assert GRAPHQL_REJECTION_REASONS == {
        category.value for category in GraphQLLimitCategory
    }
    registry = PrometheusMetricsRegistry()
    registry.record_graphql_rejection(reason="depth")
    registry.record_graphql_rejection(reason=SENTINEL)
    registry.record_graphql_rejection(reason="depth", count=0)

    rendered = registry.render()
    assert f"# TYPE {GRAPHQL_REJECTION_NAME} counter" in rendered
    assert f'{GRAPHQL_REJECTION_NAME}{{reason="depth"}} 1' in rendered
    assert f'{GRAPHQL_REJECTION_NAME}{{reason="other"}} 1' in rendered
    assert SENTINEL not in rendered


def test_violations_expose_only_category_ceiling_and_count() -> None:
    violation = GraphQLLimitViolation(
        category=GraphQLLimitCategory.ALIASES, maximum=16, observed=40
    )
    error = violation.to_graphql_error()

    assert error.message == LIMIT_MESSAGE
    assert error.extensions == _limit_extensions(GraphQLLimitCategory.ALIASES, 16, 40)
    assert error.nodes is None and error.source is None and error.path is None
    assert GraphQLLimitViolation(GraphQLLimitCategory.BATCH).to_extensions() == {
        "code": GRAPHQL_BATCH_UNSUPPORTED_CODE,
        "limit": "batch",
    }
    assert (
        GraphQLLimitViolation(GraphQLLimitCategory.DEADLINE, maximum=1.0).code
        == GRAPHQL_TIMEOUT_CODE
    )
    with pytest.raises(ValueError):
        GraphQLLimitViolation(GraphQLLimitCategory.DEPTH, observed=-1)


# Pure document checks ---------------------------------------------------------


def test_standard_introspection_fits_the_default_budgets() -> None:
    query = get_introspection_query(
        descriptions=True,
        specified_by_url=True,
        directive_is_repeatable=True,
        schema_description=True,
        input_value_deprecation=True,
    )

    assert check_request_budget(query, None) is None
    assert check_document_text(query) is None
    assert check_document(parse(query)) is None


def test_document_bytes_are_measured_as_utf8() -> None:
    limits = GraphQLLimits(max_document_bytes=16)
    query = "{ a }  # éééééé"

    violation = check_request_budget(query, None, limits)

    assert len(query) <= 16
    assert violation == GraphQLLimitViolation(
        GraphQLLimitCategory.DOCUMENT_BYTES, 16, len(query.encode("utf-8"))
    )


def test_token_scan_stops_at_the_first_token_over_the_ceiling() -> None:
    limits = GraphQLLimits(max_tokens=10)
    commented = "# " + "comment " * 1_000 + "\n{ a b c d e f g h }"
    oversized = "{ " + "a " * 10_000 + "}"

    assert check_document_text(commented, limits) is None
    assert check_document_text(oversized, limits) == GraphQLLimitViolation(
        GraphQLLimitCategory.TOKENS, 10, 11
    )
    assert check_document_text("{ a ", limits) is None  # left to the parser


def test_deep_nesting_is_rejected_before_the_recursive_parser(subtests) -> None:
    limits = GraphQLLimits(max_nesting=8, max_tokens=1_000_000)
    for name, query in [
        ("selections", "{ " + "f { " * 20_000 + "f" + " }" * 20_000 + " }"),
        ("list literal", "{ f(x: " + "[" * 20_000 + "1" + "]" * 20_000 + ") }"),
        ("object literal", "{ f(x: " + "{a: " * 20_000 + "1" + "}" * 20_000 + ") }"),
    ]:
        with subtests.test(shape=name):
            assert check_document_text(query, limits) == GraphQLLimitViolation(
                GraphQLLimitCategory.NESTING, 8, 9
            )


def test_depth_counts_fields_through_fragments_and_inline_fragments() -> None:
    document = parse(
        """
        query { a { ...One } }
        fragment One on T { b { ... on T { c { ...Two } } } }
        fragment Two on T { d { e } }
        """
    )

    assert profile_document(document).depth == 5
    assert check_document(document, GraphQLLimits(max_depth=5)) is None
    assert check_document(document, GraphQLLimits(max_depth=4)) == (
        GraphQLLimitViolation(GraphQLLimitCategory.DEPTH, 4, 5)
    )


def test_alias_amplification_is_counted_after_fragment_expansion() -> None:
    aliased = " ".join(f"a{i}: entityTypes {{ label }}" for i in range(4))
    spreads = " ".join("...Aliased" for _ in range(5))
    document = parse(f"query {{ {spreads} }} fragment Aliased on Query {{ {aliased} }}")

    profile = profile_document(document)

    assert profile.aliases == 20
    assert profile.selected_fields == 40
    assert check_document(document) == GraphQLLimitViolation(
        GraphQLLimitCategory.ALIASES, DEFAULT_GRAPHQL_LIMITS.max_aliases, 20
    )


def test_fragment_cycles_are_rejected_without_recursion(subtests) -> None:
    chain = 5_000
    long_cycle = "query { ...F0 } " + " ".join(
        f"fragment F{i} on Query {{ ...F{(i + 1) % chain} }}" for i in range(chain)
    )
    roomy = GraphQLLimits(max_fragments=10_000, max_nodes=1_000_000)
    for name, query, cycles in [
        ("self", "query { ...A } fragment A on Query { ...A }", 1),
        (
            "mutual",
            "query { ...A } fragment A on Query { ...B } "
            "fragment B on Query { entityTypes { label } ...A }",
            1,
        ),
        (
            "inline",
            "query { ...A } fragment A on Query { ... on Query { f { ...A } } }",
            1,
        ),
        ("unused", "query { a } fragment A on Query { ...A ...A }", 2),
        ("long chain", long_cycle, 1),
    ]:
        with subtests.test(shape=name):
            document = parse(query)
            assert profile_document(document).fragment_cycles == cycles
            assert check_document(document, roomy) == GraphQLLimitViolation(
                GraphQLLimitCategory.FRAGMENT_CYCLES, 0, cycles
            )


def test_fragment_expansion_bombs_saturate_instead_of_exploding() -> None:
    fragments = " ".join(
        f"fragment F{i} on Query {{ ...F{i + 1} ...F{i + 1} }}" for i in range(60)
    )
    document = parse(
        "query { ...F0 } "
        + fragments
        + " fragment F60 on Query { entityTypes { label } }"
    )

    profile = profile_document(document)

    assert profile.selected_fields == SATURATED_COUNT
    assert profile.depth == 2
    assert check_document(document) == GraphQLLimitViolation(
        GraphQLLimitCategory.SELECTED_FIELDS,
        DEFAULT_GRAPHQL_LIMITS.max_selected_fields,
        SATURATED_COUNT,
    )


def test_structural_counts_follow_a_fixed_order(subtests) -> None:
    two_operations = "query A { a } query B { b } query C { c }"
    for name, query, limits, expected in [
        (
            "operations",
            two_operations,
            GraphQLLimits(max_operations=2),
            GraphQLLimitViolation(GraphQLLimitCategory.OPERATIONS, 2, 3),
        ),
        (
            "fragments",
            "query { ...A } fragment A on Q { a } fragment B on Q { b }",
            GraphQLLimits(max_fragments=1),
            GraphQLLimitViolation(GraphQLLimitCategory.FRAGMENTS, 1, 2),
        ),
        (
            "declared variables",
            "query($a: Int, $b: Int, $c: Int) { f(a: $a, b: $b, c: $c) }",
            GraphQLLimits(max_variables=2),
            GraphQLLimitViolation(GraphQLLimitCategory.VARIABLES, 2, 3),
        ),
        (
            "nodes before operations",
            two_operations,
            GraphQLLimits(max_nodes=5, max_operations=1),
            GraphQLLimitViolation(GraphQLLimitCategory.NODES, 5, 16),
        ),
        (
            "selected fields",
            "query { a b c d }",
            GraphQLLimits(max_selected_fields=3),
            GraphQLLimitViolation(GraphQLLimitCategory.SELECTED_FIELDS, 3, 4),
        ),
    ]:
        with subtests.test(shape=name):
            assert check_document(parse(query), limits) == expected
            assert check_document(parse(query), limits) == expected


def test_variable_count_size_and_depth_are_bounded_before_parsing(subtests) -> None:
    limits = GraphQLLimits(max_variables=2, max_variable_bytes=64, max_variable_depth=3)
    for name, variables, expected in [
        (
            "count",
            {"a": 1, "b": 2, "c": 3},
            GraphQLLimitViolation(GraphQLLimitCategory.VARIABLES, 2, 3),
        ),
        (
            "bytes",
            {"input": {"text": SENTINEL * 10}},
            GraphQLLimitViolation(
                GraphQLLimitCategory.VARIABLE_BYTES,
                64,
                _compact_size({"input": {"text": SENTINEL * 10}}),
            ),
        ),
        (
            "depth",
            {"input": {"keepAlive": {"nested": [SENTINEL]}}},
            GraphQLLimitViolation(GraphQLLimitCategory.VARIABLE_DEPTH, 3, 4),
        ),
    ]:
        with subtests.test(shape=name):
            violation = check_request_budget("{ a }", variables, limits)
            assert violation == expected
            assert SENTINEL not in json.dumps(violation.to_extensions())
    assert check_request_budget("{ a }", {"input": {"text": "ok"}}, limits) is None


def test_compact_json_size_matches_the_compact_encoder() -> None:
    value = {"a": [1, 2.5, None, True, 'é\n"', {}], "b": {"c": []}}
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    assert compact_json_size(value) == (len(encoded.encode("utf-8")), 3)
    assert compact_json_size("scalar") == (8, 0)


# Property tests ---------------------------------------------------------------

_JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=30,
)


@pytest.mark.fuzz
@settings(max_examples=300, deadline=None)
@given(value=_JSON_VALUES)
def test_property_compact_json_size_equals_encoded_length(value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    size, _ = compact_json_size(value)

    assert size == len(encoded.encode("utf-8", errors="surrogatepass"))


@pytest.mark.fuzz
@settings(max_examples=200, deadline=None)
@given(depth=st.integers(min_value=1, max_value=40), ceiling=st.integers(1, 40))
def test_property_depth_decision_matches_nesting(depth: int, ceiling: int) -> None:
    limits = GraphQLLimits(max_depth=ceiling, max_nesting=64)
    query = _nested_fields(depth)

    assert check_document_text(query, limits) is None
    violation = check_document(parse(query), limits)

    assert profile_document(parse(query)).depth == depth
    if depth <= ceiling:
        assert violation is None
    else:
        assert violation == GraphQLLimitViolation(
            GraphQLLimitCategory.DEPTH, ceiling, depth
        )


@pytest.mark.fuzz
@settings(max_examples=200, deadline=None)
@given(
    local=st.integers(min_value=0, max_value=6),
    spreads=st.integers(min_value=0, max_value=6),
)
def test_property_fragment_expansion_multiplies_aliases(
    local: int, spreads: int
) -> None:
    aliased = " ".join(f"x{i}: f" for i in range(local)) or "f"
    spread_text = " ".join("...A" for _ in range(spreads)) or "g"
    document = parse(f"query {{ {spread_text} }} fragment A on Q {{ {aliased} }}")

    profile = profile_document(document)

    assert profile.aliases == local * spreads
    assert profile.selected_fields == (max(local, 1) * spreads or 1)


# HTTP surface -----------------------------------------------------------------


def _compact_size(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(encoded.encode("utf-8"))


def _adversarial_requests() -> list[tuple[str, Any, dict[str, Any]]]:
    analyze = "query($input: AnalyzeInput!) { analyze(input: $input) { text } }"
    oversized_document = "{ entityTypes { label } } # " + SENTINEL * 400
    oversized_variables = {"input": {"text": SENTINEL * 100}}
    deep_variables = json.loads("[" * 12 + json.dumps(SENTINEL) + "]" * 12)
    cycle = (
        "query { ...A } fragment A on Query { ...B "
        f'analyze(input: {{text: "{SENTINEL}"}}) {{ text }} }} '
        "fragment B on Query { ...A }"
    )
    bomb = " ".join(
        f"fragment F{i} on Query {{ ...F{i + 1} ...F{i + 1} }}" for i in range(20)
    )
    aliases = " ".join(
        f'a{i}: analyze(input: {{text: "{SENTINEL}"}}) {{ text }}' for i in range(17)
    )
    operations = " ".join(f"query Q{i} {{ entityTypes {{ label }} }}" for i in range(9))
    return [
        (
            "oversized document",
            {"query": oversized_document},
            _limit_extensions(
                GraphQLLimitCategory.DOCUMENT_BYTES,
                4096,
                len(oversized_document.encode("utf-8")),
            ),
        ),
        (
            "too many tokens",
            {"query": "{ " + "a " * 600 + "}"},
            _limit_extensions(GraphQLLimitCategory.TOKENS, 500, 501),
        ),
        (
            "excessive nesting",
            {"query": "{ " + "a { " * 40 + "b" + " }" * 40 + " }"},
            _limit_extensions(GraphQLLimitCategory.NESTING, 32, 33),
        ),
        (
            "too many operations",
            {"query": operations, "operationName": "Q0"},
            _limit_extensions(GraphQLLimitCategory.OPERATIONS, 8, 9),
        ),
        (
            "fragment cycle",
            {"query": cycle},
            _limit_extensions(GraphQLLimitCategory.FRAGMENT_CYCLES, 0, 1),
        ),
        (
            "excessive depth",
            {"query": _nested_fields(21)},
            _limit_extensions(GraphQLLimitCategory.DEPTH, 20, 21),
        ),
        (
            "alias amplification",
            {"query": "query { " + aliases + " }"},
            _limit_extensions(GraphQLLimitCategory.ALIASES, 16, 17),
        ),
        (
            "fragment bomb",
            {
                "query": "query { ...F0 } "
                + bomb
                + ' fragment F20 on Query { analyze(input: {text: "x"}) { text } }'
            },
            _limit_extensions(GraphQLLimitCategory.SELECTED_FIELDS, 1000, 2**21),
        ),
        (
            "too many variables",
            {"query": analyze, "variables": {f"v{i}": SENTINEL for i in range(33)}},
            _limit_extensions(GraphQLLimitCategory.VARIABLES, 32, 33),
        ),
        (
            "oversized variable",
            {"query": analyze, "variables": oversized_variables},
            _limit_extensions(
                GraphQLLimitCategory.VARIABLE_BYTES,
                1024,
                _compact_size(oversized_variables),
            ),
        ),
        (
            "deep variable",
            {
                "query": analyze,
                "variables": {"input": {"text": "x", "keepAlive": deep_variables}},
            },
            _limit_extensions(GraphQLLimitCategory.VARIABLE_DEPTH, 8, 9),
        ),
    ]


def test_adversarial_documents_never_reach_a_resolver_or_model_loader(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    resolver_calls: list[str],
    subtests,
) -> None:
    for name, value in {
        "OPENMED_SERVICE_METRICS_ENABLED": "true",
        "OPENMED_SERVICE_GRAPHQL_MAX_DOCUMENT_BYTES": "4096",
        "OPENMED_SERVICE_GRAPHQL_MAX_TOKENS": "500",
        "OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_BYTES": "1024",
        "OPENMED_SERVICE_GRAPHQL_MAX_VARIABLE_DEPTH": "8",
    }.items():
        monkeypatch.setenv(name, value)
    caplog.set_level(logging.DEBUG)
    app = create_app()
    cases = _adversarial_requests()

    with TestClient(app, base_url=LOOPBACK_BASE_URL) as client:
        for name, body, extensions in cases:
            with subtests.test(case=name):
                response = client.post("/graphql", json=body)
                assert response.status_code == 200
                assert response.json() == {
                    "data": None,
                    "errors": [{"message": LIMIT_MESSAGE, "extensions": extensions}],
                }
                assert SENTINEL not in response.text
        metrics_text = client.get("/metrics").text

    assert resolver_calls == []
    assert app.state.runtime._loader is None
    assert SENTINEL not in caplog.text
    assert SENTINEL not in metrics_text
    for _, _, extensions in cases:
        assert (
            f'{GRAPHQL_REJECTION_NAME}{{reason="{extensions["limit"]}"}}'
            in metrics_text
        )


def test_json_body_nested_past_the_decoder_fails_as_a_variable_limit(
    resolver_calls: list[str],
) -> None:
    nested = "[" * 50_000 + "]" * 50_000
    body = '{"query": "{ entityTypes { label } }", "variables": {"x": ' + nested + "}}"

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        response = client.post(
            "/graphql",
            content=body.encode(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 200
    assert response.json()["errors"][0]["extensions"] == _limit_extensions(
        GraphQLLimitCategory.VARIABLE_DEPTH,
        DEFAULT_GRAPHQL_LIMITS.max_variable_depth,
        DEFAULT_GRAPHQL_LIMITS.max_variable_depth + 1,
    )
    assert resolver_calls == []


def test_batched_operation_arrays_are_rejected_with_a_stable_code(
    monkeypatch: pytest.MonkeyPatch,
    resolver_calls: list[str],
    subtests,
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    operation = {
        "query": "query($input: AnalyzeInput!) { analyze(input: $input) { text } }",
        "variables": {"input": {"text": SENTINEL}},
    }
    expected = {
        "errors": [
            {
                "message": "Batched GraphQL requests are not supported.",
                "extensions": {
                    "code": GRAPHQL_BATCH_UNSUPPORTED_CODE,
                    "limit": "batch",
                },
            }
        ]
    }
    bodies = {
        "empty": b"[]",
        "single": json.dumps([operation]).encode(),
        "pair": json.dumps([operation, operation]).encode(),
        "leading whitespace": b" \r\n\t" + json.dumps([operation]).encode(),
        "byte order mark": b"\xef\xbb\xbf" + json.dumps([operation]).encode(),
    }

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        for name, body in bodies.items():
            with subtests.test(body=name):
                response = client.post(
                    "/graphql",
                    content=body,
                    headers={"content-type": "application/json"},
                )
                assert response.status_code == 400
                assert response.json() == expected
                assert SENTINEL not in response.text
        metrics_text = client.get("/metrics").text

    assert resolver_calls == []
    assert f'{GRAPHQL_REJECTION_NAME}{{reason="batch"}} {len(bodies)}' in metrics_text


def test_normal_operations_and_client_generation_still_work(
    resolver_calls: list[str],
) -> None:
    introspection = get_introspection_query(
        descriptions=True,
        specified_by_url=True,
        directive_is_repeatable=True,
        schema_description=True,
        input_value_deprecation=True,
    )

    with TestClient(create_app(), base_url=LOOPBACK_BASE_URL) as client:
        schema_response = client.post("/graphql", json={"query": introspection})
        analyze_response = client.post(
            "/graphql",
            json={
                "query": """query($input: AnalyzeInput!) {
                    first: analyze(input: $input) { modelName }
                    second: analyze(input: $input) { modelName }
                    entityTypes { label }
                }""",
                "variables": {"input": {"text": "synthetic"}},
            },
        )

    assert schema_response.status_code == 200
    assert "errors" not in schema_response.json()
    client_schema = build_client_schema(schema_response.json()["data"])
    assert set(client_schema.query_type.fields) == {
        "analyze",
        "deidentify",
        "entityTypes",
        "journeyResources",
    }
    assert analyze_response.status_code == 200
    payload = analyze_response.json()
    assert "errors" not in payload
    assert payload["data"]["first"]["modelName"] == "disease_detection_superclinical"
    assert sorted(resolver_calls) == ["analyze", "analyze", "model", "model"]


# Execution deadline -----------------------------------------------------------


def _metric_value(app: Any, prefix: str) -> int:
    for line in app.state.metrics.render().splitlines():
        if line.startswith(prefix):
            return int(line.split()[-1])
    return 0


def _orphaned_work(app: Any) -> int:
    return _metric_value(app, "openmed_service_orphaned_work ")


def _advance_loop_clock(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Make the running loop's pending timers due *seconds* early."""
    loop = asyncio.get_running_loop()
    clock = loop.time
    monkeypatch.setattr(loop, "time", lambda: clock() + seconds)


def _live_service_tasks() -> list[Any]:
    loop = asyncio.get_running_loop()
    return [task for task in _SERVICE_WORK_TASKS if task.get_loop() is loop]


async def _wait_for_work_to_finish(app: Any) -> None:
    async def wait() -> None:
        while app.state.inflight or _orphaned_work(app) or _live_service_tasks():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), 2)


def _analyze_body(text: str) -> dict[str, Any]:
    return {
        "query": "query($input: AnalyzeInput!) { analyze(input: $input) { text } }",
        "variables": {"input": {"text": text}},
    }


def _assert_deadline_response(response: httpx.Response) -> None:
    assert response.status_code == 200
    assert response.json() == {
        "data": None,
        "errors": [
            {
                "message": "The GraphQL request exceeded its execution deadline.",
                "extensions": {
                    "code": GRAPHQL_TIMEOUT_CODE,
                    "limit": "deadline",
                    "maximum": _DEADLINE_SECONDS,
                },
            }
        ],
    }
    assert SENTINEL not in response.text


def _deadline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_RATE_LIMIT_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("OPENMED_SERVICE_CONCURRENCY_WAIT_SECONDS", "0.01")
    monkeypatch.setenv("OPENMED_SERVICE_METRICS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_RETRY_MAX_ATTEMPTS", "1")
    monkeypatch.setenv(
        "OPENMED_SERVICE_GRAPHQL_DEADLINE_SECONDS", str(_DEADLINE_SECONDS)
    )


def test_deadline_cancels_a_stalled_resolver_and_releases_its_permit(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _deadline_env(monkeypatch)
    caplog.set_level(logging.DEBUG)
    cleanup: list[str] = []

    async def stalled_resolver(runtime: Any, payload: Any) -> Any:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup.append("cancelled")

    monkeypatch.setattr(graphql_schema, "_run_analyze", stalled_resolver)
    app = create_app()

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                request = asyncio.create_task(
                    client.post("/graphql", json=_analyze_body(SENTINEL))
                )
                await asyncio.wait_for(started.wait(), _WORKER_START_WAIT_SECONDS)
                _advance_loop_clock(monkeypatch, _DEADLINE_SECONDS)
                _assert_deadline_response(await request)

                assert cleanup == ["cancelled"]
                assert app.state.inflight == 0
                assert _orphaned_work(app) == 0
                assert not _live_service_tasks()
                assert not app.state.throttle.concurrency_gate._semaphores[
                    "global"
                ].locked()
                follow_up = await client.post(
                    "/graphql", json={"query": "{ entityTypes { label } }"}
                )
                assert follow_up.status_code == 200
                assert "errors" not in follow_up.json()

    started = asyncio.Event()
    asyncio.run(scenario())
    assert _metric_value(app, f'{GRAPHQL_REJECTION_NAME}{{reason="deadline"}}') == 1
    assert SENTINEL not in caplog.text


def test_deadline_expiry_leaves_no_lease_task_permit_or_inflight_counter(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _deadline_env(monkeypatch)
    caplog.set_level(logging.DEBUG)
    release = threading.Event()
    started = threading.Event()
    calls: list[str] = []

    def stalled_model(text: str, **_: Any) -> PredictionResult:
        calls.append("model")
        if len(calls) == 1:
            started.set()
            assert release.wait(_WORKER_START_WAIT_SECONDS)
        return _prediction_result(text)

    monkeypatch.setattr(openmed, "analyze_text", stalled_model)
    app = create_app()

    def active_leases() -> int:
        pool = app.state.runtime.get_loader()
        return sum(entry.active_requests for entry in pool._entries.values())

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            app.state.runtime.config.timeout = 0
            transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
            async with httpx.AsyncClient(
                transport=transport, base_url=LOOPBACK_BASE_URL
            ) as client:
                try:
                    request = asyncio.create_task(
                        client.post("/graphql", json=_analyze_body(SENTINEL))
                    )
                    assert await asyncio.to_thread(
                        started.wait, _WORKER_START_WAIT_SECONDS
                    )
                    _advance_loop_clock(monkeypatch, _DEADLINE_SECONDS)
                    _assert_deadline_response(await request)

                    # A worker thread cannot be interrupted, so its permit and
                    # warm-pool lease stay accounted until it returns.
                    assert _orphaned_work(app) == 1
                    assert app.state.inflight == 1
                    assert active_leases() == 1
                    busy = await client.post("/analyze", json={"text": "synthetic"})
                    assert busy.status_code == 503
                finally:
                    release.set()
                    await _wait_for_work_to_finish(app)

                assert active_leases() == 0
                assert not app.state.throttle.concurrency_gate._semaphores[
                    "global"
                ].locked()
                recovered = await client.post("/graphql", json=_analyze_body("ok"))
                assert recovered.status_code == 200
                assert "errors" not in recovered.json()
                await _wait_for_work_to_finish(app)

    asyncio.run(scenario())
    assert calls == ["model", "model"]
    assert _metric_value(app, f'{GRAPHQL_REJECTION_NAME}{{reason="deadline"}}') == 1
    assert SENTINEL not in caplog.text
