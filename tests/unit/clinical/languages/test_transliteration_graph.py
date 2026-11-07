"""Local synthetic terms and independent provenance graph controls."""

import json

import pytest

from openmed.clinical.languages.transliteration_graph import (
    TransliterationGraph,
    TransliterationNode,
    TransliterationPath,
    TransliterationRule,
)


def node(text, language="en", script="Latin"):
    return TransliterationNode.from_text(text, language=language, script=script)


def rule(a, b, source="synthetic_rule", confidence=0.8, reversible=False):
    return TransliterationRule(a, b, source, "a" * 64, confidence, reversible)


def test_distinct_lossy_candidates_and_provenance_are_retained():
    a, b, c = node("SyntheticA", "de"), node("SyntheticB"), node("SyntheticC")
    graph = TransliterationGraph([rule(a, b), rule(a, c, "alternate_rule", 0.9)])
    paths = graph.candidates(a, language="en", script="Latin")
    assert [p.nodes[-1] for p in paths] == [c, b]
    assert [p.rules[0].rule_source for p in paths] == [
        "alternate_rule",
        "synthetic_rule",
    ]
    assert not any(p.reversible for p in paths)


def test_inverse_only_for_explicit_reversible_rule():
    a, b = node("SyntheticA", "de"), node("SyntheticB")
    assert (
        TransliterationGraph([rule(a, b)]).candidates(b, language="de", script="Latin")
        == ()
    )
    (path,) = TransliterationGraph([rule(a, b, reversible=True)]).candidates(
        b, language="de", script="Latin"
    )
    assert path.inverse_steps == (True,)
    assert path.reversible


def test_cycles_terminate_without_dropping_parallel_provenance():
    a, b, c = node("A", "de"), node("B", "fr"), node("C")
    graph = TransliterationGraph(
        [rule(a, b), rule(b, a), rule(b, c), rule(a, c, "direct")]
    )
    paths = graph.candidates(a, language="en", script="Latin")
    assert len(paths) == 2
    assert [len(p.rules) for p in paths] == [1, 2]
    assert paths[1].confidence == pytest.approx(0.64)


def test_reordered_rules_are_identical_and_reports_have_no_spelling():
    text = "Synthetic Name Only"
    a, b = node(text, "de"), node("Synthetic Target")
    rules = [rule(a, b), rule(a, b, "alternate")]
    first = TransliterationGraph(rules).candidates(a, language="en", script="Latin")
    second = TransliterationGraph(reversed(rules)).candidates(
        a, language="en", script="Latin"
    )
    assert first == second
    assert text not in json.dumps([p.to_dict() for p in first])
    assert text not in repr(a)


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -0.1, 1.1, True, "private"]
)
def test_invalid_confidence_has_controlled_error(value):
    with pytest.raises(ValueError) as exc:
        rule(node("A"), node("B"), confidence=value)
    assert "private" not in str(exc.value)


def test_path_budget_does_not_silently_drop_alternatives():
    a, b = node("A", "de"), node("B")
    graph = TransliterationGraph([rule(a, b), rule(a, b, "alternate")])
    with pytest.raises(ValueError, match="path limit"):
        graph.candidates(a, language="en", script="Latin", max_paths=1)


def test_traversal_and_depth_bounds():
    a, b, c = node("A", "de"), node("B", "fr"), node("C")
    graph = TransliterationGraph([rule(a, b), rule(b, c)])
    assert graph.candidates(a, language="en", script="Latin", max_depth=1) == ()
    with pytest.raises(ValueError, match="budget exceeded"):
        graph.candidates(a, language="en", script="Latin", max_steps=1)
    with pytest.raises(ValueError, match="budget"):
        graph.candidates(a, language="en", script="Latin", max_depth=True)


def test_public_path_constructor_snapshots_and_validates_provenance():
    a, b = node("one"), node("two", "hi", "Devanagari")
    edge = rule(a, b)
    nodes, edges, inverses = [a, b], [edge], [False]
    path = TransliterationPath(nodes, edges, inverses)
    nodes.clear()
    edges.clear()
    inverses.clear()
    assert path.nodes == (a, b)
    assert len(path.to_dict()["rules"]) == 1
    for args in [
        ((a,), (), ()),
        ((a, b), (edge,), ()),
        ((b, a), (edge,), (False,)),
        ((b, a), (edge,), (True,)),
        ((a, b), (edge,), (1,)),
        ((a, a), (edge,), (False,)),
    ]:
        with pytest.raises(ValueError):
            TransliterationPath(*args)


def test_public_path_constructor_bounds_infinite_input():
    import itertools

    with pytest.raises(ValueError, match="limit"):
        TransliterationPath(itertools.repeat(node("one")), (), ())
