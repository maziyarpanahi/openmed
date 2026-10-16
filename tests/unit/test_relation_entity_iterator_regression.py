import pytest

from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import EvalRelation, normalize_eval_relations

ENTITIES = [
    {"id": f"e{i}", "start": i * 5, "end": i * 5 + 3, "label": "NAME"} for i in range(3)
]
RELATIONS = [
    {"type": "related_to", "head": "e0", "tail": "e1"},
    {"type": "related_to", "head": "e1", "tail": "e2"},
]


def test_single_use_entity_iterator_matches_list_input():
    expected = normalize_eval_relations(
        RELATIONS, entity_spans=ENTITIES, fixture_id="synthetic", default_language="fr"
    )
    actual = normalize_eval_relations(
        iter(RELATIONS),
        entity_spans=iter(ENTITIES),
        fixture_id="synthetic",
        default_language="fr",
    )
    assert actual == expected
    assert len(actual) == 2
    assert actual[1].tail.language == "fr"


def test_entity_generator_is_consumed_once():
    observed = []

    def entities():
        for item in ENTITIES:
            observed.append(item["id"])
            yield item

    result = normalize_eval_relations(RELATIONS, entity_spans=entities())
    assert len(result) == 2
    assert observed == ["e0", "e1", "e2"]


def test_empty_relations_do_not_consume_unused_entities():
    def entities():
        raise AssertionError("unused iterator was consumed")
        yield

    assert normalize_eval_relations([], entity_spans=entities()) == []


def test_unknown_entity_is_still_an_error():
    with pytest.raises(ValueError, match="unknown relation argument span id"):
        normalize_eval_relations(
            [{"type": "related_to", "head": "missing", "tail": "e0"}],
            entity_spans=iter(ENTITIES),
        )


def test_mapping_and_inline_arguments_remain_supported():
    mapping = {e["id"]: e for e in ENTITIES}
    assert len(normalize_eval_relations(RELATIONS, entity_spans=mapping)) == 2
    inline = [{"type": "related_to", "head": ENTITIES[0], "tail": ENTITIES[1]}]
    assert len(normalize_eval_relations(inline)) == 1


def test_typed_relations_do_not_consume_unused_entity_iterator():
    relation = EvalRelation(
        "RELATED_TO", EvalSpan(0, 3, "NAME"), EvalSpan(5, 8, "NAME")
    )

    def unused_entities():
        raise AssertionError("typed relations should not consume entities")
        yield

    result = normalize_eval_relations([relation], entity_spans=unused_entities())
    assert result == [relation]
    assert result[0] is relation


def test_mixed_typed_and_mapping_relations_share_one_lazy_lookup():
    typed = EvalRelation("RELATED_TO", EvalSpan(0, 3, "NAME"), EvalSpan(5, 8, "NAME"))
    expected = normalize_eval_relations(
        [typed, *RELATIONS], entity_spans=ENTITIES, fixture_id="synthetic"
    )
    actual = normalize_eval_relations(
        iter([typed, *RELATIONS]),
        entity_spans=iter(ENTITIES),
        fixture_id="synthetic",
    )
    assert actual == expected
    assert len(actual) == 3
