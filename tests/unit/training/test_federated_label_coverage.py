"""Per-site label coverage, loss masking, and the coordinator coverage gate."""

from __future__ import annotations

import json
import math
import random
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from openmed.core.labels import DATE, DISEASE, OTHER, PERSON, PHONE, SSN
from openmed.training.federated_label_coverage import (
    FEDERATED_LABEL_COVERAGE_EVIDENCE_SCHEMA_VERSION,
    FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION,
    FederatedLabelCoverageBand,
    FederatedLabelCoverageDeclaration,
    FederatedLabelCoverageError,
    FederatedLabelCoverageEvidence,
    FederatedLabelCoverageReasonCode,
    FederatedRoundLabelSchema,
    FederatedTextForm,
    FederatedTokenLossMask,
    build_label_coverage_evidence,
    build_label_loss_mask,
    build_label_loss_masks,
    check_round_label_coverage,
    run_masked_site_training,
    validate_label_coverage_declaration,
)
from tests.fixtures.private_learning_forbidden import FORBIDDEN_FIELD_CASES

R = FederatedLabelCoverageReasonCode
GUIDELINE = "sha256:" + "a" * 64
OTHER_GUIDELINE = "sha256:" + "b" * 64
SCHEMA = FederatedRoundLabelSchema(labels=(PHONE, PERSON), guideline_digest=GUIDELINE)
CLASSES = ("O", "B-PERSON", "I-PERSON", "B-PHONE", "I-PHONE")
O, B_PERSON, I_PERSON, B_PHONE, I_PHONE = range(5)


def _declaration(
    *labels: str,
    guideline: str = GUIDELINE,
    text_form: FederatedTextForm = FederatedTextForm.SURROGATE,
) -> FederatedLabelCoverageDeclaration:
    return FederatedLabelCoverageDeclaration(
        labels=labels, guideline_digest=guideline, text_form=text_form
    )


FULL = _declaration(PERSON, PHONE)
PERSON_ONLY = _declaration(PERSON)


def _payload() -> dict[str, Any]:
    return {
        "schema_version": FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION,
        "labels": [PHONE, PERSON],
        "guideline_digest": GUIDELINE,
        "text_form": "raw",
    }


def _reason(excinfo: pytest.ExceptionInfo[FederatedLabelCoverageError]) -> str:
    return excinfo.value.reason_code.value


# --- Loss mask semantics -------------------------------------------------------


def test_round_schema_tag_classes_are_canonical_bio_order() -> None:
    assert SCHEMA.labels == (PERSON, PHONE)
    assert SCHEMA.tag_classes == CLASSES
    assert SCHEMA.critical_labels == (PERSON, PHONE)


def test_uncovered_label_tokens_are_removed_from_that_label_negative_class() -> None:
    # Ground truth: O B-PERSON O B-PHONE I-PHONE. The site does not annotate
    # PHONE, so its observed tags show the phone tokens as O.
    observed = ["O", "B-PERSON", "O", "O", "O"]
    mask = build_label_loss_mask(observed, declaration=PERSON_ONLY, schema=SCHEMA)

    assert mask.classes == CLASSES
    negatives = mask.negative_matrix()
    for token in (3, 4):  # truly PHONE, observed O: never a PHONE negative
        assert negatives[token][B_PHONE] is False
        assert negatives[token][I_PHONE] is False
        assert negatives[token][O] is False
        assert mask.positive_targets()[token] is None
    # True outside tokens still teach every covered label as a negative.
    for token in (0, 2):
        assert negatives[token][B_PERSON] is True
        assert negatives[token][I_PERSON] is True
        assert set(mask.candidates[token]) == {O, B_PHONE, I_PHONE}
    # Covered entities stay definite positives and negatives for everything else.
    assert mask.positive_targets()[1] == B_PERSON
    assert negatives[1] == (True, False, True, True, True)


def test_full_coverage_mask_is_one_hot_and_matches_naive_training() -> None:
    observed = ["O", "B-PERSON", "I-PERSON", "O", "B-PHONE", "I-PHONE"]
    mask = build_label_loss_mask(observed, declaration=FULL, schema=SCHEMA)

    assert mask.positive_targets() == (O, B_PERSON, I_PERSON, O, B_PHONE, I_PHONE)
    for row, target in zip(mask.candidate_matrix(), mask.positive_targets()):
        assert row == tuple(index == target for index in range(len(CLASSES)))
    for row, target in zip(mask.negative_matrix(), mask.positive_targets()):
        assert row == tuple(index != target for index in range(len(CLASSES)))


def test_candidate_matrix_supports_marginal_softmax_loss() -> None:
    mask = build_label_loss_mask(
        ["O", "B-PERSON"], declaration=PERSON_ONLY, schema=SCHEMA
    )
    logits = [0.0, 1.0, -1.0, 2.0, 0.5]
    total = sum(math.exp(value) for value in logits)
    probabilities = [math.exp(value) / total for value in logits]

    outside = sum(
        p for p, keep in zip(probabilities, mask.candidate_matrix()[0]) if keep
    )
    assert outside == pytest.approx(
        probabilities[O] + probabilities[B_PHONE] + probabilities[I_PHONE]
    )
    entity = sum(
        p for p, keep in zip(probabilities, mask.candidate_matrix()[1]) if keep
    )
    assert entity == pytest.approx(probabilities[B_PERSON])


def test_empty_sequence_and_batch_builder() -> None:
    assert len(build_label_loss_mask([], declaration=FULL, schema=SCHEMA)) == 0
    masks = build_label_loss_masks(
        [["O"], ("B-PERSON", "I-PERSON")], declaration=PERSON_ONLY, schema=SCHEMA
    )
    assert [len(mask) for mask in masks] == [1, 2]
    assert isinstance(masks, tuple)


@pytest.mark.parametrize(
    ("tags", "reason"),
    [
        (["O", "B-PHONE"], R.UNDECLARED_TAG_LABEL),
        (["O", "B-SSN"], R.LABEL_NOT_IN_ROUND_SCHEMA),
        (["B-NOT_A_LABEL"], R.UNKNOWN_LABEL),
        (["B-OTHER"], R.UNKNOWN_LABEL),
        (["X-PERSON"], R.INVALID_TAG),
        (["o"], R.INVALID_TAG),
        ([3], R.INVALID_TAG),
        ("O", R.INVALID_TAG),
        (None, R.INVALID_TAG),
    ],
)
def test_mask_builder_rejects_invalid_tags_with_stable_codes(
    tags: Any, reason: FederatedLabelCoverageReasonCode
) -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        build_label_loss_mask(tags, declaration=PERSON_ONLY, schema=SCHEMA)
    assert excinfo.value.reason_code is reason
    assert "PHONE" not in str(excinfo.value)
    assert "NOT_A_LABEL" not in str(excinfo.value)


def test_batch_builder_refuses_the_whole_batch_on_one_bad_sequence() -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        build_label_loss_masks(
            [["O", "B-PERSON"], ["B-PHONE"]], declaration=PERSON_ONLY, schema=SCHEMA
        )
    assert excinfo.value.reason_code is R.UNDECLARED_TAG_LABEL


def test_mask_builder_validates_declaration_against_schema_first() -> None:
    stale = _declaration(PERSON, guideline=OTHER_GUIDELINE)
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        build_label_loss_mask(["O"], declaration=stale, schema=SCHEMA)
    assert excinfo.value.reason_code is R.GUIDELINE_VERSION_MISMATCH


def test_mask_rejects_malformed_direct_construction() -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        FederatedTokenLossMask(classes=CLASSES, candidates=((),))
    assert excinfo.value.reason_code is R.INVALID_MASK
    with pytest.raises(FederatedLabelCoverageError):
        FederatedTokenLossMask(classes=CLASSES, candidates=((7,),))
    with pytest.raises(FederatedLabelCoverageError):
        FederatedTokenLossMask(classes=CLASSES, candidates=((3, 0),))


# --- Declaration and schema validation -----------------------------------------


def test_declaration_is_canonical_and_round_trips() -> None:
    declaration = FederatedLabelCoverageDeclaration.from_dict(_payload())
    assert declaration.labels == (PERSON, PHONE)
    assert declaration.text_form is FederatedTextForm.RAW
    assert FederatedLabelCoverageDeclaration.from_json(declaration.to_json()) == (
        declaration
    )
    assert json.loads(declaration.to_json()) == {
        "guideline_digest": GUIDELINE,
        "labels": [PERSON, PHONE],
        "schema_version": FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION,
        "text_form": "raw",
    }
    with pytest.raises(FrozenInstanceError):
        declaration.labels = (PERSON,)  # type: ignore[misc]


def test_validate_declaration_accepts_a_covered_subset() -> None:
    assert (
        validate_label_coverage_declaration(PERSON_ONLY, schema=SCHEMA) is PERSON_ONLY
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda p: p.update(labels=[PERSON, "NOT_A_LABEL"]), R.UNKNOWN_LABEL),
        (lambda p: p.update(labels=["person"]), R.UNKNOWN_LABEL),
        (lambda p: p.update(labels=[OTHER]), R.UNKNOWN_LABEL),
        (lambda p: p.update(labels=[PERSON, PERSON]), R.DUPLICATE_LABEL),
        (lambda p: p.update(labels=[]), R.EMPTY_COVERAGE),
        (lambda p: p.update(labels="PERSON"), R.INVALID_DECLARATION),
        (lambda p: p.update(guideline_digest="v2"), R.INVALID_GUIDELINE_DIGEST),
        (lambda p: p.update(text_form="redacted"), R.INVALID_TEXT_FORM),
        (lambda p: p.update(schema_version="v0"), R.UNSUPPORTED_SCHEMA_VERSION),
        (lambda p: p.pop("text_form"), R.INVALID_DECLARATION),
    ],
)
def test_declaration_parsing_fails_with_stable_reason_codes(
    mutate: Any, reason: FederatedLabelCoverageReasonCode
) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        FederatedLabelCoverageDeclaration.from_dict(payload)
    assert excinfo.value.reason_code is reason
    assert "NOT_A_LABEL" not in str(excinfo.value)
    assert "redacted" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("declaration", "schema", "reason"),
    [
        (_declaration(PERSON, SSN), SCHEMA, R.LABEL_NOT_IN_ROUND_SCHEMA),
        (
            _declaration(PERSON, guideline=OTHER_GUIDELINE),
            SCHEMA,
            R.GUIDELINE_VERSION_MISMATCH,
        ),
        (
            _declaration(PERSON, text_form=FederatedTextForm.RAW),
            FederatedRoundLabelSchema(
                labels=(PERSON, PHONE),
                guideline_digest=GUIDELINE,
                allowed_text_forms=(FederatedTextForm.SURROGATE,),
            ),
            R.TEXT_FORM_NOT_ALLOWED,
        ),
        ({"labels": [PERSON]}, SCHEMA, R.INVALID_DECLARATION),
    ],
)
def test_declaration_validation_against_round_schema(
    declaration: Any,
    schema: FederatedRoundLabelSchema,
    reason: FederatedLabelCoverageReasonCode,
) -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        validate_label_coverage_declaration(declaration, schema=schema)
    assert excinfo.value.reason_code is reason
    assert GUIDELINE not in str(excinfo.value)
    assert OTHER_GUIDELINE not in str(excinfo.value)


def test_from_json_rejects_duplicate_keys_and_non_text() -> None:
    duplicated = (
        '{"schema_version": "%s", "labels": ["PERSON"], "labels": ["PHONE"], '
        '"guideline_digest": "%s", "text_form": "raw"}'
        % (FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION, GUIDELINE)
    )
    for payload in (duplicated, "{not json", b"{}", "[]"):
        with pytest.raises(FederatedLabelCoverageError) as excinfo:
            parse = FederatedLabelCoverageDeclaration.from_json
            parse(payload)  # type: ignore[arg-type]
        assert excinfo.value.reason_code is R.INVALID_DECLARATION


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"labels": ()}, R.EMPTY_COVERAGE),
        ({"labels": ("NOT_A_LABEL",)}, R.UNKNOWN_LABEL),
        ({"labels": (PERSON, PERSON)}, R.DUPLICATE_LABEL),
        ({"labels": [PERSON]}, R.INVALID_ROUND_SCHEMA),
        ({"guideline_digest": "sha256:short"}, R.INVALID_GUIDELINE_DIGEST),
        ({"allowed_text_forms": ()}, R.INVALID_TEXT_FORM),
        ({"allowed_text_forms": ("raw",)}, R.INVALID_TEXT_FORM),
        ({"additional_critical_labels": (SSN,)}, R.LABEL_NOT_IN_ROUND_SCHEMA),
    ],
)
def test_round_schema_rejects_invalid_configuration(
    kwargs: dict[str, Any], reason: FederatedLabelCoverageReasonCode
) -> None:
    arguments: dict[str, Any] = {"labels": (PERSON,), "guideline_digest": GUIDELINE}
    arguments.update(kwargs)
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        FederatedRoundLabelSchema(**arguments)
    assert excinfo.value.reason_code is reason


def test_round_schema_critical_labels_can_be_added_but_never_downgraded() -> None:
    schema = FederatedRoundLabelSchema(
        labels=(DATE, DISEASE, PHONE),
        guideline_digest=GUIDELINE,
        additional_critical_labels=(DATE,),
    )
    assert schema.critical_labels == (DATE, PHONE)
    clinical = FederatedRoundLabelSchema(labels=(DISEASE,), guideline_digest=GUIDELINE)
    assert clinical.critical_labels == ()
    # The only critical-label knob adds labels; there is no field to remove one.
    assert set(FederatedRoundLabelSchema.__dataclass_fields__) == {
        "labels",
        "guideline_digest",
        "allowed_text_forms",
        "additional_critical_labels",
    }


def test_reason_codes_are_stable() -> None:
    assert {code.value for code in R} == {
        "invalid_declaration",
        "unsupported_schema_version",
        "invalid_guideline_digest",
        "invalid_text_form",
        "unknown_label",
        "duplicate_label",
        "empty_coverage",
        "invalid_round_schema",
        "label_not_in_round_schema",
        "guideline_version_mismatch",
        "text_form_not_allowed",
        "invalid_tag",
        "undeclared_tag_label",
        "invalid_mask",
        "no_declarations",
        "critical_label_uncovered",
        "invalid_minimum_group_size",
        "invalid_evidence",
        "training_port_failed",
    }
    error = FederatedLabelCoverageError(R.UNKNOWN_LABEL, "labels")
    assert str(error) == "unknown_label: labels"
    assert error.field == "labels"


# --- Content-free sentinels ----------------------------------------------------


@pytest.mark.parametrize(
    "case", FORBIDDEN_FIELD_CASES, ids=[c.reason_code for c in FORBIDDEN_FIELD_CASES]
)
def test_declaration_rejects_identifying_or_count_fields(case: Any) -> None:
    payload = _payload()
    payload[case.field] = case.value
    for parse in (
        FederatedLabelCoverageDeclaration.from_dict,
        lambda p: FederatedLabelCoverageDeclaration.from_json(json.dumps(p)),
    ):
        with pytest.raises(FederatedLabelCoverageError) as excinfo:
            parse(payload)
        assert excinfo.value.reason_code is R.INVALID_DECLARATION
        if case.marker is not None:
            assert case.marker not in str(excinfo.value)
            assert case.marker not in repr(excinfo.value)


def test_declaration_has_no_site_text_or_count_surface() -> None:
    fields = set(FederatedLabelCoverageDeclaration.__dataclass_fields__)
    assert fields == {"labels", "guideline_digest", "text_form", "schema_version"}
    rendered = FULL.to_dict()

    def _walk(value: Any) -> None:
        assert not isinstance(value, (int, float)), "declarations carry no counts"
        if isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(rendered)


# --- Coordinator gate and evidence ---------------------------------------------


def test_round_is_blocked_when_a_critical_label_has_no_annotating_site() -> None:
    declarations = [PERSON_ONLY, PERSON_ONLY]
    evidence = build_label_coverage_evidence(declarations, schema=SCHEMA)
    assert evidence.admitted is False
    assert evidence.reason_code is R.CRITICAL_LABEL_UNCOVERED
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        check_round_label_coverage(declarations, schema=SCHEMA)
    assert excinfo.value.reason_code is R.CRITICAL_LABEL_UNCOVERED
    assert "PHONE" not in str(excinfo.value)
    with pytest.raises(FederatedLabelCoverageError):
        evidence.require_admitted()


def test_round_is_admitted_when_one_site_covers_each_critical_label() -> None:
    evidence = check_round_label_coverage([FULL, PERSON_ONLY], schema=SCHEMA)
    assert evidence.admitted is True
    assert evidence.reason_code is None
    assert evidence.require_admitted() is evidence


def test_uncovered_non_critical_label_does_not_block_the_round() -> None:
    schema = FederatedRoundLabelSchema(
        labels=(DISEASE, PHONE), guideline_digest=GUIDELINE
    )
    evidence = check_round_label_coverage([_declaration(PHONE)], schema=schema)
    bands = {entry.label: entry.band for entry in evidence.entries}
    assert bands[DISEASE] is FederatedLabelCoverageBand.NONE


def test_coordinator_refuses_the_round_on_any_invalid_declaration() -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        check_round_label_coverage(
            [FULL, _declaration(PERSON, guideline=OTHER_GUIDELINE)], schema=SCHEMA
        )
    assert excinfo.value.reason_code is R.GUIDELINE_VERSION_MISMATCH
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        check_round_label_coverage([], schema=SCHEMA)
    assert excinfo.value.reason_code is R.NO_DECLARATIONS
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        check_round_label_coverage(FULL, schema=SCHEMA)  # type: ignore[arg-type]
    assert excinfo.value.reason_code is R.INVALID_DECLARATION
    for size in (1, True, 2.0):
        with pytest.raises(FederatedLabelCoverageError) as excinfo:
            check_round_label_coverage([FULL], schema=SCHEMA, minimum_group_size=size)
        assert excinfo.value.reason_code is R.INVALID_MINIMUM_GROUP_SIZE


@pytest.mark.parametrize(
    ("full_sites", "band"),
    [
        (1, FederatedLabelCoverageBand.SUPPRESSED),
        (4, FederatedLabelCoverageBand.SUPPRESSED),
        (5, FederatedLabelCoverageBand.MINIMUM_TO_UNDER_DOUBLE),
        (9, FederatedLabelCoverageBand.MINIMUM_TO_UNDER_DOUBLE),
        (10, FederatedLabelCoverageBand.DOUBLE_TO_UNDER_FOURFOLD),
        (19, FederatedLabelCoverageBand.DOUBLE_TO_UNDER_FOURFOLD),
        (20, FederatedLabelCoverageBand.FOURFOLD_OR_MORE),
    ],
)
def test_coverage_bands_suppress_counts_below_minimum_group_size(
    full_sites: int, band: FederatedLabelCoverageBand
) -> None:
    declarations = [FULL] * full_sites + [PERSON_ONLY] * 20
    evidence = check_round_label_coverage(declarations, schema=SCHEMA)
    bands = {entry.label: entry.band for entry in evidence.entries}
    assert bands[PHONE] is band
    assert bands[PERSON] is FederatedLabelCoverageBand.FOURFOLD_OR_MORE


def test_evidence_is_deterministic_and_count_free() -> None:
    evidence = check_round_label_coverage([PERSON_ONLY, FULL], schema=SCHEMA)
    reordered = check_round_label_coverage([FULL, PERSON_ONLY], schema=SCHEMA)
    assert evidence.to_json() == reordered.to_json()
    payload = json.loads(evidence.to_json())
    assert payload == {
        "admitted": True,
        "entries": [
            {"band": "suppressed", "critical": True, "label": PERSON},
            {"band": "suppressed", "critical": True, "label": PHONE},
        ],
        "guideline_digest": GUIDELINE,
        "label_schema_digest": SCHEMA.digest,
        "minimum_group_size": 5,
        "reason_code": None,
        "schema_version": FEDERATED_LABEL_COVERAGE_EVIDENCE_SCHEMA_VERSION,
    }
    assert SCHEMA.digest.startswith("sha256:") and len(SCHEMA.digest) == 71
    integers = [v for v in payload.values() if type(v) is int]
    assert integers == [5]  # only the policy minimum, never a site count


def test_evidence_cannot_be_forged_by_direct_construction() -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        FederatedLabelCoverageEvidence(  # type: ignore[call-arg]
            label_schema_digest=SCHEMA.digest,
            guideline_digest=GUIDELINE,
            minimum_group_size=5,
            entries=(),
            admitted=True,
            reason_code=None,
        )
    assert excinfo.value.reason_code is R.INVALID_EVIDENCE


def test_schema_digest_changes_with_labels_guideline_and_critical_set() -> None:
    digests = {
        SCHEMA.digest,
        FederatedRoundLabelSchema(labels=(PERSON,), guideline_digest=GUIDELINE).digest,
        FederatedRoundLabelSchema(
            labels=(PERSON, PHONE), guideline_digest=OTHER_GUIDELINE
        ).digest,
        FederatedRoundLabelSchema(
            labels=(DATE, PERSON, PHONE),
            guideline_digest=GUIDELINE,
            additional_critical_labels=(DATE,),
        ).digest,
        FederatedRoundLabelSchema(
            labels=(DATE, PERSON, PHONE), guideline_digest=GUIDELINE
        ).digest,
    }
    assert len(digests) == 5


# --- Training port -------------------------------------------------------------


class _RecordingPort:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[FederatedTokenLossMask, ...]] = []
        self.fail = fail

    def train(self, masks: tuple[FederatedTokenLossMask, ...]) -> str:
        self.calls.append(masks)
        if self.fail:
            raise RuntimeError("placeholder port failure with label PHONE")
        return "trained"


def test_training_port_receives_masks_built_from_the_declaration() -> None:
    port = _RecordingPort()
    result = run_masked_site_training(
        port, [["O", "B-PERSON"]], declaration=PERSON_ONLY, schema=SCHEMA
    )
    assert result == "trained"
    (masks,) = port.calls
    assert masks[0].negative_matrix()[0][B_PHONE] is False


def test_training_port_is_not_called_when_inputs_are_invalid() -> None:
    port = _RecordingPort()
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        run_masked_site_training(
            port, [["B-PHONE"]], declaration=PERSON_ONLY, schema=SCHEMA
        )
    assert excinfo.value.reason_code is R.UNDECLARED_TAG_LABEL
    assert port.calls == []


def test_training_port_failures_map_to_a_stable_code_without_detail() -> None:
    with pytest.raises(FederatedLabelCoverageError) as excinfo:
        run_masked_site_training(
            _RecordingPort(fail=True),
            [["O"]],
            declaration=PERSON_ONLY,
            schema=SCHEMA,
        )
    assert excinfo.value.reason_code is R.TRAINING_PORT_FAILED
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True
    assert "placeholder" not in str(excinfo.value)


# --- Two-site toy simulation ---------------------------------------------------

_NAMES = ("Avery", "Jordan", "Morgan", "Riley", "Quinn", "Casey")
_FILLER = ("call", "at", "today", "reach", "patient", "please")


def _shape(token: str) -> str:
    """Collapse a synthetic token to a coarse character-class shape feature."""

    shape = []
    for char in token:
        kind = "d" if char.isdigit() else "X" if char.isupper() else "x"
        if char == "-":
            kind = "-"
        if not shape or shape[-1] != kind:
            shape.append(kind)
    return "".join(shape)


def _synthetic_sentences(
    rng: random.Random, count: int
) -> list[tuple[list[str], list[str]]]:
    sentences = []
    for _ in range(count):
        phone = f"{rng.randrange(200, 999)}-{rng.randrange(1000, 9999)}"
        tokens = [
            rng.choice(_FILLER),
            rng.choice(_NAMES),
            rng.choice(_FILLER),
            phone,
            rng.choice(_FILLER),
        ]
        tags = ["O", "B-PERSON", "O", "B-PHONE", "O"]
        sentences.append((tokens, tags))
    return sentences


def _hide(tags: list[str], label: str) -> list[str]:
    return ["O" if tag.endswith("-" + label) else tag for tag in tags]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _local_update(
    weights: dict[str, list[float]],
    tokens: list[list[str]],
    masks: tuple[FederatedTokenLossMask, ...],
    *,
    epochs: int = 3,
    learning_rate: float = 0.5,
) -> dict[str, list[float]]:
    """Per-class logistic terms selected by the loss mask (one-vs-rest)."""

    local = {shape: list(row) for shape, row in weights.items()}
    for _ in range(epochs):
        for sentence, mask in zip(tokens, masks):
            positives = mask.positive_targets()
            negatives = mask.negative_matrix()
            for index, token in enumerate(sentence):
                row = local.setdefault(_shape(token), [0.0] * len(mask.classes))
                for klass in range(len(mask.classes)):
                    if positives[index] == klass:
                        target = 1.0
                    elif negatives[index][klass]:
                        target = 0.0
                    else:
                        continue  # ambiguous: removed from this class's loss
                    gradient = _sigmoid(row[klass]) - target
                    row[klass] -= learning_rate * gradient
    return local


def _fed_avg(
    updates: list[tuple[dict[str, list[float]], int]], width: int
) -> dict[str, list[float]]:
    total = sum(weight for _, weight in updates)
    shapes = sorted({shape for update, _ in updates for shape in update})
    merged = {}
    for shape in shapes:
        merged[shape] = [
            sum(update.get(shape, [0.0] * width)[k] * n for update, n in updates)
            / total
            for k in range(width)
        ]
    return merged


def _recall(
    weights: dict[str, list[float]],
    sentences: list[tuple[list[str], list[str]]],
    tag: str,
) -> float:
    hits = total = 0
    for tokens, tags in sentences:
        for token, truth in zip(tokens, tags):
            if truth != tag:
                continue
            total += 1
            row = weights.get(_shape(token), [0.0] * len(CLASSES))
            hits += CLASSES[max(range(len(CLASSES)), key=row.__getitem__)] == tag
    return hits / total


def _false_positive_rate(
    weights: dict[str, list[float]],
    sentences: list[tuple[list[str], list[str]]],
    tag: str,
) -> float:
    flagged = total = 0
    for tokens, tags in sentences:
        for token, truth in zip(tokens, tags):
            if truth == tag:
                continue
            total += 1
            row = weights.get(_shape(token), [0.0] * len(CLASSES))
            flagged += CLASSES[max(range(len(CLASSES)), key=row.__getitem__)] == tag
    return flagged / total


def _simulate(*, masked: bool, site_b_covers_phone: bool = False) -> dict[str, float]:
    rng = random.Random(3851)
    site_a = _synthetic_sentences(rng, 10)
    site_b = _synthetic_sentences(rng, 40)
    test = _synthetic_sentences(rng, 25)
    b_declaration = FULL if site_b_covers_phone else PERSON_ONLY
    check_round_label_coverage([FULL, b_declaration], schema=SCHEMA)

    site_a_tokens = [tokens for tokens, _ in site_a]
    site_a_masks = build_label_loss_masks(
        [tags for _, tags in site_a], declaration=FULL, schema=SCHEMA
    )
    site_b_tokens = [tokens for tokens, _ in site_b]
    site_b_tags = [
        tags if site_b_covers_phone else _hide(tags, PHONE) for _, tags in site_b
    ]
    # Naive training treats the observed O tags as complete annotation; masked
    # training uses the declaration the site actually published.
    site_b_masks = build_label_loss_masks(
        site_b_tags,
        declaration=b_declaration if masked else FULL,
        schema=SCHEMA,
    )

    weights: dict[str, list[float]] = {}
    for _ in range(5):
        update_a = _local_update(weights, site_a_tokens, site_a_masks)
        update_b = _local_update(weights, site_b_tokens, site_b_masks)
        weights = _fed_avg(
            [(update_a, 5 * len(site_a)), (update_b, 5 * len(site_b))], len(CLASSES)
        )
    return {
        "phone_recall": _recall(weights, test, "B-PHONE"),
        "person_recall": _recall(weights, test, "B-PERSON"),
        "phone_false_positive_rate": _false_positive_rate(weights, test, "B-PHONE"),
    }


def test_toy_simulation_masked_training_preserves_unannotated_label_recall() -> None:
    naive = _simulate(masked=False)
    masked = _simulate(masked=True)

    assert naive["phone_recall"] == 0.0
    assert masked["phone_recall"] == 1.0
    # Controls: the shared covered label is learned either way, and masking
    # does not turn ordinary outside tokens into phone predictions.
    assert naive["person_recall"] == masked["person_recall"] == 1.0
    assert masked["phone_false_positive_rate"] == 0.0


def test_toy_simulation_negative_control_full_coverage_masks_change_nothing() -> None:
    naive = _simulate(masked=False, site_b_covers_phone=True)
    masked = _simulate(masked=True, site_b_covers_phone=True)
    assert naive == masked
    assert masked["phone_recall"] == 1.0
