"""Synthetic, offline contracts for patient-level DP-SGD over injected gradients."""

from __future__ import annotations

import ast
import json
import logging
import math
import random
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from openmed.training import patient_dp_sgd as dp_module
from openmed.training.patient_dp_sgd import (
    PATIENT_DP_SGD_REASON_CODES,
    PATIENT_DP_SGD_SCHEMA_VERSION,
    PatientDPSGD,
    PatientDPSGDDeclaration,
    PatientDPSGDError,
    flatten_torch_per_example_gradients,
)

RECIPE = "sha256:" + "a" * 64
KEY_SENTINEL = "PATIENT-SENTINEL-7781"
GRADIENT_SENTINEL = 123.456789
TEXT_SENTINEL = "NOTE-TEXT-SENTINEL"


def _declaration(**overrides):
    values = {
        "recipe_digest": RECIPE,
        "privacy_unit": "patient",
        "batch_sampling": "poisson",
        "amplification_claimed": True,
        "sampling_rate": 0.5,
        "expected_batch_size": 2.0,
        "noise_multiplier": 1.0,
        "clip_norm": 1.0,
        "max_steps": 10,
    }
    values.update(overrides)
    return PatientDPSGDDeclaration(**values)


def _full_batch(**overrides):
    values = {
        "amplification_claimed": False,
        "sampling_rate": 1.0,
        "expected_batch_size": 4.0,
    }
    values.update(overrides)
    return _declaration(**values)


def _test_session(declaration=None, *, dimension=3, sampling_seed=1, noise_seed=2):
    return PatientDPSGD(
        declaration or _declaration(),
        gradient_dimension=dimension,
        sampling_rng=random.Random(sampling_seed),
        noise_rng=random.Random(noise_seed),
        mode="test",
    )


def _provider(rows):
    def provide(positions):
        return [rows[position] for position in positions]

    return provide


def _norm(vector):
    return math.sqrt(math.fsum(value * value for value in vector))


def _code(excinfo):
    error = excinfo.value
    assert str(error) == error.code
    assert error.code in PATIENT_DP_SGD_REASON_CODES
    return error.code


# --------------------------------------------------------------------------
# Reference accountant used only by these tests. The accountant is injected
# into the step; OpenMed does not ship accountant math in this module.
#
# Integer-order Renyi DP of the Poisson-subsampled Gaussian mechanism under
# add/remove-one adjacency, Mironov, Talwar and Zhang, "Renyi Differential
# Privacy of the Sampled Gaussian Mechanism" (2019), arXiv:1908.10530,
# Section 3.3, with the classic RDP-to-(epsilon, delta) conversion from
# Mironov, "Renyi Differential Privacy" (2017), Proposition 3.
# --------------------------------------------------------------------------
_REFERENCE_ORDERS = tuple(range(2, 257))


def _log_moment(order, sampling_rate, noise_multiplier):
    if sampling_rate == 1.0:
        return order * (order - 1) / (2.0 * noise_multiplier**2)
    terms = [
        math.lgamma(order + 1)
        - math.lgamma(index + 1)
        - math.lgamma(order - index + 1)
        + index * math.log(sampling_rate)
        + (order - index) * math.log1p(-sampling_rate)
        + (index * index - index) / (2.0 * noise_multiplier**2)
        for index in range(order + 1)
    ]
    peak = max(terms)
    return peak + math.log(math.fsum(math.exp(term - peak) for term in terms))


def _reference_rdp(order, sampling_rate, noise_multiplier):
    return _log_moment(order, sampling_rate, noise_multiplier) / (order - 1)


def reference_accountant(*, sampling_rate, noise_multiplier, steps, delta):
    return min(
        steps * _reference_rdp(order, sampling_rate, noise_multiplier)
        + math.log(1.0 / delta) / (order - 1)
        for order in _REFERENCE_ORDERS
    )


# --------------------------------------------------------------------------
# Acceptance 1: per-patient clipping bounds the summed update's sensitivity.
# --------------------------------------------------------------------------


def _neighboring_updates(rows_with_extra, keys_with_extra, extra_count, clip=1.0):
    declaration = _full_batch(clip_norm=clip, noise_multiplier=0.8)
    dimension = len(rows_with_extra[0])
    with_patient = _test_session(
        declaration, dimension=dimension, sampling_seed=11, noise_seed=12
    )
    without_patient = _test_session(
        declaration, dimension=dimension, sampling_seed=11, noise_seed=12
    )
    full = with_patient.step(keys_with_extra, _provider(rows_with_extra))
    neighbor = without_patient.step(
        keys_with_extra[:-extra_count], _provider(rows_with_extra[:-extra_count])
    )
    batch = declaration.expected_batch_size
    return [
        (left - right) * batch
        for left, right in zip(full.update, neighbor.update, strict=True)
    ]


@pytest.mark.parametrize("note_count", [1, 2, 50, 400])
def test_patient_with_many_notes_never_exceeds_clip_bound(note_count):
    rows = [[0.1, 0.2, -0.1], [0.3, 0.0, 0.2], [-0.2, 0.1, 0.0]]
    keys = ["p1", "p1", "p2"]
    rows += [[1e6, -1e6, 1e6]] * note_count
    keys += [KEY_SENTINEL] * note_count

    difference = _neighboring_updates(rows, keys, note_count)

    assert _norm(difference) <= 1.0 * (1.0 + 1e-9)
    # The many-note patient is clipped once, not once per note.
    assert _norm(difference) == pytest.approx(1.0, rel=1e-6)


def test_patient_contribution_is_the_clipped_mean_of_its_notes():
    rows = [[0.1, 0.0], [3.0, 0.0], [1.0, 0.0]]
    keys = ["p1", "p2", "p2"]

    difference = _neighboring_updates(rows, keys, 2, clip=10.0)

    assert difference == pytest.approx([2.0, 0.0], abs=1e-9)


@pytest.mark.parametrize(
    "notes",
    [
        [[1e300, 1e300, -1e300]] * 64,
        [[5e-324, 0.0, 0.0]] * 3,
        [[3.0, 4.0, 0.0]],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.6, 0.8, 0.0]] * 1000,
    ],
)
def test_clipped_contribution_norm_is_never_above_the_bound(notes):
    clip = 1.0
    contribution = dp_module._patient_contribution(notes, clip_norm=clip)

    assert all(math.isfinite(value) for value in contribution)
    assert _norm(contribution) <= clip


def test_note_level_clipping_negative_control_exceeds_patient_bound():
    # Clipping each note separately lets a 50-note patient move the sum by 50C.
    notes = [[1e6, 0.0, 0.0]] * 50
    per_note = [
        dp_module._patient_contribution([note], clip_norm=1.0) for note in notes
    ]
    note_level_sum = [math.fsum(column) for column in zip(*per_note, strict=True)]
    patient_level = dp_module._patient_contribution(notes, clip_norm=1.0)

    assert _norm(note_level_sum) == pytest.approx(50.0)
    assert _norm(patient_level) <= 1.0


# --------------------------------------------------------------------------
# Mechanism shape: Poisson sampling, noise scale and expected-batch scaling.
# --------------------------------------------------------------------------


def test_update_is_normalized_by_expected_batch_size_with_scaled_noise():
    declaration = _full_batch(noise_multiplier=0.5, clip_norm=1.0)
    session = _test_session(declaration, dimension=2, noise_seed=7)
    expected_noise = random.Random(7)
    noise = [expected_noise.normalvariate(0.0, 0.5) for _ in range(2)]

    result = session.step(["a", "b"], _provider([[3.0, 4.0], [0.1, 0.2]]))

    assert result.update == pytest.approx(
        [(0.6 + 0.1 + noise[0]) / 4.0, (0.8 + 0.2 + noise[1]) / 4.0]
    )


def test_poisson_sampling_selects_whole_patients_with_injected_rng():
    keys = [f"p{index % 6}" for index in range(12)]
    seen = []

    def provide(positions):
        seen.append(positions)
        return [[1.0, 0.0, 0.0]] * len(positions)

    session = _test_session(_declaration(sampling_rate=0.5), sampling_seed=3)
    session.step(keys, provide)

    replay = random.Random(3)
    included = {f"p{index}" for index in range(6) if replay.random() < 0.5}
    expected = tuple(position for position, key in enumerate(keys) if key in included)
    assert seen == ([expected] if expected else [])


def test_empty_poisson_sample_still_releases_noise_of_declared_dimension():
    calls = []
    session = _test_session(_declaration(sampling_rate=1e-12), dimension=4)

    result = session.step(["p1", "p2"], lambda positions: calls.append(positions))

    assert calls == []
    assert len(result.update) == 4
    assert any(value != 0.0 for value in result.update)


def test_numpy_gradients_are_accepted_without_a_numpy_core_dependency():
    numpy = pytest.importorskip("numpy")
    declaration = _full_batch()
    from_numpy = _test_session(declaration, sampling_seed=4, noise_seed=5)
    from_lists = _test_session(declaration, sampling_seed=4, noise_seed=5)
    rows = [[0.5, -0.5, 2.0], [1.0, 1.0, 1.0]]

    left = from_numpy.step(["a", "b"], lambda positions: numpy.asarray(rows))
    right = from_lists.step(["a", "b"], _provider(rows))

    assert left.update == pytest.approx(right.update)


def test_module_does_not_import_numpy_or_torch_at_top_level():
    source = Path(dp_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert not top_level & {"numpy", "torch"}
    assert source.isascii()


# --------------------------------------------------------------------------
# Acceptance 2: unsafe configurations are refused in DP mode.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["fixed_size", "shuffle"])
def test_fixed_size_or_shuffled_batches_refused_when_amplification_claimed(sampling):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(batch_sampling=sampling)
    assert _code(excinfo) == "dp_sampling_not_poisson"


def test_unamplified_declaration_must_use_every_unit():
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(amplification_claimed=False, sampling_rate=0.5)
    assert _code(excinfo) == "dp_config_invalid"
    declaration = _full_batch(batch_sampling="fixed_size")
    assert declaration.accountant_inputs().sampling_rate == 1.0


@pytest.mark.parametrize("multiplier", [None, 0, 0.0])
def test_missing_noise_is_refused(multiplier):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(noise_multiplier=multiplier)
    assert _code(excinfo) == "dp_noise_missing"


@pytest.mark.parametrize(
    "overrides",
    [
        {"noise_multiplier": -1.0},
        {"noise_multiplier": math.inf},
        {"noise_multiplier": True},
        {"clip_norm": 0.0},
        {"clip_norm": math.nan},
        {"sampling_rate": 0.0},
        {"sampling_rate": 1.5},
        {"expected_batch_size": 0.0},
        {"max_steps": 0},
        {"max_steps": 1.5},
        {"amplification_claimed": "yes"},
        {"patient_keys_available": 1},
    ],
)
def test_invalid_numeric_configuration_is_refused(overrides):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(**overrides)
    assert _code(excinfo) == "dp_config_invalid"


@pytest.mark.parametrize("sampling", ["", "Poisson", None, "random"])
def test_unknown_batch_sampling_is_refused(sampling):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(batch_sampling=sampling)
    assert _code(excinfo) == "dp_sampling_not_poisson"


@pytest.mark.parametrize("digest", ["", "sha256:abc", "md5:" + "a" * 64, None])
def test_recipe_digest_must_be_a_sha256_reference(digest):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(recipe_digest=digest)
    assert _code(excinfo) == "dp_recipe_digest_invalid"


def test_note_level_unit_refused_when_patient_keys_are_available():
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(privacy_unit="note")
    assert _code(excinfo) == "dp_privacy_unit_note_level"


def test_note_level_unit_refused_when_step_data_carries_patient_keys():
    declaration = _declaration(privacy_unit="note", patient_keys_available=False)
    session = _test_session(declaration)

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step([None, KEY_SENTINEL], _provider([[0.0] * 3] * 2))
    assert _code(excinfo) == "dp_privacy_unit_note_level"


def test_note_level_unit_without_patient_keys_treats_each_note_as_a_unit():
    declaration = _full_batch(privacy_unit="note", patient_keys_available=False)
    session = _test_session(declaration, dimension=1)

    result = session.step([None, None], _provider([[0.25], [0.25]]))

    assert result.record.privacy_unit == "note"


@pytest.mark.parametrize("unit", ["visit", "", None, "PATIENT"])
def test_unknown_privacy_unit_is_refused(unit):
    with pytest.raises(PatientDPSGDError) as excinfo:
        _declaration(privacy_unit=unit)
    assert _code(excinfo) == "dp_privacy_unit_invalid"


@pytest.mark.parametrize("key", [None, "", 7, b"p1", "x" * 257])
def test_patient_unit_requires_opaque_string_keys_for_every_note(key):
    session = _test_session()

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1", key], _provider([[0.0] * 3] * 2))
    assert _code(excinfo) == "dp_patient_key_invalid"


class _SubclassedSystemRandom(random.SystemRandom):
    pass


@pytest.mark.parametrize(
    "factory",
    [
        lambda: random.Random(0),
        lambda: _SubclassedSystemRandom(),
        lambda: None,
        lambda: object(),
    ],
)
def test_operational_mode_accepts_only_system_random(factory):
    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=3,
            sampling_rng=factory(),
            noise_rng=random.SystemRandom(),
        )
    assert _code(excinfo) == "dp_rng_not_csprng"
    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=3,
            sampling_rng=random.SystemRandom(),
            noise_rng=factory(),
        )
    assert _code(excinfo) == "dp_rng_not_csprng"


def test_one_randomness_source_for_sampling_and_noise_is_refused():
    shared = random.SystemRandom()
    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=3,
            sampling_rng=shared,
            noise_rng=shared,
        )
    assert _code(excinfo) == "dp_rng_reused"


def test_operational_randomness_bound_to_another_session_is_refused():
    sampling = random.SystemRandom()
    PatientDPSGD(
        _declaration(),
        gradient_dimension=3,
        sampling_rng=sampling,
        noise_rng=random.SystemRandom(),
    )

    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=3,
            sampling_rng=random.SystemRandom(),
            noise_rng=sampling,
        )
    assert _code(excinfo) == "dp_rng_reused"


def test_seeded_test_randomness_cannot_be_rebound_either():
    seeded = random.Random(9)
    _test_session_with(seeded)
    with pytest.raises(PatientDPSGDError) as excinfo:
        _test_session_with(seeded)
    assert _code(excinfo) == "dp_rng_reused"


def _test_session_with(sampling_rng):
    return PatientDPSGD(
        _declaration(),
        gradient_dimension=3,
        sampling_rng=sampling_rng,
        noise_rng=random.Random(10),
        mode="test",
    )


@pytest.mark.parametrize("mode", ["", "Operational", "debug", None])
def test_unknown_mode_is_refused(mode):
    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=3,
            sampling_rng=random.SystemRandom(),
            noise_rng=random.SystemRandom(),
            mode=mode,
        )
    assert _code(excinfo) == "dp_mode_invalid"


@pytest.mark.parametrize("dimension", [0, -1, 1.0, True, None])
def test_gradient_dimension_must_be_a_positive_integer(dimension):
    with pytest.raises(PatientDPSGDError) as excinfo:
        PatientDPSGD(
            _declaration(),
            gradient_dimension=dimension,
            sampling_rng=random.Random(1),
            noise_rng=random.Random(2),
            mode="test",
        )
    assert _code(excinfo) == "dp_config_invalid"


def test_steps_beyond_the_declared_schedule_are_refused():
    session = _test_session(_full_batch(max_steps=2), dimension=1)
    for _ in range(2):
        session.step(["p1"], _provider([[0.1]]))

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1"], _provider([[0.1]]))
    assert _code(excinfo) == "dp_step_budget_exhausted"
    assert session.steps_taken == 2


@pytest.mark.parametrize(
    "rows",
    [
        [[0.0, 0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, math.nan, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, math.inf, 0.0], [0.0, 0.0, 0.0]],
        [[True, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [["0.1", 0.0, 0.0], [0.0, 0.0, 0.0]],
        None,
        "abc",
    ],
)
def test_malformed_gradients_are_refused_and_close_the_session(rows):
    session = _test_session(_full_batch())

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1", "p2"], lambda positions: rows)
    assert _code(excinfo) == "dp_gradient_invalid"
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1", "p2"], _provider([[0.0] * 3] * 2))
    assert _code(excinfo) == "dp_session_closed"
    # The failed attempt still counts against the accountant (fail closed).
    assert session.steps_taken == 1


def test_provider_failure_is_wrapped_without_its_message_or_context():
    session = _test_session(_full_batch())

    def explode(positions):
        raise RuntimeError(f"{KEY_SENTINEL} {TEXT_SENTINEL} {GRADIENT_SENTINEL}")

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1"], explode)
    assert _code(excinfo) == "dp_gradient_provider_failed"
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_non_sequence_patient_keys_are_refused():
    session = _test_session()
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(KEY_SENTINEL, _provider([]))
    assert _code(excinfo) == "dp_patient_key_invalid"


def test_test_rng_returning_out_of_range_values_is_refused():
    class Broken:
        def random(self):
            return 1.5

        def normalvariate(self, mu, sigma):
            return 0.0

    session = PatientDPSGD(
        _declaration(),
        gradient_dimension=1,
        sampling_rng=Broken(),
        noise_rng=random.Random(1),
        mode="test",
    )
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.step(["p1"], _provider([[0.0]]))
    assert _code(excinfo) == "dp_rng_invalid"


# --------------------------------------------------------------------------
# Acceptance 3: accountant inputs reproduce a reference epsilon; test mode
# never claims a guarantee.
# --------------------------------------------------------------------------

# TensorFlow Privacy MNIST DP-SGD tutorial, tutorials/README.md
# (https://github.com/tensorflow/privacy/blob/master/tutorials/README.md):
# 60,000 examples, batch 256, delta 1e-5. The table reports epsilon 3.01 for
# noise multiplier 1.1 over 60 epochs and 1.19 for 1.3 over 15 epochs. The
# exact values below were cross-checked offline against the RDP vector of
# dp-accounting 0.6.0 (PoissonSampledDpEvent of GaussianDpEvent, integer
# orders 2..256) with the same classic conversion; dp-accounting is not a
# test or runtime dependency.
_REFERENCE_SCHEDULES = [
    (1.1, 60, 3.01, 3.0090995257323585),
    (1.3, 15, 1.19, 1.1921299645499253),
]


@pytest.mark.parametrize(
    ("multiplier", "epochs", "published", "pinned"), _REFERENCE_SCHEDULES
)
def test_accountant_inputs_reproduce_reference_epsilon(
    multiplier, epochs, published, pinned
):
    population, batch = 60_000, 256
    declaration = _declaration(
        sampling_rate=batch / population,
        expected_batch_size=float(batch),
        noise_multiplier=multiplier,
        max_steps=epochs * population // batch,
    )
    inputs = declaration.accountant_inputs()

    epsilon = reference_accountant(
        sampling_rate=inputs.sampling_rate,
        noise_multiplier=inputs.noise_multiplier,
        steps=inputs.steps,
        delta=1e-5,
    )

    assert inputs.steps == epochs * population // batch
    assert inputs.sampling == "poisson"
    assert inputs.privacy_unit == "patient"
    assert round(epsilon, 2) == published
    assert epsilon == pytest.approx(pinned, rel=1e-9)


def test_reference_accountant_matches_gaussian_closed_form_without_sampling():
    # Without subsampling the RDP of the Gaussian mechanism is order / (2 z^2).
    for order in (2, 3, 10):
        assert _reference_rdp(order, 1.0, 2.0) == pytest.approx(order / 8.0)
        terms = _reference_rdp(order, 1.0 - 1e-15, 2.0)
        assert terms == pytest.approx(order / 8.0, rel=1e-6)


def test_operational_report_uses_executed_steps_and_injected_accountant():
    calls = []

    def accountant(**kwargs):
        calls.append(kwargs)
        return reference_accountant(**kwargs)

    declaration = _full_batch(noise_multiplier=4.0, max_steps=5)
    session = PatientDPSGD(
        declaration,
        gradient_dimension=2,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )
    for _ in range(3):
        session.step(["p1", "p2"], _provider([[0.1, 0.2], [0.3, 0.4]]))

    report = session.privacy_report(
        accountant, delta=1e-5, accountant_id="reference.rdp.integer"
    )

    assert calls == [
        {"sampling_rate": 1.0, "noise_multiplier": 4.0, "steps": 3, "delta": 1e-5}
    ]
    assert report.epsilon == pytest.approx(reference_accountant(**calls[0]))
    assert report.privacy_unit == "patient"
    assert report.accountant_id == "reference.rdp.integer"
    assert report.accountant_inputs.steps == 3
    assert report.recipe_digest == RECIPE
    assert report.declaration_digest == declaration.digest
    assert "per_patient_l2_clipping" in report.assumptions
    assert "no_sampling_amplification" in report.assumptions
    payload = report.to_dict()
    assert payload["guarantee_claimed"] is True
    assert payload["schema_version"] == PATIENT_DP_SGD_SCHEMA_VERSION
    json.dumps(payload)


def test_amplified_report_names_poisson_sampling_assumption():
    session = PatientDPSGD(
        _declaration(),
        gradient_dimension=1,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )
    session.step(["p1", "p2"], _provider([[0.1], [0.2]]))
    report = session.privacy_report(
        reference_accountant, delta=1e-5, accountant_id="reference"
    )
    assert "poisson_patient_sampling" in report.assumptions
    assert report.accountant_inputs.sampling_rate == 0.5


def test_test_mode_never_claims_a_guarantee():
    session = _test_session(_full_batch(), dimension=1)
    result = session.step(["p1"], _provider([[0.1]]))

    assert result.record.test_only is True
    assert result.record.to_dict()["guarantee_claimed"] is False
    assert session.accountant_inputs().test_only is True
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.privacy_report(reference_accountant, delta=1e-5, accountant_id="ref")
    assert _code(excinfo) == "dp_test_mode_no_guarantee"


def test_operational_records_do_not_claim_test_mode():
    session = PatientDPSGD(
        _full_batch(),
        gradient_dimension=1,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )
    result = session.step(["p1"], _provider([[0.1]]))
    assert result.record.test_only is False
    assert session.accountant_inputs().test_only is False


@pytest.mark.parametrize("returned", [math.nan, math.inf, -0.1, "3.0", True, None])
def test_invalid_accountant_output_is_refused(returned):
    session = PatientDPSGD(
        _full_batch(),
        gradient_dimension=1,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.privacy_report(
            lambda **kwargs: returned, delta=1e-5, accountant_id="ref"
        )
    assert _code(excinfo) == "dp_accountant_invalid"


def test_accountant_failure_is_wrapped():
    session = PatientDPSGD(
        _full_batch(),
        gradient_dimension=1,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )

    def explode(**kwargs):
        raise RuntimeError(KEY_SENTINEL)

    with pytest.raises(PatientDPSGDError) as excinfo:
        session.privacy_report(explode, delta=1e-5, accountant_id="ref")
    assert _code(excinfo) == "dp_accountant_failed"
    assert excinfo.value.__suppress_context__ is True


@pytest.mark.parametrize(
    ("accountant", "delta", "accountant_id"),
    [
        (None, 1e-5, "ref"),
        (reference_accountant, 0.0, "ref"),
        (reference_accountant, 1.0, "ref"),
        (reference_accountant, math.nan, "ref"),
        (reference_accountant, 1e-5, ""),
        (reference_accountant, 1e-5, "Has Space"),
        (reference_accountant, 1e-5, KEY_SENTINEL.lower() * 10),
    ],
)
def test_report_arguments_are_validated(accountant, delta, accountant_id):
    session = PatientDPSGD(
        _full_batch(),
        gradient_dimension=1,
        sampling_rng=random.SystemRandom(),
        noise_rng=random.SystemRandom(),
    )
    with pytest.raises(PatientDPSGDError) as excinfo:
        session.privacy_report(accountant, delta=delta, accountant_id=accountant_id)
    assert _code(excinfo) == "dp_accountant_invalid"


# --------------------------------------------------------------------------
# Acceptance 4: nothing sensitive leaves the step.
# --------------------------------------------------------------------------


def test_no_gradients_keys_text_or_noise_in_outputs_reprs_or_logs(caplog):
    caplog.set_level(logging.DEBUG)
    declaration = _full_batch(noise_multiplier=1.0)
    session = _test_session(declaration, dimension=2, noise_seed=21)
    noise_replay = random.Random(21)
    noise = [noise_replay.normalvariate(0.0, 1.0) for _ in range(2)]

    result = session.step(
        [KEY_SENTINEL, KEY_SENTINEL],
        _provider([[GRADIENT_SENTINEL, 0.0], [GRADIENT_SENTINEL, 0.0]]),
    )

    surfaces = [
        repr(declaration),
        str(declaration.to_dict()),
        repr(session),
        str(session),
        repr(result),
        str(result),
        repr(result.record),
        json.dumps(result.record.to_dict()),
        repr(session.accountant_inputs()),
        json.dumps(session.accountant_inputs().to_dict()),
        *(record.getMessage() for record in caplog.records),
    ]
    forbidden = [
        KEY_SENTINEL,
        TEXT_SENTINEL,
        str(GRADIENT_SENTINEL),
        *(repr(value) for value in noise),
        *(repr(value) for value in result.update),
    ]
    for surface in surfaces:
        for token in forbidden:
            assert token not in surface
    assert set(result.record.to_dict()) == {
        "declaration_digest",
        "gradient_dimension",
        "guarantee_claimed",
        "mode",
        "noise_applied",
        "privacy_unit",
        "schema_version",
        "step",
        "test_only",
    }


def test_errors_never_echo_inputs():
    session = _test_session(_full_batch())
    attempts = [
        lambda: session.step([KEY_SENTINEL, b"x"], _provider([[0.0] * 3] * 2)),
    ]
    for attempt in attempts:
        with pytest.raises(PatientDPSGDError) as excinfo:
            attempt()
        assert KEY_SENTINEL not in str(excinfo.value)
        assert KEY_SENTINEL not in repr(excinfo.value)
        assert excinfo.value.args == (excinfo.value.code,)


def test_records_and_declaration_are_immutable_and_digest_is_stable():
    declaration = _declaration()
    again = _declaration()

    assert declaration.digest == again.digest
    assert declaration.digest.startswith("sha256:")
    assert _declaration(clip_norm=2.0).digest != declaration.digest
    with pytest.raises(FrozenInstanceError):
        declaration.clip_norm = 5.0  # type: ignore[misc]
    payload = declaration.to_dict()
    assert payload["privacy_unit"] == "patient"
    assert payload["recipe_digest"] == RECIPE
    assert payload["schema_version"] == PATIENT_DP_SGD_SCHEMA_VERSION


def test_session_update_is_a_tuple_detached_from_provider_rows():
    rows = [[0.1, 0.2, 0.3]]
    session = _test_session(_full_batch())
    result = session.step(["p1"], _provider(rows))
    assert isinstance(result.update, tuple)
    rows[0][0] = 99.0
    assert 99.0 not in result.update


def test_reason_code_vocabulary_is_closed_and_ascii():
    assert all(
        code.isascii() and code.startswith("dp_")
        for code in PATIENT_DP_SGD_REASON_CODES
    )
    with pytest.raises(ValueError):
        PatientDPSGDError("not_a_reason_code")


# --------------------------------------------------------------------------
# Optional framework adapter.
# --------------------------------------------------------------------------


def test_torch_adapter_reports_unavailable_framework(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(dp_module, "_import_torch", lambda: missing("torch"))
    with pytest.raises(PatientDPSGDError) as excinfo:
        flatten_torch_per_example_gradients([])
    assert _code(excinfo) == "dp_adapter_unavailable"


@pytest.mark.parametrize("clip,noise", ((1e-300, 1e-300), (1e300, 1e300)))
def test_unrepresentable_noise_scale_is_refused(clip, noise):
    with pytest.raises(PatientDPSGDError, match="dp_config_invalid"):
        _declaration(clip_norm=clip, noise_multiplier=noise)


@pytest.mark.parametrize("batch,keys", ((1e-308, ["a"]), (1.0, ["a", "b"])))
def test_nonfinite_aggregate_fails_closed(batch, keys):
    session = _test_session(
        _full_batch(
            clip_norm=1e308, noise_multiplier=1e-300, expected_batch_size=batch
        ),
        dimension=1,
    )
    with pytest.raises(PatientDPSGDError, match="dp_gradient_invalid"):
        session.step(keys, _provider([[1e308] for _ in keys]))
    assert session.closed
    assert session.steps_taken == 1


def test_huge_integer_gradient_has_value_free_error_and_closes_session():
    session = _test_session(_full_batch(), dimension=1)
    with pytest.raises(PatientDPSGDError, match="dp_gradient_invalid"):
        session.step(["a"], _provider([[10**1000]]))
    assert session.closed
