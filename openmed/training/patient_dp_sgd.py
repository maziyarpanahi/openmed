"""Patient-level DP-SGD step for single-site adapter fine-tuning.

This module implements one framework-neutral DP-SGD step over per-example
gradients that the caller computes. The privacy unit is the patient: notes are
grouped by an opaque patient key, patients are Poisson-sampled with an
injected randomness source, each sampled patient's mean note gradient is
clipped to an L2 bound, Gaussian noise scaled to that bound is added to the
sum, and the result is divided by the declared expected batch size.

The module does not ship accountant math. A caller injects an accountant and
receives the accountant inputs (sampling rate, noise multiplier and executed
steps). A privacy report names the unit, the accountant and the assumptions;
synthetic utility results are not evidence of a formal guarantee.

Gradients, patient keys, note text and noise draws never appear in records,
reprs, errors or logs. Errors carry a fixed reason code only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import random
import re
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Final

PATIENT_DP_SGD_SCHEMA_VERSION: Final = "openmed.training.patient_dp_sgd.v1"

PATIENT_DP_SGD_REASON_CODES: Final = frozenset(
    {
        "dp_accountant_failed",
        "dp_accountant_invalid",
        "dp_adapter_unavailable",
        "dp_config_invalid",
        "dp_gradient_invalid",
        "dp_gradient_provider_failed",
        "dp_mode_invalid",
        "dp_noise_missing",
        "dp_patient_key_invalid",
        "dp_privacy_unit_invalid",
        "dp_privacy_unit_note_level",
        "dp_recipe_digest_invalid",
        "dp_rng_invalid",
        "dp_rng_not_csprng",
        "dp_rng_reused",
        "dp_sampling_not_poisson",
        "dp_session_closed",
        "dp_step_budget_exhausted",
        "dp_test_mode_no_guarantee",
    }
)

PRIVACY_UNITS: Final = ("patient", "note")
BATCH_SAMPLINGS: Final = ("poisson", "fixed_size", "shuffle")
DP_SGD_MODES: Final = ("operational", "test")

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ACCOUNTANT_ID: Final = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,63}\Z")
_MAX_KEY_LENGTH: Final = 256
_MAX_STEPS: Final = 10**9
_MAX_DIMENSION: Final = (1 << 31) - 1
_CLIP_SHRINK: Final = 1.0 - 2.0**-40

# Randomness sources already bound to a session in this process. Weak
# references let a discarded source disappear without keeping it alive.
_BOUND_SOURCES: weakref.WeakSet[Any] = weakref.WeakSet()


class PatientDPSGDError(ValueError):
    """Raised with a fixed reason code and no submitted values.

    Args:
        code: One of ``PATIENT_DP_SGD_REASON_CODES``.

    Attributes:
        code: The fixed reason code, also the only exception argument.
    """

    def __init__(self, code: str) -> None:
        if code not in PATIENT_DP_SGD_REASON_CODES:
            raise ValueError("unknown patient DP-SGD reason code")
        super().__init__(code)
        self.code = code


def _fail(code: str) -> PatientDPSGDError:
    return PatientDPSGDError(code)


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_finite(value: Any) -> float:
    if not _is_real(value):
        raise _fail("dp_config_invalid")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise _fail("dp_config_invalid")
    return number


def _noise_multiplier(value: Any) -> float:
    if value is None:
        raise _fail("dp_noise_missing")
    if not _is_real(value):
        raise _fail("dp_config_invalid")
    number = float(value)
    if number == 0.0:
        raise _fail("dp_noise_missing")
    if not math.isfinite(number) or number < 0.0:
        raise _fail("dp_config_invalid")
    return number


def _step_count(value: Any, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _fail("dp_config_invalid")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class PatientDPSGDAccountantInputs:
    """Inputs an injected accountant needs for one DP-SGD schedule.

    Args:
        sampling_rate: Poisson inclusion probability per privacy unit, or
            ``1.0`` when no sampling amplification is claimed.
        noise_multiplier: Gaussian standard deviation divided by the clip norm.
        steps: Number of steps to account for.
        privacy_unit: ``patient`` or ``note``.
        sampling: ``poisson`` when amplification is claimed, otherwise
            ``full_batch``.
        neighboring_relation: Adjacency the inputs assume.
        test_only: True when the inputs come from a deterministic test session.
    """

    sampling_rate: float
    noise_multiplier: float
    steps: int
    privacy_unit: str
    sampling: str
    neighboring_relation: str = "add_remove_one_unit"
    test_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the accountant inputs."""
        return {
            "neighboring_relation": self.neighboring_relation,
            "noise_multiplier": self.noise_multiplier,
            "privacy_unit": self.privacy_unit,
            "sampling": self.sampling,
            "sampling_rate": self.sampling_rate,
            "steps": self.steps,
            "test_only": self.test_only,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PatientDPSGDDeclaration:
    """Privacy-unit and mechanism declaration bound to one training recipe.

    The declaration lives beside the adapter recipe and references it by
    digest, for example ``AdapterRecipeDryRun.recipe_hash``. It is validated
    on construction and refuses configurations that cannot support the
    claimed guarantee.

    Args:
        recipe_digest: ``sha256:`` reference of the training recipe.
        sampling_rate: Poisson inclusion probability per unit, in ``(0, 1]``.
        expected_batch_size: Fixed, data-independent normalizer, normally the
            sampling rate times the declared number of units.
        noise_multiplier: Noise standard deviation divided by ``clip_norm``.
            ``None`` or zero is refused as missing noise.
        clip_norm: Per-unit L2 contribution bound.
        max_steps: Number of steps the schedule may execute.
        privacy_unit: ``patient`` (default) or ``note``.
        batch_sampling: ``poisson``, ``fixed_size`` or ``shuffle``. Only
            ``poisson`` is accepted when amplification is claimed.
        amplification_claimed: Whether the accountant may use the sampling
            rate for privacy amplification. When False the sampling rate must
            be ``1.0`` so every unit is used each step.
        patient_keys_available: Whether notes can be grouped by patient. A
            note-level unit is refused when this is True.

    Raises:
        PatientDPSGDError: With a fixed reason code for any invalid field.
    """

    recipe_digest: str
    sampling_rate: float
    expected_batch_size: float
    noise_multiplier: float | None
    clip_norm: float
    max_steps: int
    privacy_unit: str = "patient"
    batch_sampling: str = "poisson"
    amplification_claimed: bool = True
    patient_keys_available: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.recipe_digest) is not str
            or _DIGEST.fullmatch(self.recipe_digest) is None
        ):
            raise _fail("dp_recipe_digest_invalid")
        if type(self.privacy_unit) is not str or self.privacy_unit not in PRIVACY_UNITS:
            raise _fail("dp_privacy_unit_invalid")
        if type(self.patient_keys_available) is not bool:
            raise _fail("dp_config_invalid")
        if type(self.amplification_claimed) is not bool:
            raise _fail("dp_config_invalid")
        if self.privacy_unit == "note" and self.patient_keys_available:
            raise _fail("dp_privacy_unit_note_level")
        noise = _noise_multiplier(self.noise_multiplier)
        rate = _positive_finite(self.sampling_rate)
        if rate > 1.0:
            raise _fail("dp_config_invalid")
        batch = _positive_finite(self.expected_batch_size)
        clip = _positive_finite(self.clip_norm)
        _step_count(self.max_steps, minimum=1, maximum=_MAX_STEPS)
        if (
            type(self.batch_sampling) is not str
            or self.batch_sampling not in BATCH_SAMPLINGS
        ):
            raise _fail("dp_sampling_not_poisson")
        if self.amplification_claimed and self.batch_sampling != "poisson":
            raise _fail("dp_sampling_not_poisson")
        if not self.amplification_claimed and rate != 1.0:
            raise _fail("dp_config_invalid")
        object.__setattr__(self, "noise_multiplier", noise)
        object.__setattr__(self, "sampling_rate", rate)
        object.__setattr__(self, "expected_batch_size", batch)
        object.__setattr__(self, "clip_norm", clip)

    @property
    def noise_stddev(self) -> float:
        """Return the Gaussian standard deviation, noise multiplier times clip."""
        return float(self.noise_multiplier) * self.clip_norm

    @property
    def digest(self) -> str:
        """Return the ``sha256:`` digest of the canonical declaration mapping."""
        encoded = json.dumps(
            self.to_dict(), separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def accountant_inputs(
        self, steps: int | None = None
    ) -> PatientDPSGDAccountantInputs:
        """Return accountant inputs for the declared or a partial schedule.

        Args:
            steps: Steps to account for. Defaults to ``max_steps``.

        Returns:
            The sampling rate, noise multiplier and step count for an
            injected accountant.

        Raises:
            PatientDPSGDError: ``dp_config_invalid`` when ``steps`` is not an
                integer between zero and ``max_steps``.
        """
        count = self.max_steps if steps is None else steps
        _step_count(count, minimum=0, maximum=self.max_steps)
        return PatientDPSGDAccountantInputs(
            sampling_rate=self.sampling_rate,
            noise_multiplier=float(self.noise_multiplier),
            steps=count,
            privacy_unit=self.privacy_unit,
            sampling="poisson" if self.amplification_claimed else "full_batch",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible declaration, free of data and keys."""
        return {
            "amplification_claimed": self.amplification_claimed,
            "batch_sampling": self.batch_sampling,
            "clip_norm": self.clip_norm,
            "expected_batch_size": self.expected_batch_size,
            "max_steps": self.max_steps,
            "noise_multiplier": self.noise_multiplier,
            "patient_keys_available": self.patient_keys_available,
            "privacy_unit": self.privacy_unit,
            "recipe_digest": self.recipe_digest,
            "sampling_rate": self.sampling_rate,
            "schema_version": PATIENT_DP_SGD_SCHEMA_VERSION,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class PatientDPSGDStepRecord:
    """Content-free evidence for one executed step.

    The record carries no counts of sampled units or notes, because the
    realized Poisson batch size is not privatized.

    Args:
        step: One-based step index within the session.
        declaration_digest: Digest of the bound declaration.
        mode: ``operational`` or ``test``.
        privacy_unit: ``patient`` or ``note``.
        gradient_dimension: Length of the released update.
        test_only: True for a deterministic test session.
    """

    step: int
    declaration_digest: str
    mode: str
    privacy_unit: str
    gradient_dimension: int
    test_only: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible step record."""
        return {
            "declaration_digest": self.declaration_digest,
            "gradient_dimension": self.gradient_dimension,
            "guarantee_claimed": not self.test_only,
            "mode": self.mode,
            "noise_applied": True,
            "privacy_unit": self.privacy_unit,
            "schema_version": PATIENT_DP_SGD_SCHEMA_VERSION,
            "step": self.step,
            "test_only": self.test_only,
        }


@dataclass(frozen=True, slots=True, repr=False)
class PatientDPSGDStepResult:
    """The noisy, normalized update and its content-free record.

    Args:
        update: Noisy update divided by the expected batch size.
        record: Evidence for the step.
    """

    update: tuple[float, ...]
    record: PatientDPSGDStepRecord

    def __repr__(self) -> str:
        return (
            "PatientDPSGDStepResult("
            f"step={self.record.step}, dimension={len(self.update)})"
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PatientDPSGDPrivacyReport:
    """A privacy statement that names its unit, accountant and assumptions.

    Args:
        epsilon: Value the injected accountant returned.
        delta: Delta the accountant was asked for.
        accountant_id: Caller-supplied accountant identifier.
        accountant_inputs: Inputs passed to the accountant.
        declaration_digest: Digest of the bound declaration.
        recipe_digest: Digest of the training recipe.
        privacy_unit: ``patient`` or ``note``.
        assumptions: Fixed assumption codes the statement depends on.
    """

    epsilon: float
    delta: float
    accountant_id: str
    accountant_inputs: PatientDPSGDAccountantInputs
    declaration_digest: str
    recipe_digest: str
    privacy_unit: str
    assumptions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible privacy report."""
        return {
            "accountant_id": self.accountant_id,
            "accountant_inputs": self.accountant_inputs.to_dict(),
            "assumptions": list(self.assumptions),
            "declaration_digest": self.declaration_digest,
            "delta": self.delta,
            "epsilon": self.epsilon,
            "guarantee_claimed": True,
            "privacy_unit": self.privacy_unit,
            "recipe_digest": self.recipe_digest,
            "schema_version": PATIENT_DP_SGD_SCHEMA_VERSION,
        }


GradientProvider = Callable[[tuple[int, ...]], Any]
Accountant = Callable[..., float]


class PatientDPSGD:
    """Session that runs patient-level DP-SGD steps for one declaration.

    Args:
        declaration: Validated privacy-unit and mechanism declaration.
        gradient_dimension: Length of every per-example gradient row.
        sampling_rng: Randomness source for Poisson sampling.
        noise_rng: Separate randomness source for Gaussian noise.
        mode: ``operational`` (default) accepts only ``random.SystemRandom``
            instances. ``test`` accepts any object with ``random()`` and
            ``normalvariate()``, marks every record ``test_only`` and never
            issues a privacy report.

    Raises:
        PatientDPSGDError: ``dp_mode_invalid``, ``dp_config_invalid``,
            ``dp_rng_not_csprng`` or ``dp_rng_reused``.

    Example:
        >>> import random
        >>> declaration = PatientDPSGDDeclaration(
        ...     recipe_digest="sha256:" + "0" * 64,
        ...     sampling_rate=0.5,
        ...     expected_batch_size=1.0,
        ...     noise_multiplier=1.0,
        ...     clip_norm=1.0,
        ...     max_steps=2,
        ... )
        >>> session = PatientDPSGD(
        ...     declaration,
        ...     gradient_dimension=2,
        ...     sampling_rng=random.SystemRandom(),
        ...     noise_rng=random.SystemRandom(),
        ... )
        >>> result = session.step(
        ...     ["patient-a", "patient-a", "patient-b"],
        ...     lambda positions: [[0.1, 0.2]] * len(positions),
        ... )
        >>> len(result.update)
        2
    """

    __slots__ = (
        "_closed",
        "_declaration",
        "_digest",
        "_dimension",
        "_mode",
        "_noise",
        "_sampling",
        "_steps",
    )

    def __init__(
        self,
        declaration: PatientDPSGDDeclaration,
        *,
        gradient_dimension: int,
        sampling_rng: Any,
        noise_rng: Any,
        mode: str = "operational",
    ) -> None:
        if not isinstance(declaration, PatientDPSGDDeclaration):
            raise _fail("dp_config_invalid")
        if type(mode) is not str or mode not in DP_SGD_MODES:
            raise _fail("dp_mode_invalid")
        _step_count(gradient_dimension, minimum=1, maximum=_MAX_DIMENSION)
        for source in (sampling_rng, noise_rng):
            _check_source(source, mode)
        if sampling_rng is noise_rng:
            raise _fail("dp_rng_reused")
        if sampling_rng in _BOUND_SOURCES or noise_rng in _BOUND_SOURCES:
            raise _fail("dp_rng_reused")
        _BOUND_SOURCES.add(sampling_rng)
        _BOUND_SOURCES.add(noise_rng)
        self._declaration = declaration
        self._digest = declaration.digest
        self._dimension = gradient_dimension
        self._sampling = sampling_rng
        self._noise = noise_rng
        self._mode = mode
        self._steps = 0
        self._closed = False

    def __repr__(self) -> str:
        return (
            f"PatientDPSGD(mode={self._mode!r}, steps_taken={self._steps}, "
            f"max_steps={self._declaration.max_steps})"
        )

    @property
    def declaration(self) -> PatientDPSGDDeclaration:
        """Return the bound declaration."""
        return self._declaration

    @property
    def mode(self) -> str:
        """Return ``operational`` or ``test``."""
        return self._mode

    @property
    def steps_taken(self) -> int:
        """Return attempted steps, including any that failed after sampling."""
        return self._steps

    @property
    def closed(self) -> bool:
        """Return True after a failed step; a closed session runs no steps."""
        return self._closed

    def step(
        self,
        note_patient_keys: Sequence[str | None],
        gradient_provider: GradientProvider,
    ) -> PatientDPSGDStepResult:
        """Run one DP-SGD step and return only the noisy normalized update.

        Each note carries an opaque patient key (or ``None`` for a note-level
        unit). Units are Poisson-sampled in first-appearance order with one
        draw per unit. ``gradient_provider`` receives the ascending note
        positions of the sampled units and must return one gradient row per
        position, as nested sequences or an object with ``tolist()`` such as
        a numpy array. It is not called when no unit is sampled; noise of the
        declared dimension is still released.

        Any failure closes the session so a step cannot be retried until it
        succeeds. The attempt still counts toward ``steps_taken``.

        Args:
            note_patient_keys: One opaque key per note.
            gradient_provider: Callable from note positions to gradient rows.

        Returns:
            The update and a content-free step record.

        Raises:
            PatientDPSGDError: With a fixed reason code.
        """
        if self._closed:
            raise _fail("dp_session_closed")
        if self._steps >= self._declaration.max_steps:
            raise _fail("dp_step_budget_exhausted")
        self._steps += 1
        try:
            update = self._run(note_patient_keys, gradient_provider)
        except PatientDPSGDError:
            self._closed = True
            raise
        record = PatientDPSGDStepRecord(
            step=self._steps,
            declaration_digest=self._digest,
            mode=self._mode,
            privacy_unit=self._declaration.privacy_unit,
            gradient_dimension=self._dimension,
            test_only=self._mode == "test",
        )
        return PatientDPSGDStepResult(update=update, record=record)

    def accountant_inputs(self) -> PatientDPSGDAccountantInputs:
        """Return accountant inputs for the steps attempted so far."""
        inputs = self._declaration.accountant_inputs(self._steps)
        return replace(inputs, test_only=self._mode == "test")

    def privacy_report(
        self,
        accountant: Accountant,
        *,
        delta: float,
        accountant_id: str,
    ) -> PatientDPSGDPrivacyReport:
        """Ask an injected accountant for epsilon and return a privacy report.

        The accountant is called with keyword arguments ``sampling_rate``,
        ``noise_multiplier``, ``steps`` and ``delta`` and must return a finite,
        non-negative epsilon. Test sessions never issue a report.

        Args:
            accountant: Injected accountant callable.
            delta: Target delta in ``(0, 1)``.
            accountant_id: Lowercase identifier of the accountant, at most 64
                characters.

        Returns:
            A report that names the unit, accountant and assumptions.

        Raises:
            PatientDPSGDError: ``dp_test_mode_no_guarantee``,
                ``dp_accountant_invalid`` or ``dp_accountant_failed``.
        """
        if self._mode != "operational":
            raise _fail("dp_test_mode_no_guarantee")
        if not callable(accountant):
            raise _fail("dp_accountant_invalid")
        if (
            not _is_real(delta)
            or not math.isfinite(float(delta))
            or not 0.0 < float(delta) < 1.0
        ):
            raise _fail("dp_accountant_invalid")
        if type(accountant_id) is not str or not _ACCOUNTANT_ID.fullmatch(
            accountant_id
        ):
            raise _fail("dp_accountant_invalid")
        inputs = self.accountant_inputs()
        try:
            epsilon = accountant(
                sampling_rate=inputs.sampling_rate,
                noise_multiplier=inputs.noise_multiplier,
                steps=inputs.steps,
                delta=float(delta),
            )
        except Exception:
            raise _fail("dp_accountant_failed") from None
        if (
            not _is_real(epsilon)
            or not math.isfinite(float(epsilon))
            or float(epsilon) < 0.0
        ):
            raise _fail("dp_accountant_invalid")
        return PatientDPSGDPrivacyReport(
            epsilon=float(epsilon),
            delta=float(delta),
            accountant_id=accountant_id,
            accountant_inputs=inputs,
            declaration_digest=self._digest,
            recipe_digest=self._declaration.recipe_digest,
            privacy_unit=self._declaration.privacy_unit,
            assumptions=_assumptions(self._declaration),
        )

    def _run(
        self,
        note_patient_keys: Sequence[str | None],
        gradient_provider: GradientProvider,
    ) -> tuple[float, ...]:
        declaration = self._declaration
        units = _group_units(note_patient_keys, declaration.privacy_unit)
        rate = declaration.sampling_rate
        selected = [unit for unit in units if self._draw_uniform() < rate]
        columns: list[list[float]] = [[] for _ in range(self._dimension)]
        positions = tuple(sorted(position for unit in selected for position in unit))
        if positions:
            if not callable(gradient_provider):
                raise _fail("dp_gradient_provider_failed")
            try:
                provided = gradient_provider(positions)
            except Exception:
                raise _fail("dp_gradient_provider_failed") from None
            rows = _gradient_rows(provided, len(positions), self._dimension)
            row_of = {position: index for index, position in enumerate(positions)}
            for unit in selected:
                notes = [rows[row_of[position]] for position in unit]
                contribution = _patient_contribution(
                    notes, clip_norm=declaration.clip_norm
                )
                for column, value in zip(columns, contribution, strict=True):
                    column.append(value)
        stddev = declaration.noise_stddev
        batch = declaration.expected_batch_size
        return tuple(
            (math.fsum(column) + self._draw_gaussian(stddev)) / batch
            for column in columns
        )

    def _draw_uniform(self) -> float:
        try:
            value = self._sampling.random()
        except Exception:
            raise _fail("dp_rng_invalid") from None
        if not isinstance(value, float) or not 0.0 <= value < 1.0:
            raise _fail("dp_rng_invalid")
        return value

    def _draw_gaussian(self, stddev: float) -> float:
        try:
            value = self._noise.normalvariate(0.0, stddev)
        except Exception:
            raise _fail("dp_rng_invalid") from None
        if not isinstance(value, float) or not math.isfinite(value):
            raise _fail("dp_rng_invalid")
        return value


def _check_source(source: Any, mode: str) -> None:
    if mode == "operational":
        if type(source) is not random.SystemRandom:
            raise _fail("dp_rng_not_csprng")
        return
    if source is None or not all(
        callable(getattr(source, name, None)) for name in ("random", "normalvariate")
    ):
        raise _fail("dp_rng_not_csprng")
    try:
        weakref.ref(source)
    except TypeError:
        raise _fail("dp_rng_not_csprng") from None


def _group_units(
    note_patient_keys: Sequence[str | None], privacy_unit: str
) -> list[tuple[int, ...]]:
    if isinstance(note_patient_keys, (str, bytes)) or not isinstance(
        note_patient_keys, Sequence
    ):
        raise _fail("dp_patient_key_invalid")
    if privacy_unit == "note":
        if any(key is not None for key in note_patient_keys):
            raise _fail("dp_privacy_unit_note_level")
        return [(position,) for position in range(len(note_patient_keys))]
    groups: dict[str, list[int]] = {}
    for position, key in enumerate(note_patient_keys):
        if type(key) is not str or not 1 <= len(key) <= _MAX_KEY_LENGTH:
            raise _fail("dp_patient_key_invalid")
        groups.setdefault(key, []).append(position)
    return [tuple(positions) for positions in groups.values()]


def _as_list(value: Any) -> Any:
    to_list = getattr(value, "tolist", None)
    if callable(to_list) and not isinstance(value, list):
        try:
            return to_list()
        except Exception:
            raise _fail("dp_gradient_invalid") from None
    return value


def _gradient_rows(
    provided: Any, expected_rows: int, dimension: int
) -> list[tuple[float, ...]]:
    matrix = _as_list(provided)
    if isinstance(matrix, (str, bytes)) or not isinstance(matrix, Sequence):
        raise _fail("dp_gradient_invalid")
    if len(matrix) != expected_rows:
        raise _fail("dp_gradient_invalid")
    rows: list[tuple[float, ...]] = []
    for raw_row in matrix:
        row = _as_list(raw_row)
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != dimension
        ):
            raise _fail("dp_gradient_invalid")
        values = []
        for value in row:
            if not _is_real(value):
                raise _fail("dp_gradient_invalid")
            number = float(value)
            if not math.isfinite(number):
                raise _fail("dp_gradient_invalid")
            values.append(number)
        rows.append(tuple(values))
    return rows


def _l2(vector: Sequence[float]) -> float:
    return math.hypot(*vector)


def _patient_contribution(
    notes: Sequence[Sequence[float]], *, clip_norm: float
) -> tuple[float, ...]:
    """Return the mean of one unit's note gradients clipped to ``clip_norm``.

    Values are pre-scaled by their largest magnitude so large finite gradients
    cannot overflow, and the result is shrunk until its rounded norm is at
    most the bound.
    """
    count = len(notes)
    dimension = len(notes[0])
    largest = max(abs(value) for note in notes for value in note)
    if largest == 0.0:
        return (0.0,) * dimension
    unit = [
        math.fsum(note[index] / largest for note in notes) / count
        for index in range(dimension)
    ]
    unit_norm = _l2(unit)
    if unit_norm == 0.0:
        return (0.0,) * dimension
    if largest * unit_norm <= clip_norm:
        vector = [value * largest for value in unit]
    else:
        factor = clip_norm / unit_norm
        vector = [value * factor for value in unit]
    for _ in range(64):
        if _l2(vector) <= clip_norm:
            return tuple(vector)
        vector = [value * _CLIP_SHRINK for value in vector]
    raise _fail("dp_gradient_invalid")


def _assumptions(declaration: PatientDPSGDDeclaration) -> tuple[str, ...]:
    unit = declaration.privacy_unit
    sampling = (
        f"poisson_{unit}_sampling"
        if declaration.amplification_claimed
        else "no_sampling_amplification"
    )
    return (
        sampling,
        "add_remove_one_unit_adjacency",
        f"per_{unit}_l2_clipping",
        "gaussian_noise_scaled_to_clip_norm",
        "fixed_expected_batch_normalization",
        "os_csprng_randomness",
        "floating_point_gaussian_not_hardened",
        "accountant_injected_not_verified",
    )


def _import_torch() -> Any:
    return importlib.import_module("torch")


def _torch() -> Any:
    try:
        return _import_torch()
    except ImportError:
        raise _fail("dp_adapter_unavailable") from None


def flatten_torch_per_example_gradients(
    per_parameter_gradients: Sequence[Any],
) -> list[list[float]]:
    """Flatten per-example PyTorch gradients into neutral gradient rows.

    Torch is imported lazily and is optional. Each tensor holds one parameter's
    per-example gradients with the example axis first, for example
    ``grad_sample`` tensors of shape ``(examples, *parameter_shape)``. Rows
    are concatenated in the given parameter order.

    Args:
        per_parameter_gradients: One tensor per trainable parameter.

    Returns:
        One list of floats per example, suitable for ``PatientDPSGD.step``.

    Raises:
        PatientDPSGDError: ``dp_adapter_unavailable`` when torch is missing or
            ``dp_gradient_invalid`` for empty or inconsistent tensors.
    """
    torch = _torch()
    tensors = list(per_parameter_gradients)
    if not tensors or not all(isinstance(item, torch.Tensor) for item in tensors):
        raise _fail("dp_gradient_invalid")
    examples = tensors[0].shape[0] if tensors[0].dim() else 0
    if examples < 1 or any(
        item.dim() < 1 or item.shape[0] != examples for item in tensors
    ):
        raise _fail("dp_gradient_invalid")
    flat = [
        item.detach().to(device="cpu", dtype=torch.float64).reshape(examples, -1)
        for item in tensors
    ]
    return torch.cat(flat, dim=1).tolist()


def unflatten_torch_update(update: Sequence[float], like: Sequence[Any]) -> list[Any]:
    """Split a flat DP-SGD update into tensors shaped like the parameters.

    Torch is imported lazily and is optional.

    Args:
        update: ``PatientDPSGDStepResult.update``.
        like: Parameters whose shapes, dtypes and devices the result follows,
            in the order used for flattening.

    Returns:
        One tensor per parameter.

    Raises:
        PatientDPSGDError: ``dp_adapter_unavailable`` when torch is missing or
            ``dp_gradient_invalid`` when sizes do not match.
    """
    torch = _torch()
    parameters = list(like)
    if not all(isinstance(item, torch.Tensor) for item in parameters):
        raise _fail("dp_gradient_invalid")
    values = list(update)
    if sum(item.numel() for item in parameters) != len(values):
        raise _fail("dp_gradient_invalid")
    tensors = []
    offset = 0
    for item in parameters:
        size = item.numel()
        chunk = torch.tensor(values[offset : offset + size], dtype=torch.float64)
        tensors.append(
            chunk.reshape(item.shape).to(dtype=item.dtype, device=item.device)
        )
        offset += size
    return tensors


__all__ = [
    "BATCH_SAMPLINGS",
    "DP_SGD_MODES",
    "PATIENT_DP_SGD_REASON_CODES",
    "PATIENT_DP_SGD_SCHEMA_VERSION",
    "PRIVACY_UNITS",
    "PatientDPSGD",
    "PatientDPSGDAccountantInputs",
    "PatientDPSGDDeclaration",
    "PatientDPSGDError",
    "PatientDPSGDPrivacyReport",
    "PatientDPSGDStepRecord",
    "PatientDPSGDStepResult",
    "flatten_torch_per_example_gradients",
    "unflatten_torch_update",
]
