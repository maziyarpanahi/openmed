"""Dropout-aware distributed discrete Gaussian noise shares.

Each client adds an integer noise share to its encoded, clipped update before
masked secure aggregation, so an honest-but-curious coordinator observes only
the noisy modular sum. Shares are calibrated to the recovery threshold ``t``
rather than the client count ``n``: any unmaskable survivor set carries at
least the declared aggregate noise variance, and a survivor count below ``t``
aborts before unmasking.

Sampling is exact. Each share coordinate is drawn from the discrete Gaussian
with the Canonne, Kamath and Steinke algorithms (CKS20, "The Discrete Gaussian
for Differential Privacy", arXiv:2004.00010, Algorithms 1-3) using only integer
and ``fractions.Fraction`` arithmetic and an injected CSPRNG.

The bounds follow Kairouz, Liu and Steinke (KLS21, "The Distributed Discrete
Gaussian Mechanism for Federated Learning with Secure Aggregation",
arXiv:2102.06387). A sum of discrete Gaussians is not exactly a discrete
Gaussian; the calibration records the closeness term ``tau`` of Theorem 1,
eq. (3) and Proposition 14 instead of assuming it away.

Masking and recovery, the fixed-point codec and central mechanisms are owned by
separate modules. This module never logs, stores or serializes noise draws,
seeds or per-client values.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Final

FEDERATED_DISTRIBUTED_NOISE_SCHEMA_VERSION: Final = (
    "openmed.training.federated_distributed_noise.v1"
)

SHARE_VARIANCE_MARGIN: Final = Fraction(1, 2**20)
"""Relative margin added to each share scale above ``target / t``."""

SHARE_VARIANCE_DEFICIT: Final = Fraction(1, 2**21)
"""Proven relative variance deficit of one share when its scale is at least 1."""

MINIMUM_SHARE_SCALE: Final = Fraction(1)
"""Smallest per-share discrete Gaussian scale ``sigma**2`` in grid units."""

DISTRIBUTED_NOISE_REASON_CODES: Final = frozenset(
    {
        "invalid_client_count",
        "invalid_recovery_threshold",
        "invalid_target_variance",
        "invalid_dimension",
        "invalid_modulus",
        "invalid_clip_bound",
        "invalid_tail_bound",
        "share_scale_below_minimum",
        "modulus_wraparound_risk",
        "invalid_survivor_count",
        "survivors_below_threshold",
        "invalid_calibration",
        "invalid_sampler",
        "test_sampler_refused",
        "invalid_signal",
        "signal_out_of_bounds",
        "randomness_not_operational",
        "invalid_test_seed",
        "invalid_sampler_scale",
        "sampler_round_limit",
        "unmask_failed",
        "invalid_unmasked_sum",
    }
)

_MAX_CLIENTS: Final = 2**20
_MAX_DIMENSION: Final = 2**31 - 1
_MAX_RATIONAL: Final = Fraction(2**128)
_MAX_MODULUS: Final = 2**512
_MIN_TAIL_BITS: Final = 20
_MAX_TAIL_BITS: Final = 256
_MAX_SEED: Final = 2**64
_MAX_ROUNDS: Final = 10_000
_TAU_EXPONENT_CAP: Final = 4096
_DECIMAL_DIGITS: Final = 12

# Rational one-sided constants: pi**2 > 9.8696, log2(e) > 1.4426, ln(2) < 0.6932.
_PI_SQUARED_LOWER: Final = Fraction(98696, 10000)
_LOG2_E_LOWER: Final = Fraction(14426, 10000)
_LN_2_UPPER: Final = Fraction(6932, 10000)

_REFERENCES: Final = (
    "CKS20 arXiv:2004.00010 Algorithms 1-3",
    "KLS21 arXiv:2102.06387v4 Theorem 1 eq. (2)-(3)",
    "KLS21 arXiv:2102.06387v4 Proposition 14 eq. (19)",
    "KLS21 arXiv:2102.06387v4 Lemma 15",
)


class DistributedNoiseError(ValueError):
    """Raised with a fixed reason code and never with submitted values.

    Args:
        code: One entry of ``DISTRIBUTED_NOISE_REASON_CODES``.
    """

    def __init__(self, code: str) -> None:
        if code not in DISTRIBUTED_NOISE_REASON_CODES:
            code = "invalid_calibration"
        self.code = code
        super().__init__(code)


def _plain_int(value: Any) -> bool:
    return type(value) is int


def _rational(value: Any, code: str, *, maximum: Fraction) -> Fraction:
    """Convert a positive int, Fraction or finite float exactly, else fail."""
    if isinstance(value, bool):
        raise DistributedNoiseError(code)
    if type(value) is int or type(value) is Fraction:
        result = Fraction(value)
    elif type(value) is float and math.isfinite(value):
        result = Fraction(value)
    else:
        raise DistributedNoiseError(code)
    if not 0 < result <= maximum:
        raise DistributedNoiseError(code)
    return result


def _ceil_sqrt_int(value: int) -> int:
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _ceil_decimal(value: Fraction) -> str:
    """Return a decimal string that is exact or rounded up, never down."""
    if value == 0:
        return "0"
    exponent = len(str(value.numerator)) - len(str(value.denominator))
    while Fraction(10) ** exponent > value:
        exponent -= 1
    while Fraction(10) ** (exponent + 1) <= value:
        exponent += 1
    scale = Fraction(10) ** (_DECIMAL_DIGITS - 1 - exponent)
    mantissa = math.ceil(value * scale)
    if mantissa == 10**_DECIMAL_DIGITS:
        mantissa //= 10
        exponent += 1
    digits = str(mantissa).rstrip("0") or "0"
    head, tail = digits[0], digits[1:]
    text = head + ("." + tail if tail else "")
    return f"{text}e{exponent:+d}"


def _sum_closeness_tau_bound(share_scale: Fraction, client_count: int) -> Fraction:
    """Upper-bound ``tau`` of KLS21 Theorem 1, eq. (3), with rationals only.

    ``tau = 10 * sum_{k=1}^{n-1} exp(-2 pi^2 sigma^2 k / (k + 1))``. Each term
    is bounded by ``2**-floor(x * log2(e))`` with lower rational bounds for
    ``pi**2`` and ``log2(e)``, and exponents are capped at a large constant,
    which only loosens the bound.
    """
    rate = 2 * _PI_SQUARED_LOWER * _LOG2_E_LOWER * share_scale
    numerator, denominator = rate.numerator, rate.denominator
    exponents = [
        min((numerator * k) // (denominator * (k + 1)), _TAU_EXPONENT_CAP)
        for k in range(1, client_count)
    ]
    if not exponents:
        return Fraction(0)
    top = max(exponents)
    total = sum(1 << (top - exponent) for exponent in exponents)
    return Fraction(10 * total, 1 << top)


def _bernoulli_rational(rng: Any, probability: Fraction) -> bool:
    """Draw Bernoulli(p) for rational ``0 <= p <= 1`` with one exact uniform."""
    return rng.randrange(probability.denominator) < probability.numerator


def _bernoulli_exp_neg(rng: Any, gamma: Fraction) -> bool:
    """Draw Bernoulli(exp(-gamma)) exactly (CKS20 Algorithm 1), gamma >= 0."""
    whole = gamma.numerator // gamma.denominator
    for _ in range(whole):
        if not _bernoulli_exp_neg_unit(rng, Fraction(1)):
            return False
    return _bernoulli_exp_neg_unit(rng, gamma - whole)


def _bernoulli_exp_neg_unit(rng: Any, gamma: Fraction) -> bool:
    """CKS20 Algorithm 1 for ``0 <= gamma <= 1``."""
    k = 1
    for _ in range(_MAX_ROUNDS):
        if not _bernoulli_rational(rng, gamma / k):
            return k % 2 == 1
        k += 1
    raise DistributedNoiseError("sampler_round_limit")


def _discrete_laplace(rng: Any, scale: int) -> int:
    """Draw discrete Laplace with integer scale exactly (CKS20 Algorithm 2)."""
    for _ in range(_MAX_ROUNDS):
        uniform = rng.randrange(scale)
        if not _bernoulli_exp_neg(rng, Fraction(uniform, scale)):
            continue
        geometric = 0
        for _ in range(_MAX_ROUNDS):
            if not _bernoulli_exp_neg(rng, Fraction(1)):
                break
            geometric += 1
        else:
            raise DistributedNoiseError("sampler_round_limit")
        magnitude = uniform + scale * geometric
        negative = _bernoulli_rational(rng, Fraction(1, 2))
        if negative and magnitude == 0:
            continue
        return -magnitude if negative else magnitude
    raise DistributedNoiseError("sampler_round_limit")


def _discrete_gaussian(rng: Any, sigma_squared: Fraction) -> int:
    """Draw N_Z(0, sigma**2) exactly (CKS20 Algorithm 3)."""
    floor_sigma = math.isqrt(sigma_squared.numerator // sigma_squared.denominator)
    scale = floor_sigma + 1
    center = sigma_squared / scale
    for _ in range(_MAX_ROUNDS):
        candidate = _discrete_laplace(rng, scale)
        gamma = (abs(candidate) - center) ** 2 / (2 * sigma_squared)
        if _bernoulli_exp_neg(rng, gamma):
            return candidate
    raise DistributedNoiseError("sampler_round_limit")


class DiscreteGaussianShareSampler:
    """Exact discrete Gaussian sampler bound to an injected CSPRNG.

    Operational construction accepts only an instance whose type is exactly
    ``random.SystemRandom``; subclasses, ``random.Random`` and other
    generators are refused. Seeded generators exist only through
    ``for_testing``, and ``noised_client_share`` refuses them unless the caller
    opts in explicitly.

    Args:
        rng: An operating-system CSPRNG, exactly ``random.SystemRandom``.

    Raises:
        DistributedNoiseError: ``randomness_not_operational`` for any other
            generator.
    """

    __slots__ = ("_rng", "_test_mode")

    def __init__(self, rng: random.SystemRandom) -> None:
        if type(rng) is not random.SystemRandom:
            raise DistributedNoiseError("randomness_not_operational")
        self._rng: Any = rng
        self._test_mode = False

    @classmethod
    def for_testing(cls, seed: int) -> DiscreteGaussianShareSampler:
        """Return a deterministic sampler for synthetic tests only.

        Args:
            seed: Integer seed in ``[0, 2**64)``. It is never stored in records.

        Returns:
            A sampler marked as test mode.

        Raises:
            DistributedNoiseError: ``invalid_test_seed`` for a non-integer,
                boolean or out-of-range seed.
        """
        if not _plain_int(seed) or not 0 <= seed < _MAX_SEED:
            raise DistributedNoiseError("invalid_test_seed")
        sampler = cls.__new__(cls)
        sampler._rng = random.Random(seed)
        sampler._test_mode = True
        return sampler

    @property
    def is_test_mode(self) -> bool:
        """Whether this sampler uses a seeded, non-operational generator."""
        return self._test_mode

    def sample(self, sigma_squared: int | Fraction | float) -> int:
        """Draw one integer from the discrete Gaussian ``N_Z(0, sigma**2)``.

        Args:
            sigma_squared: Positive scale parameter. Floats convert exactly to
                a ``Fraction``; sampling itself uses no floating point.

        Returns:
            One exact integer sample.

        Raises:
            DistributedNoiseError: ``invalid_sampler_scale`` for an invalid
                scale, or ``sampler_round_limit`` if a bounded rejection loop
                is exhausted (never a clipped or biased draw).
        """
        scale = _rational(sigma_squared, "invalid_sampler_scale", maximum=_MAX_RATIONAL)
        return _discrete_gaussian(self._rng, scale)

    def __repr__(self) -> str:
        mode = "test" if self._test_mode else "operational"
        return f"{type(self).__name__}(mode={mode})"


@dataclass(frozen=True, slots=True, repr=False)
class DistributedNoiseMechanismRecord:
    """Value-free declaration of the trust boundary and share calibration.

    The record contains configuration and calibration-derived bounds only. It
    never contains noise draws, seeds, client updates, survivor identities or
    per-client values. Rational values are exact ``p/q`` strings; ``tau`` and
    ``rho`` are decimal upper bounds rounded up.

    Args:
        fields: Canonical record body, excluding the digest.
        digest: ``sha256:`` digest of the canonical JSON body.
    """

    fields: tuple[tuple[str, Any], ...]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping of the record body and its digest."""
        payload = {key: value for key, value in self.fields}
        payload["references"] = list(payload["references"])
        payload["digest"] = self.digest
        return payload

    def __repr__(self) -> str:
        return f"{type(self).__name__}(digest={self.digest})"


@dataclass(frozen=True, slots=True, repr=False)
class DistributedNoiseCalibration:
    """Per-client share calibration for a declared aggregate noise variance.

    All quantities are in integer grid units of the fixed-point codec. Every
    derived bound is computed in ``__post_init__`` from the declared
    configuration, so an inconsistent calibration cannot be constructed.

    Calibration (KLS21, arXiv:2102.06387):

    * Share scale: ``sigma**2 = target / t * (1 + 2**-20)``, required to be at
      least 1. For ``X ~ N_Z(0, sigma**2)``, Poisson summation gives
      ``Var[X] = sigma**2 - 8 pi^2 sigma^4 S / (1 + 2 sum_j q^(j^2))`` with
      ``q = exp(-2 pi^2 sigma^2)`` and ``S = sum_{j>=1} j^2 q^(j^2) <= 2q``,
      so ``Var[X] >= sigma**2 * (1 - 16 pi^2 sigma^2 q) >=
      sigma**2 * (1 - 2**-21)`` for ``sigma**2 >= 1`` (and ``Var[X] <
      sigma**2`` by Lemma 15). Variances of independent shares add exactly,
      so ``s >= t`` survivors carry variance at least ``target``.
    * Sensitivity after randomized rounding: ``Delta_2**2 <= (c +
      ceil(sqrt(d)))**2`` for L2 clip ``c`` in grid units (Theorem 1,
      eq. (2), probability-one branch with ``gamma = 1``). Client signals are
      refused unless they satisfy this bound and ``|x_j| <= ceil(c)``.
    * Sum of shares: ``tau_n = 10 * sum_{k=1}^{n-1} exp(-2 pi^2 sigma^2 k /
      (k + 1))`` (Theorem 1, eq. (3)). By Proposition 14, eq. (19), adding
      the sum of ``s`` shares gives ``eps**2 / 2``-zCDP with ``eps**2 <=
      Delta_2**2 / (s sigma**2) + 2 tau_s d``; since ``tau_s <= tau_n``, every
      ``t <= s <= n`` satisfies ``rho <= (Delta_2**2 / (t sigma**2) +
      2 tau_n d) / 2``, reported as ``zcdp_rho_bound`` under the stated trust
      boundary (honest-but-curious coordinator, non-colluding clients).
      Adjacency zeroes one client signal while retaining its noise share;
      the survivor set is fixed and dropout is data-independent.
    * Modular clipping: each coordinate of the sum of at most ``n`` shares is
      subgaussian with variance proxy ``n sigma**2`` (Lemma 15), so
      ``P(max_j |Z_j| > k) <= 2 d exp(-k**2 / (2 n sigma**2)) <= 2**-b`` for
      ``k**2 >= 2 n sigma**2 ln 2 (b + 1 + ceil(log2 d))``. The modulus must
      satisfy ``n ceil(c) + k <= (m - 1) // 2``, a fail-closed form of the
      wraparound condition behind Proposition 35 and Theorem 36.

    Args:
        client_count: Number of enrolled clients ``n``, from 2 to 2**20.
        recovery_threshold: Minimum recoverable quorum ``t``, ``1 <= t <= n``.
        target_aggregate_variance: Declared minimum variance of the aggregate
            noise for any recoverable survivor set.
        dimension: Encoded update length ``d``.
        modulus: Secure-aggregation modulus ``m``.
        clip_l2_norm: Per-client L2 clip ``c`` in grid units before rounding.
        tail_bound_bits: Declared wraparound probability bound ``2**-b``.
    """

    client_count: int
    recovery_threshold: int
    target_aggregate_variance: Fraction
    dimension: int
    modulus: int
    clip_l2_norm: Fraction
    tail_bound_bits: int = 64
    share_scale: Fraction = field(init=False)
    coordinate_bound: int = field(init=False)
    l2_sensitivity_squared_bound: Fraction = field(init=False)
    aggregate_signal_bound: int = field(init=False)
    noise_tail_bound: int = field(init=False)
    sum_closeness_tau_bound: Fraction = field(init=False)
    zcdp_rho_bound: Fraction = field(init=False)

    def __post_init__(self) -> None:
        n = self.client_count
        if not _plain_int(n) or not 2 <= n <= _MAX_CLIENTS:
            raise DistributedNoiseError("invalid_client_count")
        t = self.recovery_threshold
        if not _plain_int(t) or not 1 <= t <= n:
            raise DistributedNoiseError("invalid_recovery_threshold")
        target = _rational(
            self.target_aggregate_variance,
            "invalid_target_variance",
            maximum=_MAX_RATIONAL,
        )
        d = self.dimension
        if not _plain_int(d) or not 1 <= d <= _MAX_DIMENSION:
            raise DistributedNoiseError("invalid_dimension")
        m = self.modulus
        if not _plain_int(m) or not 3 <= m <= _MAX_MODULUS:
            raise DistributedNoiseError("invalid_modulus")
        clip = _rational(self.clip_l2_norm, "invalid_clip_bound", maximum=_MAX_RATIONAL)
        bits = self.tail_bound_bits
        if not _plain_int(bits) or not _MIN_TAIL_BITS <= bits <= _MAX_TAIL_BITS:
            raise DistributedNoiseError("invalid_tail_bound")

        share_scale = target / t * (1 + SHARE_VARIANCE_MARGIN)
        if share_scale < MINIMUM_SHARE_SCALE:
            raise DistributedNoiseError("share_scale_below_minimum")
        coordinate_bound = math.ceil(clip)
        sensitivity_squared = (clip + _ceil_sqrt_int(d)) ** 2
        signal_bound = n * coordinate_bound
        tail_square = (
            2 * n * share_scale * _LN_2_UPPER * (bits + 1 + (d - 1).bit_length())
        )
        noise_bound = _ceil_sqrt_int(math.ceil(tail_square))
        if signal_bound + noise_bound > (m - 1) // 2:
            raise DistributedNoiseError("modulus_wraparound_risk")
        tau = _sum_closeness_tau_bound(share_scale, n)
        rho = (sensitivity_squared / (t * share_scale) + 2 * tau * d) / 2

        values = {
            "target_aggregate_variance": target,
            "clip_l2_norm": clip,
            "share_scale": share_scale,
            "coordinate_bound": coordinate_bound,
            "l2_sensitivity_squared_bound": sensitivity_squared,
            "aggregate_signal_bound": signal_bound,
            "noise_tail_bound": noise_bound,
            "sum_closeness_tau_bound": tau,
            "zcdp_rho_bound": rho,
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def _survivors(self, survivor_count: Any) -> int:
        if not _plain_int(survivor_count) or not (
            0 <= survivor_count <= self.client_count
        ):
            raise DistributedNoiseError("invalid_survivor_count")
        if survivor_count < self.recovery_threshold:
            raise DistributedNoiseError("survivors_below_threshold")
        return survivor_count

    def require_recovery_quorum(self, survivor_count: int) -> None:
        """Refuse to proceed toward unmasking below the recovery threshold.

        Args:
            survivor_count: Number of clients whose masked shares survived.

        Raises:
            DistributedNoiseError: ``invalid_survivor_count`` or
                ``survivors_below_threshold``.
        """
        self._survivors(survivor_count)

    def aggregate_scale(self, survivor_count: int) -> Fraction:
        """Return the summed share scale ``s * sigma**2`` for ``s`` survivors.

        Args:
            survivor_count: Survivor count ``s`` with ``t <= s <= n``.

        Returns:
            The exact aggregate scale parameter.
        """
        return self._survivors(survivor_count) * self.share_scale

    def aggregate_variance_lower_bound(self, survivor_count: int) -> Fraction:
        """Return a proven lower bound on the aggregate noise variance.

        Args:
            survivor_count: Survivor count ``s`` with ``t <= s <= n``.

        Returns:
            ``s * sigma**2 * (1 - 2**-21)``, which is at least the target.
        """
        return self.aggregate_scale(survivor_count) * (1 - SHARE_VARIANCE_DEFICIT)

    def mechanism_record(self) -> DistributedNoiseMechanismRecord:
        """Return the value-free mechanism declaration with its digest.

        Returns:
            A record of the trust boundary, configuration and derived bounds.
        """
        body: dict[str, Any] = {
            "schema_version": FEDERATED_DISTRIBUTED_NOISE_SCHEMA_VERSION,
            "mechanism": "distributed_discrete_gaussian",
            "sampler": "cks20_exact_discrete_gaussian",
            "randomness": "injected_csprng",
            "trust_boundary": "coordinator_untrusted",
            "threat_model": "honest_but_curious_coordinator_non_colluding_clients",
            "noise_placement": "client_shares",
            "adjacency": "zero_one_client_update",
            "participation": "fixed_survivor_set",
            "dropout_model": "data_independent",
            "calibration_basis": "recovery_threshold",
            "survivor_rule": "abort_below_recovery_threshold",
            "wraparound_rule": "reject_modulus_below_tail_bound",
            "client_count": self.client_count,
            "recovery_threshold": self.recovery_threshold,
            "dimension": self.dimension,
            "modulus": self.modulus,
            "tail_bound_bits": self.tail_bound_bits,
            "target_aggregate_variance": str(self.target_aggregate_variance),
            "clip_l2_norm": str(self.clip_l2_norm),
            "share_scale": str(self.share_scale),
            "share_variance_margin": str(SHARE_VARIANCE_MARGIN),
            "share_variance_deficit": str(SHARE_VARIANCE_DEFICIT),
            "coordinate_bound": self.coordinate_bound,
            "l2_sensitivity_squared_bound": str(self.l2_sensitivity_squared_bound),
            "aggregate_signal_bound": self.aggregate_signal_bound,
            "noise_tail_bound": self.noise_tail_bound,
            "sum_closeness_tau_bound": _ceil_decimal(self.sum_closeness_tau_bound),
            "zcdp_rho_bound": _ceil_decimal(self.zcdp_rho_bound),
            "references": list(_REFERENCES),
        }
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        digest = "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()
        body["references"] = tuple(_REFERENCES)
        return DistributedNoiseMechanismRecord(
            fields=tuple(sorted(body.items())), digest=digest
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(values=hidden)"


def calibrate_distributed_noise(
    *,
    client_count: int,
    recovery_threshold: int,
    target_aggregate_variance: int | Fraction | float,
    dimension: int,
    modulus: int,
    clip_l2_norm: int | Fraction | float,
    tail_bound_bits: int = 64,
) -> DistributedNoiseCalibration:
    """Calibrate client noise shares to the secure-aggregation threshold.

    Args:
        client_count: Number of enrolled clients ``n``.
        recovery_threshold: Minimum recoverable quorum ``t``.
        target_aggregate_variance: Declared minimum aggregate noise variance in
            squared grid units. Floats convert exactly.
        dimension: Encoded update length ``d``.
        modulus: Secure-aggregation modulus ``m``.
        clip_l2_norm: Per-client L2 clip in grid units. Floats convert exactly.
        tail_bound_bits: Wraparound probability bound ``2**-b``, 20 to 256.

    Returns:
        An immutable calibration. See ``DistributedNoiseCalibration`` for the
        formulas and citations.

    Raises:
        DistributedNoiseError: A fixed reason code for invalid configuration,
            a share scale below 1, or a modulus that cannot hold the clipped
            signal plus the declared noise tail without wraparound.

    Example:
        >>> calibration = calibrate_distributed_noise(
        ...     client_count=5,
        ...     recovery_threshold=3,
        ...     target_aggregate_variance=300,
        ...     dimension=8,
        ...     modulus=2**32,
        ...     clip_l2_norm=16,
        ...     tail_bound_bits=40,
        ... )
        >>> calibration.aggregate_variance_lower_bound(3) >= 300
        True
    """
    return DistributedNoiseCalibration(
        client_count=client_count,
        recovery_threshold=recovery_threshold,
        target_aggregate_variance=target_aggregate_variance,  # type: ignore[arg-type]
        dimension=dimension,
        modulus=modulus,
        clip_l2_norm=clip_l2_norm,  # type: ignore[arg-type]
        tail_bound_bits=tail_bound_bits,
    )


def _require_calibration(calibration: Any) -> DistributedNoiseCalibration:
    if type(calibration) is not DistributedNoiseCalibration:
        raise DistributedNoiseError("invalid_calibration")
    return calibration


def noised_client_share(
    calibration: DistributedNoiseCalibration,
    encoded_signal: Sequence[int],
    sampler: DiscreteGaussianShareSampler,
    *,
    allow_test_sampler: bool = False,
) -> tuple[int, ...]:
    """Return one client's noised residues ready for masked aggregation.

    The signal is validated in full before any noise is drawn. The result is
    ``(x_j + y_j) mod m`` with ``y_j ~ N_Z(0, sigma**2)``; it leaves the client
    only through the masking protocol and is never recorded here.

    Args:
        calibration: Calibration from ``calibrate_distributed_noise``.
        encoded_signal: Clipped integer update from the fixed-point codec, of
            length ``d``, with ``|x_j| <= ceil(c)`` and squared L2 norm at most
            ``l2_sensitivity_squared_bound``.
        sampler: Exact discrete Gaussian sampler.
        allow_test_sampler: Accept a seeded test sampler. Synthetic tests only.

    Returns:
        A tuple of ``d`` residues in ``[0, m)``.

    Raises:
        DistributedNoiseError: ``invalid_calibration``, ``invalid_sampler``,
            ``test_sampler_refused``, ``invalid_signal`` or
            ``signal_out_of_bounds``.
    """
    calibration = _require_calibration(calibration)
    if type(sampler) is not DiscreteGaussianShareSampler:
        raise DistributedNoiseError("invalid_sampler")
    if sampler.is_test_mode and allow_test_sampler is not True:
        raise DistributedNoiseError("test_sampler_refused")
    if type(encoded_signal) not in (list, tuple):
        raise DistributedNoiseError("invalid_signal")
    if len(encoded_signal) != calibration.dimension:
        raise DistributedNoiseError("invalid_signal")
    if any(not _plain_int(value) for value in encoded_signal):
        raise DistributedNoiseError("invalid_signal")
    bound = calibration.coordinate_bound
    if any(abs(value) > bound for value in encoded_signal):
        raise DistributedNoiseError("signal_out_of_bounds")
    norm_squared = sum(value * value for value in encoded_signal)
    if norm_squared > calibration.l2_sensitivity_squared_bound:
        raise DistributedNoiseError("signal_out_of_bounds")
    modulus = calibration.modulus
    scale = calibration.share_scale
    return tuple(
        (value + _discrete_gaussian(sampler._rng, scale)) % modulus
        for value in encoded_signal
    )


def unmask_noised_aggregate(
    calibration: DistributedNoiseCalibration,
    survivor_count: int,
    unmask: Callable[[], Sequence[int]],
) -> tuple[int, ...]:
    """Check the recovery quorum, then unmask and center the modular sum.

    The injected ``unmask`` port is called only after the survivor count is
    validated against the recovery threshold. Port failures are reported with
    a fixed code and no exception text.

    Args:
        calibration: Calibration from ``calibrate_distributed_noise``.
        survivor_count: Number of clients whose masked shares survived.
        unmask: Zero-argument port returning the modular sum as ``d`` residues
            in ``[0, m)``.

    Returns:
        The centered aggregate, each value in ``[-(m // 2), (m - 1) // 2]``.

    Raises:
        DistributedNoiseError: ``invalid_calibration``,
            ``invalid_survivor_count``, ``survivors_below_threshold``,
            ``unmask_failed`` or ``invalid_unmasked_sum``.
    """
    calibration = _require_calibration(calibration)
    calibration.require_recovery_quorum(survivor_count)
    try:
        total = unmask()
    except Exception:
        raise DistributedNoiseError("unmask_failed") from None
    modulus = calibration.modulus
    if (
        type(total) not in (list, tuple)
        or len(total) != calibration.dimension
        or any(not _plain_int(value) or not 0 <= value < modulus for value in total)
    ):
        raise DistributedNoiseError("invalid_unmasked_sum")
    half = (modulus - 1) // 2
    return tuple(value if value <= half else value - modulus for value in total)


__all__ = [
    "DISTRIBUTED_NOISE_REASON_CODES",
    "FEDERATED_DISTRIBUTED_NOISE_SCHEMA_VERSION",
    "MINIMUM_SHARE_SCALE",
    "SHARE_VARIANCE_DEFICIT",
    "SHARE_VARIANCE_MARGIN",
    "DiscreteGaussianShareSampler",
    "DistributedNoiseCalibration",
    "DistributedNoiseError",
    "DistributedNoiseMechanismRecord",
    "calibrate_distributed_noise",
    "noised_client_share",
    "unmask_noised_aggregate",
]
