"""Exact integer samplers for differential-privacy noise.

The samplers in this module draw discrete Laplace and discrete Gaussian noise
with integer and :class:`fractions.Fraction` arithmetic only. They follow
Algorithms 1, 2 and 3 of Canonne, Kamath and Steinke, "The Discrete Gaussian
for Differential Privacy" (NeurIPS 2020, https://arxiv.org/abs/2004.00010).
No floating-point value is computed while sampling, so the set of reachable
outputs and their probabilities do not depend on rounding. This closes the
floating-point attack described by Mironov, "On Significance of the Least
Significant Bits for Differential Privacy" (CCS 2012), for integer releases.

Randomness comes from a :class:`DPRandomSource`. Its default constructor uses
the operating system through :mod:`secrets` and accepts no seed. Seeded draws
exist only through the explicitly named :meth:`DPRandomSource.for_testing`
constructor and are for synthetic tests; they provide no privacy guarantee.

The rejection loops have a constant expected number of rounds. Each loop is
bounded; reaching a bound raises :class:`DiscreteSamplerError` instead of
returning a biased draw. With the configured bound this happens with
probability far below ``2**-1000`` per draw. Sampler running time depends on
the drawn value; timing side channels are out of scope.
"""

from __future__ import annotations

import math
import random
import secrets
from fractions import Fraction
from numbers import Rational
from typing import Any, Literal

__all__ = [
    "DP_SAMPLERS",
    "DPRandomSource",
    "DPSampler",
    "DiscreteSamplerError",
    "sample_discrete_gaussian",
    "sample_discrete_laplace",
]

DPSampler = Literal[
    "unspecified",
    "float_laplace",
    "float_gaussian",
    "discrete_laplace",
    "discrete_gaussian",
    "grid_discrete_laplace",
]
DP_SAMPLERS: frozenset[str] = frozenset(
    {
        "unspecified",
        "float_laplace",
        "float_gaussian",
        "discrete_laplace",
        "discrete_gaussian",
        "grid_discrete_laplace",
    }
)

_MAX_ROUNDS = 10_000
_ONE = Fraction(1)


class DiscreteSamplerError(RuntimeError):
    """Raised when an exact sampler reaches its rejection bound.

    The message is fixed and never contains a parameter or a partial draw.
    """


class DPRandomSource:
    """Uniform integer randomness for the exact discrete samplers.

    ``DPRandomSource()`` draws from the operating system through
    :func:`secrets.randbelow` and accepts no seed. Use
    :meth:`for_testing` for reproducible synthetic tests only.

    The type check performed by release helpers guards against accidental
    seeding; it is not a security boundary against code that subclasses this
    type on purpose.
    """

    __slots__ = ("_generator",)

    def __init__(self) -> None:
        self._generator: random.Random | None = None

    @classmethod
    def for_testing(cls, seed: int) -> DPRandomSource:
        """Return a seeded source for synthetic, reproducible tests.

        Args:
            seed: Integer seed for a local pseudo-random generator.

        Returns:
            A source whose :attr:`test_mode` is ``True``. Its output is
            predictable and provides no privacy guarantee.

        Raises:
            TypeError: If ``seed`` is not an integer.
        """

        if type(seed) is not int:
            raise TypeError("test seed must be an integer")
        source = cls()
        source._generator = random.Random(seed)
        return source

    @property
    def test_mode(self) -> bool:
        """Return whether this source is a seeded test-only source."""

        return self._generator is not None

    def randbelow(self, upper: int) -> int:
        """Return a uniform integer in ``[0, upper)``.

        Args:
            upper: Exclusive positive integer bound.

        Returns:
            A uniformly distributed integer.

        Raises:
            ValueError: If ``upper`` is not a positive integer.
        """

        if type(upper) is not int or upper <= 0:
            raise ValueError("upper must be a positive integer")
        if self._generator is None:
            return secrets.randbelow(upper)
        return self._generator.randrange(upper)

    def __repr__(self) -> str:
        return f"DPRandomSource(test_mode={self.test_mode})"


def _dp_source(rng: Any) -> DPRandomSource:
    if rng is None:
        return DPRandomSource()
    if not isinstance(rng, DPRandomSource):
        raise TypeError(
            "rng must be a DPRandomSource; use DPRandomSource.for_testing(seed) "
            "for seeded synthetic tests"
        )
    return rng


def _positive_rational(value: Any, *, field_name: str) -> Fraction:
    # Floats convert exactly through their binary value, so no rounding occurs.
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a real number")
    if isinstance(value, Rational):
        result = Fraction(value.numerator, value.denominator)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        result = Fraction(value)
    else:
        raise TypeError(f"{field_name} must be a real number")
    if result <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return result


def _bernoulli(probability: Fraction, rng: DPRandomSource) -> bool:
    return rng.randbelow(probability.denominator) < probability.numerator


def _bernoulli_exp_unit(gamma: Fraction, rng: DPRandomSource) -> bool:
    # CKS20 Algorithm 1 for gamma in [0, 1]: K is odd with probability
    # exp(-gamma) because P(K > k) = gamma**k / k!.
    k = 1
    while _bernoulli(gamma / k, rng):
        k += 1
        if k > _MAX_ROUNDS:
            raise DiscreteSamplerError("discrete sampler exceeded its round bound")
    return k % 2 == 1


def _bernoulli_exp(gamma: Fraction, rng: DPRandomSource) -> bool:
    # CKS20 Algorithm 1 for gamma > 1: exp(-gamma) factors into floor(gamma)
    # independent exp(-1) trials and one fractional trial. Each exp(-1) trial
    # stops the loop with probability 1 - exp(-1).
    whole = gamma.numerator // gamma.denominator
    for _ in range(whole):
        if not _bernoulli_exp_unit(_ONE, rng):
            return False
    return _bernoulli_exp_unit(gamma - whole, rng)


def _discrete_laplace(scale: Fraction, rng: DPRandomSource) -> int:
    # CKS20 Algorithm 2 with scale = t / s. P(Y = y) is proportional to
    # exp(-|y| / scale) for every integer y.
    t, s = scale.numerator, scale.denominator
    for _ in range(_MAX_ROUNDS):
        u = rng.randbelow(t)
        if not _bernoulli_exp(Fraction(u, t), rng):
            continue
        v = 0
        while _bernoulli_exp_unit(_ONE, rng):
            v += 1
            if v > _MAX_ROUNDS:
                raise DiscreteSamplerError("discrete sampler exceeded its round bound")
        magnitude = (u + t * v) // s
        negative = rng.randbelow(2) == 1
        if negative and magnitude == 0:
            continue
        return -magnitude if negative else magnitude
    raise DiscreteSamplerError("discrete sampler exceeded its round bound")


def _discrete_gaussian(sigma_squared: Fraction, rng: DPRandomSource) -> int:
    # CKS20 Algorithm 3. floor(sigma) is computed exactly from sigma**2 = n/d
    # as isqrt(n * d) // d. Any positive integer t yields the exact target;
    # floor(sigma) + 1 keeps the expected number of rounds small.
    numerator, denominator = sigma_squared.numerator, sigma_squared.denominator
    t = math.isqrt(numerator * denominator) // denominator + 1
    proposal_scale = Fraction(t)
    shift = sigma_squared / t
    for _ in range(_MAX_ROUNDS):
        candidate = _discrete_laplace(proposal_scale, rng)
        gamma = (abs(candidate) - shift) ** 2 / (2 * sigma_squared)
        if _bernoulli_exp(gamma, rng):
            return candidate
    raise DiscreteSamplerError("discrete sampler exceeded its round bound")


def sample_discrete_laplace(
    scale: int | float | Fraction,
    *,
    rng: DPRandomSource | None = None,
) -> int:
    """Draw exact discrete Laplace noise.

    The draw ``Y`` satisfies ``P(Y = y) = tanh(1 / (2 * scale)) *
    exp(-|y| / scale)`` for every integer ``y``. Adding it to an integer query
    with L1 sensitivity ``sensitivity`` gives pure
    ``sensitivity / scale``-differential privacy.

    Args:
        scale: Positive scale. Floats are converted exactly via ``Fraction``.
        rng: Optional randomness source; system randomness when omitted.

    Returns:
        An integer noise draw.

    Raises:
        TypeError: If ``scale`` or ``rng`` has an unsupported type.
        ValueError: If ``scale`` is not finite and positive.
        DiscreteSamplerError: If the rejection bound is reached.
    """

    exact_scale = _positive_rational(scale, field_name="scale")
    return _discrete_laplace(exact_scale, _dp_source(rng))


def sample_discrete_gaussian(
    sigma_squared: int | float | Fraction,
    *,
    rng: DPRandomSource | None = None,
) -> int:
    """Draw exact discrete Gaussian noise.

    The draw ``Y`` satisfies ``P(Y = y)`` proportional to
    ``exp(-y**2 / (2 * sigma_squared))`` for every integer ``y``. Adding it to
    an integer query with sensitivity ``sensitivity`` gives
    ``sensitivity**2 / (2 * sigma_squared)``-zero-concentrated differential
    privacy (CKS20, Theorem 4).

    Args:
        sigma_squared: Positive variance parameter. Floats are converted
            exactly via ``Fraction``; pass a ``Fraction`` for exact rationals.
        rng: Optional randomness source; system randomness when omitted.

    Returns:
        An integer noise draw.

    Raises:
        TypeError: If ``sigma_squared`` or ``rng`` has an unsupported type.
        ValueError: If ``sigma_squared`` is not finite and positive.
        DiscreteSamplerError: If the rejection bound is reached.
    """

    exact_variance = _positive_rational(sigma_squared, field_name="sigma_squared")
    return _discrete_gaussian(exact_variance, _dp_source(rng))
