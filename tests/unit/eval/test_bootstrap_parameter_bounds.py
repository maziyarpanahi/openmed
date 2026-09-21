"""Invalid bootstrap settings must not fabricate confidence intervals."""

from unittest.mock import Mock

import pytest

from openmed.eval.metrics import bootstrap_ci


@pytest.mark.parametrize("values", [[], [4.0], [4.0, 6.0]])
@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_resample_count_rejected_before_statistic(values, count):
    statistic = Mock(return_value=5.0)
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_ci(values, statistic, n_resamples=count)
    statistic.assert_not_called()


@pytest.mark.parametrize("values", [[4.0], [4.0, 6.0]])
@pytest.mark.parametrize(
    "alpha", [-0.1, 1.1, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_probability_rejected_before_statistic(values, alpha):
    statistic = Mock(return_value=5.0)
    with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
        bootstrap_ci(values, statistic, n_resamples=20, alpha=alpha)
    statistic.assert_not_called()


@pytest.mark.parametrize("alpha", [0.0, 0.05, 1.0])
def test_finite_endpoint_probabilities_remain_supported(alpha):
    result = bootstrap_ci(
        [4.0, 6.0],
        lambda values: sum(values) / len(values),
        n_resamples=20,
        alpha=alpha,
        seed=1,
    )
    assert result.lower <= result.point <= result.upper
    assert result.alpha == alpha
    assert result.n_resamples == 20


@pytest.mark.parametrize("values", [[], [4.0]])
def test_valid_degenerate_corpus_keeps_zero_width_interval(values):
    result = bootstrap_ci(values, sum, n_resamples=10)
    assert result.degenerate is True
    assert result.point == result.lower == result.upper == sum(values)


def test_valid_resampling_is_reproducible_and_not_modified():
    values = [2.0, 4.0, 8.0, 10.0]
    statistic = lambda xs: sum(xs) / len(xs)
    first = bootstrap_ci(values, statistic, n_resamples=50, seed=42)
    second = bootstrap_ci(values, statistic, n_resamples=50, seed=42)
    assert first == second
    assert first.point == 6.0
    assert first.degenerate is False
    assert values == [2.0, 4.0, 8.0, 10.0]


def test_statistic_error_is_not_masked_for_valid_parameters():
    error = RuntimeError("synthetic statistic failure")
    with pytest.raises(RuntimeError) as caught:
        bootstrap_ci([1.0, 2.0], Mock(side_effect=error), n_resamples=10)
    assert caught.value is error
