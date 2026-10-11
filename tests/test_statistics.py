import numpy as np
import pandas as pd

from src.statistics import (
    analyse_pair,
    calculate_half_life,
    estimate_hedge_ratio,
)


def test_hedge_ratio_recovers_linear_relationship():
    rng = np.random.default_rng(42)
    x = pd.Series(rng.normal(size=500))
    y = 2.0 + 1.5 * x + pd.Series(
        rng.normal(scale=0.001, size=500)
    )

    intercept, slope = estimate_hedge_ratio(y, x)

    assert abs(intercept - 2.0) < 0.01
    assert abs(slope - 1.5) < 0.01


def test_half_life_is_positive_for_mean_reverting_series():
    rng = np.random.default_rng(7)
    values = [0.0]

    for _ in range(499):
        values.append(0.8 * values[-1] + rng.normal(scale=0.5))

    half_life = calculate_half_life(pd.Series(values))

    assert np.isfinite(half_life)
    assert half_life > 0


def test_analyse_pair_returns_train_and_test_counts():
    rng = np.random.default_rng(21)
    n = 400
    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    noise = rng.normal(0, 0.01, n)
    y = 100 * np.exp(0.7 * np.log(x / 100) + noise)

    prices = pd.DataFrame({"Y": y, "X": x})

    result = analyse_pair(prices, "Y", "X", train_fraction=0.7)

    assert result["train_observations"] == 280
    assert result["test_observations"] == 120
    assert np.isfinite(result["hedge_ratio"])
    assert 0 <= result["train_cointegration_pvalue"] <= 1
