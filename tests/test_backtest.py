import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_pairs, performance_metrics


def test_performance_metrics_are_finite():
    returns = pd.Series([0.01, -0.005, 0.003, 0.002, -0.001] * 60)
    metrics = performance_metrics(returns)

    assert np.isfinite(metrics["total_return"])
    assert np.isfinite(metrics["annualised_volatility"])
    assert np.isfinite(metrics["sharpe_ratio"])
    assert metrics["max_drawdown"] <= 0


def test_backtest_returns_have_expected_columns():
    rng = np.random.default_rng(42)
    n = 250

    x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.zeros(n)

    for i in range(1, n):
        spread[i] = 0.85 * spread[i - 1] + rng.normal(0, 0.02)

    y = np.exp(np.log(x) + spread)

    prices = pd.DataFrame({
        "Y": y,
        "X": x,
    }, index=pd.bdate_range("2020-01-01", periods=n))

    result = backtest_pairs(
        prices,
        "Y",
        "X",
        intercept=0.0,
        hedge_ratio=1.0,
        entry_z=1.5,
        exit_z=0.3,
        lookback=30,
        transaction_cost_bps=5,
    )

    expected = {
        "spread", "zscore", "signal", "y_weight", "x_weight",
        "gross_return", "turnover", "transaction_cost",
        "net_return", "equity",
    }

    assert expected.issubset(result.columns)
    assert len(result) == n
    assert result["net_return"].notna().all()
    assert (result["transaction_cost"] >= 0).all()


def test_invalid_thresholds_are_rejected():
    prices = pd.DataFrame({
        "Y": np.linspace(100, 110, 100),
        "X": np.linspace(100, 105, 100),
    })

    with pytest.raises(ValueError):
        backtest_pairs(
            prices, "Y", "X",
            intercept=0.0,
            hedge_ratio=1.0,
            entry_z=0.5,
            exit_z=1.0,
        )
def test_trading_starts_only_on_requested_date():
    import numpy as np
    import pandas as pd
    from src.backtest import backtest_pairs
    dates = pd.bdate_range("2024-01-01", periods=100)
    t = np.arange(100, dtype=float)
    prices = pd.DataFrame(
        {
            "Y": 100 * np.exp(0.002 * t + 0.03 * np.sin(t / 4)),
            "X": 100 * np.exp(0.001 * t),
        },
        index=dates,
    )
    start_date = dates[80]
    result = backtest_pairs(
        prices,
        "Y",
        "X",
        intercept=0.0,
        hedge_ratio=1.0,
        entry_z=0.5,
        exit_z=0.1,
        lookback=20,
        transaction_cost_bps=5.0,
        start_trading_at=start_date,
    )
    assert (result.loc[result.index < start_date, "signal"] == 0).all()
    assert (result.loc[result.index < start_date, "net_return"] == 0).all()
