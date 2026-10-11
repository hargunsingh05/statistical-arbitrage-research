"""Research backtester for a two-asset, dollar-neutral pairs strategy."""

import numpy as np
import pandas as pd


def performance_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate common performance metrics from daily portfolio returns."""
    returns = returns.dropna().astype(float)
    if returns.empty:
        raise ValueError("Returns series is empty.")

    equity = (1.0 + returns).cumprod()
    years = len(returns) / 252.0
    total_return = float(equity.iloc[-1] - 1.0)

    cagr = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else float("nan")
    )

    daily_std = float(returns.std(ddof=1))
    volatility = float(daily_std * np.sqrt(252.0))
    sharpe = (
        float(returns.mean() / daily_std * np.sqrt(252.0))
        if daily_std > 0
        else float("nan")
    )
    drawdown = equity / equity.cummax() - 1.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualised_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def backtest_pairs(
    prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    intercept: float,
    hedge_ratio: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    lookback: int = 60,
    transaction_cost_bps: float = 5.0,
    start_trading_at: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Backtest a simple dollar-neutral pairs strategy.

    Estimate intercept and hedge_ratio on training data, then freeze them
    before running this function on the test sample.

    Portfolio weights have equal and opposite dollar exposure, with gross
    exposure scaled to 1. This is dollar-neutral, not beta-neutral.
    Signals observed at close t are applied to the next return interval.
    Costs are charged on changes in the applied portfolio weights.
    """
    if entry_z <= exit_z or exit_z < 0:
        raise ValueError("Require entry_z > exit_z >= 0.")
    if lookback < 20:
        raise ValueError("lookback must be at least 20.")
    if transaction_cost_bps < 0:
        raise ValueError("transaction costs cannot be negative.")
    if not np.isfinite(intercept) or not np.isfinite(hedge_ratio):
        raise ValueError("intercept and hedge_ratio must be finite.")
    if hedge_ratio == 0:
        raise ValueError("hedge_ratio must be non-zero.")
    if y_ticker == x_ticker:
        raise ValueError("The two tickers must be different.")
    if y_ticker not in prices.columns or x_ticker not in prices.columns:
        raise ValueError("Both tickers must be present in prices.")

    pair = prices[[y_ticker, x_ticker]].dropna().astype(float)
    if (pair <= 0).any().any():
        raise ValueError("Prices must be positive.")
    if len(pair) <= lookback + 2:
        raise ValueError("Not enough prices for the selected lookback.")

    log_y = np.log(pair[y_ticker])
    log_x = np.log(pair[x_ticker])
    spread = log_y - intercept - hedge_ratio * log_x

    # The current close is compared only with previous observations.
    rolling_mean = spread.shift(1).rolling(lookback).mean()
    rolling_std = spread.shift(1).rolling(lookback).std(ddof=1)
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)

    signal = pd.Series(0.0, index=pair.index)
    current_position = 0.0

    for i, z in enumerate(zscore.to_numpy()):
        if (
            start_trading_at is not None
            and pair.index[i] < pd.Timestamp(start_trading_at)
        ):
            signal.iloc[i] = 0.0
            continue

        if not np.isfinite(z):
            signal.iloc[i] = current_position
            continue

        if current_position == 0.0:
            if z >= entry_z:
                current_position = -1.0
            elif z <= -entry_z:
                current_position = 1.0
        elif abs(z) <= exit_z:
            current_position = 0.0
        elif (
            current_position == 1.0 and z >= entry_z
        ) or (
            current_position == -1.0 and z <= -entry_z
        ):
            current_position = -current_position

        signal.iloc[i] = current_position

    # Equal-dollar long and short legs: gross exposure is 100%.
    y_signal_weight = signal * 0.5
    x_signal_weight = -signal * 0.5

    # A signal from close t earns returns starting in the next interval.
    applied_y = y_signal_weight.shift(1).fillna(0.0)
    applied_x = x_signal_weight.shift(1).fillna(0.0)

    y_returns = pair[y_ticker].pct_change().fillna(0.0)
    x_returns = pair[x_ticker].pct_change().fillna(0.0)
    gross_returns = applied_y * y_returns + applied_x * x_returns

    # Turnover reflects changes in the weights actually held.
    turnover = (
        applied_y.diff().abs().fillna(applied_y.abs())
        + applied_x.diff().abs().fillna(applied_x.abs())
    )
    costs = turnover * transaction_cost_bps / 10_000.0
    net_returns = gross_returns - costs

    result = pd.DataFrame({
        "spread": spread,
        "zscore": zscore,
        "signal": signal,
        "y_weight": applied_y,
        "x_weight": applied_x,
        "gross_return": gross_returns,
        "turnover": turnover,
        "transaction_cost": costs,
        "net_return": net_returns,
    })
    result["equity"] = (1.0 + result["net_return"]).cumprod()
    return result


if __name__ == "__main__":
    prices = pd.read_csv(
        "results/prices.csv",
        index_col="Date",
        parse_dates=True,
    )

    # Demonstration only; these parameters are arbitrary, not estimated.
    result = backtest_pairs(
        prices,
        "SPY",
        "DIA",
        intercept=0.0,
        hedge_ratio=1.0,
    )
    result.to_csv("results/backtest_demo.csv")

    print("Demonstration results (not a validated strategy):")
    print(performance_metrics(result["net_return"]))
    print("Saved results to results/backtest_demo.csv")
