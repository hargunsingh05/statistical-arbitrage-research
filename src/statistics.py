"""Statistical tests for candidate pairs-trading relationships."""

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> tuple[float, float]:
    """Fit y = intercept + hedge_ratio * x using OLS."""
    aligned = pd.concat(
        [y.rename("y"), x.rename("x")], axis=1
    ).dropna()

    if len(aligned) < 30:
        raise ValueError("At least 30 paired observations are required.")

    model = sm.OLS(
        aligned["y"],
        sm.add_constant(aligned["x"]),
    ).fit()

    intercept = float(model.params["const"])
    hedge_ratio = float(model.params["x"])
    return intercept, hedge_ratio


def calculate_half_life(spread: pd.Series) -> float:
    """Estimate spread half-life using a simple AR(1)-style regression."""
    spread = spread.dropna()

    if len(spread) < 30:
        return float("nan")

    lagged = spread.shift(1).dropna()
    change = spread.diff().dropna()
    lagged = lagged.loc[change.index]

    model = sm.OLS(change, sm.add_constant(lagged)).fit()
    speed = float(model.params.iloc[1])

    if speed >= 0:
        return float("inf")

    half_life = -np.log(2) / speed
    return float(half_life) if half_life > 0 else float("nan")


def analyse_pair(
    prices: pd.DataFrame,
    y_ticker: str,
    x_ticker: str,
    train_fraction: float = 0.70,
) -> dict[str, float | str | int]:
    """Evaluate one pair, fitting its hedge ratio on training data only."""
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be >= 0.5 and < 1.0.")

    pair = prices[[y_ticker, x_ticker]].dropna()

    if len(pair) < 100:
        raise ValueError("At least 100 aligned observations are required.")

    split = int(len(pair) * train_fraction)
    train = pair.iloc[:split]
    test = pair.iloc[split:]

    intercept, hedge_ratio = estimate_hedge_ratio(
        np.log(train[y_ticker]),
        np.log(train[x_ticker]),
    )

    # Apply the training-period hedge ratio to both periods.
    log_spread_train = (
        np.log(train[y_ticker]) - intercept
        - hedge_ratio * np.log(train[x_ticker])
    )
    log_spread_test = (
        np.log(test[y_ticker]) - intercept
        - hedge_ratio * np.log(test[x_ticker])
    )

    train_coint_stat, train_coint_p, _ = coint(
        np.log(train[y_ticker]),
        np.log(train[x_ticker]),
    )
    test_adf_stat, test_adf_p, *_ = adfuller(log_spread_test, result_object=False)

    return {
        "y_ticker": y_ticker,
        "x_ticker": x_ticker,
        "train_observations": len(train),
        "test_observations": len(test),
        "hedge_ratio": hedge_ratio,
        "train_cointegration_stat": float(train_coint_stat),
        "train_cointegration_pvalue": float(train_coint_p),
        "test_spread_adf_stat": float(test_adf_stat),
        "test_spread_adf_pvalue": float(test_adf_p),
        "train_half_life_days": calculate_half_life(log_spread_train),
        "test_half_life_days": calculate_half_life(log_spread_test),
    }


def scan_pairs(
    prices: pd.DataFrame,
    train_fraction: float = 0.70,
) -> pd.DataFrame:
    """Scan all unique ticker pairs and sort by training p-value."""
    rows = []

    for y_ticker, x_ticker in combinations(prices.columns, 2):
        try:
            rows.append(
                analyse_pair(
                    prices,
                    y_ticker,
                    x_ticker,
                    train_fraction=train_fraction,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            print(f"Skipping {y_ticker}/{x_ticker}: {exc}")

    results = pd.DataFrame(rows)

    if not results.empty:
        results = results.sort_values(
            "train_cointegration_pvalue"
        ).reset_index(drop=True)

    return results


if __name__ == "__main__":
    prices = pd.read_csv(
        "results/prices.csv",
        index_col="Date",
        parse_dates=True,
    )

    results = scan_pairs(prices)
    results.to_csv("results/pair_statistics.csv", index=False)

    if results.empty:
        print("No pairs could be analysed.")
    else:
        print(results.to_string(index=False))
        print("\nSaved results to results/pair_statistics.csv")
