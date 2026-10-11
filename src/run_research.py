"""Run a chronological, exploratory out-of-sample pairs experiment."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from src.backtest import backtest_pairs, performance_metrics
from src.statistics import estimate_hedge_ratio


def main():
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    prices = pd.read_csv(
        results_dir / "prices.csv",
        index_col="Date",
        parse_dates=True,
    ).sort_index()

    if prices.index.has_duplicates:
        raise ValueError("Price data contains duplicate dates.")
    if prices.isna().any().any():
        raise ValueError("Price data contains missing values.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    split = int(len(prices) * 0.70)
    train = prices.iloc[:split]
    test = prices.iloc[split:]

    if len(train) < 100 or len(test) <= 60:
        raise ValueError("Insufficient training or test observations.")

    # Rank every pair on the complete training sample only.
    scan_rows = []

    for y_ticker, x_ticker in combinations(train.columns, 2):
        log_y = np.log(train[y_ticker])
        log_x = np.log(train[x_ticker])

        intercept, hedge_ratio = estimate_hedge_ratio(log_y, log_x)
        coint_stat, coint_pvalue, _ = coint(log_y, log_x)

        scan_rows.append({
            "y_ticker": y_ticker,
            "x_ticker": x_ticker,
            "training_observations": len(train),
            "intercept": intercept,
            "hedge_ratio": hedge_ratio,
            "training_cointegration_stat": float(coint_stat),
            "training_cointegration_pvalue": float(coint_pvalue),
        })

    pair_results = pd.DataFrame(scan_rows).sort_values(
        "training_cointegration_pvalue"
    ).reset_index(drop=True)

    pair_results.to_csv(
        results_dir / "training_pair_scan.csv",
        index=False,
    )

    if pair_results.empty:
        raise RuntimeError("No candidate pairs could be analysed.")

    candidate = pair_results.iloc[0]
    y_ticker = str(candidate["y_ticker"])
    x_ticker = str(candidate["x_ticker"])
    intercept = float(candidate["intercept"])
    hedge_ratio = float(candidate["hedge_ratio"])

    # Use historical prices to warm up the rolling signal calculations.
    # Trading is disabled until the first test-period date.
    warmup = train.tail(60)[[y_ticker, x_ticker]]
    test_pair = test[[y_ticker, x_ticker]]
    combined = pd.concat([warmup, test_pair])

    full_result = backtest_pairs(
        combined,
        y_ticker,
        x_ticker,
        intercept=intercept,
        hedge_ratio=hedge_ratio,
        entry_z=2.0,
        exit_z=0.5,
        lookback=60,
        transaction_cost_bps=5.0,
        start_trading_at=test_pair.index[0],
    )

    # Report only test-period observations, never the warm-up rows.
    result = full_result.loc[test_pair.index].copy()
    strategy_metrics = performance_metrics(result["net_return"])

    # Reference benchmark only: this is long-only and not risk-matched
    # to the dollar-neutral strategy.
    benchmark_returns = (
        test_pair[y_ticker].pct_change().fillna(0.0) * 0.5
        + test_pair[x_ticker].pct_change().fillna(0.0) * 0.5
    )
    benchmark_metrics = performance_metrics(benchmark_returns)

    result.to_csv(results_dir / "oos_returns.csv")

    report = {
        "candidate_pair": f"{y_ticker}-{x_ticker}",
        "selection_method": (
            "Lowest Engle-Granger cointegration p-value among all pairs, "
            "calculated on the complete training period"
        ),
        "training_cointegration_pvalue": float(
            candidate["training_cointegration_pvalue"]
        ),
        "training_rows": len(train),
        "test_rows": len(test_pair),
        "test_start": str(test_pair.index[0].date()),
        "test_end": str(test_pair.index[-1].date()),
        "warmup_observations": len(warmup),
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "entry_z": 2.0,
        "exit_z": 0.5,
        "lookback": 60,
        "transaction_cost_bps_per_unit_turnover": 5.0,
        "strategy_test_metrics": strategy_metrics,
        "equal_weight_buy_and_hold_test_metrics": benchmark_metrics,
        "caveats": [
            "Exploratory research, not a deployable trading strategy.",
            "Selecting the lowest p-value across multiple pairs introduces selection bias.",
            "The selected pair may not be cointegrated out of sample.",
            "The long-only benchmark is not risk-matched to the dollar-neutral strategy.",
            "Transaction costs exclude market impact, borrow fees and financing costs.",
            "Dollar-neutral weights do not guarantee beta neutrality.",
        ],
    }

    with open(results_dir / "oos_report.json", "w") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))
    print("\nSaved training_pair_scan.csv, oos_returns.csv and oos_report.json")


if __name__ == "__main__":
    main()
