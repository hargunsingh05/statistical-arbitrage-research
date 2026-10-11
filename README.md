# Statistical Arbitrage Research
An exploratory quantitative research project investigating cointegration-based
pairs trading, rolling spread signals, transaction costs, and chronological
out-of-sample evaluation.
## Research question
Can a simple mean-reversion strategy applied to pairs of liquid US equity ETFs
produce attractive risk-adjusted returns after estimated transaction costs?
## Instruments
- SPY — S&P 500 ETF
- QQQ — Nasdaq-100 ETF
- IWM — Russell 2000 ETF
- DIA — Dow Jones Industrial Average ETF
Daily adjusted price data is downloaded using `yfinance` and saved locally for
reproducibility of subsequent calculations.
## Methodology
1. Validate price data for missing values, duplicate dates, and non-positive prices.
2. Split observations chronologically into a training period and a later test period.
3. Estimate log-price relationships and rank candidate pairs using the
   Engle-Granger cointegration test on training data only.
4. Calculate the spread and rolling z-score using historical observations.
5. Generate entry and exit signals using fixed z-score thresholds.
6. Apply a one-period signal lag and estimated turnover-based transaction costs.
7. Evaluate test-period returns, annualised volatility, Sharpe ratio, and
   maximum drawdown against a simple equal-weight buy-and-hold reference.
The strategy uses equal-dollar long/short weights. Dollar neutrality does not
guarantee market-beta neutrality.
## Out-of-sample result
The current experiment selected IWM–QQQ using the lowest training-period
cointegration p-value among six candidate pairs.
| Metric | Strategy | Equal-weight buy-and-hold reference |
|---|---:|---:|
| Total return | -3.36% | +106.62% |
| CAGR | -0.96% | +22.87% |
| Annualised volatility | 5.34% | 18.62% |
| Sharpe ratio | -0.15 | 1.20 |
| Maximum drawdown | -5.71% | -23.71% |
The selected pair's training cointegration p-value was approximately 0.183.
This is not statistically significant at the conventional 5% level. The result
does not establish a reliable cointegrating relationship or a profitable strategy.
The buy-and-hold reference is long-only and has different market exposure, so
it is not a risk-matched benchmark.
## Run locally
Use Python 3.11 or newer, create a virtual environment, and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

Run the test suite:

python -m pytest -v

Run the research pipeline after the price data is available:

python -m src.run_research

The research pipeline reads results/prices.csv and writes the pair scan,
out-of-sample returns, and JSON report into results/.

Limitations

* Only four ETFs and six candidate pairs are considered.
* Selecting the lowest p-value across multiple pairs creates selection bias.
* Cointegration may be unstable over time.
* The test result is historical and does not predict future performance.
* Transaction costs omit market impact, borrow fees, financing costs, and other
    implementation frictions.
* The benchmark is not risk-matched to the strategy.
* No claim is made that this strategy is suitable for live trading.

This project is for educational and research purposes only, not investment advice.
