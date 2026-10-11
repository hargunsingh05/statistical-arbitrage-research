"""Download and validate daily market prices for research."""

from pathlib import Path

import pandas as pd
import yfinance as yf


DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "DIA"]
DEFAULT_START = "2015-01-01"


def download_prices(
    tickers: list[str] | None = None,
    start: str = DEFAULT_START,
    end: str | None = None,
) -> pd.DataFrame:
    """Download adjusted daily closing prices for the requested tickers."""
    symbols = tickers or DEFAULT_TICKERS

    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise ValueError("No market data returned. Check the tickers or connection.")

    # With multiple tickers, yfinance normally returns a MultiIndex.
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain closing prices.")
        prices = raw["Close"].copy()
    else:
        # Handle a single ticker separately.
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain closing prices.")
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")

    if prices.empty:
        raise ValueError("No usable prices remain after cleaning.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")

    return prices


def validate_prices(prices: pd.DataFrame) -> dict[str, object]:
    """Return basic quality checks for a price dataset."""
    if prices.empty:
        raise ValueError("Price dataset is empty.")

    return {
        "rows": len(prices),
        "columns": list(prices.columns),
        "start": str(prices.index.min().date()),
        "end": str(prices.index.max().date()),
        "missing_values": int(prices.isna().sum().sum()),
        "duplicate_dates": int(prices.index.duplicated().sum()),
        "non_positive_prices": int((prices <= 0).sum().sum()),
    }


def save_prices(
    prices: pd.DataFrame,
    output_path: str = "results/prices.csv",
) -> Path:
    """Save prices to CSV and return the resulting path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index_label="Date")
    return path


if __name__ == "__main__":
    prices = download_prices()
    report = validate_prices(prices)
    print("Price data validation:")
    for key, value in report.items():
        print(f"{key}: {value}")

    output = save_prices(prices)
    print(f"Saved prices to: {output}")
