"""
Data loading and preprocessing for BTC volatility analysis.

Two modes:
  1. load_from_csv()  — load pre-built dataset (fast, offline)
  2. build_dataset()  — fetch from APIs and rebuild (slow, needs keys)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_from_csv(path: Path = DATA_DIR / "btc_dataset.csv") -> pd.DataFrame:
    """Load the pre-built dataset from CSV."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def compute_log_returns(prices: pd.DataFrame, column: str = "btc") -> pd.Series:
    """Compute log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices[column] / prices[column].shift(1)).dropna()


def get_btc_returns(df: pd.DataFrame, scale: float = 100.0) -> pd.Series:
    """Get scaled BTC log returns ready for GARCH modeling.

    The arch library works better with returns scaled by 100.
    """
    ret = df["dln_btc"] * scale
    ret.name = "btc_ret"
    return ret


def get_exog(df: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """Get the 5 significant exogenous variables (from general-to-specific selection).

    Selected variables (all significant at p < 0.05):
      - dln_gold:   gold returns (+)
      - d_vix:      VIX first difference (-)
      - dln_volume: BTC trading volume (+)
      - dln_n_tx:   on-chain transaction count (+)
      - dln_cpi:    CPI inflation proxy (+)
    """
    cols = ["dln_gold", "d_vix", "dln_volume", "dln_n_tx", "dln_cpi"]
    exog = df[cols] * scale
    return exog.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# Dataset builder (requires API keys + internet)
# ---------------------------------------------------------------------------

def build_dataset(start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch all data from APIs and build the full dataset.

    Requires: yfinance, fredapi, requests, pytrends, python-dotenv
    Environment: FRED_API_KEY in ~/.secrets.env
    """
    import os
    import requests
    from datetime import date
    from dotenv import load_dotenv

    load_dotenv(Path("~/.secrets.env").expanduser())

    if end is None:
        end = date.today().isoformat()

    # --- yfinance ---
    import yfinance as yf

    tickers = {
        "BTC-USD": "btc", "GC=F": "gold", "CL=F": "oil",
        "^GSPC": "sp500", "^VIX": "vix", "URTH": "msci",
        "^TNX": "tbond", "^IRX": "tbill",
    }
    raw = yf.download(list(tickers.keys()), start=start, end=end)
    prices = raw["Close"].rename(columns=tickers)
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(0, axis=1)
    btc_volume = raw["Volume"]["BTC-USD"].rename("volume")

    # --- blockchain.com ---
    def fetch_blockchain(chart_name: str, col_name: str) -> pd.Series:
        url = f"https://api.blockchain.info/charts/{chart_name}"
        params = {"timespan": "all", "format": "json", "sampled": "false"}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()["values"]
        s = pd.DataFrame(data)
        s["date"] = pd.to_datetime(s["x"], unit="s")
        return s.set_index("date")["y"].rename(col_name)

    hashrate = fetch_blockchain("hash-rate", "hashrate")
    n_addresses = fetch_blockchain("n-unique-addresses", "addresses")
    n_tx = fetch_blockchain("n-transactions", "n_tx")
    revenue = fetch_blockchain("miners-revenue", "revenue")

    # --- FRED CPI ---
    from fredapi import Fred
    fred = Fred(api_key=os.environ["FRED_API_KEY"])
    cpi_monthly = fred.get_series("CPIAUCSL", observation_start=start)
    cpi_monthly.name = "cpi"

    # --- Merge ---
    btc_dates = prices["btc"].dropna().index
    df = prices.reindex(btc_dates)
    df["volume"] = btc_volume.reindex(btc_dates)
    for name, series in [("hashrate", hashrate), ("addresses", n_addresses),
                         ("n_tx", n_tx), ("revenue", revenue)]:
        df[name] = series.reindex(btc_dates)
    df["cpi"] = cpi_monthly.reindex(btc_dates)
    df = df.ffill().dropna()

    # --- Transformations ---
    price_cols = ["btc", "gold", "oil", "sp500", "msci",
                  "hashrate", "volume", "addresses", "n_tx", "revenue"]
    diff_cols = ["tbond", "tbill", "vix"]
    cpi_cols = ["cpi"]

    log_ret = np.log(df[price_cols]).diff()
    log_ret.columns = [f"dln_{c}" for c in log_ret.columns]

    first_diff = df[diff_cols].diff()
    first_diff.columns = [f"d_{c}" for c in first_diff.columns]

    cpi_ret = np.log(df[cpi_cols]).diff()
    cpi_ret.columns = [f"dln_{c}" for c in cpi_ret.columns]

    df = pd.concat([df, log_ret, first_diff, cpi_ret], axis=1).iloc[1:]
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building dataset from APIs...")
    df = build_dataset()
    out = DATA_DIR / "btc_dataset.csv"
    df.to_csv(out)
    print(f"Saved to {out}")
    print(f"Shape: {df.shape}, Range: {df.index.min().date()} → {df.index.max().date()}")
