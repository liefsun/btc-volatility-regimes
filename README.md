# Regime-Switching Volatility Toolkit

Volatility modeling and regime detection for Bitcoin returns using GARCH-family models, Markov-switching frameworks, and machine learning.

## Background

This project extends my PhD research on Bitcoin volatility dynamics. The thesis employed MS-VAR (Markov-Switching Vector Autoregression) and DCC-GARCH (Dynamic Conditional Correlation) to study cross-asset volatility spillovers. This toolkit applies the Markov-switching framework to **univariate volatility modeling** — detecting calm vs turbulent market regimes and producing regime-aware forecasts.

## Project Structure

```
btc-garch/
├── notebooks/
│   ├── 01_data_pipeline.ipynb    # Data acquisition & preprocessing
│   ├── 02_egarch_analysis.ipynb  # GARCH/GJR/EGARCH with exogenous vars
│   ├── 03_ms_garch.ipynb         # Markov-Switching volatility (2 & 3 regimes)
│   └── 04_ml_regimes.ipynb       # ML regime classification (TODO)
├── src/
│   └── data_loader.py            # Data loading utilities
├── data/                         # Downloaded data (gitignored)
├── tests/
├── docs/
└── requirements.txt
```

## Models

| Model | Status | Description |
|-------|--------|-------------|
| GARCH(1,1) | Done | Baseline symmetric volatility |
| GJR-GARCH(1,1) | Done | Asymmetric with leverage indicator |
| EGARCH(1,1) | Done | Log-variance, no positivity constraints |
| EGARCH + Exogenous | Done | Gold, VIX, volume, tx count, CPI in mean eq |
| MS (2-regime) | Done | Markov-switching mean + variance, calm vs turbulent |
| MS (3-regime) | Done | Calm / Normal / Crisis granularity |
| ML Regime Classification | TODO | RF/XGBoost vs HMM comparison |

## Key Findings

### Single-Regime (EGARCH)
- **Best model**: EGARCH(1,1) + Student-t + 5 exogenous variables
- **Heavy tails**: Degrees of freedom ν ≈ 2.7 — extreme leptokurtosis
- **No leverage effect**: γ not significant for BTC (unlike equities)
- **Significant drivers**: Gold (+), VIX (−), trading volume (+), transaction count (+), CPI (+)

### Regime-Switching
- **2-regime model**: Calm (σ_annual ≈ 32%) vs Turbulent (σ_annual ≈ 104%), volatility ratio 3.3x
- **Regime persistence**: Calm lasts ~9 days, Turbulent ~5 days on average
- **67% calm / 33% turbulent**: BTC spends about two-thirds of time in calm regime
- **3-regime preferred** (lower AIC): Calm (27%) / Normal (81%) / Crisis (203% annualized vol)
- **Turbulent episodes align with known events**: COVID crash, China mining ban, FTX collapse

## Data

Daily observations from 2015-01-05 to present. Sources:
- **Financial markets**: yfinance (BTC, Gold, Oil, SP500, VIX, MSCI, T-Bond, T-Bill)
- **On-chain**: blockchain.com API (hashrate, addresses, transactions, revenue)
- **Macro**: FRED API (CPI)
- **Sentiment**: Google Trends via pytrends

## Tech Stack

Python, NumPy, SciPy, pandas, matplotlib, arch, statsmodels, scikit-learn

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT
