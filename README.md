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
│   ├── 04_oos_evaluation.ipynb   # Out-of-sample forecast evaluation
│   ├── 05_ml_regimes.ipynb       # HMM + RF/XGBoost regime classification
│   └── 06_trading_backtest.ipynb # Regime-based trading strategy backtest
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
| OOS Forecast Evaluation | Done | Expanding window backtest: GARCH vs EGARCH vs MS |
| Gaussian HMM | Done | 2-regime HMM, comparison with Markov-Switching |
| RF / XGBoost | Done | Supervised regime classification with feature importance |
| Trading Backtest | Done | Regime-based signal → position sizing → PnL evaluation |

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

### Out-of-Sample Forecast
- **GARCH(1,1) wins** on MAE and QLIKE — simpler model forecasts better OOS
- **DM test**: GARCH vs MS significantly different (p=0.048); EGARCH vs MS not significant
- All models have low Mincer-Zarnowitz R² — typical for daily vol forecasting

### ML Regime Classification
- **HMM vs Markov-Switching**: 94.8% agreement (Cohen's kappa = 0.88), substantial concordance
- **XGBoost best for turbulent detection**: F1 = 0.61 (precision 60%, recall 62%)
- **Top features**: rolling volatility (5d, 21d), lagged returns, VIX changes
- **1-day-ahead prediction degrades significantly**: F1 drops from 0.61 to 0.33 — regimes are easier to classify than predict

### Trading Strategy Backtest
- **Scaled strategy** (weight = 1 - P(turbulent)) achieves **Sharpe 1.26** vs Buy-and-Hold 0.95
- **Max drawdown halved**: -42% vs -83% for buy-and-hold
- **Breakeven transaction cost**: ~0.26% per trade for Scaled strategy
- **Best during crises**: COVID Sharpe 2.29 (vs 1.80 buy-and-hold), Crypto Winter -0.97 (vs -1.60)

## Data

Daily observations from 2015-01-05 to present. Sources:
- **Financial markets**: yfinance (BTC, Gold, Oil, SP500, VIX, MSCI, T-Bond, T-Bill)
- **On-chain**: blockchain.com API (hashrate, addresses, transactions, revenue)
- **Macro**: FRED API (CPI)
- **Sentiment**: Google Trends via pytrends

## Tech Stack

Python, NumPy, SciPy, pandas, matplotlib, arch, statsmodels, scikit-learn, hmmlearn, XGBoost

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT
