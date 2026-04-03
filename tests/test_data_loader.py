"""Tests for src/data_loader.py — offline functions only (no API calls)."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

# Ensure src/ is importable
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_from_csv, compute_log_returns, get_btc_returns, get_exog, DATA_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dataset():
    """Load the real dataset once for all tests."""
    return load_from_csv()


@pytest.fixture
def small_prices():
    """Tiny DataFrame for unit-testing compute_log_returns."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.DataFrame({"btc": [100, 110, 105, 115, 120]}, index=idx)


# ---------------------------------------------------------------------------
# load_from_csv
# ---------------------------------------------------------------------------

class TestLoadFromCsv:
    def test_returns_dataframe(self, dataset):
        assert isinstance(dataset, pd.DataFrame)

    def test_has_datetime_index(self, dataset):
        assert isinstance(dataset.index, pd.DatetimeIndex)

    def test_expected_columns_present(self, dataset):
        required = ["btc", "dln_btc", "dln_gold", "d_vix", "dln_volume",
                     "dln_n_tx", "dln_cpi", "vix", "sp500"]
        for col in required:
            assert col in dataset.columns, f"Missing column: {col}"

    def test_no_nans_in_core_columns(self, dataset):
        core = ["btc", "dln_btc", "gold", "vix"]
        for col in core:
            assert dataset[col].isna().sum() == 0, f"NaN in {col}"

    def test_reasonable_row_count(self, dataset):
        assert len(dataset) > 3000, "Dataset too small"

    def test_date_range_starts_2015(self, dataset):
        assert dataset.index.min().year == 2015

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            load_from_csv(Path("/nonexistent/file.csv"))


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------

class TestComputeLogReturns:
    def test_length(self, small_prices):
        ret = compute_log_returns(small_prices, "btc")
        assert len(ret) == len(small_prices) - 1

    def test_values(self, small_prices):
        ret = compute_log_returns(small_prices, "btc")
        expected_first = np.log(110 / 100)
        assert abs(ret.iloc[0] - expected_first) < 1e-10

    def test_no_nans(self, small_prices):
        ret = compute_log_returns(small_prices, "btc")
        assert ret.isna().sum() == 0

    def test_missing_column_raises(self, small_prices):
        with pytest.raises(KeyError):
            compute_log_returns(small_prices, "nonexistent")


# ---------------------------------------------------------------------------
# get_btc_returns
# ---------------------------------------------------------------------------

class TestGetBtcReturns:
    def test_default_scale_100(self, dataset):
        ret = get_btc_returns(dataset)
        raw = dataset["dln_btc"]
        assert abs(ret.iloc[0] - raw.iloc[0] * 100) < 1e-10

    def test_scale_1(self, dataset):
        ret = get_btc_returns(dataset, scale=1.0)
        raw = dataset["dln_btc"]
        assert abs(ret.iloc[0] - raw.iloc[0]) < 1e-10

    def test_name(self, dataset):
        ret = get_btc_returns(dataset)
        assert ret.name == "btc_ret"

    def test_same_length(self, dataset):
        ret = get_btc_returns(dataset)
        assert len(ret) == len(dataset)


# ---------------------------------------------------------------------------
# get_exog
# ---------------------------------------------------------------------------

class TestGetExog:
    def test_returns_5_columns(self, dataset):
        exog = get_exog(dataset)
        assert exog.shape[1] == 5

    def test_expected_columns(self, dataset):
        exog = get_exog(dataset)
        expected = {"dln_gold", "d_vix", "dln_volume", "dln_n_tx", "dln_cpi"}
        assert set(exog.columns) == expected

    def test_no_infs(self, dataset):
        exog = get_exog(dataset)
        assert not np.isinf(exog.values).any()

    def test_no_nans(self, dataset):
        exog = get_exog(dataset)
        assert exog.isna().sum().sum() == 0

    def test_scale_applied(self, dataset):
        exog_100 = get_exog(dataset, scale=100.0)
        exog_1 = get_exog(dataset, scale=1.0)
        ratio = exog_100.iloc[0, 0] / exog_1.iloc[0, 0]
        assert abs(ratio - 100.0) < 1e-6


# ---------------------------------------------------------------------------
# DATA_DIR
# ---------------------------------------------------------------------------

class TestDataDir:
    def test_data_dir_exists(self):
        assert DATA_DIR.exists(), f"DATA_DIR not found: {DATA_DIR}"

    def test_csv_exists(self):
        csv_path = DATA_DIR / "btc_dataset.csv"
        assert csv_path.exists(), f"Dataset CSV not found: {csv_path}"
