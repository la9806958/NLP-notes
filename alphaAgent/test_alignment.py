#!/usr/bin/env python3
"""
Unit tests for factor-return alignment verification.

Tests that factor resampling and forward return computation maintain
strict causal relationships without lookahead bias.
"""

import numpy as np
import pandas as pd
import pytest

# Import from local modules
from factor_evaluation import (
    make_30min_forward_return_from_price,
    resample_factor_to_30min
)

# --- Helpers ---------------------------------------------------------------

def make_session_minutes(start_date, end_date, tz='America/New_York'):
    """
    Build a tz-aware 1-minute DateTimeIndex for US regular session (09:30-16:00 ET)
    for business days in [start_date, end_date].
    """
    days = pd.bdate_range(start=start_date, end=end_date, tz=tz)
    all_minutes = []
    for d in days:
        # 09:30 to 16:00 inclusive of start, exclusive of end when resampling [label='left']
        start = (d.tz_convert(tz).normalize() + pd.Timedelta(hours=9, minutes=30))
        end   = (d.tz_convert(tz).normalize() + pd.Timedelta(hours=16))
        all_minutes.append(pd.date_range(start, end, freq='1min', inclusive='left', tz=tz))
    return all_minutes[0].append(all_minutes[1:]) if len(all_minutes) > 1 else all_minutes[0]

def per_minute_path_from_bin_returns(index_1m, bin_minutes=30, bin_returns=None):
    """
    Create a 1-minute price path such that each 30-min bin has a specified gross return.
    For each bin k with gross return g_k, we set a constant per-minute return r_k satisfying:
        (1 + r_k) ** bin_minutes = (1 + g_k)
    """
    if bin_returns is None:
        raise ValueError("bin_returns must be provided")

    # Build a bin id per minute
    # Assume bins align to the session start and are contiguous with length 'bin_minutes'
    minutes_from_open = (index_1m - index_1m.normalize() - pd.Timedelta(hours=9, minutes=30)).seconds // 60
    bin_id = minutes_from_open // bin_minutes

    # Map bin_id -> per-minute return
    unique_bins = np.unique(bin_id)
    assert len(unique_bins) == len(bin_returns), "bin_returns length must equal #bins per day"

    # Build per-minute return array
    per_minute_r = np.empty(len(index_1m), dtype=float)
    for k, g_k in enumerate(bin_returns):
        # constant per-minute return in this bin
        r_k = (1.0 + g_k) ** (1.0 / bin_minutes) - 1.0
        per_minute_r[bin_id == k] = r_k

    # Create price path starting at 100.0
    prices = np.empty(len(index_1m), dtype=float)
    prices[0] = 100.0
    for t in range(1, len(index_1m)):
        prices[t] = prices[t-1] * (1.0 + per_minute_r[t-1])

    return pd.Series(prices, index=index_1m, name="price")

def make_factor_1m_from_bin_returns(index_1m, bin_minutes=30, bin_returns=None):
    """
    Make a 1-minute factor that equals the CURRENT bin's 30-min gross return g_k
    for every minute inside that bin (i.e., piecewise-constant per bin).
    After resampling with last() and a FULL-BIN shift(1), this should equal
    the PREVIOUS bin's 30-min return.
    """
    if bin_returns is None:
        raise ValueError("bin_returns must be provided")

    minutes_from_open = (index_1m - index_1m.normalize() - pd.Timedelta(hours=9, minutes=30)).seconds // 60
    bin_id = minutes_from_open // bin_minutes
    g = np.array(bin_returns, dtype=float)
    factor_vals = g[bin_id]  # piecewise constant per current bin
    return pd.Series(factor_vals, index=index_1m, name="factor")

def last_bar_mask_per_session(idx):
    """Create boolean mask for last bar of each trading session."""
    by_day = pd.Series(1, index=idx).groupby(idx.date, group_keys=False)
    return by_day.cumcount(ascending=False) == 0

# --- Tests -----------------------------------------------------------------

@pytest.mark.parametrize("bin_minutes", [30])
def test_forward_returns_drop_last_bar_and_no_lookahead(bin_minutes):
    """
    1) Forward returns drop the last bar (overnight removed).
    2) With strict-causal factor resampling (full-bin lag), factor[t] == fwd_return[t-Δ].
       Concretely: factor_resampled == fwd_return.shift(1) on common timestamps.
    3) As a sanity check, if we *remove* the full-bin lag, correlation at lag 0 ~ 1 (lookahead).
    """
    # Two business days, so we exercise the overnight boundary
    idx_1m = make_session_minutes("2024-07-01", "2024-07-02", tz="America/New_York")

    # There are exactly 13 bins of 30 minutes in a 6.5h session (09:30–16:00)
    bins_per_day = 13
    g_day = [0.01, -0.02, 0.03, 0.005, 0.0, -0.01, 0.015, -0.005, 0.02, 0.0, -0.015, 0.01, -0.005]
    assert len(g_day) == bins_per_day

    # Build alternating pattern across two days (same pattern both days)
    # We construct prices and factor on each day separately to avoid cross-day spillover.
    days = sorted(set(pd.to_datetime(idx_1m.tz_convert("America/New_York").date)))
    all_prices = []
    all_factor = []
    for d in days:
        day_mask = (idx_1m.date == d.date())
        idx_d = idx_1m[day_mask]
        prices_d = per_minute_path_from_bin_returns(idx_d, bin_minutes, g_day)
        factor_d = make_factor_1m_from_bin_returns(idx_d, bin_minutes, g_day)
        all_prices.append(prices_d)
        all_factor.append(factor_d)

    price_1m = pd.concat(all_prices).sort_index()
    factor_1m = pd.concat(all_factor).sort_index()

    # Compute 30-min forward returns (should drop last bar of each session)
    fwd = make_30min_forward_return_from_price(price_1m, resample_freq=f"{bin_minutes}T")

    # Check: last bar of each session is NaN (overnight removed)
    mask_last = last_bar_mask_per_session(fwd.index)
    assert fwd[mask_last].isna().all(), "Last bar per session should be NaN in forward returns."

    # Resample factor with STRICT causality (your function should do full-bin shift(1))
    f_res = resample_factor_to_30min(factor_1m, resample_freq=f"{bin_minutes}T", causal_shift=True)

    # Compare on common timestamps
    idx = f_res.index.intersection(fwd.index)
    lhs = f_res.loc[idx]
    rhs = fwd.shift(1).loc[idx]  # previous-bin return

    # 1) Equality within numerical tolerance
    assert np.allclose(lhs.values, rhs.values, equal_nan=True, atol=1e-12), \
        "Factor (strict-causal) must equal previous bin's forward return."

    # 2) Correlation sanity: lag-0 ~ 0, lag+1 ~ 1 (on non-NaNs)
    valid = lhs.notna() & rhs.notna()
    if valid.sum() > 2:
        corr_lag0 = np.corrcoef(lhs[valid], fwd.loc[idx][valid])[0, 1]
        corr_lagp1 = np.corrcoef(lhs[valid], fwd.shift(-1).loc[idx][valid])[0, 1]
        assert abs(corr_lag0) < 0.2, f"Lag-0 correlation should be small; got {corr_lag0:.3f}"
        assert corr_lagp1 > 0.99, f"Lag+1 correlation should be ~1; got {corr_lagp1:.3f}"

    # 3) Negative control: if we (incorrectly) remove the full-bin lag, we get lookahead at lag 0
    #    Simulate a bad resampler: resample last() without the shift(1)
    bad_factor_no_lag = factor_1m.resample(f"{bin_minutes}T", label='left', closed='left').last()
    common = bad_factor_no_lag.index.intersection(fwd.index)
    vmask = bad_factor_no_lag.loc[common].notna() & fwd.loc[common].notna()
    if vmask.sum() > 2:
        corr_lookahead = np.corrcoef(bad_factor_no_lag.loc[common][vmask],
                                     fwd.loc[common][vmask])[0, 1]
        assert corr_lookahead > 0.99, \
            "Without the full-bin lag, lag-0 correlation ~1 (lookahead) should be observed."


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"])
