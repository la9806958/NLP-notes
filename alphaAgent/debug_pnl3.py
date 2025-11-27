#!/usr/bin/env python3
"""
Final comprehensive debug: Test all components step by step
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from factor_evaluation import (
    to_dense,
    compute_sharpe_dense,
    pnl_rowwise_optimizer,
    clean_pnl_spikes
)

print("="*60)
print("COMPREHENSIVE DEBUG TEST")
print("="*60)

# Create realistic data: 100 tickers, 1 day of 1-minute data
start_time = datetime(2024, 1, 1, 9, 30)
timestamps_1min = pd.date_range(start_time, periods=390, freq='min')
n_tickers = 100
tickers = [f'TICK{i:03d}' for i in range(n_tickers)]

np.random.seed(42)

factor_data = {}
forward_returns = {}

for ticker in tickers:
    # Factor: moderate sparsity (30% NaN)
    factor_vals = np.random.randn(len(timestamps_1min)) * 0.01
    nan_mask = np.random.choice([True, False], size=len(timestamps_1min), p=[0.3, 0.7])
    factor_vals[nan_mask] = np.nan

    # Returns: some extreme values
    returns = np.random.randn(len(timestamps_1min)) * 0.02
    extreme_idx = np.random.choice(len(timestamps_1min), size=5, replace=False)
    returns[extreme_idx] = np.random.choice([-0.6, 0.6], size=5)

    factor_data[ticker] = pd.Series(factor_vals, index=timestamps_1min)
    forward_returns[ticker] = pd.Series(returns, index=timestamps_1min)

print(f"Created data: {n_tickers} tickers, {len(timestamps_1min)} timestamps")

# Test WITHOUT resampling
print("\n" + "="*60)
print("TEST 1: 1-minute data (no resampling)")
print("="*60)

ts_index_1min, _, F_1min, R_1min = to_dense(factor_data, forward_returns, resample_freq=None)
print(f"Matrix shape: {F_1min.shape}")
valid_per_ts = np.sum(np.isfinite(F_1min) & np.isfinite(R_1min), axis=1)
print(f"Valid assets per timestamp: {valid_per_ts.min()}-{valid_per_ts.max()}")

# Compute PnL directly
print("\nComputing PnL (optimizer, min_assets=10)...")
pnl_1min = pnl_rowwise_optimizer(F_1min, R_1min, min_assets=10, n_workers=4)
pnl_series_1min = pd.Series(pnl_1min, index=ts_index_1min)
print(f"PnL before dropna: {len(pnl_series_1min)} values, {np.isfinite(pnl_1min).sum()} valid")

pnl_series_1min = pnl_series_1min.dropna()
print(f"PnL after dropna: {len(pnl_series_1min)} values")

if len(pnl_series_1min) > 0:
    print(f"PnL range: [{pnl_series_1min.min():.6f}, {pnl_series_1min.max():.6f}]")

    # Clean spikes
    pnl_cleaned = clean_pnl_spikes(pnl_series_1min, max_return=0.5)
    pnl_cleaned = pnl_cleaned.dropna()
    print(f"PnL after spike cleaning: {len(pnl_cleaned)} values")

    if len(pnl_cleaned) >= 2:
        mean = pnl_cleaned.mean()
        std = pnl_cleaned.std()
        sharpe = (mean / std) * np.sqrt(3276) if std > 0 else 0.0
        print(f"Sharpe ratio: {sharpe:.4f}")
    else:
        print("Insufficient data for Sharpe calculation")
else:
    print("No valid PnL data!")

# Test WITH resampling
print("\n" + "="*60)
print("TEST 2: 30-minute resampled data")
print("="*60)

sharpe_30min, pnl_30min = compute_sharpe_dense(
    F_1min, R_1min, ts_index_1min,
    annualize_periods=3276,
    use_optimizer=True
)

print(f"Sharpe from compute_sharpe_dense: {sharpe_30min:.4f}")
print(f"PnL series length: {len(pnl_30min)}")

# Now test with actual resampling in to_dense
print("\n" + "="*60)
print("TEST 3: Resample in to_dense, then compute")
print("="*60)

ts_index_30min, _, F_30min, R_30min = to_dense(factor_data, forward_returns, resample_freq='30min')
print(f"Matrix shape after resampling: {F_30min.shape}")

sharpe_30min_v2, pnl_30min_v2 = compute_sharpe_dense(
    F_30min, R_30min, ts_index_30min,
    annualize_periods=3276,
    use_optimizer=True
)

print(f"Sharpe: {sharpe_30min_v2:.4f}")
print(f"PnL length: {len(pnl_30min_v2)}")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"1-min data: Sharpe calculation {'succeeded' if len(pnl_series_1min) >= 2 else 'FAILED'}")
print(f"30-min (method 1): Sharpe = {sharpe_30min:.4f}")
print(f"30-min (method 2): Sharpe = {sharpe_30min_v2:.4f}")

if sharpe_30min_v2 != 0.0:
    print("\n✅ SUCCESS: 30-minute resampling produces non-zero Sharpe!")
else:
    print("\n❌ ISSUE: Sharpe is still zero after resampling")






