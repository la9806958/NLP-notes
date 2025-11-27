#!/usr/bin/env python3
"""
Debug script to understand why PnL data is insufficient despite having valid IC data.
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add path for imports
sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from factor_evaluation import (
    to_dense,
    compute_ic_series_dense,
    compute_sharpe_dense,
    rowwise_pearson,
    pnl_rowwise_optimizer,
    clean_pnl_spikes
)

# Create synthetic test data
print("="*60)
print("DEBUGGING PNL CALCULATION ISSUE")
print("="*60)

# Create timestamps (1-minute data for 2 days)
start_time = datetime(2024, 1, 1, 9, 30)
timestamps_1min = pd.date_range(start_time, periods=780, freq='1T')  # 390 min/day * 2 days
print(f"\n1. Created {len(timestamps_1min)} 1-minute timestamps")

# Create 5 tickers with factor and return data
n_tickers = 5
tickers = [f'TICK{i}' for i in range(n_tickers)]

factor_data = {}
forward_returns = {}

np.random.seed(42)

for ticker in tickers:
    # Factor values: some NaNs to simulate sparse data
    factor_vals = np.random.randn(len(timestamps_1min)) * 0.01
    # Add some NaNs (20% sparse)
    nan_mask = np.random.rand(len(timestamps_1min)) < 0.2
    factor_vals[nan_mask] = np.nan

    # Forward returns: some extreme values that will be clipped
    returns = np.random.randn(len(timestamps_1min)) * 0.02
    # Add some extreme outliers (5%)
    extreme_mask = np.random.rand(len(timestamps_1min)) < 0.05
    returns[extreme_mask] = np.random.choice([-0.6, 0.6], extreme_mask.sum())

    factor_data[ticker] = pd.Series(factor_vals, index=timestamps_1min)
    forward_returns[ticker] = pd.Series(returns, index=timestamps_1min)

print(f"2. Created factor data for {len(tickers)} tickers")
print(f"   Sample factor stats: mean={factor_data[tickers[0]].mean():.6f}, "
      f"NaN%={factor_data[tickers[0]].isna().sum()/len(timestamps_1min)*100:.1f}%")
print(f"   Sample return stats: mean={forward_returns[tickers[0]].mean():.6f}, "
      f"min={forward_returns[tickers[0]].min():.3f}, max={forward_returns[tickers[0]].max():.3f}")

# Convert to dense matrices WITHOUT resampling
print("\n" + "="*60)
print("STEP 1: Convert to dense (1-minute, NO resampling)")
print("="*60)
ts_index_1min, tickers_sorted, F_1min, R_1min = to_dense(factor_data, forward_returns, resample_freq=None)
print(f"F shape: {F_1min.shape}, R shape: {R_1min.shape}")
print(f"F valid: {np.isfinite(F_1min).sum()}/{F_1min.size} ({np.isfinite(F_1min).sum()/F_1min.size*100:.1f}%)")
print(f"R valid: {np.isfinite(R_1min).sum()}/{R_1min.size} ({np.isfinite(R_1min).sum()/R_1min.size*100:.1f}%)")
print(f"R min: {np.nanmin(R_1min):.3f}, max: {np.nanmax(R_1min):.3f}")

# Compute IC on 1-minute data
ic_1min = rowwise_pearson(F_1min, R_1min)
ic_1min_valid = ic_1min[np.isfinite(ic_1min)]
print(f"\nIC (1-min): {len(ic_1min_valid)} valid values out of {len(ic_1min)} timestamps")
print(f"IC mean: {ic_1min_valid.mean():.6f}")

# Compute PnL on 1-minute data
print("\nComputing PnL (1-min, optimizer)...")
pnl_1min = pnl_rowwise_optimizer(F_1min, R_1min, min_assets=3)  # Lower threshold for test
pnl_series_1min = pd.Series(pnl_1min, index=ts_index_1min).dropna()
print(f"PnL before cleaning: {len(pnl_series_1min)} valid values")

# Clean spikes
pnl_cleaned_1min = clean_pnl_spikes(pnl_series_1min, max_return=0.5)
pnl_cleaned_1min = pnl_cleaned_1min.dropna()
print(f"PnL after cleaning: {len(pnl_cleaned_1min)} valid values")
print(f"PnL range: [{pnl_cleaned_1min.min():.6f}, {pnl_cleaned_1min.max():.6f}]")

# Count how many values are exactly ±0.5
at_limit = (pnl_series_1min.abs() == 0.5).sum()
print(f"PnL values at exactly ±0.5: {at_limit}")

# Now test with 30-minute resampling
print("\n" + "="*60)
print("STEP 2: Convert to dense (30-minute, WITH resampling)")
print("="*60)
ts_index_30min, tickers_sorted, F_30min, R_30min = to_dense(factor_data, forward_returns, resample_freq='30T')
print(f"F shape: {F_30min.shape}, R shape: {R_30min.shape}")
print(f"F valid: {np.isfinite(F_30min).sum()}/{F_30min.size} ({np.isfinite(F_30min).sum()/F_30min.size*100:.1f}%)")
print(f"R valid: {np.isfinite(R_30min).sum()}/{R_30min.size} ({np.isfinite(R_30min).sum()/R_30min.size*100:.1f}%)")
print(f"R min: {np.nanmin(R_30min):.3f}, max: {np.nanmax(R_30min):.3f}")

# Compute IC on 30-minute data
ic_30min = rowwise_pearson(F_30min, R_30min)
ic_30min_valid = ic_30min[np.isfinite(ic_30min)]
print(f"\nIC (30-min): {len(ic_30min_valid)} valid values out of {len(ic_30min)} timestamps")
print(f"IC mean: {ic_30min_valid.mean():.6f}")

# Compute PnL on 30-minute data
print("\nComputing PnL (30-min, optimizer)...")
pnl_30min = pnl_rowwise_optimizer(F_30min, R_30min, min_assets=3)
pnl_series_30min = pd.Series(pnl_30min, index=ts_index_30min).dropna()
print(f"PnL before cleaning: {len(pnl_series_30min)} valid values")

# Clean spikes
pnl_cleaned_30min = clean_pnl_spikes(pnl_series_30min, max_return=0.5)
pnl_cleaned_30min = pnl_cleaned_30min.dropna()
print(f"PnL after cleaning: {len(pnl_cleaned_30min)} valid values")
if len(pnl_cleaned_30min) > 0:
    print(f"PnL range: [{pnl_cleaned_30min.min():.6f}, {pnl_cleaned_30min.max():.6f}]")

# Count how many values are exactly ±0.5
at_limit = (pnl_series_30min.abs() == 0.5).sum()
print(f"PnL values at exactly ±0.5: {at_limit}")

# Check the spike cleaning logic
print("\n" + "="*60)
print("SPIKE CLEANING ANALYSIS")
print("="*60)

test_pnl = pd.Series([0.1, 0.49, 0.5, 0.51, -0.49, -0.5, -0.51, 0.0])
print(f"Test PnL values: {test_pnl.values}")
print(f"Values with abs() >= 0.5: {(test_pnl.abs() >= 0.5).sum()}")
print(f"Values with abs() > 0.5: {(test_pnl.abs() > 0.5).sum()}")

cleaned = clean_pnl_spikes(test_pnl, max_return=0.5)
print(f"\nAfter cleaning (current logic with >=):")
print(f"Remaining values: {cleaned.dropna().values}")
print(f"Removed {len(test_pnl) - len(cleaned.dropna())} values")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"1-min:  IC has {len(ic_1min_valid)} points, PnL has {len(pnl_cleaned_1min)} points")
print(f"30-min: IC has {len(ic_30min_valid)} points, PnL has {len(pnl_cleaned_30min)} points")
print(f"\nThe issue: PnL has fewer valid points than IC due to:")
print(f"  1. pnl_rowwise_optimizer requires min_assets=10 per timestamp")
print(f"  2. Optimization can fail for individual timestamps")
print(f"  3. Spike cleaning with >= removes clipped values at exactly ±0.5")
