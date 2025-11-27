#!/usr/bin/env python3
"""
Debug script part 2: Test with more realistic sparse factor scenario
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from factor_evaluation import (
    to_dense,
    pnl_rowwise_optimizer,
    rowwise_pearson
)

print("="*60)
print("SCENARIO: Sparse factor with 2000 tickers")
print("="*60)

# Create 1-minute timestamps for 1 day
start_time = datetime(2024, 1, 1, 9, 30)
timestamps_1min = pd.date_range(start_time, periods=390, freq='min')
print(f"Created {len(timestamps_1min)} 1-minute timestamps")

# Create 2000 tickers
n_tickers = 2000
tickers = [f'TICK{i:04d}' for i in range(n_tickers)]

# Create a SPARSE factor (only triggers when condition is met)
# Simulate a factor like "high_volume_effect" that only has values for 5% of ticker-timestamps
factor_data = {}
forward_returns = {}

np.random.seed(42)

for ticker in tickers:
    # Factor: sparse (95% NaN)
    factor_vals = np.full(len(timestamps_1min), np.nan)
    valid_idx = np.random.choice(len(timestamps_1min), size=int(len(timestamps_1min)*0.05), replace=False)
    factor_vals[valid_idx] = np.random.randn(len(valid_idx)) * 0.01

    # Returns: always available
    returns = np.random.randn(len(timestamps_1min)) * 0.02

    factor_data[ticker] = pd.Series(factor_vals, index=timestamps_1min)
    forward_returns[ticker] = pd.Series(returns, index=timestamps_1min)

# Check sparsity
total_vals = sum(f.notna().sum() for f in factor_data.values())
total_possible = len(timestamps_1min) * n_tickers
print(f"Factor sparsity: {total_vals}/{total_possible} ({total_vals/total_possible*100:.1f}% non-NaN)")

# Per-timestamp: how many valid assets?
ts_index_1min, _, F_1min, R_1min = to_dense(factor_data, forward_returns, resample_freq=None)
valid_per_ts_1min = np.sum(np.isfinite(F_1min) & np.isfinite(R_1min), axis=1)
print(f"\n1-MINUTE DATA:")
print(f"  Valid assets per timestamp: min={valid_per_ts_1min.min()}, "
      f"mean={valid_per_ts_1min.mean():.1f}, max={valid_per_ts_1min.max()}")
print(f"  Timestamps with >= 10 valid assets: {(valid_per_ts_1min >= 10).sum()}/{len(valid_per_ts_1min)}")

# Compute IC
ic_1min = rowwise_pearson(F_1min, R_1min)
ic_valid_1min = np.isfinite(ic_1min).sum()
print(f"  IC: {ic_valid_1min} valid timestamps")

# Compute PnL
pnl_1min = pnl_rowwise_optimizer(F_1min, R_1min, min_assets=10)
pnl_valid_1min = np.isfinite(pnl_1min).sum()
print(f"  PnL: {pnl_valid_1min} valid timestamps")

# Now with 30-minute resampling
print(f"\n30-MINUTE RESAMPLED DATA:")
ts_index_30min, _, F_30min, R_30min = to_dense(factor_data, forward_returns, resample_freq='30min')
valid_per_ts_30min = np.sum(np.isfinite(F_30min) & np.isfinite(R_30min), axis=1)
print(f"  Timestamps: {len(ts_index_30min)}")
print(f"  Valid assets per timestamp: min={valid_per_ts_30min.min()}, "
      f"mean={valid_per_ts_30min.mean():.1f}, max={valid_per_ts_30min.max()}")
print(f"  Timestamps with >= 10 valid assets: {(valid_per_ts_30min >= 10).sum()}/{len(valid_per_ts_30min)}")

# Compute IC
ic_30min = rowwise_pearson(F_30min, R_30min)
ic_valid_30min = np.isfinite(ic_30min).sum()
print(f"  IC: {ic_valid_30min} valid timestamps")

# Compute PnL
pnl_30min = pnl_rowwise_optimizer(F_30min, R_30min, min_assets=10)
pnl_valid_30min = np.isfinite(pnl_30min).sum()
print(f"  PnL: {pnl_valid_30min} valid timestamps")

print(f"\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print(f"For sparse factors (5% coverage):")
print(f"  1-min: IC={ic_valid_1min} points, PnL={pnl_valid_1min} points")
print(f"  30-min: IC={ic_valid_30min} points, PnL={pnl_valid_30min} points")
print(f"\nResampling helps, but if factor is too sparse, PnL will still be 0")
print(f"because min_assets=10 requirement is not met at each timestamp.")
