#!/usr/bin/env python3
"""
Diagnose the sharp drawdown in Nov 2023 - Jan 2024 backtest period
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from data_loader import load_real_market_data
from factor_evaluation import to_dense, pnl_rowwise_optimizer, clean_pnl_spikes

print("="*80)
print("DIAGNOSING BACKTEST DRAWDOWN: Nov 2023 - Jan 2024")
print("="*80)

# Load data
print("\n1. Loading market data...")
data_dict = load_real_market_data(
    data_path="/home/lichenhui/data/1min",
    n_tickers=100,  # Use 100 tickers for diagnosis
    n_cores=10
)

tickers = data_dict['tickers']
ticker_data = data_dict['ticker_data']
print(f"   Loaded {len(tickers)} tickers")

# Create simple momentum factor for testing
print("\n2. Creating test factor (30-min close momentum)...")
factor_data = {}
forward_returns = {}

for ticker in tickers:
    if ticker not in ticker_data:
        continue

    df = ticker_data[ticker]

    # Resample to 30-min
    df_30min = df.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    if len(df_30min) < 50:
        continue

    # Simple factor: 30-period momentum
    factor = df_30min['close'].pct_change(periods=30)

    # Forward returns: 1-period ahead
    fwd_ret = df_30min['close'].shift(-1) / df_30min['close'] - 1

    factor_data[ticker] = factor
    forward_returns[ticker] = fwd_ret

print(f"   Created factors for {len(factor_data)} tickers")

# Convert to dense matrices
print("\n3. Converting to dense matrices...")
ts_index, tickers_used, F, R = to_dense(factor_data, forward_returns, resample_freq=None)
print(f"   Matrix shape: {F.shape} (T={F.shape[0]} timestamps, N={F.shape[1]} tickers)")
print(f"   Date range: {ts_index.min()} to {ts_index.max()}")

# Analyze data coverage over time
print("\n4. Analyzing data coverage by time period...")
valid_counts = np.sum(np.isfinite(F) & np.isfinite(R), axis=1)
coverage_df = pd.DataFrame({
    'timestamp': ts_index,
    'valid_assets': valid_counts
})

# Group by month
coverage_df['month'] = coverage_df['timestamp'].dt.to_period('M')
monthly_coverage = coverage_df.groupby('month')['valid_assets'].agg(['mean', 'min', 'max', 'count'])
print("\nMonthly data coverage:")
print(monthly_coverage.head(10))

# Check early period specifically
early_period_mask = ts_index < '2024-02-01'
early_coverage = coverage_df[early_period_mask]
print(f"\nEarly period (before 2024-02-01):")
print(f"  Mean valid assets: {early_coverage['valid_assets'].mean():.1f}")
print(f"  Min valid assets: {early_coverage['valid_assets'].min()}")
print(f"  Max valid assets: {early_coverage['valid_assets'].max()}")

# Compute PnL
print("\n5. Computing PnL with optimizer...")
pnl = pnl_rowwise_optimizer(F, R, risk_budget=0.1, gme_limit=2.0, min_assets=10, n_workers=4)
pnl_series = pd.Series(pnl, index=ts_index).dropna()
print(f"   Valid PnL points: {len(pnl_series)} / {len(ts_index)}")

# Analyze PnL distribution
print("\n6. Analyzing PnL distribution...")
print(f"   Mean: {pnl_series.mean():.6f}")
print(f"   Std: {pnl_series.std():.6f}")
print(f"   Min: {pnl_series.min():.6f}")
print(f"   Max: {pnl_series.max():.6f}")
print(f"   Median: {pnl_series.median():.6f}")

# Check for extreme values
extreme_positive = pnl_series[pnl_series > 0.5].count()
extreme_negative = pnl_series[pnl_series < -0.5].count()
print(f"\n   Extreme positive returns (>50%): {extreme_positive}")
print(f"   Extreme negative returns (<-50%): {extreme_negative}")

# Show worst returns
print("\n   10 Worst returns:")
worst_returns = pnl_series.nsmallest(10)
for timestamp, ret in worst_returns.items():
    # Check data coverage at this timestamp
    idx = ts_index.get_loc(timestamp)
    n_valid = valid_counts[idx]
    print(f"     {timestamp}: {ret:.4f} ({ret*100:.2f}%) - {n_valid} valid assets")

# Analyze by time period
print("\n7. Comparing early vs later periods...")
early_pnl = pnl_series[pnl_series.index < '2024-02-01']
later_pnl = pnl_series[pnl_series.index >= '2024-02-01']

print(f"\n   Early period (before 2024-02-01):")
print(f"     Count: {len(early_pnl)}")
print(f"     Mean: {early_pnl.mean():.6f}")
print(f"     Std: {early_pnl.std():.6f}")
print(f"     Min: {early_pnl.min():.6f}")
print(f"     Max: {early_pnl.max():.6f}")

print(f"\n   Later period (from 2024-02-01):")
print(f"     Count: {len(later_pnl)}")
print(f"     Mean: {later_pnl.mean():.6f}")
print(f"     Std: {later_pnl.std():.6f}")
print(f"     Min: {later_pnl.min():.6f}")
print(f"     Max: {later_pnl.max():.6f}")

# Compute cumulative PnL
print("\n8. Computing cumulative PnL...")
cumulative_wealth = (1 + pnl_series).cumprod()
cumulative_pnl = cumulative_wealth - 1

print(f"   Start: {cumulative_pnl.iloc[0]:.4f}")
print(f"   End: {cumulative_pnl.iloc[-1]:.4f}")
print(f"   Min (max drawdown): {cumulative_pnl.min():.4f}")

# Find when drawdown bottoms out
min_idx = cumulative_pnl.argmin()
min_date = cumulative_pnl.index[min_idx]
print(f"   Worst drawdown at: {min_date}")
print(f"   Cumulative PnL at that point: {cumulative_pnl.iloc[min_idx]:.4f}")

# Create visualization
print("\n9. Creating diagnostic visualization...")
fig, axes = plt.subplots(4, 1, figsize=(16, 14))

# Plot 1: Data coverage over time
ax = axes[0]
ax.plot(coverage_df['timestamp'], coverage_df['valid_assets'], linewidth=1, alpha=0.7)
ax.axvline(pd.Timestamp('2024-02-01'), color='red', linestyle='--', label='Feb 2024')
ax.set_ylabel('Valid Assets')
ax.set_title('Data Coverage Over Time')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Period returns
ax = axes[1]
colors = ['green' if x > 0 else 'red' for x in pnl_series.values]
ax.bar(pnl_series.index, pnl_series.values * 100, color=colors, alpha=0.6, width=0.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axvline(pd.Timestamp('2024-02-01'), color='red', linestyle='--', label='Feb 2024')
ax.set_ylabel('Period Return (%)')
ax.set_title('Period Returns')
ax.set_ylim(-60, 60)
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 3: Cumulative PnL
ax = axes[2]
ax.plot(cumulative_pnl.index, cumulative_pnl.values * 100, linewidth=2)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(pd.Timestamp('2024-02-01'), color='red', linestyle='--', label='Feb 2024')
ax.set_ylabel('Cumulative PnL (%)')
ax.set_title('Cumulative PnL')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Rolling 20-period mean return
ax = axes[3]
rolling_mean = pnl_series.rolling(20).mean()
ax.plot(rolling_mean.index, rolling_mean.values * 100, linewidth=2, color='blue')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(pd.Timestamp('2024-02-01'), color='red', linestyle='--', label='Feb 2024')
ax.set_ylabel('Rolling Mean Return (%)')
ax.set_xlabel('Date')
ax.set_title('20-Period Rolling Mean Return')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig('/home/lichenhui/data/alphaAgent/drawdown_diagnosis.png', dpi=150, bbox_inches='tight')
print(f"   Saved to: /home/lichenhui/data/alphaAgent/drawdown_diagnosis.png")

print("\n" + "="*80)
print("DIAGNOSIS COMPLETE")
print("="*80)
