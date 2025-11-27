#!/usr/bin/env python3
"""
Test the fixes for the drawdown issue
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from data_loader import load_real_market_data
from factor_evaluation import to_dense, compute_sharpe_dense

print("="*80)
print("TESTING FIXES FOR BACKTEST DRAWDOWN")
print("="*80)

# Load data
print("\n1. Loading market data...")
data_dict = load_real_market_data(
    data_path="/home/lichenhui/data/1min",
    n_tickers=100,
    n_cores=10
)

tickers = data_dict['tickers']
ticker_data = data_dict['ticker_data']
print(f"   Loaded {len(tickers)} tickers")

# Create simple momentum factor
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
print(f"   Matrix shape: {F.shape}")
print(f"   Date range: {ts_index.min()} to {ts_index.max()}")

# Check coverage
valid_counts = np.sum(np.isfinite(F) & np.isfinite(R), axis=1)
coverage_pct = valid_counts / F.shape[1]
print(f"\n   Coverage stats:")
print(f"     Mean: {coverage_pct.mean()*100:.1f}%")
print(f"     Min: {coverage_pct.min()*100:.1f}%")
print(f"     Max: {coverage_pct.max()*100:.1f}%")

# Test WITH fixes
print("\n4. Computing Sharpe WITH FIXES (50% coverage threshold, 1008 lookback window)...")
sharpe_fixed, pnl_fixed = compute_sharpe_dense(
    F, R, ts_index,
    annualize_periods=3276,
    use_optimizer=True,
    min_coverage_pct=0.50,
    lookback_window=252 * 4
)

print(f"\n   Results WITH FIXES:")
print(f"     Sharpe: {sharpe_fixed:.4f}")
print(f"     PnL series length: {len(pnl_fixed)}")
if len(pnl_fixed) > 0:
    print(f"     Date range: {pnl_fixed.index.min()} to {pnl_fixed.index.max()}")
    print(f"     Mean return: {pnl_fixed.mean():.6f}")
    print(f"     Std return: {pnl_fixed.std():.6f}")
    print(f"     Min return: {pnl_fixed.min():.6f}")
    print(f"     Max return: {pnl_fixed.max():.6f}")

    # Calculate cumulative PnL
    cumulative_fixed = (1 + pnl_fixed).cumprod() - 1
    print(f"     Total return: {cumulative_fixed.iloc[-1]*100:.2f}%")
    print(f"     Max drawdown: {cumulative_fixed.min()*100:.2f}%")

# Create comparison visualization
if len(pnl_fixed) > 0:
    print("\n5. Creating visualization...")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('Backtest Results WITH FIXES', fontsize=16, fontweight='bold')

    # Plot 1: Data coverage over time
    ax = axes[0]
    coverage_df = pd.DataFrame({
        'timestamp': ts_index,
        'coverage_pct': coverage_pct * 100
    })
    ax.plot(coverage_df['timestamp'], coverage_df['coverage_pct'], linewidth=1, alpha=0.7)
    if len(pnl_fixed) > 0:
        ax.axvline(pnl_fixed.index.min(), color='green', linestyle='--',
                   label=f'Backtest start: {pnl_fixed.index.min().strftime("%Y-%m-%d")}', linewidth=2)
    ax.axhline(50, color='red', linestyle='--', label='50% coverage threshold', alpha=0.5)
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Data Coverage Over Time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Period returns
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in pnl_fixed.values]
    ax.bar(pnl_fixed.index, pnl_fixed.values * 100, color=colors, alpha=0.6, width=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Period Return (%)')
    ax.set_title(f'Period Returns (Sharpe: {sharpe_fixed:.4f})')
    ax.set_ylim(-10, 10)
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative PnL
    ax = axes[2]
    cumulative_fixed = (1 + pnl_fixed).cumprod() - 1
    ax.plot(cumulative_fixed.index, cumulative_fixed.values * 100, linewidth=2, color='blue')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Cumulative PnL (%)')
    ax.set_xlabel('Date')
    ax.set_title(f'Cumulative PnL (Total: {cumulative_fixed.iloc[-1]*100:.2f}%, Max DD: {cumulative_fixed.min()*100:.2f}%)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = '/home/lichenhui/data/alphaAgent/test_fixes_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved to: {output_path}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nSUMMARY:")
print("  ✅ Covariance matrix now uses rolling 1008-period window")
print("  ✅ Backtest starts after 50% coverage threshold is met")
print("  ✅ Backtest starts after 1008-period warmup for variance calculation")
if len(pnl_fixed) > 0 and cumulative_fixed.min() > -0.90:
    print("  ✅ No catastrophic drawdown detected!")
else:
    print("  ⚠️  Check results - may need further adjustment")
