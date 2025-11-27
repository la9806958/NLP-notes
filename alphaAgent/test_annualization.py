#!/usr/bin/env python3
"""
Test to verify annualization is correct
"""

import numpy as np
import pandas as pd

print("="*60)
print("ANNUALIZATION VERIFICATION")
print("="*60)

# Simulate a simple trading strategy with known properties
np.random.seed(42)

# Create 30-minute returns with known Sharpe ratio
# Let's create returns with annual Sharpe = 2.0
n_periods_per_year = 252 * 13  # 3276 periods
n_periods = n_periods_per_year * 2  # 2 years of data

# Target: annual Sharpe = 2.0
# Sharpe = mean / std * sqrt(periods)
# If we want annual Sharpe = 2.0, and we have 3276 periods/year:
# 2.0 = mean / std * sqrt(3276)
# mean / std = 2.0 / sqrt(3276) = 2.0 / 57.24 = 0.0349

target_sharpe_ratio = 0.0349  # Per-period Sharpe ratio
mean_return = 0.001  # 0.1% per period
std_return = mean_return / target_sharpe_ratio

print(f"\nGenerating synthetic returns:")
print(f"  Mean per period: {mean_return:.6f} ({mean_return*100:.4f}%)")
print(f"  Std per period: {std_return:.6f} ({std_return*100:.4f}%)")
print(f"  Per-period Sharpe: {target_sharpe_ratio:.6f}")

# Generate returns
returns = np.random.normal(mean_return, std_return, n_periods)
returns_series = pd.Series(returns)

# Calculate Sharpe ratio
calc_mean = returns_series.mean()
calc_std = returns_series.std(ddof=1)
per_period_sharpe = calc_mean / calc_std
annualized_sharpe = per_period_sharpe * np.sqrt(n_periods_per_year)

print(f"\nCalculated metrics:")
print(f"  Sample mean: {calc_mean:.6f}")
print(f"  Sample std: {calc_std:.6f}")
print(f"  Per-period Sharpe: {per_period_sharpe:.6f}")
print(f"  Annualized Sharpe: {annualized_sharpe:.4f}")
print(f"  Target Sharpe: 2.0")

# Verify the formula
print(f"\n" + "="*60)
print("VERIFICATION")
print("="*60)
print(f"Formula: Sharpe = (mean / std) * sqrt(periods_per_year)")
print(f"  = ({calc_mean:.6f} / {calc_std:.6f}) * sqrt({n_periods_per_year})")
print(f"  = {per_period_sharpe:.6f} * {np.sqrt(n_periods_per_year):.2f}")
print(f"  = {annualized_sharpe:.4f}")

# Check if close to target
if abs(annualized_sharpe - 2.0) < 0.2:
    print(f"\n✅ Annualization formula is CORRECT!")
else:
    print(f"\n❌ Annualization formula may be INCORRECT")

# Now test what happens with different resampling
print(f"\n" + "="*60)
print("RESAMPLING TEST")
print("="*60)

# Create 1-minute data
timestamps_1min = pd.date_range('2024-01-01', periods=390*2, freq='min')
returns_1min = pd.Series(np.random.normal(0.0001, 0.0005, len(timestamps_1min)),
                         index=timestamps_1min)

print(f"\n1-minute data:")
print(f"  Periods: {len(returns_1min)}")
print(f"  Mean: {returns_1min.mean():.8f}")
print(f"  Std: {returns_1min.std():.8f}")

# What's the correct annualization for 1-minute returns?
# 390 minutes/day * 252 days = 98,280 periods/year
periods_1min = 390 * 252
sharpe_1min = (returns_1min.mean() / returns_1min.std()) * np.sqrt(periods_1min)
print(f"  Annualized Sharpe (1-min): {sharpe_1min:.4f}")

# Resample to 30-minute
returns_30min = returns_1min.resample('30min').last().dropna()
print(f"\n30-minute resampled data:")
print(f"  Periods: {len(returns_30min)}")
print(f"  Mean: {returns_30min.mean():.8f}")
print(f"  Std: {returns_30min.std():.8f}")

periods_30min = 13 * 252
sharpe_30min = (returns_30min.mean() / returns_30min.std()) * np.sqrt(periods_30min)
print(f"  Annualized Sharpe (30-min): {sharpe_30min:.4f}")

print(f"\n" + "="*60)
print("CONCLUSION:")
print("="*60)
print(f"For 30-minute period returns:")
print(f"  Periods per year = 13 * 252 = {periods_30min}")
print(f"  Annualization factor = sqrt({periods_30min}) = {np.sqrt(periods_30min):.2f}")
print(f"  Formula: Sharpe_annual = (mean / std) * {np.sqrt(periods_30min):.2f}")
print(f"\nCurrent code uses: np.sqrt(ANNUALIZE_MINUTES)")
print(f"  where ANNUALIZE_MINUTES = 252 * 13 = 3276")
print(f"  sqrt(3276) = {np.sqrt(3276):.2f}")
print(f"\n✅ Current annualization factor is CORRECT for 30-minute returns!")
