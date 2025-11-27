#!/usr/bin/env python3
"""
Diagnose why multiple factors have identical Sharpe ratios and correlations
"""
import pandas as pd
import numpy as np
import pickle
import os

# Check if we have cached factor results
if not os.path.exists('factor_matrix.pkl'):
    print("factor_matrix.pkl not found. Cannot diagnose.")
    print("The script needs the factor_matrix computed by alpha_agent_factor.py")
    exit(1)

# Load the factor matrix
print("Loading factor_matrix...")
with open('factor_matrix.pkl', 'rb') as f:
    factor_matrix = pickle.load(f)

print(f"Factor matrix shape: {factor_matrix.shape}")
print(f"Factor matrix columns (first 10): {factor_matrix.columns[:10].tolist()}")

# Get all unique factor names
all_factors = factor_matrix.columns.get_level_values('factor').unique()
print(f"\nTotal unique factor names: {len(all_factors)}")

# Check factors with identical Sharpe (0.389772) and correlation (-0.001153)
suspect_factors = [
    'high_volume_high_range_breakout',
    'high_volume_low_range_breakout',
    'high_volume_high_range_reversal',
    'high_volume_tight_range_breakout',
    'high_volume_breakout_with_wide_range',
    'high_volume_low_range_reversal',
    'high_volume_breakout_with_narrow_range',
    'high_volume_breakout_with_recent_average_tight_range',
]

print(f"\n{'='*80}")
print("ANALYZING SUSPECT FACTORS WITH IDENTICAL METRICS")
print(f"{'='*80}")

# For each suspect factor, get values across all tickers
for factor_name in suspect_factors:
    if factor_name not in all_factors:
        print(f"\n{factor_name}: NOT FOUND IN FACTOR MATRIX")
        continue

    # Extract all values for this factor across all tickers
    factor_data = factor_matrix.xs(factor_name, level='factor', axis=1)

    # Flatten to 1D array
    values = factor_data.values.flatten()

    # Remove NaN values
    values = values[np.isfinite(values)]

    print(f"\n{factor_name}:")
    print(f"  Total values: {len(values)}")
    print(f"  Unique values: {len(np.unique(values))}")
    print(f"  Min: {values.min():.6f}, Max: {values.max():.6f}")
    print(f"  Mean: {values.mean():.6f}, Std: {values.std():.6f}")
    print(f"  First 10 values: {values[:10]}")

# Now compare pairs to see if they're identical
print(f"\n{'='*80}")
print("PAIRWISE COMPARISONS")
print(f"{'='*80}")

for i, factor1 in enumerate(suspect_factors[:5]):  # Just check first 5 pairs
    if factor1 not in all_factors:
        continue
    for factor2 in suspect_factors[i+1:i+2]:  # Compare with next factor
        if factor2 not in all_factors:
            continue

        data1 = factor_matrix.xs(factor1, level='factor', axis=1).values.flatten()
        data2 = factor_matrix.xs(factor2, level='factor', axis=1).values.flatten()

        # Check if they're identical
        mask = np.isfinite(data1) & np.isfinite(data2)
        data1_clean = data1[mask]
        data2_clean = data2[mask]

        if len(data1_clean) > 0:
            correlation = np.corrcoef(data1_clean, data2_clean)[0, 1]
            max_diff = np.abs(data1_clean - data2_clean).max()

            print(f"\n{factor1} vs {factor2}:")
            print(f"  Correlation: {correlation:.10f}")
            print(f"  Max absolute difference: {max_diff:.10f}")
            print(f"  Are identical? {np.allclose(data1_clean, data2_clean)}")
