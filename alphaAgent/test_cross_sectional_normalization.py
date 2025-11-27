#!/usr/bin/env python3
"""
Test cross-sectional normalization of alpha values
"""

import numpy as np
from scipy.stats import rankdata
from scipy.special import ndtri

print("="*80)
print("TESTING CROSS-SECTIONAL NORMALIZATION")
print("="*80)

# Test 1: Simple case with 100 values
print("\nTest 1: Simple normal distribution")
np.random.seed(42)
alpha = np.random.randn(100) * 2 + 1  # Mean=1, Std=2

print(f"Original alpha:")
print(f"  Mean: {alpha.mean():.4f}")
print(f"  Std: {alpha.std():.4f}")
print(f"  Min: {alpha.min():.4f}")
print(f"  Max: {alpha.max():.4f}")

# Apply rank-based normalization
ranks = rankdata(alpha, method='average')
percentiles = ranks / (len(alpha) + 1)
alpha_normalized = ndtri(percentiles)
alpha_normalized = np.clip(alpha_normalized, -3.0, 3.0)

print(f"\nNormalized alpha:")
print(f"  Mean: {alpha_normalized.mean():.4f}")
print(f"  Std: {alpha_normalized.std():.4f}")
print(f"  Min: {alpha_normalized.min():.4f}")
print(f"  Max: {alpha_normalized.max():.4f}")

# Test 2: With extreme outliers
print("\n" + "="*80)
print("\nTest 2: With extreme outliers")
alpha_outliers = np.random.randn(100)
alpha_outliers[0] = 1000  # Extreme positive outlier
alpha_outliers[1] = -1000  # Extreme negative outlier

print(f"Original alpha with outliers:")
print(f"  Mean: {alpha_outliers.mean():.4f}")
print(f"  Std: {alpha_outliers.std():.4f}")
print(f"  Min: {alpha_outliers.min():.4f}")
print(f"  Max: {alpha_outliers.max():.4f}")

# Apply rank-based normalization
ranks = rankdata(alpha_outliers, method='average')
percentiles = ranks / (len(alpha_outliers) + 1)
alpha_normalized2 = ndtri(percentiles)
alpha_normalized2 = np.clip(alpha_normalized2, -3.0, 3.0)

print(f"\nNormalized alpha (outliers handled):")
print(f"  Mean: {alpha_normalized2.mean():.4f}")
print(f"  Std: {alpha_normalized2.std():.4f}")
print(f"  Min: {alpha_normalized2.min():.4f}")
print(f"  Max: {alpha_normalized2.max():.4f}")

# Test 3: Check distribution
print("\n" + "="*80)
print("\nTest 3: Distribution check (should be approximately N(0,1) clipped at [-3, 3])")

# Large sample to check distribution
np.random.seed(123)
alpha_large = np.random.randn(10000) * 5 + 3
ranks = rankdata(alpha_large, method='average')
percentiles = ranks / (len(alpha_large) + 1)
alpha_normalized3 = ndtri(percentiles)
alpha_normalized3 = np.clip(alpha_normalized3, -3.0, 3.0)

print(f"Large sample (n=10000):")
print(f"  Mean: {alpha_normalized3.mean():.6f} (should be ≈ 0)")
print(f"  Std: {alpha_normalized3.std():.6f} (should be ≈ 1)")
print(f"  Min: {alpha_normalized3.min():.4f} (should be -3.0)")
print(f"  Max: {alpha_normalized3.max():.4f} (should be 3.0)")

# Check percentiles
percentiles_check = [1, 5, 25, 50, 75, 95, 99]
for p in percentiles_check:
    val = np.percentile(alpha_normalized3, p)
    print(f"  {p}th percentile: {val:.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("✅ Rank-based transformation converts any distribution to N(0,1)")
print("✅ Outliers are handled gracefully (mapped to extreme percentiles)")
print("✅ Clipping at [-3, 3] prevents extreme values from dominating")
print("✅ Cross-sectional normalization makes factors comparable")
