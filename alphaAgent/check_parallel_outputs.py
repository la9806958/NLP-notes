#!/usr/bin/env python3
"""
Detailed output checking for parallelized functions
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from factor_evaluation import (
    pnl_rowwise_optimizer,
    compute_ic_series_dense,
    rowwise_pearson,
    N_WORKERS
)


def check_pnl_rowwise_optimizer():
    """Detailed check of pnl_rowwise_optimizer output"""

    print("="*80)
    print("DETAILED CHECK: pnl_rowwise_optimizer")
    print("="*80)

    # Create realistic test data
    T = 100  # 100 timestamps (smaller for detailed inspection)
    N = 50   # 50 assets

    np.random.seed(123)

    # Create factor values with some signal
    F = np.random.randn(T, N).astype(np.float32)

    # Create returns that are partially correlated with factors
    R = (0.0005 * F + 0.001 * np.random.randn(T, N)).astype(np.float32)

    # Add some NaN values (10%)
    nan_mask_f = np.random.rand(T, N) < 0.1
    nan_mask_r = np.random.rand(T, N) < 0.1
    F[nan_mask_f] = np.nan
    R[nan_mask_r] = np.nan

    print(f"\nTest Data:")
    print(f"  Shape: T={T} timestamps, N={N} assets")
    print(f"  NaN in F: {np.isnan(F).sum()} ({np.isnan(F).mean()*100:.1f}%)")
    print(f"  NaN in R: {np.isnan(R).sum()} ({np.isnan(R).mean()*100:.1f}%)")
    print(f"  Factor range: [{np.nanmin(F):.4f}, {np.nanmax(F):.4f}]")
    print(f"  Return range: [{np.nanmin(R):.6f}, {np.nanmax(R):.6f}]")

    # Run parallel version
    print(f"\n{'='*80}")
    print(f"PARALLEL VERSION (n_workers={N_WORKERS})")
    print(f"{'='*80}")

    pnl_parallel = pnl_rowwise_optimizer(
        F, R,
        risk_budget=0.1,
        gme_limit=2.0,
        min_assets=10,
        n_workers=N_WORKERS
    )

    print(f"\nOutput Statistics:")
    print(f"  Shape: {pnl_parallel.shape}")
    print(f"  Dtype: {pnl_parallel.dtype}")
    print(f"  Valid values: {np.isfinite(pnl_parallel).sum()} / {T} ({np.isfinite(pnl_parallel).mean()*100:.1f}%)")
    print(f"  NaN values: {np.isnan(pnl_parallel).sum()}")

    if np.isfinite(pnl_parallel).sum() > 0:
        print(f"\nPnL Distribution (valid values only):")
        valid_pnl = pnl_parallel[np.isfinite(pnl_parallel)]
        print(f"  Mean: {valid_pnl.mean():.8f}")
        print(f"  Std: {valid_pnl.std():.8f}")
        print(f"  Min: {valid_pnl.min():.8f}")
        print(f"  Max: {valid_pnl.max():.8f}")
        print(f"  Median: {np.median(valid_pnl):.8f}")

        # Show percentiles
        print(f"\nPercentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  {p:2d}th: {np.percentile(valid_pnl, p):.8f}")

        # Show first 10 values
        print(f"\nFirst 10 PnL values:")
        for i in range(min(10, len(pnl_parallel))):
            status = "✓" if np.isfinite(pnl_parallel[i]) else "✗ (NaN)"
            val_str = f"{pnl_parallel[i]:.8f}" if np.isfinite(pnl_parallel[i]) else "NaN"
            print(f"  t={i:3d}: {val_str:>15s}  {status}")
    else:
        print("\n⚠️  WARNING: No valid PnL values generated!")
        print("\nChecking why:")
        for t in range(min(5, T)):
            valid_assets = np.isfinite(F[t, :]) & np.isfinite(R[t, :])
            print(f"  t={t}: {valid_assets.sum()} valid assets (need ≥10)")

    # Run sequential version for comparison
    print(f"\n{'='*80}")
    print(f"SEQUENTIAL VERSION (n_workers=1)")
    print(f"{'='*80}")

    pnl_sequential = pnl_rowwise_optimizer(
        F, R,
        risk_budget=0.1,
        gme_limit=2.0,
        min_assets=10,
        n_workers=1
    )

    print(f"\nOutput Statistics:")
    print(f"  Shape: {pnl_sequential.shape}")
    print(f"  Valid values: {np.isfinite(pnl_sequential).sum()} / {T}")

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON: Parallel vs Sequential")
    print(f"{'='*80}")

    both_valid = np.isfinite(pnl_parallel) & np.isfinite(pnl_sequential)

    if both_valid.sum() > 0:
        diff = pnl_parallel[both_valid] - pnl_sequential[both_valid]

        print(f"\nDifferences (where both valid):")
        print(f"  Compared values: {both_valid.sum()}")
        print(f"  Max absolute diff: {np.abs(diff).max():.12f}")
        print(f"  Mean absolute diff: {np.abs(diff).mean():.12f}")
        print(f"  RMS diff: {np.sqrt((diff**2).mean()):.12f}")

        if both_valid.sum() > 1:
            corr = np.corrcoef(pnl_parallel[both_valid], pnl_sequential[both_valid])[0, 1]
            print(f"  Correlation: {corr:.10f}")

        # Show some examples
        print(f"\nSample comparisons (first 10 valid):")
        count = 0
        for i in range(T):
            if both_valid[i] and count < 10:
                diff_val = pnl_parallel[i] - pnl_sequential[i]
                print(f"  t={i:3d}: Parallel={pnl_parallel[i]:.10f}, Sequential={pnl_sequential[i]:.10f}, Diff={diff_val:.2e}")
                count += 1
    else:
        print("\n⚠️  No overlapping valid values to compare!")

    return pnl_parallel, pnl_sequential


def check_compute_ic_series_dense():
    """Detailed check of compute_ic_series_dense output"""

    print("\n\n" + "="*80)
    print("DETAILED CHECK: compute_ic_series_dense")
    print("="*80)

    # Create test data
    T = 200  # 200 timestamps (smaller for detailed inspection)
    N = 50   # 50 assets

    np.random.seed(456)

    # Create factor values
    F = np.random.randn(T, N).astype(np.float32)

    # Create returns with strong correlation to factor
    R = (0.3 * F + 0.7 * np.random.randn(T, N)).astype(np.float32) * 0.01

    # Add some NaN values
    nan_mask_f = np.random.rand(T, N) < 0.05
    nan_mask_r = np.random.rand(T, N) < 0.05
    F[nan_mask_f] = np.nan
    R[nan_mask_r] = np.nan

    ts_index = pd.date_range('2020-01-01', periods=T, freq='30min')

    print(f"\nTest Data:")
    print(f"  Shape: T={T} timestamps, N={N} assets")
    print(f"  NaN in F: {np.isnan(F).sum()} ({np.isnan(F).mean()*100:.1f}%)")
    print(f"  NaN in R: {np.isnan(R).sum()} ({np.isnan(R).mean()*100:.1f}%)")
    print(f"  Expected correlation: ~0.3 (30% signal)")

    # Test vectorized version (baseline)
    print(f"\n{'='*80}")
    print(f"VECTORIZED VERSION (use_parallel=False)")
    print(f"{'='*80}")

    ic_vectorized = compute_ic_series_dense(F, R, ts_index, use_parallel=False)

    print(f"\nOutput Statistics:")
    print(f"  Shape: {ic_vectorized.shape}")
    print(f"  Dtype: {ic_vectorized.dtype}")
    print(f"  Valid values: {ic_vectorized.notna().sum()} / {T} ({ic_vectorized.notna().mean()*100:.1f}%)")
    print(f"  NaN values: {ic_vectorized.isna().sum()}")

    if ic_vectorized.notna().sum() > 0:
        print(f"\nIC Distribution:")
        valid_ic = ic_vectorized.dropna()
        print(f"  Mean: {valid_ic.mean():.6f}")
        print(f"  Std: {valid_ic.std():.6f}")
        print(f"  Min: {valid_ic.min():.6f}")
        print(f"  Max: {valid_ic.max():.6f}")
        print(f"  Median: {valid_ic.median():.6f}")

        # Show percentiles
        print(f"\nPercentiles:")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  {p:2d}th: {valid_ic.quantile(p/100):.6f}")

        # Show first 10 values
        print(f"\nFirst 10 IC values:")
        for i in range(min(10, len(ic_vectorized))):
            val = ic_vectorized.iloc[i]
            status = "✓" if pd.notna(val) else "✗ (NaN)"
            val_str = f"{val:.8f}" if pd.notna(val) else "NaN"
            print(f"  t={i:3d} ({ts_index[i].strftime('%Y-%m-%d %H:%M')}): {val_str:>12s}  {status}")

    # Test parallel version (note: may use vectorized for T < 1000)
    print(f"\n{'='*80}")
    print(f"PARALLEL VERSION (use_parallel=True, n_workers={N_WORKERS})")
    print(f"{'='*80}")

    ic_parallel = compute_ic_series_dense(F, R, ts_index, n_workers=N_WORKERS, use_parallel=True)

    print(f"\nOutput Statistics:")
    print(f"  Shape: {ic_parallel.shape}")
    print(f"  Valid values: {ic_parallel.notna().sum()} / {T}")

    # Note about threshold
    if T < 1000:
        print(f"\n  ℹ️  Note: T={T} < 1000, so function used vectorized computation")
        print(f"      (Parallelization only kicks in for T ≥ 1000)")

    # Compare
    print(f"\n{'='*80}")
    print(f"COMPARISON: Parallel vs Vectorized")
    print(f"{'='*80}")

    both_valid = ic_parallel.notna() & ic_vectorized.notna()

    if both_valid.sum() > 0:
        diff = (ic_parallel - ic_vectorized).abs()

        print(f"\nDifferences:")
        print(f"  Compared values: {both_valid.sum()}")
        print(f"  Max absolute diff: {diff.max():.12f}")
        print(f"  Mean absolute diff: {diff[both_valid].mean():.12f}")
        print(f"  RMS diff: {np.sqrt((diff[both_valid]**2).mean()):.12f}")

        if both_valid.sum() > 1:
            corr = ic_parallel[both_valid].corr(ic_vectorized[both_valid])
            print(f"  Correlation: {corr:.10f}")

        # Show some examples
        print(f"\nSample comparisons (first 10 valid):")
        count = 0
        for i in range(T):
            if both_valid.iloc[i] and count < 10:
                diff_val = ic_parallel.iloc[i] - ic_vectorized.iloc[i]
                print(f"  t={i:3d}: Parallel={ic_parallel.iloc[i]:.10f}, Vectorized={ic_vectorized.iloc[i]:.10f}, Diff={diff_val:.2e}")
                count += 1
    else:
        print("\n⚠️  No overlapping valid values to compare!")

    # Test with large dataset to trigger parallelization
    print(f"\n{'='*80}")
    print(f"LARGE DATASET TEST (T=2000 to trigger parallelization)")
    print(f"{'='*80}")

    T_large = 2000
    F_large = np.random.randn(T_large, N).astype(np.float32)
    R_large = (0.3 * F_large + 0.7 * np.random.randn(T_large, N)).astype(np.float32) * 0.01
    ts_large = pd.date_range('2020-01-01', periods=T_large, freq='30min')

    print(f"\n  Testing with T={T_large} (should use parallel processing)...")
    ic_large = compute_ic_series_dense(F_large, R_large, ts_large, n_workers=N_WORKERS, use_parallel=True)

    print(f"  ✓ Completed!")
    print(f"    Valid values: {ic_large.notna().sum()} / {T_large}")
    print(f"    Mean IC: {ic_large.mean():.6f}")
    print(f"    Std IC: {ic_large.std():.6f}")

    return ic_vectorized, ic_parallel


def main():
    """Run all checks"""

    print("\n" + "="*80)
    print("PARALLEL FUNCTION OUTPUT VERIFICATION")
    print("="*80)

    # Check 1: PnL optimizer
    pnl_parallel, pnl_sequential = check_pnl_rowwise_optimizer()

    # Check 2: IC computation
    ic_vectorized, ic_parallel = check_compute_ic_series_dense()

    print("\n\n" + "="*80)
    print("ALL CHECKS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
