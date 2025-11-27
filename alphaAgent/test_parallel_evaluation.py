#!/usr/bin/env python3
"""
Test parallelized factor evaluation functions
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/lichenhui/data/alphaAgent')

from factor_evaluation import (
    pnl_rowwise_optimizer,
    compute_ic_series_dense,
    N_WORKERS
)


def test_pnl_rowwise_optimizer():
    """Test parallelized PnL computation"""

    print("="*70)
    print("TEST: pnl_rowwise_optimizer (Parallelized)")
    print("="*70)

    # Create test data
    T = 5000  # 5000 timestamps
    N = 100   # 100 assets

    print(f"\nGenerating test data: T={T} timestamps, N={N} assets")

    np.random.seed(42)
    F = np.random.randn(T, N).astype(np.float32)
    R = np.random.randn(T, N).astype(np.float32) * 0.01  # Small returns

    # Add some NaN values
    F[np.random.rand(T, N) < 0.1] = np.nan
    R[np.random.rand(T, N) < 0.1] = np.nan

    print(f"Data generated with {np.isnan(F).sum()} NaN values in F, {np.isnan(R).sum()} in R")

    # Test with parallel processing
    print(f"\nRunning with {N_WORKERS} workers...")
    start_time = time.time()
    pnl_parallel = pnl_rowwise_optimizer(F, R, n_workers=N_WORKERS)
    parallel_time = time.time() - start_time

    print(f"✅ Parallel computation completed in {parallel_time:.3f} seconds")
    print(f"   Result shape: {pnl_parallel.shape}")
    print(f"   Valid PnL values: {np.isfinite(pnl_parallel).sum()} / {T}")
    print(f"   Mean PnL: {np.nanmean(pnl_parallel):.6f}")
    print(f"   Std PnL: {np.nanstd(pnl_parallel):.6f}")

    # Test with sequential (1 worker) for comparison
    print(f"\nRunning sequentially (1 worker) for comparison...")
    start_time = time.time()
    pnl_sequential = pnl_rowwise_optimizer(F, R, n_workers=1)
    sequential_time = time.time() - start_time

    print(f"✅ Sequential computation completed in {sequential_time:.3f} seconds")
    print(f"   Speedup: {sequential_time / parallel_time:.2f}x")

    # Verify results are similar (not identical due to floating point order)
    valid_mask = np.isfinite(pnl_parallel) & np.isfinite(pnl_sequential)
    if valid_mask.sum() > 0:
        correlation = np.corrcoef(pnl_parallel[valid_mask], pnl_sequential[valid_mask])[0, 1]
        print(f"   Correlation between parallel and sequential: {correlation:.6f}")
        if correlation > 0.99:
            print("   ✅ Results are highly correlated!")
        else:
            print("   ⚠️  Results differ more than expected")

    return pnl_parallel


def test_compute_ic_series_dense():
    """Test parallelized IC computation"""

    print("\n" + "="*70)
    print("TEST: compute_ic_series_dense (Parallelized)")
    print("="*70)

    # Create test data
    T = 5000  # 5000 timestamps
    N = 100   # 100 assets

    print(f"\nGenerating test data: T={T} timestamps, N={N} assets")

    np.random.seed(42)
    F = np.random.randn(T, N).astype(np.float32)
    R = np.random.randn(T, N).astype(np.float32) * 0.01

    # Add some correlation
    R = R + 0.3 * F  # Add 30% signal

    # Add some NaN values
    F[np.random.rand(T, N) < 0.1] = np.nan
    R[np.random.rand(T, N) < 0.1] = np.nan

    ts_index = pd.date_range('2020-01-01', periods=T, freq='30T')

    print(f"Data generated with added correlation")

    # Test with parallel processing
    print(f"\nRunning with parallel processing ({N_WORKERS} workers)...")
    start_time = time.time()
    ic_parallel = compute_ic_series_dense(F, R, ts_index, n_workers=N_WORKERS, use_parallel=True)
    parallel_time = time.time() - start_time

    print(f"✅ Parallel computation completed in {parallel_time:.3f} seconds")
    print(f"   Result shape: {ic_parallel.shape}")
    print(f"   Valid IC values: {ic_parallel.notna().sum()} / {T}")
    print(f"   Mean IC: {ic_parallel.mean():.6f}")
    print(f"   Std IC: {ic_parallel.std():.6f}")

    # Test without parallel processing (vectorized)
    print(f"\nRunning without parallelization (vectorized)...")
    start_time = time.time()
    ic_vectorized = compute_ic_series_dense(F, R, ts_index, use_parallel=False)
    vectorized_time = time.time() - start_time

    print(f"✅ Vectorized computation completed in {vectorized_time:.3f} seconds")

    # Compare results
    diff = (ic_parallel - ic_vectorized).abs()
    print(f"\nComparison:")
    print(f"   Max absolute difference: {diff.max():.10f}")
    print(f"   Mean absolute difference: {diff.mean():.10f}")

    if diff.max() < 1e-6:
        print("   ✅ Results are virtually identical!")
    elif diff.max() < 1e-3:
        print("   ✅ Results are very close!")
    else:
        print("   ⚠️  Results differ more than expected")

    return ic_parallel


def main():
    """Run all tests"""

    print("\n" + "="*70)
    print("PARALLELIZED FACTOR EVALUATION TESTS")
    print(f"Using {N_WORKERS} workers for parallelization")
    print("="*70)

    # Test 1: PnL computation
    pnl_result = test_pnl_rowwise_optimizer()

    # Test 2: IC computation
    ic_result = test_compute_ic_series_dense()

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
