#!/usr/bin/env python3
"""
Parameter Sweep Script
Sweeps K_factors and NUM_FACTORS to measure formulation and solving times.
"""

import pandas as pd
import numpy as np
from singlePeriodOptimizer import SinglePeriodOptimizer
import logging
import time
from utils import ewma_covariance_with_gc, cs_rank_gaussianize, compute_forward_ewma_returns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define parameter ranges
K_FACTORS_RANGE = [10, 15, 20, 25, 30, 35, 40, 45, 50]
NUM_FACTORS_RANGE = [20, 30, 40, 50, 60, 70, 80, 90, 100]

# Data loading parameters
CUTOFF = pd.Timestamp("2022-01-01")
ret_csv = "/data/csv/hourly_close_to_close_returns_matrix.csv"
RANDOM_SEED = 42

def load_sample_data():
    """Load a small sample of data for testing"""
    print("Loading sample data...")

    # Load returns
    returns = (
        pd.read_csv(ret_csv, index_col=0, parse_dates=True)
        .loc[CUTOFF:]
        .fillna(0.0)
        .clip(lower=-1.0, upper=1.0)
    )

    # Select random subset of tickers
    np.random.seed(RANDOM_SEED)
    available_tickers = returns.columns.tolist()
    n_tickers_to_select = min(3000, len(available_tickers))
    selected_tickers = np.random.choice(available_tickers, size=n_tickers_to_select, replace=False)
    returns = returns[selected_tickers]

    print(f"Selected {n_tickers_to_select} tickers")

    # Compute alphas (simplified version)
    signal = compute_forward_ewma_returns(returns, start_lag=1, horizon_hours=40, halflife_hours=20.0)
    signal_norm = cs_rank_gaussianize(signal).fillna(0.0)

    # Rescale signal
    returns_std = returns.std().mean()
    signal_norm = signal_norm * returns_std

    # Take a single date slice for testing
    test_date = signal_norm.index[500]  # Use a date in the middle

    # Get alpha for test date
    alpha_today = signal_norm.loc[test_date].dropna()

    # Compute covariance at test date
    returns_past = returns.shift(1)
    HL = 128.0
    PERIODS_PER_YEAR = 8760  # approximate
    EPSILON = 1e-10

    print("Computing covariance matrix...")
    cov_matrices = ewma_covariance_with_gc(
        returns_df=returns_past,
        hl_hours=HL,
        periods_per_year=PERIODS_PER_YEAR,
        epsilon=EPSILON,
        clip_abs=None
    )

    if test_date not in cov_matrices:
        # Find the nearest date with a covariance matrix
        available_dates = list(cov_matrices.keys())
        test_date = available_dates[len(available_dates) // 2]
        alpha_today = signal_norm.loc[test_date].dropna()

    C_full = cov_matrices[test_date]

    # Get common tickers
    common_tickers = alpha_today.index.intersection(returns.columns)
    alpha_today = alpha_today.loc[common_tickers]

    # Extract covariance submatrix
    all_tickers = returns.columns
    common_indices = [all_tickers.get_loc(ticker) for ticker in common_tickers]
    C = C_full[np.ix_(common_indices, common_indices)]

    print(f"Data prepared: N={len(common_tickers)} assets")
    print(f"Alpha shape: {alpha_today.shape}")
    print(f"Covariance shape: {C.shape}")

    return alpha_today.values, C, len(common_tickers)


def run_parameter_sweep():
    """Run the parameter sweep"""
    print("="*80)
    print("PARAMETER SWEEP: K_factors vs NUM_FACTORS")
    print("="*80)

    # Load data once
    alpha, C, N = load_sample_data()

    # Initialize results storage
    results = []

    # Parameter sweep
    total_runs = len(K_FACTORS_RANGE) * len(NUM_FACTORS_RANGE)
    run_count = 0

    print(f"\nStarting sweep: {len(K_FACTORS_RANGE)} K_factors × {len(NUM_FACTORS_RANGE)} NUM_FACTORS = {total_runs} runs")
    print("-"*80)

    for k_factors in K_FACTORS_RANGE:
        for num_factors in NUM_FACTORS_RANGE:
            run_count += 1
            print(f"\nRun {run_count}/{total_runs}: K_factors={k_factors}, NUM_FACTORS={num_factors}")

            try:
                # Create optimizer with specific K_factors
                sp_optimizer = SinglePeriodOptimizer()
                sp_optimizer.K_factors = k_factors

                # Create factor exposure matrix B for neutrality constraints
                np.random.seed(RANDOM_SEED)
                B = np.random.randn(num_factors, N)

                # Create starting portfolio (zeros)
                x0 = np.zeros(N)

                # Run optimization
                start_time = time.perf_counter()
                w = sp_optimizer.solve_long_short_portfolio_with_risk_penalty(
                    alpha=alpha,
                    C=C,
                    x0=x0,
                    B=B
                )
                total_time = time.perf_counter() - start_time

                # Extract timing information
                formulation_time = sp_optimizer.last_formulation_time
                solve_time = sp_optimizer.last_solve_time

                # Store results
                results.append({
                    'K_factors': k_factors,
                    'NUM_FACTORS': num_factors,
                    'N': N,
                    'formulation_time_s': formulation_time,
                    'solve_time_s': solve_time,
                    'total_time_s': total_time,
                    'success': True
                })

                print(f"  ✓ Formulation: {formulation_time:.4f}s | Solve: {solve_time:.4f}s | Total: {total_time:.4f}s")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'K_factors': k_factors,
                    'NUM_FACTORS': num_factors,
                    'N': N,
                    'formulation_time_s': None,
                    'solve_time_s': None,
                    'total_time_s': None,
                    'success': False
                })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_filename = "parameter_sweep_results.csv"
    results_df.to_csv(csv_filename, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to {csv_filename}")
    print(f"{'='*80}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df[['K_factors', 'NUM_FACTORS', 'formulation_time_s', 'solve_time_s']].describe())

    # Create pivot tables for easier analysis
    formulation_pivot = results_df.pivot(index='K_factors', columns='NUM_FACTORS', values='formulation_time_s')
    solve_pivot = results_df.pivot(index='K_factors', columns='NUM_FACTORS', values='solve_time_s')

    print("\n" + "="*80)
    print("FORMULATION TIME (seconds)")
    print("="*80)
    print(formulation_pivot.to_string())

    print("\n" + "="*80)
    print("SOLVE TIME (seconds)")
    print("="*80)
    print(solve_pivot.to_string())

    # Save pivot tables
    formulation_pivot.to_csv("formulation_time_grid.csv")
    solve_pivot.to_csv("solve_time_grid.csv")
    print(f"\nPivot tables saved to formulation_time_grid.csv and solve_time_grid.csv")

    return results_df


if __name__ == "__main__":
    try:
        results_df = run_parameter_sweep()
        print("\n" + "="*80)
        print("PARAMETER SWEEP COMPLETE")
        print("="*80)
    except Exception as e:
        print(f"\nERROR during parameter sweep: {e}")
        raise
