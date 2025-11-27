#!/usr/bin/env python3
"""
Forward EWMA Returns Backtest Script
Computes alphas as exponential weighted average of forward returns (T to T+40h)
where closer terms receive higher weights.
"""

import pandas as pd
import numpy as np
from singlePeriodOptimizer import SinglePeriodOptimizer
from multiPeriodOptimizer import MultiPeriodOptimizer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import logging

# Import utility functions
from utils import (
    signal_target_dot_series,
    sharpe_from_series,
    sharpe_ci_from_series,
    cs_rank_gaussianize,
    clean_price_jumps,
    calculate_trading_year_length,
    compute_ann_sharpe,
    compute_sharpe_confidence_interval,
    shift_weights,
    compute_forward_ewma_returns,
    ewma_variance,
    ewma_covariance_with_gc
)

# Define constants
CUTOFF = pd.Timestamp("2022-01-01")
CUTOFFMAX = pd.Timestamp("2023-01-03")
ret_csv = "/data/csv/hourly_close_to_close_returns_matrix.csv"

def run_backtest():
    """Run complete backtest using forward EWMA returns as alphas"""

    # Portfolio optimization setup - define early
    # Toggle between single-period and multi-period optimization
    USE_MULTI_PERIOD = False  # Set to True to use multi-period optimizer with multiple lags

    # Multi-period specific: future lags to use for alpha computation
    # These represent the ending point for forward-looking windows starting at T+1
    MP_LAGS = [1, 2, 8, 16]  # Compute alphas from T+1 to T+2, T+1 to T+8, and T+1 to T+16

    # Fixed random seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # Alpha noise mixing parameters
    ALPHA_TRUE_WEIGHT = 0.02  # 10% true alpha
    ALPHA_NOISE_WEIGHT = 0.98  # 90% white noise

    print(f"\n{'='*80}")
    print(f"Computing alphas from forward exponentially weighted returns")
    print(f"{'='*80}")

    # Load returns data
    returns = (
        pd.read_csv(ret_csv, index_col=0, parse_dates=True)
          .loc[CUTOFF:]
          .fillna(0.0)
          .clip(lower=-1.0, upper=1.0)
    )

    # Randomly select 3000 tickers
    np.random.seed(RANDOM_SEED)
    available_tickers = returns.columns.tolist()
    n_tickers_to_select = min(3000, len(available_tickers))
    selected_tickers = np.random.choice(available_tickers, size=n_tickers_to_select, replace=False)
    returns = returns[selected_tickers]
    print(f"Randomly selected {n_tickers_to_select} tickers from {len(available_tickers)} available tickers")


    # Compute alphas as exponential weighted average of returns from T to T+40 hours
    # More recent returns get higher weight (exponential decay looking forward)
    print("Computing alphas as exponential weighted average of forward returns (T to T+40 hours)...")

    # Compute forward EWMA alphas at multiple future lags if using multi-period
    if USE_MULTI_PERIOD:
        print(f"Computing alphas from T+1 to T+lag for lags: {MP_LAGS}")
        print(f"Alpha mixing: {ALPHA_TRUE_WEIGHT*100:.0f}% true alpha + {ALPHA_NOISE_WEIGHT*100:.0f}% white noise (seed={RANDOM_SEED})")
        # Dictionary to store alphas at different lags
        alphas_dict = {}
        for lag in MP_LAGS:
            # Compute from T+1 to T+lag (inclusive)
            horizon = lag  # horizon_hours to include all periods from T+1 to T+lag
            print(f"  Computing alpha from T+1 to T+{lag} (horizon={horizon}h)...")
            halflife = lag
            alpha_true = compute_forward_ewma_returns(returns, start_lag=1, horizon_hours=horizon, halflife_hours=halflife)

            # Add 10% true alpha + 90% white noise
            # Scale white noise to match variance of alpha_true
            white_noise = np.random.randn(*alpha_true.shape) * np.std(alpha_true.values)
            alpha_noisy = ALPHA_TRUE_WEIGHT * alpha_true + ALPHA_NOISE_WEIGHT * white_noise

            alphas_dict[lag] = alpha_noisy
            print(f"    Non-zero values (before noise): {(alpha_true != 0).sum().sum()}")
            print(f"    Mean (true): {alpha_true.mean().mean():.6f}, Std (true): {alpha_true.std().mean():.6f}")
            print(f"    Mean (noisy): {alpha_noisy.mean().mean():.6f}, Std (noisy): {alpha_noisy.std().mean():.6f}")

        # Aggregate alphas across all horizons (equal-weighted average)
        print(f"Aggregating alphas across horizons {MP_LAGS}...")
        signal = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
        for lag in MP_LAGS:
            signal += alphas_dict[lag]
        signal = signal / len(MP_LAGS)
        print(f"  Aggregated alpha computed as equal-weighted average of {len(MP_LAGS)} horizons")
    else:
        # Single-period: compute only at lag 1
        signal = compute_forward_ewma_returns(returns, start_lag=1, horizon_hours=40, halflife_hours=20.0)

    print(f"\nAlpha EWMA stats (primary signal):")
    print(f"  Shape: {signal.shape}")
    print(f"  Non-zero values: {(signal != 0).sum().sum()}")
    print(f"  Mean: {signal.mean().mean():.6f}")
    print(f"  Std: {signal.std().mean():.6f}")

    # Apply cross-sectional rank gaussianization and fill NaNs
    signal_norm = cs_rank_gaussianize(signal)
    signal_norm = signal_norm.fillna(0.0)

    # Rescale signal_norm to match the returns distribution scale (std only, not mean)
    returns_std = returns.std().mean()
    signal_norm = signal_norm * returns_std
    print(f"\nRescaled signal_norm to match returns std:")
    print(f"  Returns std: {returns_std:.6f}")
    print(f"  Signal_norm mean: {signal_norm.mean().mean():.6f}, std: {signal_norm.std().mean():.6f}")

    # If multi-period, normalize all alpha signals
    if USE_MULTI_PERIOD:
        alphas_norm_dict = {}
        for lag in MP_LAGS:
            alpha_norm = cs_rank_gaussianize(alphas_dict[lag])
            alphas_norm_dict[lag] = alpha_norm.fillna(0.0)
        print(f"Normalized alphas for all lags: {list(alphas_norm_dict.keys())}")


    print("Using raw signals without time series normalization...")
    print("Raw signal stats:")
    print(signal_norm.describe().T.loc[:, ['mean', 'std', 'min', 'max']].head())

    # Extend matrices to common date range
    returns = returns.loc[CUTOFF:CUTOFFMAX]
    full_date_range = returns.index
    all_tickers = signal_norm.columns.intersection(returns.columns)

    signal_extended = signal_norm.reindex(index=full_date_range, columns=all_tickers, fill_value=0.0)
    returns_extended = returns.reindex(index=full_date_range, columns=all_tickers, fill_value=0.0)

    signal = signal_extended
    returns = returns_extended

    print(f"Extended signal shape: {signal.shape}")
    print(f"Extended returns shape: {returns.shape}")
    print(f"Date range: {signal.index.min()} to {signal.index.max()}")

    # Using risk penalty optimizer with the following parameters:
    RISK_LAMBDA = 1.0     # Risk penalty multiplier
    CAPITAL = 1.0         # Capital scaling
    GME_LIMIT = 2.0       # Gross market exposure limit
    BOX = 1.0             # Box constraint per asset

    # Define parameter ranges for sweep
    K_FACTORS_RANGE = [10, 15, 20, 25, 30, 35, 40, 45, 50] # number of PCA factors
    NUM_FACTORS_RANGE = [20, 30, 40, 50, 60, 70, 80, 90, 100] # number of factor exposure constraints

    # K_FACTORS_RANGE = [10]
    # NUM_FACTORS_RANGE = [20]

    # Create multi-period optimizer instance if needed
    mp_optimizer = MultiPeriodOptimizer() if USE_MULTI_PERIOD else None

    # Set variant name based on optimizer mode
    if USE_MULTI_PERIOD:
        variant_name = f"forward_ewma_mp_T1_to_T{'-'.join(map(str, MP_LAGS))}_noise{int(ALPHA_NOISE_WEIGHT*100)}_seed{RANDOM_SEED}"
    else:
        variant_name = "forward_ewma_40h_sp"

    print(f"VARIANT: {variant_name}")

    # Create output directory for this variant
    output_dir = f'backtest_plots/forward_ewma_{variant_name}'
    os.makedirs(output_dir, exist_ok=True)

    alpha_cache = signal
    returns_aligned = returns.loc[alpha_cache.index]

    # Calculate trading year length and setup optimization
    trading_year_length = calculate_trading_year_length(ret_csv)
    lookback_cov = trading_year_length
    all_dates = alpha_cache.index.sort_values()

    EPSILON = 1e-10
    returns_aligned_past = returns_aligned.shift(1)

    # Compute EWMA variance for risk model
    HL = 128.0
    PERIODS_PER_YEAR = lookback_cov
    EPSILON = 1e-10
    CLIP_ABS = None  # e.g., 0.05

    var_hourly = ewma_variance(
        returns_df=returns_aligned_past,
        hl_hours=HL,
        periods_per_year=PERIODS_PER_YEAR,
        epsilon=EPSILON,
        clip_abs=CLIP_ABS,
    )

    var_rolling = var_hourly.add(EPSILON)

    # Compute full EWMA covariance matrices
    print("Computing full EWMA covariance matrices...")


    cov_matrices = ewma_covariance_with_gc(
        returns_df=returns_aligned_past,
        hl_hours=HL,
        periods_per_year=PERIODS_PER_YEAR,
        epsilon=EPSILON,
        clip_abs=CLIP_ABS
    )


    print(f"Computed {len(cov_matrices)} full covariance matrices")

    print("Computed rolling full covariance matrices...")

    # Function to run optimization with specific parameters
    def run_single_parameter_combination(k_factors, num_factors):
        """Run optimization for a single parameter combination"""
        print(f"\n{'='*80}")
        print(f"Running optimization with K_FACTORS={k_factors}, NUM_FACTORS={num_factors}")
        print(f"{'='*80}")

        # Create optimizer instance with specific k_factors for this sweep
        sp_optimizer_sweep = SinglePeriodOptimizer()
        sp_optimizer_sweep.K_factors = k_factors  # Set the number of PCA factors

        if USE_MULTI_PERIOD:
            print(f"Using MULTI-PERIOD optimizer with lags: {MP_LAGS}")
        else:
            print("Using SINGLE-PERIOD optimizer")

        # Dictionary to track previous weights for warm starting
        previous_weights = {}

        # Accumulators for timing information
        total_formulation_time = 0.0
        total_solve_time = 0.0
        solve_count = 0

        def process_date(i: int):
            nonlocal total_formulation_time, total_solve_time, solve_count
            d = all_dates[i]

            if USE_MULTI_PERIOD:
                # Multi-period optimization with alphas from different future horizons
                alphas_list = []
                common = None

                # Pull current date variance (same for all alpha horizons)
                variances = var_rolling.loc[d].dropna()
                if variances.empty:
                    return None

                for lag in MP_LAGS:
                    # Pull alpha computed from future lag horizon (T+lag to T+lag+40h)
                    # All alphas are indexed at the same timestamp d, but represent different future horizons
                    alpha_lag_signal = alphas_norm_dict[lag]
                    alpha_today = alpha_lag_signal.loc[d, variances.index].dropna()

                    # Find common symbols across all alpha signals (on first iteration, or intersect)
                    if common is None:
                        common = alpha_today.index.intersection(variances.index)
                    else:
                        common = common.intersection(alpha_today.index).intersection(variances.index)

                    if common.empty:
                        return None

                # Create a single covariance matrix to be used across all periods
                σ2 = variances.loc[common].values.astype(float)
                cov_matrix = np.diag(σ2)

                # Now collect alphas using common symbols
                # Use the SAME covariance matrix for all periods
                # Use absolute alphas for all periods (not deltas)
                for lag in MP_LAGS:
                    alpha_lag_signal = alphas_norm_dict[lag]
                    alpha_today = alpha_lag_signal.loc[d, common].values.astype(float)
                    alphas_list.append(alpha_today)

                # Create covs_list with the same covariance matrix repeated for each period
                covs_list = [cov_matrix for _ in MP_LAGS]

                # Use multi-period optimizer
                # Returns shape (N, T) where T = len(MP_LAGS)
                w_all = mp_optimizer.solve_long_short_portfolio_mp_with_risk_penalty(
                    alphas_list,
                    covs_list,
                    risk_lambda=RISK_LAMBDA,
                    capital=CAPITAL,
                    gme_limit=GME_LIMIT,
                    box=BOX
                )

                # Use the first period's weights (corresponding to lag 1 - nearest future horizon)
                w = w_all[:, 0]

            else:
                # Single-period optimization (original logic)
                # Check if we have a covariance matrix for this date
                if d not in cov_matrices:
                    return None

                # Get the full covariance matrix for this date
                C_full = cov_matrices[d]

                # Pull today's diag(Σ) - already annualised & ε-adjusted
                variances = var_rolling.loc[d].dropna()
                if variances.empty:
                    return None

                # Pull today's α (expected returns), keep common symbols
                alpha_today = alpha_cache.loc[d, variances.index].dropna()
                common = alpha_today.index.intersection(variances.index)
                if common.empty:
                    return None

                r = alpha_today.loc[common].values.astype(float)
                N = len(common)

                # Get the indices of common symbols in the original returns columns
                all_tickers = returns_aligned_past.columns
                common_indices = [all_tickers.get_loc(ticker) for ticker in common]

                # Extract the submatrix of the full covariance matrix for common symbols
                C = C_full[np.ix_(common_indices, common_indices)]

                # Get previous weights (x0) for the common symbols, default to zeros
                x0 = np.zeros(N)
                if d in previous_weights:
                    prev_w = previous_weights[d]
                    # Align previous weights to current common symbols
                    for idx, sym in enumerate(common):
                        if sym in prev_w:
                            x0[idx] = prev_w[sym]

                # Initialize factor exposure matrix B as N(0,1) with shape (K, N)
                np.random.seed(42)  # For reproducibility
                B = np.random.randn(num_factors, N)

                # Use single-period risk penalty optimizer with x0 and B
                w = sp_optimizer_sweep.solve_long_short_portfolio_with_risk_penalty(
                    alpha=r,
                    C=C,
                    x0=x0,
                    B=B
                )

                # Capture and accumulate timing information
                formulation_time = sp_optimizer_sweep.last_formulation_time
                solve_time = sp_optimizer_sweep.last_solve_time
                total_formulation_time += formulation_time
                total_solve_time += solve_time
                solve_count += 1

                # Store current weights for next iteration
                previous_weights[d] = dict(zip(common, w))

            rec = {"date": d}
            rec.update(dict(zip(common, w)))
            return rec

        # Serial optimization (required for both SPO and MPO)
        print("Starting serial optimization...")
        results = []
        total_dates = len(all_dates) - lookback_cov
        for idx, i in enumerate(range(lookback_cov, len(all_dates))):
            if idx % 100 == 0:
                print(f"Progress: {idx}/{total_dates} dates processed ({100*idx/total_dates:.1f}%)")
            result = process_date(i)
            if result is not None:
                results.append(result)

        records = [r for r in results if r is not None]
        if not records:
            print("ERROR: No valid records were produced by process_date.")
            return None, None, None, None, None, None

        df = pd.DataFrame(records)
        if 'date' not in df.columns:
            raise ValueError("'date' column missing from results.")

        weights_df = df.set_index('date').sort_index()
        csv_filename = f"target_price_hourly_{variant_name}_K{k_factors}_N{num_factors}.csv"
        weights_df.to_csv(csv_filename)
        print(f"Saved portfolio weights to {csv_filename}")

        # Compute average timing per solve
        avg_formulation_time = total_formulation_time / solve_count if solve_count > 0 else 0.0
        avg_solve_time = total_solve_time / solve_count if solve_count > 0 else 0.0

        print(f"Timing: {solve_count} solves, Avg Formulation={avg_formulation_time:.4f}s, Avg Solve={avg_solve_time:.4f}s")

        return weights_df, csv_filename, k_factors, num_factors, avg_formulation_time, avg_solve_time

    # Parameter sweep loop
    sweep_results = []
    print(f"\n{'='*80}")
    print(f"Starting parameter sweep over {len(K_FACTORS_RANGE)} K_FACTORS × {len(NUM_FACTORS_RANGE)} NUM_FACTORS")
    print(f"K_FACTORS_RANGE: {K_FACTORS_RANGE}")
    print(f"NUM_FACTORS_RANGE: {NUM_FACTORS_RANGE}")
    print(f"{'='*80}")

    for k_idx, k_factors in enumerate(K_FACTORS_RANGE):
        for n_idx, num_factors in enumerate(NUM_FACTORS_RANGE):
            combo_num = k_idx * len(NUM_FACTORS_RANGE) + n_idx + 1
            total_combos = len(K_FACTORS_RANGE) * len(NUM_FACTORS_RANGE)

            print(f"\n{'='*80}")
            print(f"Parameter combination {combo_num}/{total_combos}")
            print(f"{'='*80}")

            # Run optimization for this parameter combination
            result = run_single_parameter_combination(k_factors, num_factors)

            if result[0] is None:
                print(f"Warning: No results for K={k_factors}, NUM_FACTORS={num_factors}")
                continue

            weights_df, csv_filename, k, n, avg_form_time, avg_solve_time = result

            # Store timing results
            sweep_results.append({
                'k_factors': k_factors,
                'num_factors': num_factors,
                'formulation_time_s': avg_form_time,
                'solve_time_s': avg_solve_time,
                'total_time_s': avg_form_time + avg_solve_time,
                'csv_filename': csv_filename
            })

    # Save parameter sweep results to CSV
    if sweep_results:
        sweep_df = pd.DataFrame(sweep_results)
        sweep_csv_filename = f"parameter_sweep_timing_{variant_name}.csv"
        sweep_df.to_csv(sweep_csv_filename, index=False)
        print(f"\n{'='*80}")
        print(f"Parameter sweep complete!")
        print(f"Timing results saved to: {sweep_csv_filename}")
        print(f"{'='*80}")
        print(f"\nParameter sweep timing summary:")
        print(sweep_df.to_string(index=False))

        # Use the first parameter combination for visualization
        best_result = sweep_results[0]
        best_csv = best_result['csv_filename']
        print(f"\n{'='*80}")
        print(f"Using first parameter combination for detailed analysis:")
        print(f"  K_FACTORS={best_result['k_factors']}, NUM_FACTORS={best_result['num_factors']}")
        print(f"{'='*80}")
    else:
        print("Warning: No valid sweep results to save")
        return None

    # VISUALIZATION AND BACKTESTING (using first parameters)
    print(f"\nStarting visualization and backtesting for first parameter combination...")

    try:
        weights_hourly = pd.read_csv(best_csv, index_col=0, parse_dates=True).fillna(0.0)
        print(f"Weights shape: {weights_hourly.shape}")
        print(f"Weights date range: {weights_hourly.index.min()} to {weights_hourly.index.max()}")

        returns_pivot = pd.read_csv(ret_csv, index_col=0, parse_dates=True).fillna(0.0)
        returns_pivot = clean_price_jumps(returns_pivot)

        print(f"Returns shape: {returns_pivot.shape}")
        print(f"Returns date range: {returns_pivot.index.min()} to {returns_pivot.index.max()}")

        # Find common dates and tickers
        common_dates = weights_hourly.index.intersection(returns_pivot.index)
        common_tickers = weights_hourly.columns.intersection(returns_pivot.columns)

        print(f"Common dates: {len(common_dates)}")
        print(f"Common tickers: {len(common_tickers)}")

        if len(common_dates) == 0:
            raise ValueError("No common dates between weights and returns data")
        if len(common_tickers) == 0:
            raise ValueError("No common tickers between weights and returns data")

        # Align data to common dates and tickers
        weights_hourly = weights_hourly.loc[common_dates, common_tickers].fillna(0.0)
        returns_pivot = returns_pivot.loc[common_dates, common_tickers].fillna(0.0)

        print(f"Aligned weights shape: {weights_hourly.shape}")
        print(f"Aligned returns shape: {returns_pivot.shape}")

    except Exception as e:
        print(f"Data loading error: {e}")
        raise

    # === NEW: Target·Prediction alignment Sharpe (pre-optimization) ===========
    print("\nComputing alignment (∑ returns · signal) Sharpe vs lag...")

    # Use the same returns grid; for signals we use the raw alpha you optimized on.
    # Reindex to the current common grid to avoid mismatches.
    signal_for_alignment = signal_norm

    # Evaluate across a set of lags (in hours). 0 means contemporaneous,
    # 1 means use signal_{t-1} against returns_t, etc.
    align_lags = list(range(0, 80))  # up to 10 trading days if 8h/day
    align_results = []

    for L in align_lags:
        s_series = signal_target_dot_series(signal_for_alignment, returns_pivot, lag_hours=L)
        if s_series.empty:
            align_results.append((L, 0.0, 0.0, 0.0))
            continue
        s_hat, ci_lo, ci_hi = sharpe_ci_from_series(s_series, alpha=0.05, periods_per_year=2016)
        align_results.append((L, s_hat, ci_lo, ci_hi))

    # Print first few for sanity
    print("\nAlignment Sharpe Ratios (pre-optimization) at Different Lags:")
    for L, s_hat, ci_lo, ci_hi in align_results[:41]:
        print(f"  Lag={L}h ({L//8}d): Sharpe={s_hat:.3f}, CI=({ci_lo:.3f}, {ci_hi:.3f})")

    # Plot Alignment Sharpe vs Lag with error bars
    lags_ = np.array([x[0] for x in align_results])
    sharpes_ = np.array([x[1] for x in align_results])
    ci_lo_ = sharpes_ - np.array([x[2] for x in align_results])
    ci_hi_ = np.array([x[3] for x in align_results]) - sharpes_

    plt.figure(figsize=(8, 5))
    plt.errorbar(lags_ / 8.0, sharpes_, yerr=[ci_lo_, ci_hi_], fmt='o--', capsize=5, label='Alignment Sharpe w/ CI')
    plt.xlabel('Lag (days)')
    plt.ylabel('Annualized Sharpe')
    plt.title(f'Alignment Sharpe (∑ r·ŷ) vs Lag - {variant_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alignment_sharpe_vs_lag.png')
    plt.close()

    # Also plot cumulative alignment score (raw sum of ∑ r·ŷ) for a few lags
    plt.figure(figsize=(12, 6))
    demo_lags = [1, 3, 4, 8, 16, 24, 40]  # 0h, 1d, 2d, 3d, 5d
    for L in demo_lags:
        s_series = signal_target_dot_series(signal_for_alignment, returns_pivot, lag_hours=L)
        if s_series.empty:
            continue
        cum_score = s_series.cumsum()
        plt.plot(cum_score.index, cum_score, label=f'Lag={L}h ({L//8}d)')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.title(f'Cumulative Alignment Score (∑ r·ŷ cumulative) - {variant_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative ∑ r·ŷ', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cumulative_alignment_scores.png')
    plt.close()
    # === END NEW ===============================================================

    # 2. Compute hourly portfolio returns FIRST (moved from later in code)
    print("\nComputing portfolio returns...")

    hourly_portfolio_returns_shifted = (returns_pivot * weights_hourly.shift(1).fillna(0)).sum(axis=1)

    # Remove any NaN or inf values
    hourly_portfolio_returns_shifted = hourly_portfolio_returns_shifted.replace([np.inf, -np.inf], 0).fillna(0)

    # === NEW: Alignment-score-based PnL charts =================================
    # Match execution lag used in your backtest (weights shift(1) -> 1 hour)
    EXEC_LAG_HOURS = 1

    # 1) Build alignment series aligned to the same grid used for portfolio PnL
    #    alignment_t = sum_i r_{t,i} * signal_{t-EXEC_LAG,i}
    alignment_1h = signal_target_dot_series(
        signal_for_alignment.reindex_like(returns_pivot),
        returns_pivot,
        lag_hours=EXEC_LAG_HOURS
    ).reindex(returns_pivot.index)  # ensure same index

    # 2) Align with realized portfolio returns (already using shift(1) in your code)
    port_ret = hourly_portfolio_returns_shifted.reindex(alignment_1h.index).fillna(0.0)

    # Basic sanity diagnostics
    try:
        pear = port_ret.corr(alignment_1h)
    except Exception:
        pear = np.nan
    try:
        spear = port_ret.corr(alignment_1h, method='spearman')
    except Exception:
        spear = np.nan
    print(f"[Alignment Diagnostics] Pearson(port_ret, alignment) = {pear:.4f}, "
          f"Spearman = {spear:.4f}")

    # 3) Alignment-weighted cumulative PnL: sum_t port_ret_t * zscore(alignment_t)
    a = alignment_1h.replace([np.inf, -np.inf], np.nan)
    a = (a - a.mean()) / (a.std(ddof=0) + 1e-12)  # z-score (global)
    align_weighted_ret = (port_ret * a).fillna(0.0)

    cum_align_weighted = (1.0 + align_weighted_ret).cumprod() - 1.0
    plt.figure(figsize=(12, 6))
    plt.plot(cum_align_weighted.index, cum_align_weighted, label='Alignment-weighted PnL')
    plt.title(f'Alignment-Weighted Cumulative PnL (lag={EXEC_LAG_HOURS}h) - {variant_name}')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/alignment_weighted_cum_pnl.png')
    plt.close()

    # 4) PnL by alignment quintile (ex-post diagnostic)
    #    We bucket times by the cross-sample distribution of alignment_t.
    valid_mask = alignment_1h.replace([np.inf, -np.inf], np.nan).notna()
    qvals = pd.qcut(alignment_1h[valid_mask].rank(method='first'),
                    q=5, labels=[1, 2, 3, 4, 5])
    quintile_series = pd.Series(index=alignment_1h.index, dtype='float64')
    quintile_series.loc[qvals.index] = qvals.astype('float64')

    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, 5))
    for q, c in zip([1, 2, 3, 4, 5], colors):
        mask = quintile_series == float(q)
        pnl_q = (1.0 + port_ret.where(mask, 0.0)).cumprod() - 1.0
        plt.plot(pnl_q.index, pnl_q, label=f'Q{int(q)}', linewidth=1.6, color=c)
    plt.title(f'Cumulative PnL by Alignment Quintile (ex-post) - {variant_name}')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend(title='Alignment')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pnl_by_alignment_quintile.png')
    plt.close()

    # 5) Gate by sign of alignment (ex-post diagnostic)
    pos_mask = alignment_1h >= 0
    neg_mask = alignment_1h < 0
    pnl_pos = (1.0 + port_ret.where(pos_mask, 0.0)).cumprod() - 1.0
    pnl_neg = (1.0 + port_ret.where(neg_mask, 0.0)).cumprod() - 1.0

    plt.figure(figsize=(12, 6))
    plt.plot(pnl_pos.index, pnl_pos, label='Alignment ≥ 0')
    plt.plot(pnl_neg.index, pnl_neg, label='Alignment < 0', alpha=0.8)
    plt.title(f'Cumulative PnL Gated by Alignment Sign (ex-post) - {variant_name}')
    plt.xlabel('Date'); plt.ylabel('Cumulative Return')
    plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pnl_by_alignment_sign.png')
    plt.close()
    # === END NEW: Alignment-score-based PnL charts =============================

    # 1. Plot total weight over time
    weights_hourly['total_weight'] = weights_hourly.sum(axis=1)
    plt.figure(figsize=(12, 6))
    weights_hourly['total_weight'].plot()
    plt.title(f'Total Weight Over Time - {variant_name}')
    plt.xlabel('Date')
    plt.ylabel('Total Weight')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_weight_over_time.png')
    plt.close()

    print(f"Portfolio returns stats:")
    print(f"  Non-zero returns: {(hourly_portfolio_returns_shifted != 0).sum()}")
    print(f"  Mean return: {hourly_portfolio_returns_shifted.mean():.6f}")
    print(f"  Std return: {hourly_portfolio_returns_shifted.std():.6f}")
    print(f"  Min return: {hourly_portfolio_returns_shifted.min():.6f}")
    print(f"  Max return: {hourly_portfolio_returns_shifted.max():.6f}")

    # 3. Compute overall annualized Sharpe ratio
    trading_hours_per_year = 2016
    mean_ret = hourly_portfolio_returns_shifted.mean() * trading_hours_per_year
    std_ret = hourly_portfolio_returns_shifted.std() * np.sqrt(trading_hours_per_year)

    if np.isnan(mean_ret) or np.isnan(std_ret) or std_ret < 1e-12:
        overall_sharpe = 0.0
        print(f"Warning: Invalid Sharpe calculation - mean_ret: {mean_ret}, std_ret: {std_ret}")
    else:
        overall_sharpe = mean_ret / std_ret

    print(f"\\nOverall Annualized Sharpe for {variant_name}: {overall_sharpe:.4f}")

    # 4. Rolling Sharpe ratio
    window_size = 1008  # 126 days * 8 hours
    rolling_returns_mean = hourly_portfolio_returns_shifted.rolling(window_size).mean()
    rolling_returns_std = hourly_portfolio_returns_shifted.rolling(window_size).std()

    rolling_sharpe = rolling_returns_mean / (rolling_returns_std + 1e-12)
    rolling_sharpe = rolling_sharpe * np.sqrt(2016)
    rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], 0).fillna(0)

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe', color='blue')
    plt.xlabel('Date')
    plt.ylabel('Rolling Sharpe')
    plt.title(f'Annualized {window_size}-Hour Rolling Sharpe - {variant_name}')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_sharpe.png')
    plt.close()

    # 5. Rolling volatility
    rolling_volatility = hourly_portfolio_returns_shifted.rolling(window_size).std() * np.sqrt(2016)
    rolling_volatility = rolling_volatility.replace([np.inf, -np.inf], 0).fillna(0)
    plt.figure(figsize=(10, 5))
    plt.plot(rolling_volatility.index, rolling_volatility, label='Rolling Volatility', color='red')
    plt.xlabel('Date')
    plt.ylabel('Rolling Volatility')
    plt.title(f'Annualized {window_size}-Hour Rolling Volatility - {variant_name}')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rolling_volatility.png')
    plt.close()

    # 6. Hourly returns plot
    plt.figure(figsize=(12, 6))
    plt.plot(hourly_portfolio_returns_shifted.index, hourly_portfolio_returns_shifted,
             label="Hourly Returns", color='blue', alpha=0.7)
    plt.axhline(y=0, color='black', linewidth=0.8, linestyle='--')
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.title(f"Hourly Portfolio Returns - {variant_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hourly_portfolio_returns.png')
    plt.close()

    # 7. Sharpe at different lags
    lag_results = []
    possible_lags = [i for i in range(0, 160)]
    for lag in possible_lags:
        lagged_weights = shift_weights(weights_hourly, lag)
        hourly_portfolio_returns_lag = (returns_pivot * lagged_weights).fillna(0).sum(axis=1)
        hourly_portfolio_returns_lag = hourly_portfolio_returns_lag.replace([np.inf, -np.inf], 0).fillna(0)
        sharpe_est, ci_lower, ci_upper = compute_sharpe_confidence_interval(hourly_portfolio_returns_lag)
        lag_results.append((lag, sharpe_est, ci_lower, ci_upper))

    print(f"\\nSharpe Ratios at Different Lags for {variant_name}:")
    for lag, sharpe_est, ci_low, ci_high in lag_results[:80]:  # Show first 20
        print(f"  Lag={lag}h ({lag//8}d): Sharpe={sharpe_est:.3f}, CI=({ci_low:.3f}, {ci_high:.3f})")

    lags = [x[0] for x in lag_results]
    sharpes = [x[1] for x in lag_results]
    ci_low = [x[1] - x[2] for x in lag_results]
    ci_high = [x[3] - x[1] for x in lag_results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(np.array(lags)/8, sharpes, yerr=[ci_low, ci_high], fmt='o--', capsize=5, label='Sharpe w/ CI')
    plt.xlabel('Lag (days)')
    plt.ylabel('Annualized Sharpe')
    plt.title(f'Annualized Sharpe at Different Lags - {variant_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sharpe_vs_lag.png')
    plt.close()

    # 8. Cumulative returns for different lags
    plt.figure(figsize=(12, 6))
    possible_lags_cum = [1, 2, 3, 5, 7, 9]
    n_lags = len(possible_lags_cum)
    colors = plt.cm.viridis(np.linspace(0.3, 1, n_lags))
    linewidths = np.linspace(0.5, 2, n_lags)
    for lag, color, lw in zip(possible_lags_cum, colors, linewidths):
        lagged_weights = weights_hourly.shift(lag).fillna(0.0)
        hourly_portfolio_returns = (returns_pivot * lagged_weights).sum(axis=1)

        active_dates = lagged_weights.any(axis=1)
        hourly_portfolio_returns = hourly_portfolio_returns.loc[active_dates]

        cumulative_returns = (1.0 + hourly_portfolio_returns).cumprod() - 1.0
        plt.plot(cumulative_returns.index, cumulative_returns,
                 label=f'Lag={lag}h ({lag//8}d)', linewidth=lw, color=color)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.title(f'Cumulative Portfolio Returns at Different Lags - {variant_name}', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.savefig(f'{output_dir}/cumulative_returns_lags.png')
    plt.close()

    # Print final range summary
    print("\\n" + "="*50)
    print(f"BACKTEST SUMMARY - {variant_name}")
    print("="*50)
    print(f"Backtest start date: {common_dates.min()}")
    print(f"Backtest end date: {common_dates.max()}")
    print(f"Total backtest period: {(common_dates.max() - common_dates.min()).days} days")
    print(f"Number of trading hours: {len(common_dates)}")
    print(f"Average trading hours per day: {len(common_dates) / (common_dates.max() - common_dates.min()).days:.1f}")
    print(f"Overall Annualized Sharpe: {overall_sharpe:.4f}")
    print("="*50)

    print(f"\\nAll plots saved to {output_dir}/")
    print(f"Backtest complete for variant: {variant_name}")

    # Save summary results to a file
    with open(f'{output_dir}/backtest_summary_{variant_name}.txt', 'w') as f:
        f.write(f"BACKTEST SUMMARY - {variant_name}\\n")
        f.write("="*50 + "\\n")
        f.write(f"Alpha computation: Forward EWMA of returns (T to T+40h)\\n")
        f.write(f"Backtest start date: {common_dates.min()}\\n")
        f.write(f"Backtest end date: {common_dates.max()}\\n")
        f.write(f"Total backtest period: {(common_dates.max() - common_dates.min()).days} days\\n")
        f.write(f"Number of trading hours: {len(common_dates)}\\n")
        f.write(f"Overall Annualized Sharpe: {overall_sharpe:.4f}\\n")
        f.write(f"Portfolio weights saved to: {csv_filename}\\n")


    return variant_name, overall_sharpe

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('optimizer_timing.log'),
            logging.StreamHandler()
        ]
    )

    print(f"\\n{'='*80}")
    print("STARTING FORWARD EWMA BACKTEST")
    print(f"{'='*80}")

    try:
        result = run_backtest()
        if result is not None:
            variant_name, sharpe = result
            print(f"\\n{'='*80}")
            print("BACKTEST COMPLETE")
            print(f"{'='*80}")
            print(f"Variant: {variant_name}")
            print(f"Annualized Sharpe: {sharpe:.4f}")
            print(f"{'='*80}")
        else:
            print(f"ERROR: run_backtest returned None")
    except Exception as e:
        print(f"ERROR during backtest: {e}")
        raise
