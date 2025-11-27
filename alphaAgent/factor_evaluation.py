#!/usr/bin/env python3
"""
Factor Evaluation Module

This module contains all evaluation functions for computing factor performance metrics:
- Information Coefficient (IC) calculations (Pearson correlation)
- Sharpe ratio and PnL computations
- Visualization utilities

Extracted from alpha_agent_factor.py for better modularity.
"""

import os
import sys
import logging
import multiprocessing as mp
from typing import Dict, Tuple
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import rankdata, spearmanr
from scipy.special import ndtri  # inverse normal CDF

# Add backtest module to path for Optimizer
sys.path.insert(0, '/home/lichenhui')
from backtest.optimizer import Optimizer

# Import session filtering utility
from data_loader import keep_us_regular_session

logger = logging.getLogger(__name__)

# Number of workers for parallelization
N_WORKERS = 20


# =============================================================================
# PARALLEL PROCESSING HELPERS
# =============================================================================

def _compute_pnl_for_timestamp_batch(args):
    """
    Helper function to compute PnL for a batch of timestamps.
    Used for parallel processing in pnl_rowwise_optimizer.

    Args:
        args: Tuple of (batch_indices, F_batch, R_batch, R_hist, risk_budget, gme_limit, min_assets, lookback_window)

    Returns:
        List of (timestamp_index, pnl_value) tuples
    """
    batch_indices, F_batch, R_batch, R_hist, risk_budget, gme_limit, min_assets, lookback_window = args

    # Create optimizer instance
    optimizer = Optimizer(risk_budget=risk_budget, gme_limit=gme_limit)

    results = []
    error_counts = {}  # Track error types

    for i, t in enumerate(batch_indices):
        alpha_t = F_batch[i, :]
        returns_t = R_batch[i, :]

        # Replace factor NaNs with 0 (neutral signal) to allow more assets to participate
        # We still exclude assets where returns are NaN (can't trade them)

        # Find valid assets: must have valid returns (factor NaNs already filled with 0)
        valid_mask = np.isfinite(alpha_t) & np.isfinite(returns_t)


        if valid_mask.sum() < min_assets:
            error_counts['insufficient_assets'] = error_counts.get('insufficient_assets', 0) + 1
            results.append((t, np.nan))
            continue

        # Extract valid data
        alpha_valid = alpha_t[valid_mask]
        returns_valid = returns_t[valid_mask]

        # Check for zero variance in factor (degenerate case)
        if np.std(alpha_valid) < 1e-10:
            error_counts['zero_variance'] = error_counts.get('zero_variance', 0) + 1
            results.append((t, np.nan))
            continue

        # Cross-sectional normalization: rank-based transform to N(0,1) and clip [-3, 3]
        n_valid = len(alpha_valid)
        if n_valid > 2:
            # Rank transform: convert to percentiles then to normal distribution
            ranks = rankdata(alpha_valid, method='average')
            percentiles = ranks / (n_valid + 1)  # Avoid 0 and 1
            alpha_normalized = ndtri(percentiles)  # Convert to N(0,1)
            alpha_normalized = np.clip(alpha_normalized, -3.0, 3.0)
        else:
            # Not enough data for rank transform, use raw values
            alpha_normalized = alpha_valid

        # Create diagonal covariance matrix using historical rolling window
        # FIX: Use rolling window of past returns instead of forward returns
        alpha_valid_f64 = alpha_normalized.astype(np.float64)
        returns_valid_f64 = returns_valid.astype(np.float64)

        # Calculate variances from historical returns (rolling window)
        # Use lookback_window samples (default: 252*4 = 1008 periods)
        start_hist_idx = max(0, t - lookback_window)
        end_hist_idx = t
        R_window = R_hist[start_hist_idx:end_hist_idx, :]  # Shape: (window_size, N_total)
        R_window_valid = R_window[:, valid_mask]  # Select only valid assets

        # Compute variance for each asset from historical data
        if len(R_window_valid) >= 20:  # Need minimum history
            variances = np.nanvar(R_window_valid, axis=0, ddof=1)
            # Ensure positive variances with minimum threshold
            variances = np.clip(variances, 1e-6, None)
        else:
            # Insufficient history - use simple estimate
            variances = np.abs(returns_valid_f64) + 1e-6

        C = np.diag(variances)

        try:
            # Solve for optimal weights
            weights = optimizer.solve_long_short_portfolio(alpha_valid_f64, C)
            pnl_t = np.dot(weights, returns_valid_f64)
            results.append((t, pnl_t))
        except Exception as e:
            error_type = type(e).__name__
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            # Log the first few errors for debugging
            if i < 3:
                logger.debug(f"Optimization failed at t={t}: {error_type}: {e}")
            results.append((t, np.nan))

    # Log error summary if there were failures
    if error_counts and len(error_counts) > 0:
        error_summary = ", ".join([f"{k}={v}" for k, v in sorted(error_counts.items())])
        logger.warning(f"  Batch optimization errors: {error_summary} (batch size={len(batch_indices)})")

    return results


def _compute_ic_for_row_batch(args):
    """
    Helper function to compute IC for a batch of rows.
    Used for parallel processing in compute_ic_series_dense.

    Args:
        args: Tuple of (batch_indices, F_batch, R_batch)

    Returns:
        List of (row_index, ic_value) tuples
    """
    batch_indices, F_batch, R_batch = args

    results = []
    for i, row_idx in enumerate(batch_indices):
        F_row = F_batch[i, :]
        R_row = R_batch[i, :]

        # Compute Pearson correlation for this row
        mask = np.isfinite(F_row) & np.isfinite(R_row)
        count = mask.sum()

        if count <= 2:
            results.append((row_idx, np.nan))
            continue

        # Filter to valid values
        F_valid = F_row[mask]
        R_valid = R_row[mask]

        # Compute correlation
        if len(F_valid) > 2:
            mean_F = F_valid.mean()
            mean_R = R_valid.mean()
            num = ((F_valid - mean_F) * (R_valid - mean_R)).sum()
            denom_F = np.sqrt(((F_valid - mean_F) ** 2).sum())
            denom_R = np.sqrt(((R_valid - mean_R) ** 2).sum())
            denom = denom_F * denom_R

            if denom > 1e-12:
                ic = num / denom
                results.append((row_idx, ic))
            else:
                results.append((row_idx, np.nan))
        else:
            results.append((row_idx, np.nan))

    return results


# =============================================================================
# CORE COMPUTATION FUNCTIONS (DENSE MATRIX APPROACH)
# =============================================================================

def rowwise_pearson(F: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Compute row-wise Pearson correlation between factor matrix F and returns matrix R.

    Args:
        F: Factor matrix of shape (T, N), dtype float32, may contain NaN
        R: Returns matrix of shape (T, N), dtype float32, may contain NaN

    Returns:
        Array of shape (T,) containing IC values for each timestamp
    """
    # F, R: shape (T, N), float32, may contain NaN
    # Compute means ignoring NaN
    mask = np.isfinite(F) & np.isfinite(R)
    count = mask.sum(axis=1).astype(np.float64)

    # Replace NaNs with 0 for sums
    Fz = np.where(mask, F, 0.0)
    Rz = np.where(mask, R, 0.0)

    sum_x = Fz.sum(axis=1)
    sum_y = Rz.sum(axis=1)
    sum_x2 = (Fz * Fz).sum(axis=1)
    sum_y2 = (Rz * Rz).sum(axis=1)
    sum_xy = (Fz * Rz).sum(axis=1)

    numerator = count * sum_xy - sum_x * sum_y
    denom_x = count * sum_x2 - sum_x * sum_x
    denom_y = count * sum_y2 - sum_y * sum_y
    denom = np.sqrt(denom_x) * np.sqrt(denom_y)
    ic = np.where((count > 2) & (denom > 1e-12), numerator / denom, np.nan)
    return ic


def pnl_rowwise(F: np.ndarray, R: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute row-wise PnL by normalizing factors and computing weighted returns.

    Args:
        F: Factor matrix of shape (T, N), dtype float32, may contain NaN
        R: Returns matrix of shape (T, N), dtype float32, may contain NaN
        eps: Small epsilon for numerical stability

    Returns:
        Array of shape (T,) containing PnL values for each timestamp
    """
    T, N = F.shape
    logger.info(f"  Memory footprint: F={F.nbytes/1024**2:.2f}MB, R={R.nbytes/1024**2:.2f}MB (shape: {T}x{N})")

    # Compute cross-sectional Z-scores
    mean = np.nanmean(F, axis=1)
    std = np.nanstd(F, axis=1, ddof=1)
    std_safe = np.where(std > eps, std, np.nan)

    Z = (F - mean[:, None]) / std_safe[:, None]  # keep NaNs
    Z = np.clip(Z, -3.0, 3.0)

    # Only consider names with both weight and return defined
    valid = np.isfinite(Z) & np.isfinite(R)

    # Demean within the *valid* cross-section
    cs_mean = np.nanmean(np.where(valid, Z, np.nan), axis=1)[:, None]
    Z = Z - cs_mean

    # Normalize to unit gross exposure on the valid set
    gross = np.nansum(np.abs(np.where(valid, Z, np.nan)), axis=1)
    W = Z / gross[:, None]

    # PnL: use sum (portfolio return), not mean (changes with N)
    pnl = np.nansum(np.where(valid, W * R, np.nan), axis=1)

    # Require at least 10 valid names
    valid_count = np.sum(valid, axis=1)
    pnl = np.where(valid_count >= 10, pnl, np.nan)

    # Compute Spearman correlation between weights and original signal for diagnostics
    spearman_corrs = []
    for t in range(T):
        w_t = W[t, :]
        f_t = F[t, :]
        # Only compute correlation where both are finite
        valid_mask = np.isfinite(w_t) & np.isfinite(f_t)
        if valid_mask.sum() >= 10:  # Need at least 10 points for meaningful correlation
            try:
                corr, _ = spearmanr(f_t[valid_mask], w_t[valid_mask])
                if np.isfinite(corr):
                    spearman_corrs.append(corr)
            except:
                pass

    if len(spearman_corrs) > 0:
        mean_spearman = np.mean(spearman_corrs)
        median_spearman = np.median(spearman_corrs)
        logger.info(f"  Spearman(signal, weights): mean={mean_spearman:.4f}, median={median_spearman:.4f}, n={len(spearman_corrs)} timestamps")
    else:
        logger.warning(f"  Could not compute Spearman correlation (insufficient valid data)")

    return pnl


def pnl_rowwise_optimizer(F: np.ndarray, R: np.ndarray,
                          risk_budget: float = 0.1,
                          gme_limit: float = 2.0,
                          min_assets: int = 10,
                          n_workers: int = N_WORKERS,
                          lookback_window: int = 252 * 4) -> np.ndarray:
    """
    Compute row-wise PnL using the Optimizer's long-short portfolio solver.

    PARALLELIZED VERSION: Processes timestamps in parallel using multiprocessing.

    This uses the same optimization approach as brokerClassificationBacktest.py:
    - Maximizes signal-weighted returns
    - Subject to risk budget constraint (via Cholesky decomposition)
    - Market neutral (long-short balanced)
    - Gross market exposure limit

    FIX: Uses rolling window of historical returns for covariance estimation.

    Args:
        F: Factor matrix of shape (T, N), dtype float32, may contain NaN
        R: Returns matrix of shape (T, N), dtype float32, may contain NaN
        risk_budget: Risk budget constraint (default: 0.1 = 10% volatility target)
        gme_limit: Gross market exposure limit (default: 2.0)
        min_assets: Minimum number of valid assets required (default: 10)
        n_workers: Number of parallel workers (default: N_WORKERS = 20)
        lookback_window: Number of periods for rolling variance calculation (default: 252*4 = 1008)

    Returns:
        Array of shape (T,) containing PnL values for each timestamp
    """
    T, _ = F.shape
    pnl = np.full(T, np.nan, dtype=np.float32)

    # Split timestamps into batches for parallel processing
    batch_size = max(1, T // n_workers)
    batches = []

    for i in range(0, T, batch_size):
        end_idx = min(i + batch_size, T)
        batch_indices = list(range(i, end_idx))
        F_batch = F[i:end_idx, :]
        R_batch = R[i:end_idx, :]
        # Pass entire R history for rolling window calculation
        batches.append((batch_indices, F_batch, R_batch, R, risk_budget, gme_limit, min_assets, lookback_window))

    # Process batches in parallel
    try:
        with mp.Pool(processes=n_workers) as pool:
            batch_results = pool.map(_compute_pnl_for_timestamp_batch, batches)

        # Collect results
        n_valid = 0
        for batch_result in batch_results:
            for t, pnl_t in batch_result:
                pnl[t] = pnl_t
                if np.isfinite(pnl_t):
                    n_valid += 1

        logger.info(f"  Parallelized PnL computation completed with {n_workers} workers: {n_valid}/{T} valid")

    except (OSError, RuntimeError) as e:
        # Fallback to sequential processing if multiprocessing fails
        logger.warning(f"Parallel PnL computation failed ({e}), falling back to sequential")

        optimizer = Optimizer(risk_budget=risk_budget, gme_limit=gme_limit)

        for t in range(T):
            alpha_t = F[t, :]
            returns_t = R[t, :]

            # Replace factor NaNs with 0 (neutral signal)
            valid_mask = np.isfinite(alpha_t) & np.isfinite(returns_t)

            if valid_mask.sum() < min_assets:
                continue

            alpha_valid = alpha_t[valid_mask]
            returns_valid = returns_t[valid_mask]

            # Cross-sectional normalization: rank-based transform to N(0,1) and clip [-3, 3]
            n_valid = len(alpha_valid)
            if n_valid > 2:
                # Rank transform: convert to percentiles then to normal distribution
                ranks = rankdata(alpha_valid, method='average')
                percentiles = ranks / (n_valid + 1)  # Avoid 0 and 1
                alpha_normalized = ndtri(percentiles)  # Convert to N(0,1)
                alpha_normalized = np.clip(alpha_normalized, -3.0, 3.0)
            else:
                # Not enough data for rank transform, use raw values
                alpha_normalized = alpha_valid

            alpha_valid_f64 = alpha_normalized.astype(np.float64)
            returns_valid_f64 = returns_valid.astype(np.float64)

            # Use rolling window for variance calculation
            start_hist_idx = max(0, t - lookback_window)
            end_hist_idx = t
            R_window = R[start_hist_idx:end_hist_idx, :]  # Shape: (window_size, N_total)
            R_window_valid = R_window[:, valid_mask]  # Select only valid assets

            if len(R_window_valid) >= 20:
                variances = np.nanvar(R_window_valid, axis=0, ddof=1)
                variances = np.clip(variances, 1e-6, None)
            else:
                variances = np.abs(returns_valid) + 1e-6

            C = np.diag(variances)

            try:
                weights = optimizer.solve_long_short_portfolio(alpha_valid, C)
                pnl[t] = np.dot(weights, returns_valid)
            except Exception as e:
                logger.debug(f"Optimization failed at t={t}: {e}")
                continue

    return pnl


def make_30min_forward_return_from_price(price_1m: pd.Series, resample_freq: str = '30T') -> pd.Series:
    """
    Resample 1-minute prices to specified frequency and compute forward returns.

    Alignment:
      - Returns are labeled at the start of the interval [t, t+Δ)
      - The last bar of each session is removed to avoid overnight returns

    Args:
        price_1m: 1-minute price series with DatetimeIndex
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)

    Returns:
        Forward return series at resampled frequency (intraday only, no overnight)
    """
    # 1) Keep US regular session (09:30–16:00 ET)
    price_session = keep_us_regular_session(price_1m.to_frame('price'))['price']

    # 2) Resample to bar-close price; bins labeled at left edge: [t, t+Δ)
    P_resampled = price_session.resample(resample_freq, label='left', closed='left').last()

    # 3) Forward return labeled at t for [t, t+Δ)
    fwd_return = (P_resampled.shift(-1) / P_resampled - 1)

    # Clip extreme returns to [-0.5, 0.5] range
    fwd_return = fwd_return.clip(-0.5, 0.5)

    # 4) Remove the LAST bar of each session to avoid any overnight span
    by_day = fwd_return.groupby(fwd_return.index.date, group_keys=False)
    last_bar_mask = by_day.cumcount(ascending=False) == 0
    fwd_return[last_bar_mask] = np.nan

    return fwd_return


def forward_return_intraday(close: pd.Series, horizon: int = 30) -> pd.Series:
    """
    Compute forward returns within each trading session only (US regular hours).

    Ensures forward returns don't span across session boundaries by:
    1. Grouping by date
    2. Computing forward returns within each day's session
    3. Marking the last `horizon` minutes of each session as NaN

    Args:
        close: Close price series with DatetimeIndex
        horizon: Forward horizon in minutes (default: 30)

    Returns:
        Forward return series with session-aware computation
    """
    by_day = close.groupby(close.index.date, group_keys=False)
    fwd = by_day.apply(lambda s: s.shift(-horizon) / s - 1)
    # Drop the final horizon minutes (no next bar inside session)
    last_mask = by_day.cumcount(ascending=False) < horizon
    fwd[last_mask] = np.nan
    return fwd


def resample_factor_to_30min(factor_1m: pd.Series,
                             resample_freq: str = '30T',
                             minute_lag: int = 1) -> pd.Series:
    """
    Resample 1-minute factor to specified frequency with configurable minute lag.

    - Filters to US regular session (09:30–16:00 ET)
    - Shifts by specified minute lag to avoid using current bar info
    - Takes the last value in each bin (snapshot at bar close)
    - Drops the last bar of each session (no forward return available)

    Args:
        factor_1m: 1-minute factor series with DatetimeIndex
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)
        minute_lag: Number of minutes to lag the factor (default: 1)

    Returns:
        Resampled factor series
    """
    # Step 1: Filter to regular session hours (09:30-16:00 ET)
    factor_session = keep_us_regular_session(factor_1m.to_frame('factor'))['factor']

    # Shift by specified minute lag
    factor_session = factor_session.shift(minute_lag)

    factor_binclose = factor_session.resample(resample_freq, label='left', closed='left').last()

    # FULL-BIN lag to prevent within-bin info
    factor_resampled = factor_binclose.shift(1)

    by_day = factor_resampled.groupby(factor_resampled.index.date)
    last_bar_mask = by_day.cumcount(ascending=False) == 0
    factor_resampled[last_bar_mask] = np.nan
    return factor_resampled


def resample_from_prices(
    factor_data: Dict[str, pd.Series],
    price_data: Dict[str, pd.Series],
    resample_freq: str = '30T'
) -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    """
    Resample factors and compute forward returns from prices with provably correct alignment.

    This is the recommended approach:
    1. Resample prices to target frequency
    2. Compute forward returns on resampled prices
    3. Resample factors to target frequency
    4. Intersect indices to ensure perfect alignment

    Args:
        factor_data: Dict mapping ticker -> Series of factor values (1-min resolution)
        price_data: Dict mapping ticker -> Series of prices (1-min resolution)
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)

    Returns:
        Tuple of (resampled_factor_data, resampled_forward_returns)
    """
    resampled_factor_data = {}
    resampled_forward_returns = {}

    total_before = 0
    total_after = 0

    for ticker in factor_data.keys():
        if ticker not in price_data:
            continue

        # Count valid values before resampling
        n_before = factor_data[ticker].notna().sum()
        total_before += n_before

        # Resample factor and compute forward returns from prices
        F_resampled = resample_factor_to_30min(factor_data[ticker], resample_freq)
        R_resampled = make_30min_forward_return_from_price(price_data[ticker], resample_freq)

        # Intersect indices to ensure perfect alignment
        idx = F_resampled.index.intersection(R_resampled.index)
        F_aligned = F_resampled.reindex(idx).dropna()
        R_aligned = R_resampled.reindex(idx).dropna()

        # Only keep if we have aligned data
        if len(F_aligned) > 0 and len(R_aligned) > 0:
            # Further intersect to ensure both have values at same timestamps
            common_idx = F_aligned.index.intersection(R_aligned.index)
            resampled_factor_data[ticker] = F_aligned.reindex(common_idx)
            resampled_forward_returns[ticker] = R_aligned.reindex(common_idx)

            # Count valid values after resampling
            total_after += len(common_idx)

    logger.info(f"  Resampling from prices: {total_before} valid 1-min data points -> {total_after} valid {resample_freq} data points")

    return resampled_factor_data, resampled_forward_returns


def resample_from_prices_multiple_lags(
    factor_data: Dict[str, pd.Series],
    price_data: Dict[str, pd.Series],
    resample_freq: str = '30T',
    minute_lags: list = None
) -> Dict[int, Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]]:
    """
    Resample factors at multiple minute lags and compute forward returns from prices.

    Args:
        factor_data: Dict mapping ticker -> Series of factor values (1-min resolution)
        price_data: Dict mapping ticker -> Series of prices (1-min resolution)
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)
        minute_lags: List of minute lags to test (default: [1, 3, 5, 10])

    Returns:
        Dict mapping lag -> (resampled_factor_data, resampled_forward_returns)
    """
    if minute_lags is None:
        minute_lags = [1]

    results = {}

    for lag in minute_lags:
        logger.info(f"  Processing lag={lag} minutes...")
        resampled_factor_data = {}
        resampled_forward_returns = {}

        for ticker in factor_data.keys():
            if ticker not in price_data:
                continue

            # Resample factor with specified lag and compute forward returns from prices
            F_resampled = resample_factor_to_30min(factor_data[ticker], resample_freq, minute_lag=lag)
            R_resampled = make_30min_forward_return_from_price(price_data[ticker], resample_freq)

            # Intersect indices to ensure perfect alignment
            idx = F_resampled.index.intersection(R_resampled.index)
            F_aligned = F_resampled.reindex(idx).dropna()
            R_aligned = R_resampled.reindex(idx).dropna()

            # Only keep if we have aligned data
            if len(F_aligned) > 0 and len(R_aligned) > 0:
                # Further intersect to ensure both have values at same timestamps
                common_idx = F_aligned.index.intersection(R_aligned.index)
                resampled_factor_data[ticker] = F_aligned.reindex(common_idx)
                resampled_forward_returns[ticker] = R_aligned.reindex(common_idx)

        results[lag] = (resampled_factor_data, resampled_forward_returns)

    return results


def to_dense_from_prices(
    factor_data: Dict[str, pd.Series],
    price_data: Dict[str, pd.Series],
    resample_freq: str = '30T'
) -> Tuple[pd.Index, list, np.ndarray, np.ndarray]:
    """
    Convert factor_data and price_data dicts to dense T×N matrices with provably correct alignment.

    This is the recommended approach:
    1. Resample prices to target frequency and compute forward returns
    2. Resample factors to target frequency
    3. Intersect indices to ensure perfect alignment
    4. Convert to dense matrices

    Args:
        factor_data: Dict mapping ticker -> Series of factor values (1-min resolution)
        price_data: Dict mapping ticker -> Series of prices (1-min resolution)
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)

    Returns:
        Tuple of (ts_index, tickers, F, R) where:
        - ts_index: pd.Index of timestamps (length T)
        - tickers: List of ticker symbols (length N)
        - F: Factor matrix of shape (T, N), dtype float32
        - R: Returns matrix of shape (T, N), dtype float32
    """
    # Resample factors and compute returns from prices with provably correct alignment
    factor_data_resampled, forward_returns_resampled = resample_from_prices(
        factor_data, price_data, resample_freq
    )

    # Convert to dense matrices
    tickers = sorted(set(factor_data_resampled) & set(forward_returns_resampled))
    ts = sorted(set().union(*[s.index for s in factor_data_resampled.values()],
                            *[s.index for s in forward_returns_resampled.values()]))
    T, N = len(ts), len(tickers)
    F = np.full((T, N), np.nan, dtype=np.float32)
    R = np.full((T, N), np.nan, dtype=np.float32)
    ts_index = pd.Index(ts)

    for j, tkr in enumerate(tickers):
        f = factor_data_resampled[tkr].reindex(ts_index).to_numpy(dtype=np.float32)
        r = forward_returns_resampled[tkr].reindex(ts_index).to_numpy(dtype=np.float32)
        F[:, j] = f
        R[:, j] = r

    return ts_index, tickers, F, R


def compute_ic_series_dense(
    F: np.ndarray,
    R: np.ndarray,
    ts_index: pd.Index,
    n_workers: int = N_WORKERS,
    use_parallel: bool = True
) -> pd.Series:
    """
    Compute IC timeseries using dense matrices with row-wise Pearson correlation.

    PARALLELIZED VERSION: Can optionally process rows in parallel for large datasets.

    Args:
        F: Factor matrix of shape (T, N), dtype float32
        R: Returns matrix of shape (T, N), dtype float32
        ts_index: pd.Index of timestamps (length T)
        n_workers: Number of parallel workers (default: N_WORKERS = 20)
        use_parallel: If True, use parallel processing (default: True)

    Returns:
        pd.Series of IC values indexed by timestamp
    """
    T = F.shape[0]

    # Use vectorized computation for small datasets (faster than parallelization overhead)
    if not use_parallel or T < 1000:
        ic = rowwise_pearson(F, R)
        return pd.Series(ic, index=ts_index)

    # Parallel processing for large datasets
    ic = np.full(T, np.nan, dtype=np.float32)

    # Split rows into batches for parallel processing
    batch_size = max(1, T // n_workers)
    batches = []

    for i in range(0, T, batch_size):
        end_idx = min(i + batch_size, T)
        batch_indices = list(range(i, end_idx))
        F_batch = F[i:end_idx, :]
        R_batch = R[i:end_idx, :]
        batches.append((batch_indices, F_batch, R_batch))

    # Process batches in parallel
    try:
        with mp.Pool(processes=n_workers) as pool:
            batch_results = pool.map(_compute_ic_for_row_batch, batches)

        # Collect results
        for batch_result in batch_results:
            for row_idx, ic_val in batch_result:
                ic[row_idx] = ic_val

        logger.info(f"  Parallelized IC computation completed with {n_workers} workers")

    except (OSError, RuntimeError) as e:
        # Fallback to vectorized computation if multiprocessing fails
        logger.warning(f"Parallel IC computation failed ({e}), falling back to vectorized")
        ic = rowwise_pearson(F, R)

    return pd.Series(ic, index=ts_index)


def clean_pnl_spikes(pnl_series: pd.Series, max_return: float = 0.5) -> pd.Series:
    """
    Clean PNL series by removing large spikes (outlier returns).

    Args:
        pnl_series: PnL timeseries (period returns)
        max_return: Maximum absolute return threshold (default: 0.5 = 50%)

    Returns:
        Cleaned PnL series with spikes removed (replaced with NaN)
    """
    # Create a copy to avoid modifying the original
    cleaned = pnl_series.copy()

    # Identify spikes: absolute returns > max_return threshold
    # Using > instead of >= to avoid removing values at exactly the clip boundary
    # (Returns are clipped to ±0.5 in to_dense, so we shouldn't remove those)
    spikes = cleaned.abs() > max_return
    n_spikes = spikes.sum()

    if n_spikes > 0:
        logger.info(f"  Cleaning {n_spikes} PnL spikes (|return| > {max_return*100:.0f}%)")
        # Replace spikes with NaN
        cleaned[spikes] = np.nan

    return cleaned


def compute_sharpe_dense(
    F: np.ndarray,
    R: np.ndarray,
    ts_index: pd.Index,
    annualize_periods: int = 13 * 252,
    use_optimizer: bool = False,
    risk_budget: float = 0.1,
    gme_limit: float = 2.0,
    min_coverage_pct: float = 0.50,
    lookback_window: int = 252 * 4
) -> Tuple[float, pd.Series]:
    """
    Compute Sharpe ratio using dense matrices with row-wise PnL calculation.

    Args:
        F: Factor matrix of shape (T, N), dtype float32
        R: Returns matrix of shape (T, N), dtype float32
        ts_index: pd.Index of timestamps (length T)
        annualize_periods: Number of periods per year for annualization (default: 3276)
                          For 30-min returns: 13 periods/day × 252 trading days = 3276
                          Sharpe = (mean / std) * sqrt(annualize_periods)
        use_optimizer: If True, use optimizer-based portfolio construction (default: True)
        risk_budget: Risk budget for optimizer (default: 0.1 = 10% vol target)
        gme_limit: Gross market exposure limit for optimizer (default: 2.0)
        min_coverage_pct: Minimum percentage of assets with valid data to start backtest (default: 0.50)
        lookback_window: Number of periods for rolling variance calculation (default: 252*4 = 1008)

    Returns:
        Tuple of (sharpe_ratio, pnl_series)
    """
    # FIX: Filter out early period with insufficient coverage
    T, N = F.shape

    # Handle empty data case
    if T == 0 or N == 0:
        logger.warning(f"  Empty factor matrix (T={T}, N={N}), returning NaN")
        return np.nan, pd.Series(dtype=float)

    valid_counts = np.sum(np.isfinite(F) & np.isfinite(R), axis=1)
    coverage_pct = valid_counts / N

    # Find first timestamp where coverage meets threshold AND we have sufficient lookback
    # Check if coverage_pct array is non-empty before calling argmax
    if len(coverage_pct) == 0:
        logger.warning(f"  Empty coverage array, returning NaN")
        return np.nan, pd.Series(dtype=float)

    start_idx = max(lookback_window, np.argmax(coverage_pct >= min_coverage_pct))

    if start_idx == 0 and coverage_pct[0] < min_coverage_pct:
        logger.warning(f"  Coverage never reaches {min_coverage_pct*100:.0f}% threshold!")
        logger.warning(f"  Max coverage: {coverage_pct.max()*100:.1f}% at index {coverage_pct.argmax()}")
        # Use the point with maximum coverage instead
        start_idx = max(lookback_window, coverage_pct.argmax())

    logger.info(f"  Coverage threshold: {min_coverage_pct*100:.0f}% ({min_coverage_pct*N:.0f}/{N} assets)")
    logger.info(f"  Starting backtest at index {start_idx}/{T} ({ts_index[start_idx]})")
    logger.info(f"  Coverage at start: {coverage_pct[start_idx]*100:.1f}% ({valid_counts[start_idx]}/{N} assets)")
    logger.info(f"  Skipping first {start_idx} periods for warmup and coverage threshold")

    # Filter matrices to start from coverage threshold
    F_filtered = F[start_idx:, :]
    R_filtered = R[start_idx:, :]
    ts_index_filtered = ts_index[start_idx:]

    # Diagnostic: Check factor data quality
    F_std_per_timestamp = np.nanstd(F_filtered, axis=1)
    n_zero_var = np.sum(F_std_per_timestamp < 1e-10)
    if n_zero_var > len(F_filtered) * 0.5:
        logger.warning(f"  Factor has zero/low variance in {n_zero_var}/{len(F_filtered)} timestamps ({n_zero_var/len(F_filtered)*100:.1f}%) - this may cause failures")
        logger.warning(f"  Factor std range: [{np.nanmin(F_std_per_timestamp):.6f}, {np.nanmax(F_std_per_timestamp):.6f}], median: {np.nanmedian(F_std_per_timestamp):.6f}")

    # Use optimizer-based PnL calculation if requested
    if use_optimizer:
        logger.info(f"  Using optimizer-based portfolio construction (risk_budget={risk_budget}, gme_limit={gme_limit})")
        # Need to pass full R matrix for lookback window, but adjust indices
        # Create modified optimizer call that handles the offset
        pnl_full = np.full(len(F_filtered), np.nan, dtype=np.float32)

        # Call optimizer with adjusted batch processing
        T_filtered = F_filtered.shape[0]
        n_workers = N_WORKERS
        batch_size = max(1, T_filtered // n_workers)
        batches = []

        for i in range(0, T_filtered, batch_size):
            end_idx = min(i + batch_size, T_filtered)
            # Batch indices are relative to start_idx in full matrix
            batch_indices = list(range(start_idx + i, start_idx + end_idx))
            F_batch = F_filtered[i:end_idx, :]
            R_batch = R_filtered[i:end_idx, :]
            # Pass full R matrix for historical lookback
            batches.append((batch_indices, F_batch, R_batch, R, risk_budget, gme_limit, 10, lookback_window))

        try:
            with mp.Pool(processes=n_workers) as pool:
                batch_results = pool.map(_compute_pnl_for_timestamp_batch, batches)

            # Collect results (adjust indices back to filtered range)
            n_valid = 0
            for batch_result in batch_results:
                for t_abs, pnl_t in batch_result:
                    t_rel = t_abs - start_idx
                    if 0 <= t_rel < len(pnl_full):
                        pnl_full[t_rel] = pnl_t
                        if np.isfinite(pnl_t):
                            n_valid += 1

            logger.info(f"  Parallelized PnL computation completed with {n_workers} workers: {n_valid}/{len(pnl_full)} valid")

            # Warn if no valid PnL values computed
            if n_valid == 0:
                logger.warning(f"  Optimizer produced 0 valid PnL values out of {len(pnl_full)} timestamps - check factor quality")

            pnl = pnl_full
        except (OSError, RuntimeError) as e:
            logger.warning(f"Parallel processing failed, using standard optimizer call: {e}")
            # Fallback: pass full R, but optimize only filtered portion
            pnl = pnl_rowwise_optimizer(F_filtered, R_filtered, risk_budget=risk_budget,
                                       gme_limit=gme_limit, lookback_window=lookback_window)
    else:
        logger.info(f"  Using simple PnL calculation (normalized equal-weighted)")
        pnl = pnl_rowwise(F_filtered, R_filtered)

    pnl_series = pd.Series(pnl, index=ts_index_filtered).dropna()

    # Clean large spikes (>=50% returns to prevent compounding issues)
    pnl_series = clean_pnl_spikes(pnl_series, max_return=0.5)
    pnl_series = pnl_series.dropna()

    # Allow calculation even with just 1 data point (but Sharpe will be 0)
    if len(pnl_series) == 0:
        logger.warning("  No valid PnL data after cleaning - returning empty series")
        return 0.0, pnl_series

    if len(pnl_series) == 1:
        logger.warning("  Only 1 PnL data point - cannot compute meaningful Sharpe ratio")
        return 0.0, pnl_series

    mean = pnl_series.mean()
    std = pnl_series.std(ddof=1)
    sharpe = (mean / std) * np.sqrt(annualize_periods) if std > 0 else 0.0

    logger.info(f"  PnL stats: n={len(pnl_series)}, mean={mean:.6f}, std={std:.6f}, Sharpe={sharpe:.4f}")

    # Compute cross-lag diagnostics (Spearman IC across lags)
    logger.info(f"  Cross-lag diagnostics (Spearman IC):")
    lag_ics = compute_cross_lag_ic(F_filtered, R_filtered, ts_index_filtered, lags=range(-2, 3))
    for lag, ic in sorted(lag_ics.items()):
        logger.info(f"    Lag {lag:+d}: IC={ic:.4f}")

    return sharpe, pnl_series


def _nw_tstat_from_series(x: pd.Series, maxlags: int = 5) -> float:
    """HAC t-stat for mean!=0 using OLS on a constant (Newey-West)."""
    # y_t = x_t; regress on const
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 10:
        return 0.0
    X = np.ones((len(x), 1), dtype=np.float16)
    y = x.values.astype(np.float16)
    try:
        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        # t-value of the constant is the t-stat of the mean
        return float(model.tvalues[0])
    except:
        return 0.0


def compute_cross_lag_ic(F: np.ndarray, R: np.ndarray, ts_index: pd.Index,
                         lags: range = range(-2, 3)) -> Dict[int, float]:
    """Compute cross-lag Spearman IC to diagnose lead-lag relationships.

    Args:
        F: Factor matrix of shape (T, N), dtype float32
        R: Returns matrix of shape (T, N), dtype float32
        ts_index: pd.Index of timestamps (length T)
        lags: Range of lags to test (default: -2 to +2)
              Positive lag means factor leads returns
              Negative lag means factor lags returns

    Returns:
        Dictionary mapping lag to Spearman IC
    """
    # Convert to DataFrames for easier shifting
    F_df = pd.DataFrame(F, index=ts_index)
    R_df = pd.DataFrame(R, index=ts_index)

    lag_ics = {}
    for lag in lags:
        try:
            # Shift factor by lag periods
            F_shifted = F_df.shift(lag)

            # Stack to get aligned pairs - need to ensure both have the same valid indices
            # Create a mask for valid (non-NaN) pairs
            valid_mask = np.isfinite(F_shifted.values) & np.isfinite(R_df.values)

            # Flatten arrays and select only valid pairs
            F_flat = F_shifted.values[valid_mask]
            R_flat = R_df.values[valid_mask]

            # Need at least 10 pairs to compute meaningful correlation
            if len(F_flat) < 10:
                lag_ics[lag] = 0.0
                continue

            # Compute Spearman correlation on properly aligned data
            ic, _ = spearmanr(F_flat, R_flat)
            lag_ics[lag] = ic if np.isfinite(ic) else 0.0
        except Exception as e:
            lag_ics[lag] = 0.0

    return lag_ics


def compute_sharpe_multiple_lags(
    factor_data: Dict[str, pd.Series],
    price_data: Dict[str, pd.Series],
    resample_freq: str = '30T',
    minute_lags: list = None,
    annualize_periods: int = 13 * 252,
    use_optimizer: bool = False,
    risk_budget: float = 0.1,
    gme_limit: float = 2.0,
    min_coverage_pct: float = 0.50,
    lookback_window: int = 252 * 4
) -> Dict[int, Tuple[float, pd.Series]]:
    """
    Compute Sharpe ratios for multiple minute lags.

    Args:
        factor_data: Dict mapping ticker -> Series of factor values (1-min resolution)
        price_data: Dict mapping ticker -> Series of prices (1-min resolution)
        resample_freq: Resampling frequency (default: '30T' for 30 minutes)
        minute_lags: List of minute lags to test (default: [1, 3, 5, 10])
        annualize_periods: Number of periods per year for annualization
        use_optimizer: If True, use optimizer-based portfolio construction
        risk_budget: Risk budget for optimizer
        gme_limit: Gross market exposure limit for optimizer
        min_coverage_pct: Minimum percentage of assets with valid data to start backtest
        lookback_window: Number of periods for rolling variance calculation

    Returns:
        Dict mapping lag -> (sharpe_ratio, pnl_series)
    """
    if minute_lags is None:
        minute_lags = [1]

    logger.info(f"Computing Sharpe ratios for lags: {minute_lags}")

    # Resample factors at all lags
    lag_data = resample_from_prices_multiple_lags(factor_data, price_data, resample_freq, minute_lags)

    results = {}

    for lag in minute_lags:
        logger.info(f"\n=== Processing lag={lag} minutes ===")
        resampled_factor_data, resampled_forward_returns = lag_data[lag]

        # Convert to dense matrices
        ts_index, tickers, F, R = to_dense_from_prices_impl(
            resampled_factor_data, resampled_forward_returns
        )

        # Compute Sharpe
        sharpe, pnl_series = compute_sharpe_dense(
            F, R, ts_index,
            annualize_periods=annualize_periods,
            use_optimizer=use_optimizer,
            risk_budget=risk_budget,
            gme_limit=gme_limit,
            min_coverage_pct=min_coverage_pct,
            lookback_window=lookback_window
        )

        results[lag] = (sharpe, pnl_series)
        logger.info(f"  Lag {lag} min: Sharpe = {sharpe:.4f}, PnL points = {len(pnl_series)}")

    return results


def to_dense_from_prices_impl(
    factor_data_resampled: Dict[str, pd.Series],
    forward_returns_resampled: Dict[str, pd.Series]
) -> Tuple[pd.Index, list, np.ndarray, np.ndarray]:
    """
    Convert already-resampled factor and return data to dense matrices.
    Helper function for compute_sharpe_multiple_lags.

    Args:
        factor_data_resampled: Dict mapping ticker -> resampled factor series
        forward_returns_resampled: Dict mapping ticker -> resampled return series

    Returns:
        Tuple of (ts_index, tickers, F, R)
    """
    tickers = sorted(set(factor_data_resampled) & set(forward_returns_resampled))
    ts = sorted(set().union(*[s.index for s in factor_data_resampled.values()],
                            *[s.index for s in forward_returns_resampled.values()]))
    T, N = len(ts), len(tickers)
    F = np.full((T, N), np.nan, dtype=np.float32)
    R = np.full((T, N), np.nan, dtype=np.float32)
    ts_index = pd.Index(ts)

    for j, tkr in enumerate(tickers):
        f = factor_data_resampled[tkr].reindex(ts_index).to_numpy(dtype=np.float32)
        r = forward_returns_resampled[tkr].reindex(ts_index).to_numpy(dtype=np.float32)
        F[:, j] = f
        R[:, j] = r

    return ts_index, tickers, F, R


# =============================================================================
# VISUALIZATION
# =============================================================================

def save_factor_pnl_visualization(
    pnl_series: pd.Series,
    factor_name: str,
    sharpe: float,
    output_dir: str = "/home/lichenhui/data/alphaAgent/factor_pnl_charts"
) -> None:
    """
    Save PNL visualization for a factor to a PNG file.

    Args:
        pnl_series: PnL timeseries (period returns)
        factor_name: Name of the factor
        sharpe: Sharpe ratio
        output_dir: Directory to save the chart
    """
    # Always attempt visualization, even with limited data
    if pnl_series.empty:
        logger.warning(f"  Cannot create visualization for {factor_name} - empty PnL series")
        return

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Calculate cumulative PnL
        cumulative_pnl = (1 + pnl_series).cumprod() - 1
        total_return = cumulative_pnl.iloc[-1]

        # Calculate drawdown
        cumulative_wealth = (1 + pnl_series).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle(f'Factor PNL Analysis: {factor_name}\nSharpe Ratio: {sharpe:.3f}',
                    fontsize=16, fontweight='bold')

        # Plot 1: Cumulative PNL
        ax1 = axes[0]
        ax1.plot(cumulative_pnl.index, cumulative_pnl.values * 100,
                linewidth=2, color='#2E86AB', label='Cumulative PNL')
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.set_title(f'Cumulative PNL | Total Return: {total_return*100:.2f}%', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Plot 2: Period Returns
        ax2 = axes[1]
        colors = ['#06A77D' if x > 0 else '#D62828' for x in pnl_series.values]
        ax2.bar(pnl_series.index, pnl_series.values * 100,
               color=colors, alpha=0.6, width=max(1, len(pnl_series) // 1000))
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Period Return (%)', fontsize=12)
        ax2.set_title('Period-by-Period Returns', fontsize=13)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Drawdown
        ax3 = axes[2]
        ax3.fill_between(drawdown.index, drawdown.values * 100, 0,
                        color='#D62828', alpha=0.4, label='Drawdown')
        ax3.plot(drawdown.index, drawdown.values * 100,
                color='#8B0000', linewidth=1.5)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title(f'Drawdown | Max DD: {max_drawdown*100:.2f}%', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')

        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        safe_name = factor_name.replace('/', '_').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved PNL chart to: {filepath}")

    except Exception as e:
        logger.warning(f"  Failed to save visualization for {factor_name}: {e}")


def save_multi_lag_pnl_visualization(
    lag_results: Dict[int, Tuple[float, pd.Series]],
    factor_name: str,
    output_dir: str = "/home/lichenhui/data/alphaAgent/factor_pnl_charts"
) -> None:
    """
    Save multi-lag PNL visualization comparing different minute lags.

    Args:
        lag_results: Dict mapping lag -> (sharpe_ratio, pnl_series)
        factor_name: Name of the factor
        output_dir: Directory to save the chart
    """
    if not lag_results:
        logger.warning(f"  Cannot create multi-lag visualization for {factor_name} - no results")
        return

    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Define color palette for different lags
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        fig.suptitle(f'Multi-Lag PNL Analysis: {factor_name}', fontsize=16, fontweight='bold')

        # Sort lags for consistent plotting
        sorted_lags = sorted(lag_results.keys())

        # Plot 1: Cumulative PNL for all lags
        ax1 = axes[0]
        sharpe_info = []
        for i, lag in enumerate(sorted_lags):
            sharpe, pnl_series = lag_results[lag]
            if len(pnl_series) == 0:
                continue

            cumulative_pnl = (1 + pnl_series).cumprod() - 1
            color = colors[i % len(colors)]
            ax1.plot(cumulative_pnl.index, cumulative_pnl.values * 100,
                    linewidth=2, color=color, label=f'Lag {lag}min (SR={sharpe:.3f})', alpha=0.8)
            sharpe_info.append(f"Lag {lag}min: SR={sharpe:.3f}")

        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax1.set_title('Cumulative PNL Comparison Across Lags', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10)

        # Plot 2: Sharpe Ratio Bar Chart
        ax2 = axes[1]
        sharpes = [lag_results[lag][0] for lag in sorted_lags]
        bar_colors = [colors[i % len(colors)] for i in range(len(sorted_lags))]
        bars = ax2.bar([f'{lag}min' for lag in sorted_lags], sharpes, color=bar_colors, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Sharpe Ratio', fontsize=12)
        ax2.set_xlabel('Minute Lag', fontsize=12)
        ax2.set_title('Sharpe Ratio by Lag', fontsize=13)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, sharpe in zip(bars, sharpes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{sharpe:.3f}',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

        # Plot 3: Rolling Sharpe (30-day window) for each lag
        ax3 = axes[2]
        window = 30  # 30 periods rolling window

        for i, lag in enumerate(sorted_lags):
            sharpe, pnl_series = lag_results[lag]
            if len(pnl_series) < window:
                continue

            # Calculate rolling Sharpe
            rolling_mean = pnl_series.rolling(window=window).mean()
            rolling_std = pnl_series.rolling(window=window).std()
            rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(13 * 252)  # Annualized

            color = colors[i % len(colors)]
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values,
                    linewidth=1.5, color=color, label=f'Lag {lag}min', alpha=0.7)

        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Rolling Sharpe Ratio', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_title(f'Rolling {window}-Period Sharpe Ratio', fontsize=13)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best', fontsize=10)

        # Format x-axis for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save figure
        safe_name = factor_name.replace('/', '_').replace(' ', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_multi_lag_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"  Saved multi-lag PNL chart to: {filepath}")
        logger.info(f"  Sharpe ratios: {', '.join(sharpe_info)}")

    except Exception as e:
        logger.warning(f"  Failed to save multi-lag visualization for {factor_name}: {e}")


