"""
Utility functions for backtesting
"""

import pandas as pd
import numpy as np
from scipy.stats import norm


def signal_target_dot_series(signal_df: pd.DataFrame,
                             returns_df: pd.DataFrame,
                             lag_hours: int = 0) -> pd.Series:
    """
    Build the alignment series s_t = sum_i returns_{t,i} * signal_{t-lag,i}.
    - signal_df: predictions/scores (index = timestamps, columns = tickers)
    - returns_df: realized returns aligned to the same grid
    - lag_hours: how many hours to shift the signal into the past (execution delay)
    Returns a pd.Series indexed by time.
    """
    # Align dates and tickers
    common_dates = signal_df.index.intersection(returns_df.index)
    common_tickers = signal_df.columns.intersection(returns_df.columns)
    if len(common_dates) == 0 or len(common_tickers) == 0:
        return pd.Series(dtype=float)

    s = signal_df.loc[common_dates, common_tickers]
    r = returns_df.loc[common_dates, common_tickers]

    if lag_hours > 0:
        s = s.shift(lag_hours)

    # Elementwise multiply then sum across tickers
    align_series = (r * s).sum(axis=1)
    # Clean
    align_series = align_series.replace([np.inf, -np.inf], np.nan).dropna()
    return align_series


def sharpe_from_series(x: pd.Series, periods_per_year: int = 2016) -> float:
    """
    Annualized Sharpe from a (hourly) series x_t.
    """
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 2:
        return 0.0
    mu = x.mean()
    sd = x.std()
    if not np.isfinite(mu) or not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return (mu * periods_per_year) / (sd * np.sqrt(periods_per_year))


def sharpe_ci_from_series(x: pd.Series,
                          alpha: float = 0.05,
                          periods_per_year: int = 2016) -> tuple[float, float, float]:
    """
    Annualized Sharpe + Wald-style CI (approx se ≈ 1/sqrt(n)).
    """
    s = sharpe_from_series(x, periods_per_year)
    n = x.replace([np.inf, -np.inf], np.nan).dropna().shape[0]
    if n < 2 or s == 0.0:
        return (s, s, s)
    z = norm.ppf(1 - alpha / 2.0)
    se = 1.0 / np.sqrt(n)
    return (s, s - z * se, s + z * se)


def cs_rank_gaussianize(df: pd.DataFrame, alpha_clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional rank gaussianization with optional clipping"""
    def _one_ts(s):
        r = s.rank(method='average', na_option='keep')
        n = r.notna().sum()
        if n <= 1:
            return s * 0.0
        u = (r - 0.5) / n
        z = pd.Series(np.nan, index=s.index)
        z.loc[u.notna()] = norm.ppf(u[u.notna()].clip(1e-6, 1-1e-6))
        return z

    signal_norm = df.apply(_one_ts, axis=1)
    if alpha_clip is not None:
        signal_norm = signal_norm.clip(lower=-alpha_clip, upper=alpha_clip)
        print(f"Clipped alphas to ±{alpha_clip}")
    return signal_norm


def clean_price_jumps(returns_df, price_jump_cutoff=1.0, winsor_clip=0.30):
    """Remove artificial jumps and winsorize fat tails"""
    mask_jump = returns_df.abs() > price_jump_cutoff
    if mask_jump.any().any():
        print(f"[clean_price_jumps] Found {(mask_jump).sum().sum()} jump rows >±{price_jump_cutoff:.0%}")
    returns_df = returns_df.mask(mask_jump)
    returns_df = returns_df.fillna(0.0)
    return returns_df


def calculate_trading_year_length(returns_matrix_path):
    """Calculate trading year length from 2023 data in returns matrix."""
    df = pd.read_csv(returns_matrix_path, index_col=0, parse_dates=True)

    # Filter for 2023 only (index is already datetime)
    df_2023 = df[(df.index >= '2023-01-01') & (df.index < '2024-01-01')]
    trading_year_periods = len(df_2023)

    print(f"Trading year length from 2023 data: {trading_year_periods} periods")
    print(f"2023 date range: {df_2023.index.min()} to {df_2023.index.max()}")
    return trading_year_periods


def compute_ann_sharpe(hourly_returns, trading_hours_per_year=2016):
    """Compute annualized Sharpe ratio"""
    clean_returns = hourly_returns.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean_returns) < 2:
        return 0.0

    mean_hourly = clean_returns.mean()
    std_hourly = clean_returns.std()

    if np.isnan(mean_hourly) or np.isnan(std_hourly) or std_hourly < 1e-12:
        return 0.0

    mean_annualized = mean_hourly * trading_hours_per_year
    std_annualized = std_hourly * np.sqrt(trading_hours_per_year)

    return mean_annualized / std_annualized


def compute_sharpe_confidence_interval(hourly_returns, alpha=0.05, trading_hours_per_year=2016):
    """Compute Sharpe ratio with confidence interval"""
    sharpe_est = compute_ann_sharpe(hourly_returns, trading_hours_per_year)

    clean_returns = hourly_returns.replace([np.inf, -np.inf], np.nan).dropna()
    n = len(clean_returns)

    if n < 2 or sharpe_est == 0.0:
        return (sharpe_est, sharpe_est, sharpe_est)

    se_sharpe = 1.0 / np.sqrt(n)
    z_value = norm.ppf(1 - alpha / 2.0)
    ci_lower = sharpe_est - z_value * se_sharpe
    ci_upper = sharpe_est + z_value * se_sharpe
    return (sharpe_est, ci_lower, ci_upper)


def shift_weights(weights_df, lag):
    """Shift weights by lag periods"""
    return weights_df.shift(lag).fillna(0.0)


def compute_forward_ewma_returns(returns_df: pd.DataFrame,
                                 start_lag: int = 1,
                                 horizon_hours: int = 40,
                                 halflife_hours: float = 20.0) -> pd.DataFrame:
    """
    Compute forward-looking exponential weighted average of returns from T+start_lag to T+start_lag+horizon_hours.
    Closer (more recent) terms get higher weight.

    Parameters:
    -----------
    returns_df : pd.DataFrame
        Returns matrix (timestamps x tickers)
    start_lag : int
        Starting lag in hours (e.g., 1 for T+1, 8 for T+8, 16 for T+16)
    horizon_hours : int
        Forward-looking horizon in hours (default: 40)
    halflife_hours : float
        Half-life for exponential weighting (default: 20.0)

    Returns:
    --------
    pd.DataFrame
        Forward EWMA returns for each timestamp and ticker
    """
    alpha_ewma = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    # Compute decay rate from half-life: weight = exp(-λ * Δt) where λ = ln(2) / halflife
    lambda_decay = np.log(2.0) / halflife_hours

    # For each timestamp, compute weighted average of forward returns
    max_needed = start_lag + horizon_hours
    for i in range(len(returns_df.index) - max_needed):
        current_time = returns_df.index[i]

        # Get returns from T+start_lag to T+start_lag+horizon_hours
        forward_returns = returns_df.iloc[i+start_lag:i+start_lag+horizon_hours]

        if len(forward_returns) == 0:
            continue

        # Compute weights (closer periods get higher weight)
        # Δt = 1, 2, 3, ..., horizon_hours (relative to start of window)
        delta_t = np.arange(1, len(forward_returns) + 1)
        weights = np.exp(-lambda_decay * delta_t)

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Compute weighted average for each ticker
        # Shape: (horizon_hours, n_tickers)
        weighted_sum = (forward_returns.values * weights[:, np.newaxis]).sum(axis=0)

        alpha_ewma.iloc[i] = weighted_sum

    return alpha_ewma


def ewma_variance(
    returns_df: pd.DataFrame,
    hl_hours: float = 128.0,           # half-life in hours
    periods_per_year: int = 2016,      # your annualization factor
    epsilon: float = 1e-10,            # variance floor
    clip_abs: float | None = None,     # optional robust clipping, e.g., 0.05
) -> pd.DataFrame:
    """
    Compute exponentially weighted moving variance per column (asset), then annualize.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Hourly returns, indexed by datetime, columns = tickers.
    hl_hours : float
        Half-life in hours for EWMA.
    periods_per_year : int
        Annualization factor for hourly to annual.
    epsilon : float
        Floor for stability.
    clip_abs : float | None
        If provided, clip returns to [-clip_abs, +clip_abs] before variance.

    Returns
    -------
    pd.DataFrame
        Annualized EWMA variance per asset, same shape as input.
    """
    if clip_abs is not None:
        r = returns_df.clip(lower=-clip_abs, upper=clip_abs)
    else:
        r = returns_df

    # Convert half-life to alpha: alpha = 1 - 2^(-1/HL)
    alpha = 1.0 - 2.0 ** (-1.0 / float(hl_hours))

    # EWMA variance (per column); bias=False gives sample-like normalization
    var_ewma = r.ewm(alpha=alpha, adjust=False).var(bias=False)

    # Annualize and apply a small floor
    var_annual = (var_ewma * periods_per_year).clip(lower=epsilon)

    return var_annual


def ewma_covariance(
    returns_df: pd.DataFrame,
    hl_hours: float = 128.0,           # half-life in hours
    periods_per_year: int = 2016,      # your annualization factor
    epsilon: float = 1e-10,            # covariance floor (added to diagonal)
    clip_abs: float | None = None,     # optional robust clipping, e.g., 0.05
) -> dict:
    """
    Compute exponentially weighted moving covariance matrices, then annualize.

    Uses recursive EWMA update: Σ_t = (1-α)*Σ_{t-1} + α*r_t*r_t'

    Parameters
    ----------
    returns_df : pd.DataFrame
        Hourly returns, indexed by datetime, columns = tickers.
    hl_hours : float
        Half-life in hours for EWMA.
    periods_per_year : int
        Annualization factor for hourly to annual.
    epsilon : float
        Floor added to diagonal for stability.
    clip_abs : float | None
        If provided, clip returns to [-clip_abs, +clip_abs] before covariance.

    Returns
    -------
    dict
        Dictionary mapping date -> (N, N) covariance matrix, annualized.
    """
    if clip_abs is not None:
        r = returns_df.clip(lower=-clip_abs, upper=clip_abs)
    else:
        r = returns_df

    # Convert half-life to alpha: alpha = 1 - 2^(-1/HL)
    alpha = 1.0 - 2.0 ** (-1.0 / float(hl_hours))

    N = r.shape[1]  # number of assets
    cov_matrices = {}

    # Initialize covariance matrix with identity scaled by epsilon
    S = np.eye(N) * epsilon

    # Iterate through each timestamp
    for date in r.index:
        # Get returns vector for this date
        r_t = r.loc[date].values.reshape(-1, 1)

        # EWMA update: S_t = (1-α)*S_{t-1} + α*r_t*r_t'
        S = (1 - alpha) * S + alpha * (r_t @ r_t.T)

        # Annualize and add epsilon to diagonal for stability
        S_annual = S * periods_per_year
        S_annual += np.eye(N) * epsilon

        # Store the covariance matrix for this date
        cov_matrices[date] = S_annual

    return cov_matrices

import gc
def ewma_covariance_with_gc(returns_df, hl_hours=128.0, periods_per_year=2016,
                            epsilon=1e-10, clip_abs=None, dtype=np.float32,
                            checkpoint=1,  # store every k steps; set 1 to store all
                            gc_every=1000):
    r = returns_df.to_numpy(dtype=dtype, copy=True)
    # Replace NaNs with 0 safely
    np.nan_to_num(r, copy=False, nan=0.0, posinf=None, neginf=None)

    if clip_abs is not None:
        np.clip(r, -clip_abs, clip_abs, out=r)

    alpha = 1.0 - 2.0 ** (-1.0 / float(hl_hours))
    one_minus_alpha = 1.0 - alpha

    T, N = r.shape
    S = np.eye(N, dtype=dtype) * epsilon

    cov_matrices = {}
    outer = np.empty((N, N), dtype=dtype)  # reuse buffer

    for t in range(T):
        rt = r[t]
        # In-place decay
        np.multiply(S, one_minus_alpha, out=S)
        # In-place outer product into buffer, then axpy into S
        np.outer(rt, rt, out=outer)
        S += alpha * outer

        if (t % checkpoint == 0) or (t == T - 1):
            # Make a temporary view to write scaled + bumped version
            Sa = S.copy()  # copy only when storing
            Sa *= periods_per_year
            Sa.flat[::N+1] += epsilon
            cov_matrices[returns_df.index[t]] = Sa  # keep the copy
            # Remove reference to Sa (not necessary, but explicit)
            del Sa

        # Periodic GC to return freed memory to the allocator
        if gc_every and (t % gc_every == 0) and t > 0:
            gc.collect()

    # Optional: drop r to free input array now
    del r, outer
    gc.collect()
    return cov_matrices
