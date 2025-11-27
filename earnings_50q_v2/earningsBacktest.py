#!/usr/bin/env python3
"""
Classification Prediction Matrix Backtest Script
Modified version of brokerBacktest.py to handle classification prediction matrices
"""

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from backtest.optimizer import Optimizer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm
import os
import glob

# Define constants
CUTOFF = pd.Timestamp("2022-01-01")
CUTOFFMAX = pd.Timestamp("2024-07-01")
ret_csv = "hourly_close_to_close_returns_matrix.csv"

# Find all classification prediction matrix files
classification_files = glob.glob("/home/lichenhui/prediction_matrix_embeddings_new.csv")
print(f"Found {len(classification_files)} classification prediction matrix files:")
for f in classification_files:
    print(f"  {f}")

if not classification_files:
    raise ValueError("No classification prediction matrix files found!")

# === NEW: helpers for target·prediction alignment Sharpe ======================
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
# === END NEW ==================================================================

def extract_variant_name(signal_file):
    """Extract variant name from filename"""
    filename = Path(signal_file).name
    # Pattern: broker_report_classification_prediction_matrix_{variant}_{timestamp}.csv
    parts = filename.replace("regression_prediction_matrix_", "").replace(".csv", "").split("_")
    # Take everything except the timestamp (last part)
    variant_name = "_".join(parts[:-1])
    return variant_name

def decay_fill_exponential(df: pd.DataFrame, halflife_hours: float, max_gap_hours: int | None = None) -> pd.DataFrame:
    """Forward-decay fill NaN gaps using an exponential half-life (in hours for hourly data)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    # 1) Forward-fill the last observed value (to carry it across NaN gaps)
    v_ffill = df.ffill()

    # 2) Build a matrix of the last observation timestamps, forward-filled
    obs_mask = ~df.isna().values  # ndarray for speed
    date_matrix = np.tile(df.index.values[:, None], (1, df.shape[1]))  # (n_rows, n_cols) datetime64[ns]
    last_obs_dates = np.where(obs_mask, date_matrix, np.datetime64("NaT"))

    # Forward-fill the last_obs_dates down each column
    last_obs_dates = pd.DataFrame(last_obs_dates, index=df.index, columns=df.columns).ffill().values

    # 3) Compute Δhours since last observation (NaN where there was no prior observation)
    dt_hours = (df.index.values[:, None] - last_obs_dates).astype("timedelta64[h]").astype("float")

    # 4) Exponential decay factor and decayed values
    decay = np.power(0.5, dt_hours / float(halflife_hours))
    decayed = v_ffill.values * decay

    # 5) Fill rule: only fill where original is NaN, we have a valid last_obs (not NaN dt), and (optionally) within max gap
    fill_mask = df.isna().values & ~np.isnan(dt_hours)
    if max_gap_hours is not None:
        fill_mask &= (dt_hours <= float(max_gap_hours))

    filled = df.values.copy()
    filled[fill_mask] = decayed[fill_mask]

    return pd.DataFrame(filled, index=df.index, columns=df.columns)

def identify_poor_coverage_year_months(signal_data: pd.DataFrame, coverage_threshold: int = 500) -> pd.PeriodIndex:
    """Identify year–month pairs with poor coverage (< threshold total non-NaN entries)."""
    if not isinstance(signal_data.index, pd.DatetimeIndex):
        raise ValueError("signal_data.index must be a DatetimeIndex")

    # Total non-NaNs per year–month
    monthly_coverage = signal_data.groupby(signal_data.index.to_period('M')).apply(
        lambda g: g.notna().sum().sum()
    )

    poor = monthly_coverage[monthly_coverage < coverage_threshold]
    print("Monthly coverage analysis:")
    for period, cnt in monthly_coverage.items():
        status = "POOR" if period in poor.index else "GOOD"
        print(f"  {period}: {cnt} signals ({status})")

    return pd.PeriodIndex(poor.index, freq='M')

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

def run_backtest_for_file(signal_file):
    """Run complete backtest for a single classification prediction matrix file"""

    variant_name = extract_variant_name(signal_file)
    print(f"\n{'='*80}")
    print(f"PROCESSING VARIANT: {variant_name}")
    print(f"Signal file: {signal_file}")
    print(f"{'='*80}")

    # Create output directory for this variant
    output_dir = f'backtest_plots/EarningsCall_risk_fixed_{variant_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Load signal data (already in weighted score terms)
    signal = (
        pd.read_csv(signal_file, index_col=0, parse_dates=True)
          .loc[CUTOFF:]
    )

    # Load returns data
    returns = (
        pd.read_csv(ret_csv, index_col=0, parse_dates=True)
          .loc[CUTOFF:]
          .fillna(0.0)
          .clip(lower=-1.0, upper=1.0)
    )

    # Align signal to returns index
    aligned = signal.reindex(returns.index)

    # NaN out signals before signal start date
    SIGNAL_START = pd.Timestamp("2023-01-01")
    aligned.loc[aligned.index < SIGNAL_START] = np.nan

    print(f"NaN'd out signals before {SIGNAL_START}")
    poor_periods = identify_poor_coverage_year_months(aligned, coverage_threshold=100)

    signal = decay_fill_exponential(aligned, halflife_hours=40, max_gap_hours=80)
    # signal = aligned
    # Apply cross-sectional rank gaussianization and fill NaNs
    '''
    mu  = signal.rolling(252*8, min_periods=252*8).mean()
    sig = signal.rolling(252*8, min_periods=252*8).std()
    signal_ts = (signal - mu) / (sig + 1e-8)
    '''
    signal_norm = cs_rank_gaussianize(signal)


    # Identify and handle poor coverage periods
    """
    if len(poor_periods) > 0:
        mask_to_nan = signal_norm.index.to_period('M').isin(poor_periods)
        signal_norm.loc[mask_to_nan] = np.nan
        print(f"NaN'd out {mask_to_nan.sum()} rows across {len(poor_periods)} year–month pairs: {list(poor_periods.astype(str))}")
    else:
        print("No poor-coverage year–month pairs found.")
    """
    signal_norm = signal_norm.fillna(0.0)


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

    # Portfolio optimization setup
    Optimizer_obj = Optimizer(risk_budget=0.1, gme_limit=2)

    alpha_raw = signal
    alpha_cache = alpha_raw
    returns_aligned = returns.loc[alpha_cache.index]

    # Calculate trading year length and setup optimization
    trading_year_length = calculate_trading_year_length(ret_csv)
    lookback_cov = trading_year_length
    all_dates = alpha_cache.index.sort_values()

    EPSILON = 1e-10
    TARGET_RISK = 0.10

    PERIODS_PER_YEAR = 2016
    returns_aligned_past = returns_aligned.shift(1)


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

    # Example usage in your pipeline:
    # returns_aligned_past = returns_aligned.shift(1)
    # Choose a half-life (e.g., 64–256 hours). Start with 128.
    HL = 128.0
    PERIODS_PER_YEAR = lookback_cov
    EPSILON = 1e-10

    # Optional robust clipping, tune to your data if you have jump filtering already:
    CLIP_ABS = None  # e.g., 0.05

    var_hourly = ewma_variance(
        returns_df=returns_aligned_past,
        hl_hours=HL,
        periods_per_year=PERIODS_PER_YEAR,
        epsilon=EPSILON,
        clip_abs=CLIP_ABS,
    )

    # var_rolling_ewma now replaces your previous var_rolling
    # and can be fed into the optimizer as diag(Σ_t) at each date.


    '''
    # Diagonal covariance matrix calculation
    var_hourly = (
        returns_aligned_past
          .rolling(window=lookback_cov, min_periods=lookback_cov)
          .var(ddof=0)
    )
    '''
    var_rolling = var_hourly.add(EPSILON)

    print("Computed rolling diagonal covariance (variance) matrices...")


    def process_date(i: int):
        d = all_dates[i]

        # Pull today's diag(Σ) - already annualised & ε-adjusted
        variances = var_rolling.loc[d].dropna()
        if variances.empty:
            return None

        # Pull today's α (expected returns), keep common symbols
        alpha_today = alpha_cache.loc[d, variances.index].dropna()
        common = alpha_today.index.intersection(variances.index)
        if common.empty:
            return None

        r = alpha_today.values.astype(float)
        σ2 = variances.loc[common].values.astype(float)

        w = Optimizer_obj.solve_long_short_portfolio(r, np.diag(σ2))

        rec = {"date": d}
        rec.update(dict(zip(common, w)))
        return rec

    # Parallel optimization
    print("Starting parallel optimization...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(process_date)(i) for i in range(lookback_cov, len(all_dates))
    )

    records = [r for r in results if r is not None]
    if not records:
        print("ERROR: No valid records were produced by process_date.")
        return None

    df = pd.DataFrame(records)
    if 'date' not in df.columns:
        raise ValueError("'date' column missing from results.")

    weights_df = df.set_index('date').sort_index()
    csv_filename = f"target_price_hourly_{variant_name}.csv"
    weights_df.to_csv(csv_filename)
    print(f"Saved portfolio weights to {csv_filename}")

    # VISUALIZATION AND BACKTESTING
    print(f"\nStarting visualization and backtesting for {variant_name}...")

    try:
        weights_hourly = pd.read_csv(csv_filename, index_col=0, parse_dates=True).fillna(0.0)
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

    hourly_portfolio_returns = (returns_pivot * weights_hourly).fillna(0).sum(axis=1)
    hourly_portfolio_returns_shifted = (returns_pivot * weights_hourly.shift(1).fillna(0)).sum(axis=1)

    # Remove any NaN or inf values
    hourly_portfolio_returns = hourly_portfolio_returns.replace([np.inf, -np.inf], 0).fillna(0)
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
        f.write(f"Signal file: {signal_file}\\n")
        f.write(f"Backtest start date: {common_dates.min()}\\n")
        f.write(f"Backtest end date: {common_dates.max()}\\n")
        f.write(f"Total backtest period: {(common_dates.max() - common_dates.min()).days} days\\n")
        f.write(f"Number of trading hours: {len(common_dates)}\\n")
        f.write(f"Overall Annualized Sharpe: {overall_sharpe:.4f}\\n")
        f.write(f"Portfolio weights saved to: {csv_filename}\\n")


    return variant_name, overall_sharpe

# Main execution
if __name__ == "__main__":
    print(f"\\n{'='*80}")
    print("STARTING CLASSIFICATION BACKTESTS")
    print(f"{'='*80}")

    results_summary = []

    for i, signal_file in enumerate(classification_files):
        try:
            print(f"\\nProcessing file {i+1}/{len(classification_files)}: {Path(signal_file).name}")
            result = run_backtest_for_file(signal_file)
            if result is not None:
                variant_name, sharpe = result
                results_summary.append((variant_name, sharpe))
            else:
                print(f"ERROR: run_backtest_for_file returned None for {signal_file}")
        except Exception as e:
            print(f"ERROR processing {signal_file}: {e}")
            continue

    print(f"\\n{'='*80}")
    print("ALL CLASSIFICATION BACKTESTS COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed {len(results_summary)} variants:")
    for variant_name, sharpe in results_summary:
        print(f"  {variant_name}: Sharpe = {sharpe:.4f}")
    print(f"{'='*80}")
