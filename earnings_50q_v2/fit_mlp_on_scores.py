#!/usr/bin/env python3
"""
Eleven‑class quantile **classifier** for **1‑day** forward returns with rolling volatility normalization.

Modified version to work with CSV input from portfolio_50q_full_run_output/portfolio_results_full_2023_2025.csv
OPTIMIZED with multiprocessing for dataset preparation.
Uses simple MLP with sklearn instead of complex PyTorch model.

Enhanced with weighted scoring and proper handling of multiple predictions per ticker-date pair.

New in this revision
────────────────────
• Logs the **date cut‑offs** between Train, Validation, and Test splits so you
  can verify the chronological partitioning.
• Modified to load CSV format instead of JSON
• Added multiprocessing for dataset preparation to use all CPU cores
• Enhanced weighted scoring calculation (probability-weighted scores)
• Proper averaging of multiple predictions per ticker-date pair
• Uses simple sklearn MLPClassifier instead of custom PyTorch model
"""

import json, pickle, logging, csv, math
from pathlib import Path
from datetime import datetime, timedelta  # noqa: F401 – retained for completeness
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# ────────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────── Constants ────────────────────────────
N_CLASSES    = 11                                # 11 buckets
LABEL_MAP    = np.arange(-5, 6)                  # 0→−5, 1→−4, ..., 10→+5
DAYS_FWD     = 1                                 # 1 trading day
ROLLING_WINDOW = 63                              # 63 days for rolling std normalization
# Date-based splits
TRAIN_END_DATE = "2024-02-01"
VAL_START_DATE = "2024-02-02"
VAL_END_DATE = "2024-05-30"
TEST_START_DATE = "2024-06-01"
ARTIFACT_DIR = Path("/home/lichenhui")          # change if needed

# ────────────────────────────── IO helpers ───────────────────────────

def _prep_scores(scores):
    """Ensure 50-d float array centered at 5, shape (50,)."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.shape[0] != 50:
        raise ValueError(f"Each score vector must have 50 dims, got {x.shape[0]}")
    # clip to [0, 10] just in case upstream deviates
    np.clip(x, 0.0, 10.0, out=x)
    # Center the scores at 5 by subtracting 5
    x = x - 5.0
    return x

def to_scores(lbl: np.ndarray):
    return LABEL_MAP[lbl]
    
def load_portfolio_allocations_csv(path: str) -> pd.DataFrame:
    """Load portfolio allocations from CSV format."""
    logger.info("Loading portfolio allocations from CSV…")
    df = pd.read_csv(path)
    
    # Extract the 50 question scores into a list format
    score_columns = [f'q{i}' for i in range(1, 51)]
    
    rows = []
    for _, row in df.iterrows():
        scores = [row[col] for col in score_columns]
        rows.append({
            "date": row["date"],
            "ticker": row["ticker"], 
            "scores": scores,
        })
    
    logger.info("Loaded %d rows", len(rows))
    return pd.DataFrame(rows)

def load_close_prices(path: str) -> pd.DataFrame:
    logger.info("Loading close‑price matrix…")
    df = (pd.read_csv(path)
            .assign(DateTime=lambda x: pd.to_datetime(x["DateTime"]))
            .set_index("DateTime"))
    logger.info("Shape: %s", df.shape)
    return df

# ────────────────────────────── Feature / label prep ─────────────────

def fit_scaler_and_transform(X_tr, X_va, X_te, X_all):
    """
    Fit StandardScaler on TRAIN ONLY and transform train/val/test/all.
    Returns: scaler, Xtr, Xva, Xte, Xall
    Also logs and saves diagnostics.
    """
    scaler = StandardScaler(with_mean=False, with_std=True)
    scaler.fit(X_tr)  # <-- train-only

    Xtr  = scaler.transform(X_tr)
    Xva  = scaler.transform(X_va) if X_va is not None else None
    Xte  = scaler.transform(X_te) if X_te is not None else None
    Xall = scaler.transform(X_all) if X_all is not None else None

    # Diagnostics: per-feature train mean/std and validation distribution summaries
    train_means = Xtr.mean(axis=0)
    train_stds  = Xtr.std(axis=0, ddof=0)
    val_means   = Xva.mean(axis=0) if Xva is not None else np.array([])
    val_stds    = Xva.std(axis=0, ddof=0) if Xva is not None else np.array([])

    logger.info("Per-feature TRAIN mean (first 5): %s", np.round(train_means[:5], 4))
    logger.info("Per-feature TRAIN std  (first 5): %s", np.round(train_stds[:5],  4))
    if Xva is not None:
        logger.info("Per-feature VAL   mean (first 5): %s", np.round(val_means[:5], 4))
        logger.info("Per-feature VAL   std  (first 5): %s", np.round(val_stds[:5],  4))

    # Save detailed stats per feature
    stats = pd.DataFrame({
        "feature": [f"q{i}" for i in range(1, 51)],
        "train_mean": train_means,
        "train_std": train_stds,
        "val_mean": val_means if Xva is not None else np.nan,
        "val_std": val_stds if Xva is not None else np.nan,
        "scaler_train_mean_raw": scaler.mean_,     # raw means before standardization
        "scaler_train_std_raw": scaler.scale_,     # raw stds before standardization
    })
    stats_path = ARTIFACT_DIR / "scaling_stats_elevenclass_simple.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("Saved scaling diagnostics to %s", stats_path)

    return scaler, Xtr, Xva, Xte, Xall


def calc_forward_return_normalized(px: pd.DataFrame, ticker: str,
                                   ts: str, days_fwd: int = DAYS_FWD, 
                                   rolling_window: int = ROLLING_WINDOW):
    try:
        ts = pd.to_datetime(ts)
        if ts not in px.index:
            fut = px.index[px.index >= ts]
            prev = px.index[px.index < ts]
            prevP = prev[-1]
            if not len(fut):
                return None
            ts = fut[0]

        if ticker not in px:
            return None

        prev = px.index[px.index < ts]
        prevP = prev[-1] # close T - 2
        p0 = px.at[prevP, ticker]
        if np.isnan(p0):
            return None

        fut = px.index[px.index > ts]
        if len(fut) < days_fwd:
            return None

        ts1 = fut[0] # + 21
        p1 = px.at[ts1, ticker]
        if np.isnan(p1):
            return None
        
        # Calculate raw forward return
        raw_return = (p1 - p0) / p0
        
        # Calculate rolling standard deviation for normalization
        # Get 63 days of price data before the current date
        past_dates = px.index[px.index <= ts]
        if len(past_dates) < rolling_window:
            # Not enough historical data for rolling std, return raw return
            return raw_return
            
        # Get the last 63 days of data
        start_idx = max(0, len(past_dates) - rolling_window)
        rolling_dates = past_dates[start_idx:]
        
        # Calculate returns for the rolling window
        rolling_prices = px.loc[rolling_dates, ticker]
        rolling_returns = rolling_prices.pct_change(fill_method=None).dropna()
        
        if len(rolling_returns) < 5:  # Need minimum data points
            return raw_return
            
        rolling_std = rolling_returns.std()
        if rolling_std == 0 or np.isnan(rolling_std):
            return raw_return
            
        # Normalize the forward return by rolling standard deviation
        normalized_return = raw_return / rolling_std
        return normalized_return
        
    except Exception as exc:
        return None

def process_row_chunk(chunk_data):
    """Process a chunk of rows for multiprocessing."""
    chunk_rows, px_dict, px_index = chunk_data
    
    # Reconstruct DataFrame from dict (needed for multiprocessing)
    px = pd.DataFrame(px_dict, index=px_index)
    
    results = []
    for row_data in chunk_rows:
        r = calc_forward_return_normalized(px, row_data["ticker"], row_data["date"], days_fwd=1)
        results.append({
            "date": row_data["date"],
            "ticker": row_data["ticker"],
            "scores": _prep_scores(row_data["scores"]),
            "returns": r,
        })
    return results

def prepare_dataset_parallel(df_a, df_b, px):
    logger.info("Preparing dataset with multiprocessing – duplicates preserved…")
    if df_b is not None:
        df_all = pd.concat([df_a, df_b], ignore_index=True)
    else:
        df_all = df_a

    # Convert DataFrame to serializable format for multiprocessing
    px_dict = px.to_dict('series')
    px_index = px.index
    
    # Convert rows to list of dicts
    rows_data = []
    for _, row in df_all.iterrows():
        rows_data.append({
            "date": row["date"],
            "ticker": row["ticker"],
            "scores": row["scores"]
        })
    
    # Split into chunks for parallel processing
    n_cores = cpu_count()
    logger.info(f"Using {n_cores} CPU cores for parallel processing")
    chunk_size = max(1, len(rows_data) // (n_cores * 4))  # 4x cores for better load balancing
    
    chunks = []
    for i in range(0, len(rows_data), chunk_size):
        chunk = rows_data[i:i + chunk_size]
        chunks.append((chunk, px_dict, px_index))
    
    logger.info(f"Processing {len(rows_data)} rows in {len(chunks)} chunks of ~{chunk_size} rows each")
    
    # Process chunks in parallel
    with Pool(processes=n_cores) as pool:
        chunk_results = pool.map(process_row_chunk, chunks)
    
    # Flatten results
    rows = []
    for chunk_result in chunk_results:
        rows.extend(chunk_result)
    
    df_meta = pd.DataFrame(rows)
    df_meta["returns"] = df_meta["returns"].astype(float)
    logger.info("Final dataset: %d rows (%.0f%% labelled)",
                len(df_meta), 100*df_meta["returns"].notna().mean())

    X     = np.vstack(df_meta["scores"].values)
    y     = df_meta["returns"].to_numpy()
    dates = pd.to_datetime(df_meta["date"]).to_numpy()

    return X, y, dates, df_meta

# ────────────────────────────── Quantile helpers ─────────────────────

def quantile_bounds(y_train, k=N_CLASSES):
    q = np.quantile(y_train, np.linspace(0, 1, k + 1))
    logger.info("Quantile boundaries: %s", np.round(q, 6))
    return q


def assign_labels(y, q):
    return np.digitize(y, q[1:-1], right=True).astype(int)


def to_mapped_scores(labels):
    return LABEL_MAP[labels]

# ────────────────────────────── Split chronologically ────────────────

def chrono_split(X, y, dates):
    """Split data by specific date cutoffs."""
    idx = np.argsort(dates)
    X, y, dates = X[idx], y[idx], dates[idx]
    
    # Convert dates to pandas datetime for comparison
    dates_pd = pd.to_datetime(dates)
    train_end = pd.to_datetime(TRAIN_END_DATE)
    val_start = pd.to_datetime(VAL_START_DATE)
    val_end = pd.to_datetime(VAL_END_DATE)
    test_start = pd.to_datetime(TEST_START_DATE)
    
    # Create masks for each split
    train_mask = dates_pd <= train_end
    val_mask = (dates_pd >= val_start) & (dates_pd <= val_end)
    test_mask = dates_pd >= test_start
    
    return ((X[train_mask], y[train_mask], dates[train_mask]),
            (X[val_mask], y[val_mask], dates[val_mask]),
            (X[test_mask], y[test_mask], dates[test_mask]))

# ────────────────────────────── Model training ───────────────────────

def train_mlp(X_tr, y_tr, X_va, y_va):
    """Train simple MLP using sklearn MLPClassifier."""
    scaler = StandardScaler().fit(X_tr)
    Xtr, Xva = scaler.transform(X_tr), scaler.transform(X_va)

    logger.info("Scaler (train) mean[:5]=%s std[:5]=%s",
                np.round(scaler.mean_[:5], 4),
                np.round(scaler.scale_[:5], 4))
    
    mlp = MLPClassifier(hidden_layer_sizes=(256, 128),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        batch_size=256,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        n_iter_no_change=10,
                        random_state=42,
                        verbose=False)

    mlp.fit(Xtr, y_tr)
    
    # Print training error throughout training
    logger.info("Training loss history: %s", mlp.loss_curve_)
    for i, loss in enumerate(mlp.loss_curve_, 1):
        if i % 20 == 0 or i == len(mlp.loss_curve_):
            logger.info("Epoch %d - Training loss: %.6f", i, loss)
    
    logger.info("Best val score (internal): %.4f", mlp.best_validation_score_)
    logger.info("Stopped after %d epochs", mlp.n_iter_)

    va_pred = mlp.predict(Xva)
    logger.info("\nValidation report:\n%s", classification_report(y_va, va_pred))
    return mlp, scaler

# ────────────────────────────── Weighted scoring helpers ─────────────

def calculate_weighted_scores(probs, label_map=LABEL_MAP):
    """
    Calculate probability-weighted scores.
    
    Args:
        probs: Array of shape (n_samples, n_classes) with probability distributions
        label_map: Array mapping class indices to score values
    
    Returns:
        Array of weighted scores where each score is the expected value
        of the probability distribution
    
    Example:
        If probs = [[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]
        And label_map = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        Then weighted_score = 0.5 * (-5) + 0.5 * (5) = 0
    """
    return np.dot(probs, label_map)

def aggregate_predictions_by_ticker_date(df_predictions):
    """
    Aggregate multiple predictions for the same ticker-date pair by taking the mean.
    
    Args:
        df_predictions: DataFrame with columns ['date', 'ticker', 'pred_score', 'pred_cls', etc.]
    
    Returns:
        DataFrame with aggregated predictions, one row per unique ticker-date pair
    """
    logger.info("Aggregating multiple predictions per ticker-date pair...")
    
    # Group by ticker and date, then calculate means for numerical columns
    numeric_columns = ['pred_score']
    if 'pred_cls' in df_predictions.columns:
        numeric_columns.append('pred_cls')
    if 'actual_cls' in df_predictions.columns:
        numeric_columns.append('actual_cls')
    if 'returns' in df_predictions.columns:
        numeric_columns.append('returns')
    
    # Calculate aggregations
    agg_dict = {col: 'mean' for col in numeric_columns}
    
    # Handle probability columns specially if they exist
    if 'pred_probs' in df_predictions.columns:
        # For probabilities, we need to average the probability distributions
        def avg_probs(prob_list):
            # Convert list of probability arrays to numpy array and take mean
            probs_array = np.array([eval(p) if isinstance(p, str) else p for p in prob_list])
            return probs_array.mean(axis=0).tolist()
        agg_dict['pred_probs'] = avg_probs
    
    aggregated = (df_predictions.groupby(['date', 'ticker'], as_index=False)
                  .agg(agg_dict))
    
    # Add count of predictions per ticker-date pair
    counts = df_predictions.groupby(['date', 'ticker']).size().reset_index(name='num_predictions')
    aggregated = aggregated.merge(counts, on=['date', 'ticker'])
    
    # Log statistics about aggregation
    original_count = len(df_predictions)
    aggregated_count = len(aggregated)
    duplicate_pairs = original_count - aggregated_count
    
    logger.info(f"Original predictions: {original_count}")
    logger.info(f"Unique ticker-date pairs: {aggregated_count}")
    logger.info(f"Duplicate predictions aggregated: {duplicate_pairs}")
    
    if duplicate_pairs > 0:
        multi_pred_stats = aggregated[aggregated['num_predictions'] > 1]['num_predictions']
        logger.info(f"Ticker-date pairs with multiple predictions: {len(multi_pred_stats)}")
        if len(multi_pred_stats) > 0:
            logger.info(f"Max predictions per pair: {multi_pred_stats.max()}")
            logger.info(f"Mean predictions per pair (for pairs with >1): {multi_pred_stats.mean():.2f}")
    
    return aggregated

# ────────────────────────────── Main pipeline ────────────────────────

def main():
    paths = {
        "alloc":  ARTIFACT_DIR / "portfolio_50q_full_run_output/portfolio_results_full_2023_2025.csv",
        "prices": ARTIFACT_DIR / "close_prices_matrix.csv",
    }

    # Load data
    df_a   = load_portfolio_allocations_csv(paths["alloc"])
    prices = load_close_prices(paths["prices"])

    # Build dataset with parallel processing
    X, y, dates, meta = prepare_dataset_parallel(df_a, None, prices)

    # Keep labelled rows only for supervised training
    labelled = ~np.isnan(y)
    X_lbl, y_lbl, dates_lbl = X[labelled], y[labelled], dates[labelled]

    # Chrono split
    (X_tr, y_tr, d_tr), (X_va, y_va, d_va), (X_te, y_te, d_te) = chrono_split(
        X_lbl, y_lbl, dates_lbl)

    # ---- Log cut-off dates
    logger.info("Split dates: Train≤%s | Val≤%s | Test≥%s",
                pd.to_datetime(d_tr[-1]).date(),
                pd.to_datetime(d_va[-1]).date(),
                pd.to_datetime(d_te[0]).date())

    # Quantile bins from training returns (labels)
    q        = quantile_bounds(y_tr, N_CLASSES)
    y_tr_lbl = assign_labels(y_tr, q)
    y_va_lbl = assign_labels(y_va, q)
    y_te_lbl = assign_labels(y_te, q)

    # === Standardize 50 features feature-wise using TRAIN stats ===
    scaler, Xtr_s, Xva_s, Xte_s, Xall_s = fit_scaler_and_transform(
        X_tr, X_va, X_te, X
    )

    # === Train using simple sklearn MLP ===
    mlp, train_scaler = train_mlp(X_tr, y_tr_lbl, X_va, y_va_lbl)

    # Predict everywhere with the same model - using probability-weighted scores
    logger.info("Generating predictions with weighted scoring...")
    
    # Get probabilities for all predictions
    all_probs = mlp.predict_proba(Xall_s)
    
    # Calculate probability-weighted scores using helper function
    weighted_scores = calculate_weighted_scores(all_probs, LABEL_MAP)
    
    # Also keep hard predictions for comparison
    pred_lbl_all = mlp.predict(Xall_s)
    
    meta["pred_cls"]   = pred_lbl_all
    meta["pred_score"] = weighted_scores
    meta["pred_probs"] = [prob.tolist() for prob in all_probs]  # Save full probability distributions

    # Actual class where available
    actual_lbl = np.full_like(pred_lbl_all, np.nan, dtype=float)
    actual_lbl[labelled] = assign_labels(y[labelled], q)
    meta["actual_cls"] = actual_lbl

    # Test accuracy on labelled TEST split (hold-out)
    test_pred = mlp.predict(Xte_s)
    test_acc = accuracy_score(y_te_lbl, test_pred)
    logger.info("Test accuracy (hold-out): %.4f", test_acc)

    # ---- Aggregate multiple predictions per ticker-date pair ----
    meta_aggregated = aggregate_predictions_by_ticker_date(meta)

    # ---- Persist artefacts
    with open(ARTIFACT_DIR / "mlp_elevenclass_csv_parallel_simple.pkl", "wb") as fh:
        pickle.dump({"scaler": scaler, "quantiles": q, "mlp": mlp, "train_scaler": train_scaler}, fh)

    # Save both original and aggregated predictions
    meta.to_csv(ARTIFACT_DIR / "full_pred_elevenclass_fwd1_csv_parallel_simple.csv", index=False)
    meta_aggregated.to_csv(ARTIFACT_DIR / "full_pred_elevenclass_fwd1_csv_parallel_simple_aggregated.csv", index=False)

    # Create probability-weighted matrix using aggregated data
    logger.info("Creating weighted score matrix from aggregated predictions...")
    weighted_matrix = (
        meta_aggregated.pivot(index="date", columns="ticker", values="pred_score")
            .sort_index()
    )
    weighted_matrix.to_csv(ARTIFACT_DIR / "prob_weighted_matrix_elevenclass_fwd1_csv_parallel_simple.csv")
    
    # Also save traditional hard prediction matrix for comparison using aggregated data
    hard_pred_matrix = (
        meta_aggregated.pivot(index="date", columns="ticker", values="pred_cls")
            .sort_index()
    )
    hard_pred_matrix.to_csv(ARTIFACT_DIR / "pred_matrix_elevenclass_fwd1_csv_parallel_simple.csv")

    # Log some examples of weighted scoring
    logger.info("Examples of weighted scoring:")
    sample_indices = np.random.choice(len(meta_aggregated), min(5, len(meta_aggregated)), replace=False)
    for idx in sample_indices:
        row = meta_aggregated.iloc[idx]
        if 'pred_probs' in row and isinstance(row['pred_probs'], list):
            probs = np.array(row['pred_probs'])
            weighted_score = row['pred_score']
            hard_pred = row['pred_cls']
            logger.info(f"Ticker: {row['ticker']}, Date: {row['date']}")
            logger.info(f"  Probabilities: {probs}")
            logger.info(f"  Weighted score: {weighted_score:.4f}")
            logger.info(f"  Hard prediction: {hard_pred} (score: {LABEL_MAP[int(hard_pred)]})")
            logger.info(f"  Verification: {np.dot(probs, LABEL_MAP):.4f}")

    logger.info("Pipeline complete ✔ – artefacts saved in %s", ARTIFACT_DIR)
    logger.info("Key outputs:")
    logger.info(f"  - Full predictions: full_pred_elevenclass_fwd1_csv_parallel_simple.csv")
    logger.info(f"  - Aggregated predictions: full_pred_elevenclass_fwd1_csv_parallel_simple_aggregated.csv")
    logger.info(f"  - Weighted score matrix: prob_weighted_matrix_elevenclass_fwd1_csv_parallel_simple.csv")
    logger.info(f"  - Hard prediction matrix: pred_matrix_elevenclass_fwd1_csv_parallel_simple.csv")


if __name__ == "__main__":
    main()