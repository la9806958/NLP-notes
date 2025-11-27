#!/usr/bin/env python3
"""
LightGBM **5-class classification** for **21‑day** forward returns.

Modified version to work with CSV input from portfolio_50q_full_run_output/portfolio_results_full_2023_2025.csv
OPTIMIZED with multiprocessing for dataset preparation.

New in this revision
────────────────────
• Uses LightGBM classifier for 5-class quantile classification
• Logs the **date cut‑offs** between Train, Validation, and Test splits so you
  can verify the chronological partitioning.
• Modified to load CSV format instead of JSON
• Added multiprocessing for dataset preparation to use all CPU cores
"""

# pip install lightgbm
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

import json, pickle, logging, csv, math
from pathlib import Path
from datetime import datetime, timedelta  # noqa: F401 – retained for completeness
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ────────────────────────────── Logging ──────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────────── Constants ────────────────────────────
N_CLASSES    = 5                                 # 5 class classification
LABEL_MAP    = np.arange(-2, 3)                  # 0→−2, 1→−1, 2→0, 3→1, 4→2
DAYS_FWD     = 21                                # 21 trading days
# Date-based splits
TRAIN_END_DATE = "2024-02-01"
VAL_START_DATE = "2024-03-01"
VAL_END_DATE = "2024-05-01"
TEST_START_DATE = "2024-06-01"
ARTIFACT_DIR = Path("/home/lichenhui")          # change if needed

# ────────────────────────────── LightGBM Training Function ───────────

def train_lgbm_classifier(X_tr, y_tr, X_va, y_va, feature_names=None):
    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
    valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set, feature_name=feature_names, free_raw_data=False)

    params = dict(
        objective="multiclass",
        num_class=N_CLASSES,
        metric="multi_logloss",
        boosting_type="gbdt",
        num_leaves=256,          # plenty of capacity for 274k rows
        max_depth=-1,
        learning_rate=0.03,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        min_data_in_leaf=200,    # regularization for stability
        lambda_l1=0.0,
        lambda_l2=5.0,
        verbosity=-1,
        seed=42,
    )
    model = lgb.train(
      params, train_set, num_boost_round=12000,
      valid_sets=[valid_set],
      callbacks=[lgb.early_stopping(400, first_metric_only=True),
                 lgb.log_evaluation(200)]
    )

    # Metrics on val
    va_pred_proba = model.predict(X_va, num_iteration=model.best_iteration)
    va_pred = np.argmax(va_pred_proba, axis=1)
    val_acc = accuracy_score(y_va, va_pred)
    val_report = classification_report(y_va, va_pred, output_dict=True)
    return model, dict(val_acc=val_acc, val_report=val_report)

# ────────────────────────────── IO helpers ───────────────────────────

def _prep_scores(scores):
    """Ensure 50-d float array in [0, 10], shape (50,)."""
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.shape[0] != 50:
        raise ValueError(f"Each score vector must have 50 dims, got {x.shape[0]}")
    # clip to [0, 10] just in case upstream deviates
    np.clip(x, 0.0, 10.0, out=x)
    return x
    
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
    scaler = StandardScaler(with_mean=True, with_std=True)
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
    stats_path = ARTIFACT_DIR / "scaling_stats_lgbm_5class.csv"
    stats.to_csv(stats_path, index=False)
    logger.info("Saved scaling diagnostics to %s", stats_path)

    return scaler, Xtr, Xva, Xte, Xall


def calc_forward_return(px: pd.DataFrame, ticker: str,
                        ts: str, days_fwd: int = DAYS_FWD):
    try:
        ts = pd.to_datetime(ts)
        if ts not in px.index:
            fut = px.index[px.index >= ts]
            if not len(fut):
                return None
            ts = fut[0]

        if ticker not in px:
            return None

        p0 = px.at[ts, ticker]
        if np.isnan(p0):
            return None

        fut = px.index[px.index > ts]
        if len(fut) < days_fwd:
            return None

        ts1 = fut[days_fwd - 1]
        p1 = px.at[ts1, ticker]
        if np.isnan(p1):
            return None
        
        # Calculate raw forward return without normalization
        raw_return = (p1 - p0) / p0
        return raw_return
        
    except Exception as exc:
        return None

def process_row_chunk(chunk_data):
    """Process a chunk of rows for multiprocessing."""
    chunk_rows, px_dict, px_index = chunk_data
    
    # Reconstruct DataFrame from dict (needed for multiprocessing)
    px = pd.DataFrame(px_dict, index=px_index)
    
    results = []
    for row_data in chunk_rows:
        r = calc_forward_return(px, row_data["ticker"], row_data["date"])
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

    # === Standardize 50 features feature-wise using TRAIN stats ===
    scaler, Xtr_s, Xva_s, Xte_s, Xall_s = fit_scaler_and_transform(
        X_tr, X_va, X_te, X
    )

    # Quantile bins from training returns (labels)
    q        = quantile_bounds(y_tr, N_CLASSES)
    y_tr_lbl = assign_labels(y_tr, q)
    y_va_lbl = assign_labels(y_va, q)
    y_te_lbl = assign_labels(y_te, q)

    # === Train LightGBM classifier on TRAIN, validate on HOLD-OUT VAL ===
    feature_names = [f"q{i}" for i in range(1, 51)]
    model, metrics = train_lgbm_classifier(Xtr_s, y_tr_lbl, Xva_s, y_va_lbl, feature_names)
    
    logger.info("Validation metrics: %s", metrics)

    # Predict everywhere with the same scaler/model - using probability-weighted scores
    pred_proba_all = model.predict(Xall_s, num_iteration=model.best_iteration)
    pred_lbl_all = np.argmax(pred_proba_all, axis=1)
    
    # Calculate probability-weighted scores
    # Each class corresponds to a score in LABEL_MAP (e.g., [-2, -1, 0, 1, 2])
    weighted_scores = np.dot(pred_proba_all, LABEL_MAP)
    
    meta["pred_cls"]   = pred_lbl_all
    meta["pred_score"] = weighted_scores
    meta["pred_probs"] = [prob.tolist() for prob in pred_proba_all]  # Save full probability distributions

    # Actual class where available
    actual_lbl = np.full_like(pred_lbl_all, np.nan, dtype=float)
    actual_lbl[labelled] = assign_labels(y[labelled], q)
    meta["actual_cls"] = actual_lbl

    # Test accuracy on labelled TEST split (hold-out)
    test_pred_proba = model.predict(Xte_s, num_iteration=model.best_iteration)
    test_pred = np.argmax(test_pred_proba, axis=1)
    test_acc = accuracy_score(y_te_lbl, test_pred)
    
    logger.info("Test accuracy (hold-out): %.4f", test_acc)
    
    # Generate detailed test report
    test_report = classification_report(y_te_lbl, test_pred)
    logger.info("\nTest classification report (hold-out):\n%s", test_report)
    with open(ARTIFACT_DIR / "test_classification_report.txt", "w") as f:
        f.write(test_report)
    logger.info("Saved test classification report to %s", ARTIFACT_DIR / "test_classification_report.txt")

    # ---- Persist artefacts
    # Save LightGBM model and other artifacts
    model.save_model(str(ARTIFACT_DIR / "lgbm_model.txt"))
    
    with open(ARTIFACT_DIR / "lgbm_classifier_csv_parallel.pkl", "wb") as fh:
        pickle.dump({"scaler": scaler, "quantiles": q, "metrics": metrics}, fh)

    meta.to_csv(ARTIFACT_DIR / "full_pred_lgbm_5class_fwd21_csv_parallel.csv", index=False)

    # Create probability-weighted score matrix
    (
        meta.groupby(["date", "ticker"], as_index=False)["pred_score"].mean()
            .pivot(index="date", columns="ticker", values="pred_score")
            .sort_index()
            .to_csv(ARTIFACT_DIR / "prob_weighted_matrix_lgbm_5class_fwd21_csv_parallel.csv")
    )
    
    # Also save traditional hard prediction matrix for comparison
    (
        meta.groupby(["date", "ticker"], as_index=False)["pred_cls"].mean()
            .pivot(index="date", columns="ticker", values="pred_cls")
            .sort_index()
            .to_csv(ARTIFACT_DIR / "pred_matrix_lgbm_5class_fwd21_csv_parallel.csv")
    )

    # Save test metrics
    test_metrics = {
        "test_accuracy": test_acc,
        "test_report": classification_report(y_te_lbl, test_pred, output_dict=True),
        "val_metrics": metrics
    }
    with open(ARTIFACT_DIR / "test_metrics_lgbm_5class.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Pipeline complete ✔ – artefacts saved in %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()