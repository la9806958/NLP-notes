#!/usr/bin/env python3
"""
End‑to‑end **Train → Inference → Back‑test** pipeline that now also
**dynamically builds the returns matrix** (`returns_pivot.parquet`) from the
same ticker universe that each model sees at inference‑time.

Stages per model
----------------
1. **train_model**    – supervised training + checkpointing
2. **infer_signals**  – streaming inference → `<run_dir>/features_pivot.parquet`
3. **build_returns**  – reshape raw forward‑return files for *that* ticker set
4. **run_backtest**   – diagonal‑Σ Max‑Sharpe → `<run_dir>/weights.csv`

Run examples
------------
$ python pipeline.py                # full pipeline all models
$ python pipeline.py MLPLOB         # single model
$ python pipeline.py --skip-train   # reuse existing checkpoints
"""

from __future__ import annotations
import argparse, gc, logging, os, sys
from pathlib import Path
from typing import Callable, Dict, Sequence, List

import pandas as pd
import numpy as np
import psutil, torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ─────────────────────────────── CONFIG ───────────────────────────────
DATA_ROOT        = "transformed_features"
BACKTEST_CACHE   = Path("backtest_cache")
BACKTEST_CACHE.mkdir(exist_ok=True)
RETURNS_PARQUET  = BACKTEST_CACHE / "returns_pivot.parquet"  # built on the fly
BATCH_TRAIN_LG   = 32_768
BATCH_TRAIN_SM   = 8_192
BATCH_INFER      = 16_384
EPOCHS           = 5
LR               = 1e-4
WEIGHT_DECAY     = 1e-3
DEVICE           = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEQ_LEN          = 100
FEATURE_DIM      = 142
LOG_FMT          = "%(asctime)s [%(levelname)s] %(message)s"

# ─────────────────────────────── imports ──────────────────────────────
from data_loaders.LOBFullDataset import LOBFullDataset
from train.trainer import batch_gd
from backtest.optimizer import Optimizer
from backtest.backtestRunner import evaluate_model_streaming, reshape_return_matrix, select_tickers  # ← existing helpers

from models.DeepLOB     import *
from models.MLPLOB      import *
from models.TLOB        import *

# ─────────────────────────────── helpers ──────────────────────────────
logging.basicConfig(level=logging.INFO, format=LOG_FMT, stream=sys.stdout)

def log_mem(tag=""):
    m = psutil.Process(os.getpid()).memory_info().rss / 1024**3
    logging.info(f"[{tag}] RSS ≈ {m:.2f} GB")

# ---------- volume‑weighted loss ----------
VOLUME_CSV = Path("data_loaders/volume_summary.log")
if VOLUME_CSV.exists():
    _df = pd.read_csv(VOLUME_CSV)
    _df["w"] = _df["total_volume"].rank(pct=True).clip(lower=0.1)
    _df["ticker"] = _df["file"].str.extract(r"(\d+)_stitched\.csv")
    TICKER_W = dict(zip(_df["ticker"].astype(str), _df["w"]))
else:
    TICKER_W = {}

def weighted_l1(pred: torch.Tensor, tgt: torch.Tensor, tkrs: Sequence[str]):
    pred, tgt = pred.view(-1), tgt.view(-1)
    w = torch.as_tensor([TICKER_W.get(str(t), .1) for t in tkrs],
                        dtype=pred.dtype, device=pred.device)
    return (F.l1_loss(pred, tgt, reduction="none") * w).mean()

# ---------- dataloaders ----------

def loaders(batch_size: int):
    tr = LOBFullDataset(DATA_ROOT, "train", rank_transform=False, normalize_y=True)
    va = LOBFullDataset(DATA_ROOT, "val",   rank_transform=False, normalize_y=True,
                        train_mean=tr.train_mean, train_std=tr.train_std,
                        sorted_train_targets=tr.sorted_train_targets)
    te = LOBFullDataset(DATA_ROOT, "test",  rank_transform=False, normalize_y=True,
                        train_mean=tr.train_mean, train_std=tr.train_std,
                        sorted_train_targets=tr.sorted_train_targets)
    kw = dict(num_workers=32, pin_memory=True, persistent_workers=True)
    return (
        DataLoader(tr, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(te, batch_size=BATCH_INFER, shuffle=False, **kw),
    )

# ---------- model registry ----------
class SimpleBaselineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SEQ_LEN * FEATURE_DIM, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        if x.ndim == 4 and x.size(1) == 1: x = x.squeeze(1)
        return self.mlp(x)

MODEL_REGISTRY: Dict[str, Callable[[], nn.Module]] = {
    "DeepLOB":    lambda: DeepLOB(y_len=1, feature_dim_in=FEATURE_DIM),
    "MLPLOB":     lambda: MLPLOB(num_features=FEATURE_DIM, seq_size=SEQ_LEN),
    "TLOB":       lambda: TLOB(num_features=FEATURE_DIM, seq_size=SEQ_LEN),
}

# ─────────────────────── build returns matrix ────────────────────────

def build_returns_matrix(universe: List[str]) -> Path:
    """Using helpers in **backtest.backtestRunner**, reshape forward‑return CSVs
    into a pivoted (date × ticker) parquet aligned with *universe* and cache it.
    Only rebuilt when the cached file is missing *or* missing tickers."""
    if RETURNS_PARQUET.exists():
        existing = pd.read_parquet(RETURNS_PARQUET).columns.astype(str)
        if set(universe).issubset(existing):
            logging.info("[returns] cached matrix covers universe – reuse")
            return RETURNS_PARQUET
        logging.info("[returns] cache missing tickers – rebuilding")

    # 1) reshape raw files → DataFrame
    returns_df = reshape_return_matrix(universe)  # heavy I/O helper already optimised
    returns_df.to_parquet(RETURNS_PARQUET, compression="snappy")
    logging.info(f"[returns] built {RETURNS_PARQUET}  {returns_df.shape}")
    return RETURNS_PARQUET

# ─────────────────────────────── stages ──────────────────────────────

# ────────────────────────────────────────────────────────────────
def train_model(model_key: str,
                run_dir: Path,
                train_dl: DataLoader,
                val_dl:   DataLoader):

    ckpt = run_dir / "ckpt.pth"
    best = run_dir / "best.pth"

    model = MODEL_REGISTRY[model_key]().to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    opt  = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = lambda p, t, tk: weighted_l1(p, t, tk)

    # ------------------ call now matches new batch_gd signature ------------------
    batch_gd(
        model,
        crit,
        opt,
        train_dl,
        val_dl,
        epochs=EPOCHS,                       # ← keyword, first after loaders
        checkpoint_path=str(ckpt),
        best_full_model_path=str(best)
    )
    # -----------------------------------------------------------------------------

    return best
def infer_signals(model_path: Path, run_dir: Path, test_dl) -> Path:
    raw = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = raw if isinstance(raw, nn.Module) else raw["model_state"]
    if not isinstance(model, nn.Module):
        model = MODEL_REGISTRY[raw["arch"]]()
        model.load_state_dict(raw)
    model.to(DEVICE).eval()

    signals = evaluate_model_streaming(model, test_dl.dataset.tickers, device=DEVICE)
    feat_pivot = signals.pivot(index="datetime", columns="ticker", values="returns")
    out_path = run_dir / "features_pivot.parquet"
    feat_pivot.to_parquet(out_path, compression="snappy")
    return out_path

def run_backtest(features_parquet: Path, returns_parquet: Path, run_dir: Path):
    returns = pd.read_parquet(returns_parquet)
    alpha   = pd.read_parquet(features_parquet)

    common_idx = alpha.index.intersection(returns.index)
    common_col = alpha.columns.intersection(returns.columns)
    alpha, returns = alpha.loc[common_idx, common_col], returns.loc[common_idx, common_col]

    opt = Optimizer(risk_budget=0.10, gme_limit=2)
    alpha_sm = opt.smooth_alpha_cache(alpha, halflife=5)

    lookback = 2000
    σ2 = (returns.rolling(lookback).var(ddof=0) * lookback).add(1e-10)

    recs = {}
    for dt in alpha_sm.index[lookback:]:
        var = σ2.loc[dt].dropna(); r = alpha_sm.loc[dt, var.index].values
        w   = np.clip(r / var.values, 0, None)
        if w.sum() == 0: continue
        recs[dt] = w / w.sum()
    weights = pd.DataFrame.from_dict(recs, orient="index", columns=var.index)
    csv = run_dir / "weights.csv"; weights.to_csv(csv)
    return csv

# ─────────────────────────────── main ────────────────────────────────
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", help="single model key to run")
    parser.add_argument("--skip-train", action="store_true", help="reuse best.pth if present")
    args = parser.parse_args()

    todo = [args.model] if args.model else list(MODEL_REGISTRY)
    for key in todo:
        run_dir = Path(key); run_dir.mkdir(exist_ok=True)
        log_mem(f"{key} start")

        # 1) loaders
        bs = BATCH_TRAIN_SM if key == "TLOB" else BATCH_TRAIN_LG
        tr_dl, va_dl, te_dl = loaders(bs)

        # 2) ensure returns matrix for *this* ticker universe
        universe = list(map(str, te_dl.dataset.tickers))
        rpq = build_returns_matrix(universe)

        # 3) train
        best_ckpt = run_dir / "best.pth"
        if not args.skip_train or not best_ckpt.exists():
            best_ckpt = train_model(key, run_dir, tr_dl, va_dl)
        else:
            logging.info(f"[skip‑train] Using existing {best_ckpt}")

        # 4) inference
        #fp_parquet = infer_signals(best_ckpt, run_dir, te_dl)

        # 5) back‑test
        #weights_csv = run_backtest(fp_parquet, rpq, run_dir)
        #logging.info(f"✔ {key}: back‑test weights → {weights_csv}")

        #gc.collect(); torch.cuda.empty_cache(); log_mem(f"{key} done")
