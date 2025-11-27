import os
import random
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import torch
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader


DATA_DIR = "/data/2023/"
FEATURES_DIR = "/data/transformed_features/"
RETURNS_ROOT = "/data"   # folder that contains year sub-dirs
RETURNS_YEAR = "2023"             # sub-directory name inside root


def select_tickers(n, data_dir=DATA_DIR):
    """
    Randomly selects n tickers from a folder and returns cleaned ticker names.
    Example: 'SZ.002521.csv' → '002521.csv'
    """
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    selected_files = random.sample(all_files, n)
    # Convert 'SZ.002521.csv' → '002521.csv'
    tickers = [f for f in selected_files]
    return tickers

# ────────────────── helpers ──────────────────
def _normalise_ticker(tkr: str, add_ext: bool = True) -> str:
    """
    Map                       Returns
    -----------------------   -------------------
    '000046'                  'SZ.000046.csv'
    '603387'                  'SH.603387.csv'
    'SH.603387'               'SH.603387.csv'
    'SZ.002521.csv'           'SZ.002521.csv'  (unchanged)

    Exchange rule of thumb (A-shares):
      • codes starting with 6/9 → Shanghai (SH.)
      • everything else        → Shenzhen (SZ.)
      • adapt if you also store Beijing 'BJ.' etc.
    """
    if tkr.endswith(".csv"):
        return tkr                        # already in final form

    if "." in tkr:                       # prefixed but no .csv
        return f"{tkr}.csv" if add_ext else tkr

    code = tkr.zfill(6)                  # zero-pad just in case
    prefix = "SH" if code[0] in {"6", "9"} else "SZ"
    return f"{prefix}.{code}.csv" if add_ext else f"{prefix}.{code}"
# ---------------------------------------------


############# Compute and reshape target return #############

def load_target_return(
        ticker: str,
        root: str = RETURNS_ROOT,
        year: str = RETURNS_YEAR
) -> pd.DataFrame:
    fname = _normalise_ticker(ticker)     # <── NEW
    full_path = os.path.join(root, str(year), fname)
    if not os.path.exists(full_path):
        print(f"[WARNING] Ticker {ticker} does not exist in dir")
        return pd.DataFrame()

    df = pd.read_csv(full_path)         # rest unchanged

    # Step 1: Try to identify the datetime column robustly
    datetime_col = None
    for candidate in ['datetime', 'Datetime', 'date', '时间', 'timestamp']:
        if candidate in df.columns:
            datetime_col = candidate
            break

    if datetime_col is None:
        print(f"[WARNING] Ticker {ticker} file is missing a datetime column. Columns are: {df.columns.tolist()}")
        return pd.DataFrame()

    # Step 2: Parse and clean datetime
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    df = df.dropna(subset=[datetime_col])
    df = df.sort_values(datetime_col).set_index(datetime_col)

    # Step 3: Filter trading hours
    t1, t2 = datetime.strptime("09:30:00", "%H:%M:%S").time(), datetime.strptime("11:30:00", "%H:%M:%S").time()
    t3, t4 = datetime.strptime("13:00:00", "%H:%M:%S").time(), datetime.strptime("14:57:00", "%H:%M:%S").time()
    df = df[(df.index.time >= t1) & (df.index.time < t2) | (df.index.time >= t3) & (df.index.time < t4)]

    # Step 4: Resample to 3-minute bins and compute returns
    if "close" not in df.columns:
        print(f"[WARNING] Ticker {ticker} has no 'close' column.")
        return pd.DataFrame()

    df_resampled = df[['close']].shift(-1).resample('3min').last().dropna()
    df_resampled["returns"] = df_resampled["close"].pct_change().fillna(0)
    df_resampled["ticker"] = ticker
    df_resampled.reset_index(inplace=True)
    return df_resampled


def reshape_return_matrix(tickers):
    """
    Loads target returns in parallel for all tickers,
    concatenates into returns_df, and constructs returns_pivot.
    """
    with ThreadPoolExecutor() as executor:
        returns_list = list(executor.map(load_target_return, tickers))

    # Concatenate all returns DataFrames
    returns_df = pd.concat(returns_list)

    # Pivot table to get 'returns_pivot' matrix (Date as index, Tickers as columns)
    returns_pivot = returns_df.pivot(index='datetime', columns='ticker', values='returns')

    return returns_pivot


############# Read Features #############

def ticker_to_parquet_filename(ticker):
    """
    Convert 'SZ.301365.csv' → '301365_transformed.parquet'
    """
    code = ticker.split('.')[1].replace('.csv', '')
    return f"{code}_transformed.parquet"

def getFeatures(ticker, data_dir=FEATURES_DIR):
    """
    Load parquet for one ticker, extract rolling windows over z-score features + target1.
    Returns a list of samples: each is a dict with {ticker, datetime, X, Y}.
    """
    fname = ticker_to_parquet_filename(ticker) # the fname is something like 000006, if so, check for the existence of a unique file ending in 000006.csv

    full_path = os.path.join(data_dir, fname)

    if not os.path.exists(full_path):
        print(f"[WARN] Missing: {fname}")
        return []

    df = pd.read_parquet(full_path)
    if df.empty or "target1" not in df.columns:
        return []

    df['datetime'] = pd.to_datetime(df['datetime'])

    zscore_cols = [col for col in df.columns if col.endswith("_zscore")]
    if not zscore_cols:
        return []


    df[zscore_cols] = df[zscore_cols].ffill()
    X = df[zscore_cols].to_numpy(dtype=np.float32)
    Y = df["target1"].to_numpy(dtype=np.float32)
    dt = df["datetime"].reset_index(drop=True)

    T = 100
    stride = 1
    N = len(X)
    end_indices = np.arange(T, N + 1, stride)
    if len(end_indices) == 0:
        return []

    window_idx = end_indices[:, None] - T + np.arange(T)
    dataX = X[window_idx, :]
    dataY = Y[end_indices - 1]
    dataD = dt[end_indices - 1]


    # Remove rows with any NaNs in X or Y
    mask = ~np.isnan(dataX).any(axis=(1, 2)) & ~np.isnan(dataY)
    dataX = dataX[mask]
    dataY = np.nan_to_num(dataY[mask], nan=0.0)
    dataD = dataD[mask]

    results = []
    for i in range(len(dataX)):
        results.append({
            "ticker": ticker,
            "datetime": dataD.iloc[i],
            "X": dataX[i],
            "Y": dataY[i]
        })

    return results

def helper(ticker):
    return getFeatures(ticker, data_dir=FEATURES_DIR)

def load_parquet_files(tickers, data_dir=FEATURES_DIR):
    """
    Parallel load: Convert tickers to parquet filenames, call getFeatures() in parallel,
    and return all samples.
    """

    with Pool(cpu_count()) as pool:
        all_results = pool.map(helper, tickers)

    # Flatten the list of lists
    all_samples = [sample for sublist in all_results for sample in sublist]
    return all_samples

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        item = self.features[idx]
        key = (item['ticker'], item['datetime'])
        x = torch.tensor(item['X'], dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
        y = torch.tensor(item['Y'], dtype=torch.float32)
        return key, x, y

def collate_fn(batch):
    keys, x_list, y_list = zip(*batch)
    x_batch = torch.stack(x_list)  # (B, 1, seq, feat)
    y_batch = torch.stack(y_list)
    return keys, x_batch, y_batch


def evaluate_model_from_features(model, features, device='cuda', batch_size=4000*4, num_workers=32):
    """
    Efficiently evaluates the model on a list of feature dicts using DataLoader.

    Args:
        model: PyTorch model
        features: list of dicts with keys 'ticker', 'datetime', 'X', 'Y'
        device: device string
        batch_size: evaluation batch size
        num_workers: number of parallel workers for DataLoader

    Returns:
        prediction_dict: {(ticker, datetime): {'x': ..., 'y': ..., 'pred': ...}}
    """
    print("Evaluating Model from Features")
    dataset = FeatureDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    model.eval()
    prediction_dict = {}

    with torch.no_grad():
        for keys, x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu()

            for i, key in enumerate(keys):
                prediction_dict[tuple(key)] = {  # ✅ Ensure key is hashable
                    'x': x_batch[i].squeeze(0).cpu().numpy(),
                    'y': y_batch[i].item(),
                    'pred': preds[i].item() if preds[i].numel() == 1 else preds[i].numpy()
                }

    return prediction_dict


def load_model_and_evaluate(model_name, features, device='cuda'):
    """
    Loads a full model object and evaluates it on the given features.

    Model path: /data/{model_name}/Full_model_{model_name}.pth

    Args:
        model_name: str, e.g., "DeepLOB_v1"
        features: list of dicts with 'ticker', 'datetime', 'X', 'Y'
        device: 'cuda' or 'cpu'

    Returns:
        prediction_dict: {(ticker, datetime): {'x':..., 'y':..., 'pred':...}}
    """
    checkpoint_path = f"/data/{model_name}/Full_model_{model_name}.pth"
    model = torch.load(checkpoint_path, weights_only=False)
    model.to(device)
    model.eval()

    prediction_dict = evaluate_model_from_features(model, features, device=device)
    return prediction_dict


import os, random, logging, numpy as np, pandas as pd, torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader

# ----------------------------------------------------------------------
# 0.  LOGGING SET-UP  (place this near the top of your main script)
# ----------------------------------------------------------------------
LOG_PATH = Path("prediction_stream.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),                     # stdout
        logging.FileHandler(LOG_PATH, mode="w"),     # file
    ],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 1.  Data helpers (unchanged except for logging hooks)
# ----------------------------------------------------------------------
DATA_DIR     = "/data/2023/"
FEATURES_DIR = "/data/transformed_features/"

def select_tickers(n, data_dir=DATA_DIR):
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    sample = random.sample(files, n)
    logger.info(f"Selected {len(sample)} tickers.")
    return sample

def ticker_to_parquet_filename(t):
    return f"{t.split('.')[1].replace('.csv','')}_transformed.parquet"

# ----------------------------------------------------------------------
# 2.  Iterable dataset with per-ticker logs
# ----------------------------------------------------------------------
class FeatureIterableDataset(IterableDataset):
    def __init__(self, tickers, T=100, stride=1):
        self.tickers, self.T, self.stride = tickers, T, stride

    def _windows_for_ticker(self, tkr):
        pq_file = Path(FEATURES_DIR) / ticker_to_parquet_filename(tkr)
        if not pq_file.exists():
            logger.warning(f"Missing parquet: {pq_file.name}")
            return
        df = pd.read_parquet(pq_file, memory_map=True)
        zcols = [c for c in df.columns if c.endswith("_zscore")]
        if "target1" not in df.columns or not zcols:
            return

        df[zcols] = df[zcols].ffill()
        X, Y = df[zcols].to_numpy(np.float32), df["target1"].to_numpy(np.float32)
        D    = pd.to_datetime(df["datetime"]).to_numpy()

        ends = np.arange(self.T, len(X) + 1, self.stride)
        if not len(ends):
            return
        Xv = np.lib.stride_tricks.sliding_window_view(X, (self.T, X.shape[1]))[::self.stride, 0]
        keep = ~np.isnan(Xv).any((1,2)) & ~np.isnan(Y[ends-1])
        Xv, Yv, Dv = Xv[keep], np.nan_to_num(Y[ends-1][keep]), D[ends-1][keep]

        logger.info(f"[{tkr}] yielding {len(Xv):,} windows")
        for x, y, dt in zip(Xv, Yv, Dv):
            yield (tkr, dt, torch.from_numpy(x).unsqueeze(0), torch.tensor(y))
        logger.info(f"[{tkr}] finished")

    def __iter__(self):
        w = torch.utils.data.get_worker_info()
        shard = (
            self.tickers
            if w is None else
            self.tickers[w.id::w.num_workers]
        )
        for tkr in shard:
            yield from self._windows_for_ticker(tkr)

# ----------------------------------------------------------------------
# 3.  Streaming evaluation with batch-level logs
# ----------------------------------------------------------------------

# ----------------------------------------------------------
# 0.  NEW: small helper – turns one batch into (meta, X, y)
# ----------------------------------------------------------
def collate_stream(batch):
    """
    batch = list[ (ticker, dt, X, y), ... ]  length == B
    – ticker and dt are kept as-is in a list (no stacking)
    – X  -> stacked into (B, 1, seq, feat)  tensor
    – y  -> stacked into (B,)               tensor
    """
    tickers, dts, xs, ys = zip(*batch)

    x_batch = torch.stack(xs, 0)                    # (B, 1, seq, feat)
    y_batch = torch.tensor(ys, dtype=torch.float32) # (B,)

    meta = list(zip(tickers, dts))                  # keep raw objects
    return meta, x_batch, y_batch

from collections import defaultdict
import torch, pandas as pd, numpy as np
# sklearn only for the final balanced-accuracy computation in case you prefer it;
# everything inside the loop is pure-PyTorch to avoid extra Python overhead.
from sklearn.metrics import balanced_accuracy_score    # optional

def evaluate_model_streaming(
    model,
    tickers,
    device: str = "cuda",
    batch_size: int = 8_192,
    num_workers: int = 8,
    log_every: int = 50,
):
    """
    Stream batches through *model* and log running MSE / MAE / accuracy /
    balanced-accuracy without ever storing all predictions in memory.
    """
    # ── 0. Data loader ───────────────────────────────────────────────────
    ds = FeatureIterableDataset(tickers)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_stream,
    )

    # ── 1. Metric accumulators (scalar) ──────────────────────────────────
    n_obs        = 0
    sq_err_sum   = 0.0        # Σ (ŷ − y)²
    abs_err_sum  = 0.0        # Σ |ŷ − y|
    n_correct    = 0          # Σ 𝟙{sign(ŷ) == sign(y)}
    per_class_n  = defaultdict(int)   # {-1: Nneg, 1: Npos}
    per_class_tp = defaultdict(int)   # {-1: TPneg, 1: TPpos}

    # ── 2. Main loop ─────────────────────────────────────────────────────
    model.to(device).eval()
    recs = []       # still gather per-window records to return later

    with torch.no_grad():
        for b_idx, (meta, x, y_true) in enumerate(dl, 1):
            x      = x.to(device, non_blocking=True)
            y_true = y_true.to("cpu", non_blocking=True).squeeze()   # (B,)
            y_pred = model(x).cpu().squeeze()                        # (B,)

            # ── 2a. Store per-window record (unchanged) ───────────────
            for (tkr, dt), pred in zip(meta, y_pred):
                recs.append(
                    dict(
                        ticker   = tkr,
                        datetime = pd.Timestamp(dt),
                        returns  = pred.item() if pred.ndim == 0 else pred.numpy(),
                    )
                )

            # ── 2b. Update running sums for regression metrics ────────
            diff         = y_pred - y_true
            sq_err_sum  += torch.sum(diff ** 2).item()
            abs_err_sum += torch.sum(diff.abs()).item()

            # ── 2c. Update classification metrics on sign(returns) ───
            #  > 0  :=  1  (positive)
            #  < 0  := -1  (negative)
            sign_pred = torch.sign(y_pred)
            sign_true = torch.sign(y_true)

            n_correct += torch.count_nonzero(sign_pred == sign_true).item()
            for cls in (-1, 1):
                cls_mask = sign_true == cls
                per_class_n[cls]  += cls_mask.sum().item()
                per_class_tp[cls] += torch.count_nonzero(
                    cls_mask & (sign_pred == cls)
                ).item()

            # ── 2d. Running denominators ──────────────────────────────
            n_obs += y_true.numel()

            # ── 2e. Periodic logging ──────────────────────────────────
            if b_idx % log_every == 0:
                mse_running  = sq_err_sum / n_obs
                mae_running  = abs_err_sum / n_obs
                acc_running  = n_correct / n_obs

                # balanced-accuracy  = ½ (TPR_pos + TPR_neg)
                tpr_pos = (per_class_tp[1]  / per_class_n[1])  if per_class_n[1]  else 0.0
                tpr_neg = (per_class_tp[-1] / per_class_n[-1]) if per_class_n[-1] else 0.0
                bal_acc_running = 0.5 * (tpr_pos + tpr_neg)

                logger.info(
                    f"Batch {b_idx:>6} | windows {n_obs:,} | "
                    f"MSE {mse_running:.6e} | MAE {mae_running:.6e} | "
                    f"Acc {acc_running:.2%} | BalAcc {bal_acc_running:.2%}"
                )

    # ── 3. Final summary log ────────────────────────────────────────────
    mse_final  = sq_err_sum / n_obs
    mae_final  = abs_err_sum / n_obs
    acc_final  = n_correct / n_obs
    tpr_pos    = (per_class_tp[1]  / per_class_n[1])  if per_class_n[1]  else 0.0
    tpr_neg    = (per_class_tp[-1] / per_class_n[-1]) if per_class_n[-1] else 0.0
    bal_acc_final = 0.5 * (tpr_pos + tpr_neg)

    logger.info(
        f"All done – {n_obs:,} windows | "
        f"MSE {mse_final:.6e} | MAE {mae_final:.6e} | "
        f"Acc {acc_final:.2%} | BalAcc {bal_acc_final:.2%}"
    )

    # ── 4. Return predictions dataframe (unchanged) ────────────────────
    return pd.DataFrame(recs)
# ------------
def prediction_dict_to_pivot(prediction_dict):
    """
    Converts prediction_dict into a pivoted DataFrame with:
    index   = datetime
    columns = ticker
    values  = pred (used as 'returns' here)
    """
    records = []

    for (ticker, dt), data in prediction_dict.items():
        pred = data['pred']
        if isinstance(pred, (np.ndarray, list)):
            pred = pred  # assume first value if vector

        records.append({
            'ticker': ticker,
            'datetime': dt,
            'returns': pred
        })

    returns_df = pd.DataFrame(records)
    returns_pivot = returns_df.pivot(index='datetime', columns='ticker', values='returns')

    return returns_pivot
