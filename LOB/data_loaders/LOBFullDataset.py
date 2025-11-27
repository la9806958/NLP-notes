import os
import gc  # <-- Import garbage collector
import bisect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from multiprocessing import get_context, cpu_count
from tqdm import tqdm
import polars as pl
from scipy.stats import mstats

########################################
# Rank Helper
########################################
def rank_percentile_stream(value: float, sorted_array: np.ndarray) -> float:
    """
    Given a sorted array of training targets (ascending),
    return the rank-percentile of 'value' in [0, 1].
    """
    idx = bisect.bisect_left(sorted_array, value)
    return idx / len(sorted_array)

def get_top_tickers_from_log(log_path: str, n: int) -> list:
    """
    Extracts the top `n` tickers from a volume summary log file.
    Assumes each line in the log starts with a ticker name, followed by volume data.
    """
    df = pd.read_csv(log_path, delim_whitespace=True, header=None, usecols=[0], names=["ticker"])

    
    df["ticker"] = df["ticker"].str.replace("_stitched.csv", "").str.zfill(6)
    return df["ticker"].head(n).tolist()

def filter_parquet_files_by_tickers(data_dir: str, top_tickers: list) -> list:
    """
    Returns a filtered list of parquet files corresponding to the top tickers.
    """
    all_files = os.listdir(data_dir)

    # Skip header and split by comma
    parsed_rows = [line.split(',') for line in top_tickers[1:]]
    
    # Create DataFrame
    df = pd.DataFrame(parsed_rows, columns=["ticker", "volume_ask_1_sum", "volume_bid_1_sum", "total_volume"])
    
    # Convert numeric columns to float
    df[["volume_ask_1_sum", "volume_bid_1_sum", "total_volume"]] = df[["volume_ask_1_sum", "volume_bid_1_sum", "total_volume"]].astype(float)
    
    # If you want just the ticker column as a Series:
    ticker_series = df["ticker"]
    
    filtered = [f for f in all_files if f.endswith("_transformed.parquet") and any(ticker in f for ticker in ticker_series)]
    return filtered

########################################
# LOBFullDataset
########################################
class LOBFullDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        rank_transform: bool = False,
        train_mean: float = None,
        train_std: float = None,
        normalize_y: bool = True, 
        sorted_train_targets: np.ndarray = None,
        num_workers: int = None
    ):
        """
        Arguments:
            data_dir: directory with dataset (parquet) files
            split: one of {'train','val','test'}
            rank_transform: boolean to enable/disable rank transform on Y
            train_mean, train_std: used for z-score transform (only needed at inference time)
            sorted_train_targets: used for rank transform at val/test
            num_workers: override the default # of processes for parallel loading
        """
        super().__init__()
        self.normalize_y = normalize_y
        self.data_dir = data_dir
        self.split = split
        self.rank_transform = rank_transform
        self.train_mean = train_mean
        self.train_std = train_std
        self.sorted_train_targets = sorted_train_targets
        # Limit parallels to half CPU or user-specified
        self.num_workers = num_workers if num_workers else max(1, cpu_count() // 2)

        self.X, self.Y, self.computed_mean, self.computed_std, self.ticker_list = self._load_all_data()
        # per-sample ticker list (already built)
        
        # unique universe for the rest of the pipeline  ← NEW
        self.tickers = sorted(set(self.ticker_list))      # or np.unique(ticker_list)

        # Store computed mean/std if this is the training set and rank transform is off
        if not self.rank_transform and split == 'train':
            self.train_mean = self.computed_mean
            self.train_std = self.computed_std
    
    def _load_file_pandas(self, file: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        A vectorized loader that processes each month individually, then concatenates
        the results. This helps ensure any month-by-month slicing/aggregation is handled
        cleanly (e.g., rolling windows won't cross month boundaries).
        """
        ticker = file.replace("_transformed.parquet", "")
        
        path = os.path.join(self.data_dir, file)
        try:
            # 1) Read parquet file
            df = pd.read_parquet(path)
            # Convert 'datetime' if necessary
            df['datetime'] = pd.to_datetime(df['datetime'])
            df['month'] = df['datetime'].dt.month
            #print(df)
            # 2) Split by month for train/val/test
            if self.split == 'train':
                df = df[df['month'] % 2 == 1].sort_values("datetime")
            else:
                even_df = df[df['month'] % 2 == 0].sort_values("datetime")
                mid = len(even_df) // 2
                if self.split == 'val':
                    df = even_df.iloc[:mid]
                elif self.split == 'test':
                    df = even_df.iloc[mid:]
    
            # If empty or missing target, return None
            if df.empty or "target1" not in df.columns:
                return None
    
            # 3) Identify z-score columns
            zscore_cols = [col for col in df.columns if col.endswith("_zscore")]
            if not zscore_cols:
                return None
    
            # Drop rows with NaNs in z-score columns
            #df = df.dropna(subset=zscore_cols)
            df.loc[:, zscore_cols] = df[zscore_cols].ffill()
            
            if df.empty:
                return None
    
            # We'll aggregate each month's data here
            dataX_all = []
            dataY_all = []
            # 4) Unique months in the (already-split) df
            months = sorted(df['month'].unique())
    
            for month_val in months:
                sub_df = df[df['month'] == month_val]
                if sub_df.empty:
                    continue
    
                # Convert this month's data to NumPy
                X = sub_df[zscore_cols].to_numpy(dtype=np.float32)
                Y = sub_df["target1"].to_numpy(dtype=np.float32)
    
                # Rolling window parameters
                T = 100
                stride = 10
                N = len(X)
    
                # The "end_indices" are positions where each window ends
                end_indices = np.arange(T, N + 1, stride)  # e.g. [100, 110, 120, ...]
                M = len(end_indices)
                if M == 0:
                    # No valid windows for this month, skip
                    continue
    
                # Build the 2D array of indices for vectorized rolling
                window_idx = end_indices[:, None] - T + np.arange(T)  # shape: (M, T)
    
                # Gather the data in one shot
                dataX_month = X[window_idx, :]        # shape: (M, T, D)
                dataY_month = Y[end_indices - 1]      # shape: (M,)
    
                # Remove rows with NaNs
                mask = ~np.isnan(dataX_month).any(axis=(1, 2))
                dataX_month = dataX_month[mask]
                dataY_month = dataY_month[mask]
    
                # Check dataY for NaNs just in case
                mask_y = ~np.isnan(dataY_month)
                dataX_month = dataX_month[mask_y]
                dataY_month = np.nan_to_num(dataY_month[mask_y], nan=0.0)
    
                # Append to the aggregated list if there's anything left
                if len(dataX_month) > 0:
                    dataX_all.append(dataX_month)
                    dataY_all.append(dataY_month)
                #print(dataX_all)
                #print(dataX_all)
                
            # If no data across all months, return None
            if not dataX_all:
                return None
    
            # Concatenate results from all months
            dataX_agg = np.concatenate(dataX_all, axis=0)
            dataY_agg = np.concatenate(dataY_all, axis=0)
    
            # Return final arrays
            return (dataX_agg, dataY_agg, ticker) if len(dataX_agg) > 0 else None
    
        except Exception as e:
            print(f"❌ Error loading file (vectorized): {e}")
            return None

    def _load_file_pandas2(self, file: str) -> tuple[np.ndarray, np.ndarray, str] | None:
        """
        Order-book loader with odd/even *time-block* splitting.
    
        * `nBlocks` contiguous chronological blocks over the file’s dates
          (≈ equal #days per block).
        * Odd-indexed blocks ➜ train.
        * Even-indexed blocks ➜ first half val, second half test.
        * Each block is processed independently: z-score ffill + rolling windows.
        """
    
        ticker = file.replace("_transformed.parquet", "")
        path   = os.path.join(self.data_dir, file)
    
        # ------------------------------------------------------------------ cfg
        nBlocks = 60            # total blocks
        T       = 100           # window length
        stride  = 10            # hop
        # ----------------------------------------------------------------------
    
        try:
            # ── 1) Read parquet & basic prep ────────────────────────────────
            df = pd.read_parquet(path)
            if "datetime" not in df.columns:
                raise KeyError("'datetime' column missing")
    
            df["datetime"] = pd.to_datetime(df["datetime"])
            df["date"]     = df["datetime"].dt.floor("D")
    
            # ── 2) Build `nBlocks` equal-length chronological blocks ────────
            unique_days = np.sort(df["date"].unique())
            if len(unique_days) < nBlocks:
                return None                              # skip short files
    
            edges      = np.linspace(0, len(unique_days), nBlocks + 1, dtype=int)
            day_number = np.searchsorted(unique_days, df["date"].to_numpy())
            df["block"] = np.searchsorted(edges, day_number, side="right") - 1
    
            # ── 3) Odd / even split ─────────────────────────────────────────
            odd_blocks   = set(range(1, nBlocks, 2))         # 1,3,5,…
            even_blocks  = list(range(0, nBlocks, 2))        # 0,2,4,…
            half         = len(even_blocks) // 2
    
            if   self.split == "train":
                keep_blocks = odd_blocks
            elif self.split == "val":
                keep_blocks = set(even_blocks[:half])
            elif self.split == "test":
                keep_blocks = set(even_blocks[half:])
            else:
                raise ValueError(f"Unknown split: {self.split}")
    
            df = df[df["block"].isin(keep_blocks)].sort_values("datetime")
            if df.empty or "target1" not in df.columns:
                return None
    
            # ── 4) z-score columns ─────────────────────────────────────────
            zscore_cols = [c for c in df.columns if c.endswith("_zscore")]
            if not zscore_cols:
                return None
    
            # ── 5) Process each block separately ───────────────────────────
            dataX_all, dataY_all = [], []
    
            for blk in sorted(keep_blocks):
                blk_df = df[df["block"] == blk]
                if blk_df.empty:
                    continue
    
                # forward-fill *within this block only*
                blk_df.loc[:, zscore_cols] = blk_df[zscore_cols].ffill()
                if blk_df[zscore_cols].isna().all().all():
                    continue
    
                X = blk_df[zscore_cols].to_numpy(np.float32)
                Y = blk_df["target1"].to_numpy(np.float32)
    
                N = len(X)
                end_idx = np.arange(T, N + 1, stride)
                if end_idx.size == 0:
                    continue
    
                idx_matrix = end_idx[:, None] - T + np.arange(T)
                dataX_blk  = X[idx_matrix]                 # (M,T,D)
                dataY_blk  = Y[end_idx - 1]                # (M,)
    
                mask   = ~np.isnan(dataX_blk).any(axis=(1, 2))
                dataX_blk = dataX_blk[mask]
                dataY_blk = np.nan_to_num(dataY_blk[mask], nan=0.0)
    
                if len(dataX_blk):
                    dataX_all.append(dataX_blk)
                    dataY_all.append(dataY_blk)
    
            if not dataX_all:
                return None
    
            dataX = np.concatenate(dataX_all, axis=0)
            dataY = np.concatenate(dataY_all, axis=0)
    
            return (dataX, dataY, ticker) if len(dataX) else None
    
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            return None
    def _load_file_wrapper(self, file: str) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Simple wrapper that chooses between polars or pandas loader.
        """
        try:
            return self._load_file_pandas(file)
        except Exception as e:
            print(f"[ERROR] Failed loading {file}: {e}")
            return None
    

    def _load_all_data(self) -> Tuple[np.ndarray, np.ndarray, float, float, List[str]]:
        """
        Load data from all files in parallel using multiprocessing.
        Also tracks ticker for each sample and enables per-sample loss weighting.
        """
        log_path = "/home/lichenhui/data_loaders/volume_summary.log"
        top_n = 3000
        top_tickers = get_top_tickers_from_log(log_path, top_n)
        files = filter_parquet_files_by_tickers(self.data_dir, top_tickers)
    
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        ticker_list: List[str] = []
    
        ctx = get_context("spawn")
        with ctx.Pool(processes=self.num_workers) as pool:
            for result in tqdm(
                pool.imap(self._load_file_wrapper, files, chunksize=max(1, len(files) // self.num_workers // 4)),
                total=len(files),
                desc=f"Loading {self.split.upper()} data in parallel"
            ):
                if result is not None:
                    dataX, dataY, ticker = result  # Make sure _load_file_wrapper returns ticker
                    X_list.append(dataX)
                    Y_list.append(dataY)
                    ticker_list.extend([ticker] * len(dataY))  # One ticker per row
    
                del result
                gc.collect()
    
        if len(X_list) == 0:
            return (
                np.empty((0, 1, 100, 1), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                None,
                None,
                []
            )
    
        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
    
        computed_mean, computed_std = None, None
    
        # ---------------------------------------------------------------------
        # ---- Rank-percentile transform *with* verbose stats -----------------
        # ---------------------------------------------------------------------
        if self.rank_transform:
            if self.split == 'train':
                # 1) Fit
                self.sorted_train_targets = np.sort(Y_all).astype(np.float32)
        
                # 2) Transform
                from scipy.stats import rankdata
                Y_all = (rankdata(Y_all, method="average") - 0.5) / len(Y_all)
        
                # 3) Log
                print(f"[{self.split.upper()}] Rank-percentile stats:")
                print("  ↳ mean = {:.4f}, std = {:.4f}, min = {:.2f}, max = {:.2f}".format(
                      np.mean(Y_all), np.std(Y_all), np.min(Y_all), np.max(Y_all)))
        
            else:  # val / test
                if self.sorted_train_targets is None or len(self.sorted_train_targets) == 0:
                    raise ValueError(
                        f"[{self.split.upper()}] rank_transform=True but "
                        "sorted_train_targets not supplied from the training split."
                    )
        
                idx = np.searchsorted(
                    self.sorted_train_targets,
                    Y_all,
                    side="left"
                ).astype(np.float32)
                Y_all = (idx + 0.5) / len(self.sorted_train_targets)
        
                # Log
                print(f"[{self.split.upper()}] Rank-percentile stats:")
                print("  ↳ mean = {:.4f}, std = {:.4f}, min = {:.2f}, max = {:.2f}".format(
                      np.mean(Y_all), np.std(Y_all), np.min(Y_all), np.max(Y_all)))
        # ---------------------------------------------------------------------

        elif self.normalize_y:
            if self.split == 'train':
                print(f"[{self.split.upper()}] Y_all shape: {Y_all.shape}")
                Y_all_winsorized = Y_all
                computed_mean = float(np.mean(Y_all_winsorized))
                computed_std = float(np.std(Y_all_winsorized))
    
                print(f"[TRAIN] Target normalization stats (winsorized):")
                print(f"  ↳ mean = {computed_mean:.6f}, std = {computed_std:.6f}")
    
                if computed_std > 1e-12:
                    Y_all = (Y_all_winsorized ) / (computed_std + 1e-12)
                    self.train_mean = computed_mean
                    self.train_std = computed_std
    
                    print("After normalization (no clipping):")
                    print("  ↳ mean = {:.4f}, std = {:.4f}, min = {:.2f}, max = {:.2f}".format(
                        np.mean(Y_all), np.std(Y_all), np.min(Y_all), np.max(Y_all)))
                else:
                    print("  ⚠️ Warning: Std too small — skipping normalization.")
    
            else:
                print(f"[{self.split.upper()}] Y_all shape: {Y_all.shape}")
                if self.train_mean is not None and self.train_std is not None and abs(self.train_std) > 1e-12:
                    print(f"[{self.split.upper()}] Applying train normalization stats:")
                    print(f"  ↳ mean = {self.train_mean:.6f}, std = {self.train_std:.6f}")
                    Y_all = (Y_all) / (self.train_std + 1e-12)
    
                    print("After normalization (no clipping):")
                    print("  ↳ mean = {:.4f}, std = {:.4f}, min = {:.2f}, max = {:.2f}".format(
                        np.mean(Y_all), np.std(Y_all), np.min(Y_all), np.max(Y_all)))
                else:
                    print(f"[{self.split.upper()}] ⚠️ Warning: No valid train mean/std available — skipping normalization.")


        else:
            computed_mean = None
            computed_std = None

        
        if X_all.ndim == 3:
            X_all = X_all[:, None, :, :]  # [B, 1, T, D]
    
        return X_all, Y_all, computed_mean, computed_std, ticker_list
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert arrays to torch.Tensors on-the-fly
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx]), self.ticker_list[idx]
