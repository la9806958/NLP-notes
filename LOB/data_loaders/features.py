import os
import numpy as np
import pandas as pd
from typing import List
import torch
import re

########################################
# Rolling Z-Score Helper
########################################


def compute_rolling_zscore(df: pd.DataFrame, columns: List[str], horizon: int = 10) -> pd.DataFrame:
    """
    Compute rolling z-scores using a fixed-size rolling window.

    Parameters:
    - df: Input DataFrame
    - columns: List of columns to compute z-scores for
    - horizon: Rolling window size (number of rows)

    Returns:
    - DataFrame with additional columns for rolling z-scores
    """
    # Calculate rolling mean and standard deviation for specified columns
    rolling_mean = df[columns].rolling(window=horizon, min_periods=1).mean()
    rolling_std = df[columns].rolling(window=horizon, min_periods=1).std()

    # Replace zeros in rolling_std to avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    # Compute z-scores
    z_scores = (df[columns] - rolling_mean) / rolling_std

    # Rename z-score columns
    z_scores.columns = [f"{col}_{horizon}_zscore" for col in columns]

    # Concatenate original DataFrame with z-score columns
    return pd.concat([df, z_scores], axis=1)



########################################
# Feature Engineering Helpers
########################################

def compute_order_imbalance_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Computes imbalance features for each level and aggregated across levels.
    - imbalance_i = (volume_bid_i - volume_ask_i) / (volume_bid_i + volume_ask_i)
    - cum_imbalance = sum of numerator and denominator across levels, then ratio
    Adds:
      - imbalance_1, ..., imbalance_N
      - cum_imbalance
    """
    imbalances = []
    numerators = []
    denominators = []

    for i in range(1, levels + 1):
        bid_col = f'volume_bid_{i}_sum'
        ask_col = f'volume_ask_{i}_sum'
        imbalance_col = f'imbalance_{i}'

        # Compute safe imbalance
        numerator = df[bid_col] - df[ask_col]
        denominator = df[bid_col] + df[ask_col]
        df[imbalance_col] = numerator / denominator.replace({0: np.nan})

        numerators.append(numerator)
        denominators.append(denominator)

        imbalances.append(imbalance_col)

    # Cumulative imbalance
    total_numerator = sum(numerators)
    total_denominator = sum(denominators).replace({0: np.nan})
    df['cum_imbalance'] = total_numerator / total_denominator

def compute_imbalance_momentum(df: pd.DataFrame, levels: int = 10,
                                short_window: int = 5, long_window: int = 20) -> pd.DataFrame:
    """
    Add momentum indicators (EMA, MACD) over order book imbalance features.

    For each imbalance_i and cum_imbalance:
        - Compute EMA over short and long windows
        - Compute MACD = EMA_short - EMA_long
        - Compute MACD signal line (EMA of MACD)
        - Compute MACD histogram = MACD - signal

    Resulting columns:
        - imbalance_{i}_macd
        - imbalance_{i}_macd_signal
        - imbalance_{i}_macd_hist
        - cum_imbalance_macd...
    """
    all_imbalance_cols = [f'imbalance_{i}' for i in range(1, levels + 1)] + ['cum_imbalance']

    for col in all_imbalance_cols:
        ema_short = df[col].ewm(span=short_window, adjust=False).mean()
        ema_long = df[col].ewm(span=long_window, adjust=False).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal

        df[f"{col}_macd"] = macd
        #df[f"{col}_macd_signal"] = signal
        #df[f"{col}_macd_hist"] = hist

    return df


def compute_basic_mid(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the basic mid-price using top-level ask/bid VWAP."""
    df['mid_price'] = (
        df['last_price_ask_1'] + df['last_price_bid_1']
    ) / 2.0
    return df

def compute_spreads_and_mids(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    for i in range(1, levels+1):
        df[f'spread_{i}'] = (
            df[f'last_price_ask_{i}'] - df[f'last_price_bid_{i}']
        )
        df[f'mid_{i}'] = (
            df[f'last_price_ask_{i}'] + df[f'last_price_bid_{i}']
        ) / 2.0
    return df
    

def compute_price_differences(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    df['ask_diff_n'] = df[f'last_price_ask_{levels}'] - df['last_price_ask_1']
    df['bid_diff_n'] = df[f'last_price_bid_{levels}'] - df['last_price_bid_1']
    for i in range(1, levels):
        df[f'ask_diff_{i}'] = (
            df[f'last_price_ask_{i+1}'] - df[f'last_price_ask_{i}']
        )
        df[f'bid_diff_{i}'] = (
            df[f'last_price_bid_{i+1}'] - df[f'last_price_bid_{i}']
        )
    return df
    
def compute_price_volume_means(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    ask_cols = [f'last_price_ask_{i}' for i in range(1, levels+1)]
    bid_cols = [f'last_price_bid_{i}' for i in range(1, levels+1)]
    ask_vol_cols = [f'volume_ask_{i}_sum' for i in range(1, levels+1)]
    bid_vol_cols = [f'volume_bid_{i}_sum' for i in range(1, levels+1)]
    df['avg_ask_price'] = df[ask_cols].mean(axis=1)
    df['avg_bid_price'] = df[bid_cols].mean(axis=1)
    df['avg_ask_volume'] = df[ask_vol_cols].mean(axis=1)
    df['avg_bid_volume'] = df[bid_vol_cols].mean(axis=1)
    return df

def compute_accumulated_differences(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    price_diff_matrix = (
        df[[f'last_price_ask_{i}' for i in range(1, levels+1)]].values
        - df[[f'last_price_bid_{i}' for i in range(1, levels+1)]].values
    )
    df['accum_price_diff'] = price_diff_matrix.sum(axis=1)
    vol_diff_matrix = (
        df[[f'volume_ask_{i}_sum' for i in range(1, levels+1)]].values
        - df[[f'volume_bid_{i}_sum' for i in range(1, levels+1)]].values
    )
    df['accum_volume_diff'] = vol_diff_matrix.sum(axis=1)
    return df


def compute_target(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """
    Example: compute 'm_plus' as the average future mid-price over k steps,
    then define a 'target1' as that relative return.
    """
    mid = df['mid_price'].values
    # --- replace only this block -----------------------------------------------
    df['m_plus'] = pd.Series(
        [mid[i + k] if i + k < len(mid) else np.nan   # value k steps ahead
         for i in range(len(mid))],
        index=df.index
    )
    
    # absolute difference (forward – current)
    df['target1'] = (df['m_plus'] - df['mid_price']) / df['mid_price']  # ← keep if you want the diff
    # df['target1'] = (df['m_plus'] - df['mid_price']) / df['mid_price']  # relative
    # ---------------------------------------------------------------------------
    return df

def compute_target_multi(df: pd.DataFrame, max_horizon: int = 10) -> pd.DataFrame:
    """
    Computes multiple target columns: target1, target2, ..., target{max_horizon},
    each defined as the relative change in mid_price over the next N time steps.

    Example:
        target3 = (mid[t+3] - mid[t]) / mid[t]
    """
    mid = df['mid_price'].values

    for h in range(1, max_horizon + 1):
        df[f'target{h}'] = pd.Series([
            (mid[i + h] - mid[i]) / mid[i] if i + h < len(mid) else np.nan
            for i in range(len(mid))
        ], index=df.index)

    return df

def compute_order_imbalance_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """
    Computes imbalance features for each level and aggregated across levels.
    - imbalance_i = (volume_bid_i - volume_ask_i) / (volume_bid_i + volume_ask_i)
    - cum_imbalance = sum of numerator and denominator across levels, then ratio
    Adds:
      - imbalance_1, ..., imbalance_N
      - cum_imbalance
    """
    imbalances = []
    numerators = []
    denominators = []

    for i in range(1, levels + 1):
        bid_col = f'volume_bid_{i}_sum'
        ask_col = f'volume_ask_{i}_sum'
        imbalance_col = f'imbalance_{i}'

        # Compute safe imbalance
        numerator = df[bid_col] - df[ask_col]
        denominator = df[bid_col] + df[ask_col]
        df[imbalance_col] = numerator / denominator.replace({0: np.nan})

        numerators.append(numerator)
        denominators.append(denominator)

        imbalances.append(imbalance_col)

    # Cumulative imbalance
    total_numerator = sum(numerators)
    total_denominator = sum(denominators).replace({0: np.nan})
    df['cum_imbalance'] = total_numerator / total_denominator

    return df

########################################
# Moving Averages & MACD / PPO Helpers
########################################

def compute_sma(df: pd.DataFrame, column: str, window: int) -> pd.DataFrame:
    """
    Compute Simple Moving Average (SMA) over 'window' periods 
    for the given 'column'. Adds new column: f"{column}_sma_{window}".
    """
    colname_sma = f"{column}_sma_{window}"
    df[colname_sma] = (
        df[column]
        .rolling(window=window, min_periods=1)
        .mean()
    )
    return df

def compute_ema(df: pd.DataFrame, column: str, span: int) -> pd.DataFrame:
    """
    Compute Exponential Moving Average (EMA) with 'span' for the given 'column'.
    Adds new column: f"{column}_ema_{span}".
    """
    colname_ema = f"{column}_ema_{span}"
    df[colname_ema] = (
        df[column]
        .ewm(span=span, adjust=False)
        .mean()
    )
    return df

def compute_macd(df: pd.DataFrame, column: str,
                 fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence/Divergence) on `column`.
    By default, uses the typical (12, 26, 9) periods.
    Adds: 
      column_macd, column_macd_signal, column_macd_hist
    """
    macd_col = f"{column}_macd"
    signal_col = f"{column}_macd_signal"
    hist_col = f"{column}_macd_hist"

    ema_fast = df[column].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = df[column].ewm(span=slowperiod, adjust=False).mean()
    df[macd_col] = ema_fast - ema_slow
    df[signal_col] = df[macd_col].ewm(span=signalperiod, adjust=False).mean()
    df[hist_col] = df[macd_col] - df[signal_col]

    return df

def compute_ppo(df: pd.DataFrame, column: str,
                fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> pd.DataFrame:
    """
    Compute PPO (Percentage Price Oscillator) on `column`.
    By default, uses (12, 26, 9).
    Adds:
      column_ppo, column_ppo_signal, column_ppo_hist
    """
    ppo_col = f"{column}_ppo"
    ppo_signal_col = f"{column}_ppo_signal"
    ppo_hist_col = f"{column}_ppo_hist"

    ema_fast = df[column].ewm(span=fastperiod, adjust=False).mean()
    ema_slow = df[column].ewm(span=slowperiod, adjust=False).mean()

    df[ppo_col] = (ema_fast - ema_slow) / ema_slow * 100.0
    df[ppo_signal_col] = df[ppo_col].ewm(span=signalperiod, adjust=False).mean()
    df[ppo_hist_col] = df[ppo_col] - df[ppo_signal_col]

    return df

def drop_last_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops all columns starting with 'vwap_ask_' or 'vwap_bid_'.
    """
    cols_to_drop = [col for col in df.columns if col.startswith('vwap_ask_') or col.startswith('vwap_bid_')]
    return df.drop(columns=cols_to_drop)


def rename_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns like 'vwap_ask_3' → 'price_ask_3_mean'
    and 'vwap_bid_7' → 'price_bid_7_mean'.
    """
    def rename(col: str) -> str:
        if re.match(r"vwap_ask_\d+", col):
            return col.replace("vwap_ask", "price_ask") + "_mean"
        elif re.match(r"vwap_bid_\d+", col):
            return col.replace("vwap_bid", "price_bid") + "_mean"
        else:
            return col

    df = df.rename(columns={col: rename(col) for col in df.columns})
    return df

########################################
# Main Transform Function
########################################
import numpy as np
import pandas as pd
from typing import Tuple

class LOBLoader:
    def __init__(self, T: int = 100, k: int = 10):
        self.T = T
        self.k = k

    def transform_orderbook(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy upfront (good practice so you don't mutate the original)
        #df = rename_price_columns(df)
        #df = drop_last_price_columns(df)

        ########################################
        # 4) Target computation
        ########################################

        df = compute_basic_mid(df)
        df = df.copy()  # Force defrag
        
        df = compute_target(df, k=self.k)
        df = df.copy()
        
        # Define base and LOB column names
        base_cols = ['datetime']
        lob_cols = [
            f'{prefix}_{i}' for i in range(1, 11)
            for prefix in ['vwap_ask', 'volume_ask', 'vwap_bid', 'volume_bid', 'last_price_ask', 'last_price_bid']
        ]
        # Append "_sum" to volume columns only
        lob_cols = [f"{col}_sum" if "volume" in col else col for col in lob_cols]
        lob_cols = lob_cols + ['mid_price', 'target1']
        # Identify the columns that exist in the dataframe
        existing_lob_cols = [col for col in lob_cols if col in df.columns]
        keep_cols = base_cols + list(set(existing_lob_cols))  # remove duplicates, preserve datetime
        
        # Drop rows where any LOB prerequisite columns are NaN
        df = df.dropna(subset=existing_lob_cols)
        
        # Retain only the desired columns
        df = df[keep_cols]
    
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True, errors='coerce')
    
        # Replace zeros with NaN; forward-fill to handle missing
        for i in range(1, 11):
            for col in (f'vwap_ask_{i}', f'volume_ask_{i}_sum',
                        f'vwap_bid_{i}', f'volume_bid_{i}_sum', f'last_price_ask_{i}', f'last_price_bid_{i}'):
                df[col] = df[col].replace(0, np.nan).ffill()
    
        # Sort by time
        df.set_index('datetime', inplace=True)
        df.sort_index(inplace=True)
    
        ########################################
        # 1) Feature computations
        ########################################
    
        df = compute_spreads_and_mids(df, levels=10)
        df = df.copy()
    
        df = compute_price_differences(df, levels=10)
        df = df.copy()
    
        df = compute_price_volume_means(df, levels=10)
        df = df.copy()
    
        df = compute_accumulated_differences(df, levels=10)
        df = df.copy()

        df = compute_order_imbalance_features(df, levels=10)
        df = df.copy()
        
        df = compute_imbalance_momentum(df, levels=10)
        df = df.copy()
        
        ########################################
        # 2) Moving averages & MACD/PPO
        ########################################
    
        df = compute_sma(df, "mid_price", window=5)
        df = df.copy()
    
        df = compute_ema(df, "mid_price", span=12)
        df = df.copy()
    
        df = compute_macd(df, "mid_price", fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.copy()
    
        df = compute_ppo(df, "mid_price", fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.copy()
    
        #df = compute_sma(df, "vwap_bid_1", window=5)
        #df = df.copy()
    
        df = compute_macd(df, "vwap_ask_1", fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.copy()

        df = compute_macd(df, "vwap_bid_1", fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.copy()

        
        ########################################
        # 3) Rolling z-score
        ########################################
        all_cols = list(df.columns)
        exclude_cols = ["m_plus", "target1", "ticker", "mid_price"]
        columns_to_zscore = [c for c in all_cols if c not in exclude_cols]
    
        df = compute_rolling_zscore(df, columns=columns_to_zscore, horizon="30min")
        df = df.copy()

        #df = compute_rolling_zscore(df, columns=columns_to_zscore, window="2H")
        #df = df.copy()
        
        ########################################
        # 5) Final housekeeping
        ########################################
        df['ticker'] = ticker
        df.reset_index(inplace=True)
        #df.set_index(['ticker', 'datetime'], inplace=True)
    
        # Final copy to ensure a contiguous layout
        return df.copy()


    def extract_features_and_labels(self, df: pd.DataFrame, stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features and labels from the DataFrame using all columns ending with 'zscore'
        as input features. Labels are taken from the 'target1' column.
    
        :param df: DataFrame with time-series data.
        :param stride: Step size for sliding window.
        :return: Tuple (X, y) with shapes (num_sequences, T, num_features) and (num_sequences,)
        """
        # Automatically detect all columns ending in 'zscore'
        ordered_cols = [col for col in df.columns if col.endswith('zscore')]
    
        X = df[ordered_cols].values.astype(np.float32)
        y = df['target1'].values.astype(np.float32)
    
        N, D = X.shape
        end_indices = range(self.T, N + 1, stride)
        M = len(end_indices)
    
        dataX = np.zeros((M, self.T, D), dtype=np.float32)
        dataY = np.zeros((M,), dtype=np.float32)
    
        for i, end in enumerate(end_indices):
            start = end - self.T
            dataX[i] = X[start:end]
            dataY[i] = y[end - 1]
    
        mask = ~np.isnan(dataX).any(axis=(1, 2))
        return dataX[mask], np.nan_to_num(dataY[mask], nan=0.0)


    def extract_features_and_labels(self, df: pd.DataFrame, stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scans the DataFrame for all columns ending with '_zscore' and uses them as input features.
        Targets are taken from 'target1' as before.
        Applies the same windowing logic as extract_features_and_labels_simple.
        """
        # Gather all columns ending with '_zscore'
        ordered_cols = [col for col in df.columns if col.endswith('_zscore')]

        df = df.dropna(subset=ordered_cols).reset_index(drop=True)
        
        # Convert to float32 arrays
        X = df[ordered_cols].values.astype(np.float32)
        y = df['target1'].values.astype(np.float32)
    
        N, D = X.shape
        end_indices = range(self.T, N + 1, stride)
        M = len(end_indices)
    
        dataX = np.zeros((M, self.T, D), dtype=np.float32)
        dataY = np.zeros((M,), dtype=np.float32)
    
        # Windowing the time series
        for i, end in enumerate(end_indices):
            start = end - self.T
            dataX[i] = X[start:end]
            dataY[i] = y[end - 1]
    
        # Filter out windows that contain NaNs
        mask = ~np.isnan(dataX).any(axis=(1, 2))
        # Replace any remaining NaNs in the labels with 0.0
        return dataX[mask], np.nan_to_num(dataY[mask], nan=0.0)
        #return dataX, dataY
        
    def drop_last_price_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops all columns starting with 'vwap_ask_' or 'vwap_bid_'.
        """
        cols_to_drop = [col for col in df.columns if col.startswith('vwap_ask_') or col.startswith('vwap_bid_')]
        return df.drop(columns=cols_to_drop)
    

    def rename_price_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns like 'vwap_ask_3' → 'price_ask_3_mean'
        and 'vwap_bid_7' → 'price_bid_7_mean'.
        """
        def rename(col: str) -> str:
            if re.match(r"vwap_ask_\d+", col):
                return col.replace("vwap_ask", "price_ask") + "_mean"
            elif re.match(r"vwap_bid_\d+", col):
                return col.replace("vwap_bid", "price_bid") + "_mean"
            else:
                return col
    
        df = df.rename(columns={col: rename(col) for col in df.columns})
        return df


    def extract_features_and_labels_simple(self, df: pd.DataFrame, stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        ordered_cols = []
        for i in range(1, 11):
            ordered_cols.extend([
                f'vwap_ask_{i}_zscore',
                f'volume_ask_{i}_sum_zscore',
                f'vwap_bid_{i}_zscore',
                f'volume_bid_{i}_sum_zscore'
            ])

        X = df[ordered_cols].values.astype(np.float32)
        y = df['target1'].values.astype(np.float32)

        N, D = X.shape
        end_indices = range(self.T, N + 1, stride)
        M = len(end_indices)

        dataX = np.zeros((M, self.T, D), dtype=np.float32)
        dataY = np.zeros((M,), dtype=np.float32)

        for i, end in enumerate(end_indices):
            start = end - self.T
            dataX[i] = X[start:end]
            dataY[i] = y[end - 1]

        mask = ~np.isnan(dataX).any(axis=(1, 2))
        return dataX[mask], np.nan_to_num(dataY[mask], nan=0.0)
